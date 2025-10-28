#!/usr/bin/env python3
import os
import argparse
import ray
import numpy as np
import torch
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from gym_pybullet_drones.rllib_envs.marl_wrapper import RLlibMultiAgentWrapper
from gym_pybullet_drones.rllib_envs.MARLCattleAviary import MARLCattleAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import str2bool

def make_env_creator(base_env_config):
    def _create(env_context=None):
        cfg = dict(base_env_config or {})
        if env_context is not None:
            cfg.update(env_context.get("env_config", {}))
        return RLlibMultiAgentWrapper(cfg)
    return _create

def find_latest_checkpoint(base_dir="ray_results"):
    candidates = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.startswith("checkpoint_"):
                candidates.append(os.path.join(root, d))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def evaluate(checkpoint_path, num_episodes=3, gui=True, num_drones=6, num_cattle=8, shared_policy=True):
    ray.init(ignore_reinit_error=True)

    env_config = {
        "num_drones": num_drones,
        "num_cattle": num_cattle,
        "obs": ObservationType("cokin"),
        "act": ActionType("vel"),
        "gui": gui,
        "record": False,
    }

    register_env("marl_cattle_env", make_env_creator(env_config))
    print(f"\n✅ Restoring from checkpoint: {checkpoint_path}")

    # --- Build the algorithm with new API ---
    test_env = MARLCattleAviary(**env_config)
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    del test_env  # free memory

    policies = {
        "shared_policy": (None, obs_space, act_space, {})
    } if shared_policy else {
        f"policy_{i}": (None, obs_space, act_space, {}) for i in range(num_drones)
    }
    policy_mapping_fn = (lambda aid, *args: "shared_policy") if shared_policy else (lambda aid, *args: f"policy_{int(aid.split('_')[1])}")

    config = (
        PPOConfig()
        .environment(env="marl_cattle_env", env_config=env_config)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    algo = config.build()
    algo.restore(checkpoint_path)

    # Create environment for playback
    env = MARLCattleAviary(**env_config)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = {aid: False for aid in obs.keys()}
        done["__all__"] = False
        ep_reward = 0.0
        step = 0
        print(f"\n🎬 Starting episode {ep+1}")

        while not done["__all__"]:
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id)
                # Wrap observation into dict as expected by RLlib
                obs_dict = {"obs": agent_obs}
                # compute action(s) for this obs dict
                # Prefer new RLlib API: use Algorithm.get_module().forward_inference
                action = None
                try:
                    module = algo.get_module()
                    # flatten multi-agent obs dict into single vector the RLModule expects
                    import numpy as _np
                    def _agent_index(k):
                        try:
                            return int(k)  # numeric keys
                        except Exception:
                            # agent_0 or similar
                            s = str(k)
                            if "agent_" in s:
                                return int(s.split("agent_")[-1])
                            # fallback: try to parse trailing digits
                            import re
                            m = re.search(r"(\d+)$", s)
                            return int(m.group(1)) if m else 0

                    keys = sorted(list(obs_dict.keys()), key=_agent_index)
                    parts = [_np.asarray(obs_dict[k]).ravel() for k in keys]
                    flat = _np.concatenate(parts, axis=0).astype(_np.float32)
                    batch_obs = _np.expand_dims(flat, 0)
                    out = module.forward_inference({"obs": batch_obs})
                    # extract actions from common output keys
                    if isinstance(out, dict):
                        for k in ("actions", "action", "actions_sampled", "outputs"):
                            if k in out:
                                action_batch = out[k]; break
                        else:
                            # fallback: first array-like value
                            action_batch = next((v for v in out.values() if isinstance(v, (_np.ndarray, list, tuple))), None)
                    elif isinstance(out, (list, tuple)):
                        action_batch = out[0] if len(out) > 0 else None
                    else:
                        action_batch = None
                    if action_batch is None:
                        raise RuntimeError(f"RLModule.forward_inference returned unexpected output keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")
                    action = action_batch[0] if isinstance(action_batch, (list, tuple, _np.ndarray)) else action_batch
                except Exception:
                    # Fallback to older RLlib APIs using local_worker/policy
                    try:
                        # obtain local worker again safely
                        lw = None
                        if hasattr(algo, "workers"):
                            w = getattr(algo, "workers")
                            lw = w.local_worker() if callable(getattr(w, "local_worker", None)) else getattr(w, "local_worker", None)
                        if lw is None and callable(getattr(algo, "local_worker", None)):
                            lw = algo.local_worker()
                        if lw is None:
                            raise RuntimeError("no local worker available for fallback inference")
                        # if the policy object supports compute_single_action_from_input_dict, use it
                        pol = lw.for_policy(policy_id) if hasattr(lw, "for_policy") else lw.get_policy(policy_id) if hasattr(lw, "get_policy") else None
                        if pol is None:
                            raise RuntimeError("no policy object on local worker for fallback inference")
                        if hasattr(pol, "compute_single_action_from_input_dict"):
                            action = pol.compute_single_action_from_input_dict(obs_dict)
                        elif hasattr(pol, "compute_single_action"):
                            # try converting dict -> array as above
                            import numpy as _np
                            keys = sorted(list(obs_dict.keys()), key=_agent_index)
                            parts = [_np.asarray(obs_dict[k]).ravel() for k in keys]
                            flat = _np.concatenate(parts, axis=0).astype(_np.float32)
                            action = pol.compute_single_action(flat)
                        else:
                            raise RuntimeError("policy has no compatible compute method")
                    except Exception as e:
                        print("Inference failed (both RLModule and local_worker fallback):", e)
                        raise
                a = action
                # apply actions to env (assumes actions for all drones)
                env.step(a)
            obs, rewards, dones, truncs, infos = env.step(actions)
            ep_reward += sum(float(r) for r in rewards.values())
            done = dones
            step += 1

        print(f"Episode {ep+1} finished — steps={step}, total reward={ep_reward:.3f}")

    env.close()
    algo.stop()
    ray.shutdown()
    print("\n✅ Playback finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--gui", default=False, type=str2bool)
    parser.add_argument("--episodes", default=3, type=int)
    parser.add_argument("--num_drones", default=6, type=int)
    parser.add_argument("--num_cattle", default=8, type=int)
    parser.add_argument("--shared_policy", default=True, type=str2bool)
    args = parser.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint("ray_results")
    if ckpt is None:
        raise SystemExit("No checkpoint found under ray_results/")
    evaluate(ckpt, num_episodes=args.episodes, gui=args.gui,
             num_drones=args.num_drones, num_cattle=args.num_cattle,
             shared_policy=args.shared_policy)
