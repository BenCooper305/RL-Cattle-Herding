# python -c '...'
import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPO
from gym_pybullet_drones.rllib_envs.MARLCattleAviary import MARLCattleAviary
import numpy as np

ray.init(ignore_reinit_error=True)

# register your env (must run before creating the Algorithm)
register_env("marl_cattle_env", lambda cfg: MARLCattleAviary(**(cfg.get("env_config", {}) if hasattr(cfg, "get") else cfg)))

env_cfg = {"num_drones":6, "num_cattle":8, "gui":False}
algo = PPO(env="marl_cattle_env", config={"env_config": env_cfg, "num_workers": 0})

ckpt = "/home/ben/ros_ws/src/RL-Cattle-Herding/gym_pybullet_drones/simulator/ray_results/DTDE_MARL_Cattle_TEST/PPO_marl_cattle_env_b579c_00000_0_2025-10-27_23-16-04/checkpoint_000000"
algo.restore(ckpt)

print("state keys:", list(algo.get_state().keys()))
try:
    print("weights keys (top):", list(algo.get_weights().keys())[:10])
except Exception as e:
    print("get_weights failed:", e)

weights = algo.get_weights()                     # returns dict of policies
policy_weights = weights['default_policy']       # or 'shared_policy' in other checkpoints
import pickle
with open("policy_weights.pkl", "wb") as fh:
    pickle.dump(policy_weights, fh)
print("saved policy keys:", list(policy_weights.keys())[:10])

from gym_pybullet_drones.rllib_envs.MARLCattleAviary import MARLCattleAviary
env = MARLCattleAviary(num_drones=6, num_cattle=8, gui=False)
obs, info = env.reset(seed=0, options={})

# DEBUG: inspect obs structure and a small sample, then stop (paste output here)
import pprint, sys
print("DEBUG obs type:", type(obs))
if isinstance(obs, dict):
    print("DEBUG obs keys:", list(obs.keys())[:20])
    # show one key's type and a small sample
    k = list(obs.keys())[0] if len(obs) > 0 else None
    if k is not None:
        print("DEBUG sample key:", k, "type:", type(obs[k]))
        pprint.pprint(str(obs[k])[:1000])
else:
    print("DEBUG obs repr (truncated):", str(obs)[:1000])
# exit so we don't hit forward_inference yet
sys.exit(0)

# Use the RLModule forward_inference API (avoids env_runner.get_policy)
module = algo.get_module()  # RLModule instance
if isinstance(obs, dict):
    batch_obs = {k: np.expand_dims(np.asarray(v), 0) for k, v in obs.items()}
else:
    batch_obs = np.expand_dims(np.asarray(obs), 0)

inputs = {"obs": batch_obs}
out = module.forward_inference(inputs)
print("module.forward_inference keys:", list(out.keys()) if isinstance(out, dict) else type(out))

# extract actions from common keys or fallback to the first array-like value
action_batch = None
if isinstance(out, dict):
    for k in ("actions", "action", "actions_sampled", "outputs", "output"):
        if k in out:
            action_batch = out[k]
            break
    if action_batch is None:
        for v in out.values():
            if isinstance(v, (list, tuple, np.ndarray)):
                action_batch = v
                break
elif isinstance(out, (list, tuple)):
    action_batch = out[0]

if action_batch is None:
    raise RuntimeError("Could not find actions in module.forward_inference output. See printed keys above.")

action = action_batch[0] if isinstance(action_batch, (list, tuple, np.ndarray)) else action_batch
next_obs, reward, terminated, truncated, info = env.step(action)
print("reward", reward, "terminated", terminated, "truncated", truncated)

env.close()

algo.stop()
ray.shutdown()