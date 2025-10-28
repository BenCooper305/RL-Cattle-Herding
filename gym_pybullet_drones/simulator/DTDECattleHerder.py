# DTDECattleHerder_rllib_multi_fixed.py
import os
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune import register_env
from ray.tune.logger import TBXLoggerCallback
from ray.tune import Tuner, RunConfig, CheckpointConfig

from gym_pybullet_drones.rllib_envs.marl_wrapper import RLlibMultiAgentWrapper
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# defaults
DEFAULT_GUI = False
DEFAULT_OUTPUT_FOLDER = "ray_results"
DEFAULT_DRONES = 6   # if using per-agent distinct policies, must be fixed
DEFAULT_CATTLE = 8
DEFAULT_NUM_ENVS = 4
MAX_TIMESTEPS = 1_000
NUM_EVAL_EPISODES = 5


# Choose policy mode:
# True = one shared policy for all agents (recommended)
# False = distinct policy per agent (heavier, requires fixed NUM_DRONES)
USE_SHARED_POLICY = True


def make_env_creator(env_config):
    """Returns a callable that RLlib uses to create env instances."""
    def _create(env_context):
        cfg = env_config.copy() if env_config is not None else {}
        # Allow overrides from RLlib EnvContext
        cfg.update(env_context.get("env_config", {}))
        return RLlibMultiAgentWrapper(cfg)
    return _create


def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, num_drones=DEFAULT_DRONES, num_cattle=DEFAULT_CATTLE):
    ray.init(ignore_reinit_error=True)

    # ------------------ ENV CONFIG ------------------ #
    env_config = {
        "num_drones": num_drones,
        "num_cattle": num_cattle,
        "obs": ObservationType("cokin"),
        "act": ActionType("vel"),
        "gui": gui,
        "record": False,
    }

    register_env("marl_cattle_env", make_env_creator(env_config))

    # ------------------ POLICIES ------------------ #
    test_env = RLlibMultiAgentWrapper(env_config)
    obs_space = test_env.env.observation_space
    act_space = test_env.env.action_space
    del test_env  # free memory

    if USE_SHARED_POLICY:
        # one shared policy for all agents
        policies = {
            "shared_policy": (None, obs_space, act_space, {})
        }
        policy_mapping_fn = lambda agent_id, *args: "shared_policy"
    else:
        # distinct policy per agent
        policies = {
            f"policy_{i}": (None, obs_space, act_space, {})
            for i in range(num_drones)
        }
        policy_mapping_fn = lambda agent_id, *args: f"policy_{int(agent_id.split('_')[1])}"

    # ------------------ RLlib PPO CONFIG ------------------ #
    config = (
        PPOConfig()
        .environment(env="marl_cattle_env", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=DEFAULT_NUM_ENVS, num_envs_per_env_runner=1, rollout_fragment_length="auto")
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=4096,
            num_sgd_iter=10,
            vf_loss_coeff=0.7,
            entropy_coeff=0.01,
            clip_param=0.1
        )
        .evaluation(
            evaluation_interval=0,
            evaluation_duration=NUM_EVAL_EPISODES,
            evaluation_config={"gui": False, "record": False}
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
    )

    # ------------------ LOGGING / TUNER ------------------ #
    storage_path = os.path.abspath(output_folder)
    tb_logger = TBXLoggerCallback()

    tuner = tune.Tuner(
        PPO,
        run_config=RunConfig(
            name="DTDE_MARL_Cattle_TEST",
            storage_path=storage_path,
            #stop={"timesteps_total": MAX_TIMESTEPS},
            stop={"training_iteration": 500},
            checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
            callbacks=[tb_logger],
        ),
        param_space=config.to_dict(),
    )

    # ------------------ RUN ------------------ #
    results = tuner.fit()
    print("Training finished")
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool)
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument("--num_drones", default=DEFAULT_DRONES, type=int)
    parser.add_argument("--num_cattle", default=DEFAULT_CATTLE, type=int)
    parser.add_argument("--use_shared_policy", default=True, type=str2bool)
    ARGS = parser.parse_args()
    USE_SHARED_POLICY = ARGS.use_shared_policy
    run(output_folder=ARGS.output_folder, gui=ARGS.gui, num_drones=ARGS.num_drones, num_cattle=ARGS.num_cattle)
