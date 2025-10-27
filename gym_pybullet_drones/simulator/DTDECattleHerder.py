"""
-------
Setup:

    pip install ray[rllib]
    pip install requests
    pip install tensorboard

Run training:

    $ conda activate drones
    $ python DTDECattleHerder.py

Monitor logs:

    $ tensorboard --logdir ~/ray_results/DTDECattleHerder
-------
"""

import os
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune import CheckpointConfig
from ray.tune.registry import register_env
from ray.tune.logger import TBXLoggerCallback
from gym_pybullet_drones.utils.enums import DroneModel

from gym_pybullet_drones.rllib_envs.MARLCattleAviary import MARLCattleAviary
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ---------------- DEFAULTS ---------------- #
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "ray_results"
DEFAULT_OBS = ObservationType("cokin")
DEFAULT_ACT = ActionType("vel")
DEFAULT_DRONES = 12
DEFAULT_CATTLE = 16
MAX_TIMESTEPS = 500_000
NUM_EVAL_EPISODES = 2


# ---------------- TRAINING ---------------- #
def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, local=True):
    ray.init(ignore_reinit_error=True)

    # Environment config
    env_config = {
        "drone_model": DroneModel.CF2X,
        "num_drones": DEFAULT_DRONES,
        "num_cattle": DEFAULT_CATTLE,
        "obs": DEFAULT_OBS,
        "act": DEFAULT_ACT,
        "gui": gui,
        "record": record_video,
    }

    # ✅ Register the environment so RLlib knows how to instantiate it
    def env_creator(config):
        return MARLCattleAviary(**config)

    register_env("MARLCattleAviary", env_creator)

    # RLlib PPO configuration (modern API)
    config = (
        PPOConfig()
        .environment(env="MARLCattleAviary", env_config=env_config)
        .framework("torch")
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",   # RLlib will compute this automatically
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=2048,
            vf_loss_coeff=0.7,
            entropy_coeff=0.1,
            clip_param=0.1,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=NUM_EVAL_EPISODES,
            evaluation_config={"gui": False, "record": False},
        )
    )

    storage_path = os.path.abspath(output_folder)

    # Logging + checkpoint setup
    tb_logger = TBXLoggerCallback()
    tuner = tune.Tuner(
        PPO,
        run_config=tune.RunConfig(
            name="DTDECattleHerder",
            storage_path=storage_path,                 # Absolute path to output folder
            stop={"timesteps_total": MAX_TIMESTEPS},
            checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
            callbacks=[tb_logger],
        ),
        param_space=config.to_dict(),
    )

    results = tuner.fit()
    print("[LOG] Training finished")

    ray.shutdown()


# ---------------- CLI ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-drone DTDE cattle herding with RLlib PPO")
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool)
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VIDEO, type=str2bool)
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str)
    ARGS = parser.parse_args()

    run(**vars(ARGS))
