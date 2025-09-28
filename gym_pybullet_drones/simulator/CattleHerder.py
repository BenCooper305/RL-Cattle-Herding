"""
-------
Setup:

    pip install tensorboard

Run training:

    $ conda activate drones
    $ python CattleHerder.py

Monitor logs:

    $ tensorboard --logdir models/model-v9-0/tb

    tensorboard --logdir /home/ben/ros_ws/src/RL-Cattle-Herding/gym_pybullet_drones/simulator/models/model-v3-1
-------
"""

import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.CattleAviary import CattleAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'models'
DEFAULT_COLAB = False
TARGET_REWARD = 99999   # reward threshold to stop training
LOAD_FILE = "model-v12-2-3/best_model.zip" #"model-v11-1-1/best_model.zip" #model-v10-2-4/best_model.zip"        # e.g., "model-v6-2/best_model.zip" or None

DEFAULT_OBS = ObservationType('cokin') # collaborative kinematics
DEFAULT_ACT = ActionType('vel')        # 'rpm' | 'pid' | 'vel' | 'one_d_rpm' | 'one_d_pid'
DEFAULT_NUM_ENVS = 20
DEFAULT_DRONES = 4
DEFAULT_CATTLE = 16

MAX_TIMESTEPS = 500000
EVAL_FILE = "model-v12-2-2"
EVALUATION_FREQUENCY = 2048

EVALUATE_ONLY = True  # skip training, run evaluation only
NUM_EVALUTION_EPS = 2

#Main Runner
def run(
        target_reward=TARGET_REWARD,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, 
        plot=False, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        local=True):

    # Unique folder for this training run
    model_dir = os.path.join(output_folder, 'model-v12-2-4')
    os.makedirs(model_dir, exist_ok=True)


    # Environment setup
    env_kwargs = dict(
        num_drones=DEFAULT_DRONES,
        num_cattle=DEFAULT_CATTLE,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT
    )

    train_env = make_vec_env(
        CattleAviary, 
        env_kwargs=env_kwargs,
        n_envs=DEFAULT_NUM_ENVS, 
        vec_env_cls=SubprocVecEnv, 
        seed=0
    )

    eval_env = DummyVecEnv([lambda: Monitor(CattleAviary(**env_kwargs))])

    # Load existing model or create new one
    if LOAD_FILE is not None and os.path.exists(os.path.join(output_folder, LOAD_FILE)):
        model_path = os.path.join(output_folder, LOAD_FILE)
        model = PPO.load(model_path, env=train_env)
        print(f"[LOG] Loaded existing PPO model from {model_path}")
    else:
        model = PPO(
            "MlpPolicy", 
            train_env,
            learning_rate=3e-5,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.1,
            vf_coef=0.7,
            max_grad_norm=0.5,
            tensorboard_log=os.path.join(model_dir, "tb"),
            policy_kwargs=dict(
                log_std_init=-1.0,   # lower action variance initially
                ortho_init=False,
                net_arch=[dict(pi=[128, 128], vf=[128, 128])]
            ),
            verbose=1
        )
        print("[LOG] Created new PPO model")

    # Configure logger
    new_logger = configure(os.path.join(model_dir, "tb"), ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Training
    if not EVALUATE_ONLY:
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=target_reward, verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            verbose=1,
            best_model_save_path=model_dir,
            log_path=model_dir,
            eval_freq=EVALUATION_FREQUENCY,
            deterministic=True,
            render=False
        )

        model.learn(total_timesteps=MAX_TIMESTEPS, callback=eval_callback, log_interval=100)
        model.save(os.path.join(model_dir, "final_model.zip"))
        print(f"[LOG] Training finished. Model saved in {model_dir}")

        # Load best model for evaluation
        best_model_path = os.path.join(model_dir, "best_model.zip")
    else:
         model_dir = os.path.join(output_folder, EVAL_FILE)
         best_model_path = os.path.join(EVAL_FILE, "best_model.zip")

    if os.path.isfile(best_model_path):
        model = PPO.load(best_model_path)
        print(f"[LOG] Loaded best model from {best_model_path}")
    else:
        print(f"[ERROR] No best model found at {best_model_path}, using final_model.zip")
        model = PPO.load(os.path.join(model_dir, "final_model.zip"))

    # Evaluation environments
    test_env = CattleAviary(
        gui=gui,
        num_drones=DEFAULT_DRONES,
        num_cattle=DEFAULT_CATTLE,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        record=record_video
    )
    test_env_nogui = CattleAviary(
        num_drones=DEFAULT_DRONES,
        num_cattle=DEFAULT_CATTLE,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT
    )
    test_env_nogui.is_evaluating = True

    # Evaluate trained policy
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=NUM_EVALUTION_EPS)
    print(f"\n[RESULT] Mean reward {mean_reward:.2f} ± {std_reward:.2f}\n")
    test_env_nogui.evaluation_save()

    if local:
        input("Press Enter to continue...")

    # Rollout for logging
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_DRONES, 
                    output_folder=output_folder,
                    colab=colab)
    
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"reward recieved: {reward}, terminated: {terminated}, trunacted: {truncated}")

        obs2 = np.atleast_2d(obs)
        act2 = np.atleast_2d(action)

        for d in range(DEFAULT_DRONES):
            logger.log(
                drone=d,
                timestamp=i / test_env.CTRL_FREQ,
                state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                control=np.zeros(12)
            )

        #test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated or truncated:
            obs, _ = test_env.reset(seed=42, options={})
            
    test_env.close()
    if plot:
        logger.plot()


# ---------------- CLI ---------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-drone cattle herding with PPO")
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help="Enable PyBullet GUI (default: True)")
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help="Record a video (default: False)")
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Output folder for logs/models')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help="Notebook mode (default: False)")
    ARGS = parser.parse_args()

    run(**vars(ARGS))
