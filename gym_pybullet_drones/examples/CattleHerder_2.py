import os
import time
import torch
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.CattleAviary import CattleAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ----------------- DEFAULT CONFIG -----------------
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'models'
DEFAULT_COLAB = False
TARGET_REWARD = 99999
LOAD_FILE = 'no'  # path to pre-trained model, if any

DEFAULT_OBS = ObservationType('cokin')
DEFAULT_ACT = ActionType('vel')
DEFAULT_DRONES = 10
DEFAULT_CATTLE = 8

MAX_TIMESTEPS = 100_000
EVALUATION_FREQUENCY = 2048  # PPO agent steps

# ----------------- RUN FUNCTION -----------------
def run(target_reward=TARGET_REWARD,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        plot=False,
        colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO,
        local=True):
    
    filename = os.path.join(output_folder, 'model-v5-4')
    os.makedirs(filename, exist_ok=True)

    # ----------------- CREATE ENVIRONMENTS -----------------
    env_kwargs = dict(num_drones=DEFAULT_DRONES, num_cattel=DEFAULT_CATTLE, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    train_env = make_vec_env(CattleAviary, env_kwargs=env_kwargs, n_envs=1, seed=0)
    eval_env = DummyVecEnv([lambda: Monitor(CattleAviary(**env_kwargs))])

    # ----------------- LOAD OR CREATE PPO MODEL -----------------
    load_path = os.path.join('models', LOAD_FILE)
    if os.path.exists(load_path):
        model = PPO.load(load_path, env=train_env)
        print("[LOG] Loaded PPO model from:", load_path)
    else:
        model = PPO(
            'MlpPolicy',
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
            tensorboard_log=os.path.join(filename, 'tb'),
            policy_kwargs=dict(
                log_std_init=-1.0,
                ortho_init=False,
                net_arch=[dict(pi=[128, 128], vf=[128, 128])]
            ),
            verbose=1
        )
        print("[LOG] Created new PPO model")

    # ----------------- SETUP LOGGER -----------------
    new_logger = configure(os.path.join(filename, 'tb'), ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # ----------------- CALLBACKS -----------------
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=filename,
        log_path=filename,
        eval_freq=EVALUATION_FREQUENCY,
        deterministic=True,
        render=False,
        verbose=1
    )

    # ----------------- TRAIN -----------------
    model.learn(total_timesteps=MAX_TIMESTEPS, callback=eval_callback, log_interval=100)
    model.save(os.path.join(filename, 'final_model.zip'))
    print("[LOG] Training complete. Model saved at", filename)

    # ----------------- EVALUATE -----------------
    test_env = CattleAviary(gui=gui,
                            num_drones=DEFAULT_DRONES,
                            num_cattel=DEFAULT_CATTLE,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            record=record_video)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_DRONES,
                    output_folder=output_folder,
                    colab=colab)

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for t in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        # ----------------- DECENTRALISED ACTIONS -----------------
        actions = []
        for d in range(test_env.NUM_DRONES):
            obs_d = obs[d]
            action, _ = model.predict(obs_d, deterministic=True)
            actions.append(action)
        actions = np.array(actions)

        obs, reward, terminated, truncated, info = test_env.step(actions)

        # ----------------- LOGGING -----------------
        obs2 = np.atleast_2d(obs)
        act2 = np.atleast_2d(actions)
        for d in range(test_env.NUM_DRONES):
            logger.log(drone=d,
                       timestamp=t / test_env.CTRL_FREQ,
                       state=np.hstack([obs2[d][0:3],
                                        np.zeros(4),
                                        obs2[d][3:15],
                                        act2[d]
                                        ]),
                       control=np.zeros(12)
                       )

        test_env.render()
        sync(t, start, test_env.CTRL_TIMESTEP)

        if terminated.any():
            obs = test_env.reset(seed=42, options={})

    test_env.close()
    if plot:
        logger.plot()

    # ----------------- EVALUATE POLICY -----------------
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=1)
    print(f"\nMean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")


# ----------------- MAIN -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decentralised multi-drone RL script')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Use PyBullet GUI')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Record video')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Output folder')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Run in Colab')
    args = parser.parse_args()

    run(**vars(args))
