"""
-------
pip install tensorboard

In a terminal, run as:

    $ conda activate drones
    $ python CattleHerder.py
"""
import os
import time
import torch
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.CattleAviary import CattleAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'models'
DEFAULT_COLAB = False
TARGET_REWARD = 8800
LOAD_FILE = 'no' #'save-08.18.2025_19.11.25/best_model.zip'

DEFAULT_OBS = ObservationType('cokin') #collabrative kinematicsa
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_DRONES = 4
DEFAULT_CATTLE = 3

MAX_TIMESTEPS = 10000
EVALUATION_FREQUENCY = 1000

EVALUTE_ONLY = False

def run(
        target_reward = TARGET_REWARD,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, 
        plot=False, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    env_kwargs = dict(num_drones=DEFAULT_DRONES, num_cattel=DEFAULT_CATTLE, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    # Training environment
    train_env = make_vec_env(CattleAviary, env_kwargs=env_kwargs, n_envs=1, seed=0)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(CattleAviary(**env_kwargs))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)


    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Create the model #######################################
    loadfile = 'models/'+LOAD_FILE
    print(loadfile)
    if os.path.exists(loadfile):
        model = PPO.load(loadfile, env=train_env)
        print("[LOG] PPO Model Loaded")
    else:
        model = PPO('MlpPolicy', 
                    train_env,
                    learning_rate=1e-4,
                    n_steps=4096,
                    batch_size=1024,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.1,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    tensorboard_log=filename+'/tb/',
                    verbose=1)
        print("[LOG] Created New Model")

    new_logger = configure(filename+'/tb/', ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    #### Train the model #######################################
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(EVALUATION_FREQUENCY),
                                 deterministic=True,
                                 render=False)
    
    model.learn(total_timesteps=int(MAX_TIMESTEPS), callback=eval_callback, log_interval=100)

    #### Save the model ########################################
    train_env.save("vecnormalize_stats.pkl")
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################

    if local:
        input("Press Enter to continue...")

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    # For recording / GUI test
    test_env = CattleAviary(gui=gui,
                            num_drones=DEFAULT_DRONES,
                            num_cattel=DEFAULT_CATTLE,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            record=record_video)

    test_env_nogui = CattleAviary(num_drones=DEFAULT_DRONES,
                                num_cattel=DEFAULT_CATTLE,
                                obs=DEFAULT_OBS,
                                act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_DRONES, 
                    output_folder=output_folder,
                    colab=colab
                    )

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = np.atleast_2d(obs) #.squeeze()
        act2 = np.atleast_2d(action)
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

        for d in range(DEFAULT_DRONES):
            logger.log(drone=d,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[d][0:3],
                                np.zeros(4),
                                obs2[d][3:15],
                                act2[d]
                                ]),
                control=np.zeros(12)
                )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated: #or truncated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
