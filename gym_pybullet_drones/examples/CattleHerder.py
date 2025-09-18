"""
-------
pip install tensorboard

In a terminal, run as:

    $ conda activate drones
    $ python CattleHerder.py

    tensorboard --logdir /home/ben/ros_ws/src/RL-Cattle-Herding/gym_pybullet_drones/examples/models/model-v3-1
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
TARGET_REWARD = 99999 #reward to reach to end training
LOAD_FILE = 'model-v5-3/best_model.zip' #'model-v3-1' #'save-08.19.2025_15.25.22/best_model.zip'

DEFAULT_OBS = ObservationType('cokin') #collabrative kinematicsa
DEFAULT_ACT = ActionType('vel') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_DRONES = 10 #number of herding drones - DO NOT EXCEED 10
DEFAULT_CATTLE = 8 #number of cattle to herd - DO NOT CHANGE

MAX_TIMESTEPS = 1000  #max number of time steps before learning stops
EVALUATION_FREQUENCY = 2048 #number of steps to collect before updating policy

EVALUTE_ONLY = False #not used - skips training and replays a saved model

def run(
        target_reward = TARGET_REWARD,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, 
        plot=False, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        local=True):
    

    filename = os.path.join(output_folder, 'model-v5-4')
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #create environments
    env_kwargs = dict(num_drones=DEFAULT_DRONES, num_cattel=DEFAULT_CATTLE, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    train_env = make_vec_env(CattleAviary, env_kwargs=env_kwargs, n_envs=1, seed=0)
    eval_env = DummyVecEnv([lambda: Monitor(CattleAviary(**env_kwargs))])

    #### Create the model #######################################
    loadfile = 'models/'+LOAD_FILE
    print(loadfile)
    if os.path.exists(loadfile):
        model = PPO.load(loadfile, env=train_env)
        print("[LOG] PPO Model Loaded")
    else:
        model = PPO('MlpPolicy', 
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
                    tensorboard_log=filename+'/tb/',
                    policy_kwargs=dict(
                        log_std_init=-1.0,     # start with lower action variance
                        ortho_init=False,
                        net_arch=[dict(pi=[128, 128], vf=[128, 128])],),
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

    test_env_nogui.is_evaluating = True
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=1)
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
