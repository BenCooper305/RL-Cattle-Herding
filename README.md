# RL Cattle Herding

This repo containts the code for a rlenvirment leanring model using PPO to herd cattle as part of research.

This repo contains two simualtors: multi_robot_herding_edited and gym_pybullet_drones.

## multi_robot_herding

multi_robot_herding is bassed off of the multi_robot_herding which can be found here:
https://github.com/dkhoanguyen/multi_robot_herding

The version in this repo fixes an image read file, adds pickel evalation code and spawns cattle reading from a seperate yamal file

## gym_pybullet_drones

gym_pybullet_drones is based off of the gym-pubullet-drones repo which can be found here:
https://github.com/utiasDSL/gym-pybullet-drones



<img src="gym_pybullet_drones/assets/helix.gif" alt="formation flight" width="325"> <img src="gym_pybullet_drones/assets/helix.png" alt="control info" width="425">

## Installation

Developed for x64/Ubuntu 22.04

```sh
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

#for DETE you will also need:
pip install lz4
pip install ray[rllib]
```

## Centrlised Learning Decentroilsed Execution (CTDE) 

Folder: sb3_envs
PPO Library: Stable-Baselines3
Simulator: CTDECattleHerder.py

### Usage

```sh
conda activate drones
cd gym_pybullet_drones/simulator
python CTDECattleHerder.py

#view tensorboard after training
tensorboard --logdir models/your_model_to_view/tb
```


## Decentroilsed Learning Decentroilsed Execution (DTDE)

Folder: rllib_envs
PPO Library: RLlib
Simulator: DTDECattleHerder.py


### Usage

```sh
conda activate drones
cd gym_pybullet_drones/simulator
python CTDECattleHerder.py

#view tensorboard after training
tensorboard --logdir models/your_model_to_view/tb
```
python DTDECattleHerder.py --new_model_name "DTDE_v2"

python DTDECattleHerder.py --load_checkpoint "ray_results/DTDE_v2/checkpoint_000100"

python DTDECattleHerder.py --evaluate_only True --load_checkpoint "ray_results/DTDE_v2/checkpoint_000100"

tensorboard --logdir ray_results/DTDE_v2


python DTDEModelPlayback.py \
    --checkpoint "ray_results/DTDE_MARL_Cattle_TEST/PPO_marl_cattle_env_*/checkpoint_000500" \
    --num_episodes 3 \
    --gui True



## Development /Working Files
envs -> BaseAviary - handles bulk of simulation 
        BaseRLAvairy - handles action and opbservation spaces
        CattleAviary - computes reward, termination, truncated conditions
        models - saved trained models
examples -> CattleHerder - main file for running cattle herding

utils -> flockUtils - 
         mathUtils -
         utils

