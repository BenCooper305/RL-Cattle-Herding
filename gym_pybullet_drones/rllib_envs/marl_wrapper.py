from ray.rllib.env import MultiAgentEnv
import numpy as np
from typing import Dict, Any
from gym_pybullet_drones.rllib_envs.MARLCattleAviary import MARLCattleAviary
from gymnasium.spaces import Box

class RLlibMultiAgentWrapper(MultiAgentEnv):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        self.env = MARLCattleAviary(**env_config)
        self.possible_agents = [f"agent_{i}" for i in range(self.env.NUM_DRONES)]
        self.agents = self.possible_agents.copy()  # active agents

    # RLlib calls these for validation
    def get_action_space(self, agent_id):
        return self.env.action_space

    def get_observation_space(self, agent_id):
        return self.env.observation_space

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents.copy()  # reset active agents
        obs = {aid: self.env._computeObs(i) for i, aid in enumerate(self.agents)}
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, action_dict: Dict[str, np.ndarray]):
        obs, rewards, dones, truncs, infos = {}, {}, {}, {}, {}

        # Step only active agents
        for i, aid in enumerate(self.agents):
            action = action_dict.get(aid, np.zeros(self.env.action_space.shape, dtype=np.float32))
            self.env.step_for_agent(i, action)  # per-agent step

            obs[aid] = self.env._computeObs(i)
            rewards[aid] = float(self.env._computeReward(i))
            dones[aid] = bool(self.env._computeTerminated(i))
            truncs[aid] = bool(self.env._computeTruncated(i))
            infos[aid] = self.env._computeInfo(i)

        # Remove done agents
        self.agents = [aid for aid in self.agents if not dones[aid]]

        # RLlib expects "__all__"
        dones["__all__"] = len(self.agents) == 0
        truncs["__all__"] = all(truncs.values())

        return obs, rewards, dones, truncs, infos

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
