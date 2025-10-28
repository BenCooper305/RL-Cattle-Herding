from ray.rllib.env import MultiAgentEnv
import numpy as np
from typing import Dict, Any
from gym_pybullet_drones.rllib_envs.MARLCattleAviary import MARLCattleAviary
from gymnasium.spaces import Box
import inspect

class RLlibMultiAgentWrapper(MultiAgentEnv):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        # ensure env is created in headless mode inside Ray workers.
        # Only pass parameters accepted by MARLCattleAviary.__init__ and prefer headless / no-gui settings.
        cfg = dict(env_config) if env_config is not None else {}
        sig = inspect.signature(MARLCattleAviary.__init__)
        params = set(sig.parameters.keys())
        accepts_kwargs = "kwargs" in sig.parameters

        # Prefer headless modes by setting common aliases only when supported.
        if "pb_mode" in params:
            cfg.setdefault("pb_mode", "DIRECT")
        elif "connection_mode" in params:
            cfg.setdefault("connection_mode", "DIRECT")
        elif "physicsClient" in params:
            cfg.setdefault("physicsClient", "DIRECT")

        if "gui" in params:
            cfg.setdefault("gui", False)
        elif "use_gui" in params:
            cfg.setdefault("use_gui", False)
        elif "render" in params:
            cfg.setdefault("render", False)

        # Filter out unsupported keys unless __init__ accepts **kwargs.
        if not accepts_kwargs:
            filtered_cfg = {k: v for k, v in cfg.items() if k in params}
            ignored = set(cfg.keys()) - set(filtered_cfg.keys())
        else:
            filtered_cfg = cfg
            ignored = set()

        try:
            self.env = MARLCattleAviary(**filtered_cfg)
        except Exception as e:
            msg = f"Failed to create MARLCattleAviary in RLlib wrapper: {e}"
            if ignored:
                msg += f" (ignored unsupported keys: {sorted(list(ignored))})"
            raise RuntimeError(msg) from e
        # set agent ids first, then expose per-agent spaces (RLlib expects iterable/dict)
        self.possible_agents = [f"agent_{i}" for i in range(self.env.NUM_DRONES)]
        # active agents (RLlib requires both possible_agents and agents)
        self.agents = self.possible_agents.copy()
        # expose per-agent space mappings so RLlib can iterate / index them
        # keep single-space getters simple; RLlib will call get_action_space/get_observation_space
        self.action_space = {aid: self.env.action_space for aid in self.possible_agents}
        self.observation_space = {aid: self.env.observation_space for aid in self.possible_agents}

    # RLlib calls these for validation
    def get_action_space(self, agent_id):
        return self.env.action_space

    def get_observation_space(self, agent_id):
        return self.env.observation_space

    def reset(self, *, seed=None, options=None):
        # reset underlying env and rebuild agent ids/spaces to match current NUM_DRONES
        underlying_obs, underlying_info = self.env.reset(seed=seed, options=options)
        # rebuild agent lists and per-agent spaces to reflect possibly changed NUM_DRONES
        self.possible_agents = [f"agent_{i}" for i in range(self.env.NUM_DRONES)]
        self.agents = self.possible_agents.copy()
        self.action_space = {aid: self.env.action_space for aid in self.possible_agents}
        self.observation_space = {aid: self.env.observation_space for aid in self.possible_agents}
        # build wrapper-keyed obs/info dicts (only for current agents)
        obs = {f"agent_{i}": self.env._computeObs(i) for i in range(self.env.NUM_DRONES)}
        infos = {f"agent_{i}": {} for i in range(self.env.NUM_DRONES)}
        return obs, infos

    def step(self, action_dict: Dict[str, np.ndarray]):
        # Step all agents
        # Build full action array in canonical order (agent_i -> index i)
        action_array = np.zeros((self.env.NUM_DRONES, self.env.action_space.shape[0]), dtype=np.float32)
        # Only apply actions for currently active agents to avoid stepping already-done agents.
        for i, aid in enumerate(self.possible_agents):
            if aid in action_dict and aid in self.agents:
                action_array[i] = np.asarray(action_dict[aid], dtype=np.float32)

        # Step the underlying environment (expects full-array)
        try:
            self.env.step(action_array)
        except Exception as e:
            import traceback
            with open("/tmp/env_worker_exc.log", "a") as fh:
                fh.write("=== Exception in env.step() ===\n")
                fh.write(traceback.format_exc())
                fh.write("\n")
            raise

        obs: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        dones: Dict[str, bool] = {}
        truncs: Dict[str, bool] = {}
        infos: Dict[str, Any] = {}

        # Populate results for currently active agents
        for aid in list(self.agents):
            idx = int(aid.split("_")[1])
            obs[aid] = self.env._computeObs(idx)
            rewards[aid] = float(self.env._computeReward(idx))
            dones[aid] = bool(self.env._computeTerminated(idx))
            truncs[aid] = bool(self.env._computeTruncated(idx))
            infos[aid] = self.env._computeInfo(idx)

        # Remove finished agents from active list
        self.agents = [aid for aid in self.agents if not dones.get(aid, False)]

        # RLlib expects "__all__" keys
        dones["__all__"] = len(self.agents) == 0
        truncs["__all__"] = len(self.agents) == 0

        return obs, rewards, dones, truncs, infos

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
