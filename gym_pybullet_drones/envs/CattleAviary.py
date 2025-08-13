import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class CattleAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.COKIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        
        #Setting Goals
        self.TARGET_POS = self.INIT_XYZS + np.array([[2/(i+1),2/(i+1),2/(i+1)] for i in range(num_drones)])
        self.GOAL_POS = np.array([-2,-2,0.5])
        self.SpawnHerd(1)
        self.SpawnGoal(self.GOAL_POS)

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
    
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        reward = 0
        threshold = 3.5
        max_reward = threshold ** 2  # 12.25
        totalDistance = 0

        for i in range(self.NUM_DRONES):
            dist = np.linalg.norm(self.GOAL_POS - states[i][0:3])
            totalDistance += dist
            reward += max(0, max_reward - dist**2)

        #print(f"[LOG] Average distace: {totalDistance/self.NUM_DRONES}")
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist = 0
        for i in range(self.NUM_DRONES):
            dist += np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
        if dist < .0001:
            #print("[LOG] TERMINACTED!")
            return True
        else:
            return False

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        # states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # for i in range(self.NUM_DRONES):
        #     if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
        #      or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
        #     ):
        #         return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            #print("[LOG] TRUNCTAED!")
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def GetDroneDistances(self):
        distances = []
        """Computes the current info dict(s).
        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return distances
    
    ################################################################################

    def GetDistanceToHerd(self):
        distance = []
        """Computes the current info dict(s).
        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return distance

     ################################################################################

    def SpawnHerd(self,herdSize):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        p.loadURDF("cube_no_rotation.urdf",
                   [2, 2, .1],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   globalScaling=0.2,
                   physicsClientId=self.CLIENT
                   )
    
    ################################################################################

    def SpawnGoal(self, goalPos):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        p.loadURDF("sphere2.urdf",
                   [goalPos[0], goalPos[1], goalPos[2]],
                   p.getQuaternionFromEuler([0,0,0]),
                   globalScaling=0.2,
                   physicsClientId=self.CLIENT
                   )
        
    
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        p.loadURDF("sphere2.urdf",
            [1, 1, 1],
            p.getQuaternionFromEuler([0,0,0]),
            globalScaling=0.6,
            physicsClientId=self.CLIENT
            )

        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass