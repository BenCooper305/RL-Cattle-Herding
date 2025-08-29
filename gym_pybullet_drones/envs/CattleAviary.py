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
                 num_cattel: int=1,
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
        self.EPISODE_LEN_SEC = 30
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         num_cattel=num_cattel,
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
        
        self._last_dist_to_centroid = None
        self.MAX_VEL = 10
        self.MAX_DIST = 10
        self.prev_dists = None
        self.MAX_ALT_ERROR = self.DRONE_TARGET_ALTITUDE * 0.4
        self.REWARD_WEIGHTS = dict(proximity=0.5, approach=1, altitude=0.5)
    ################################################################################
    

    def _computeReward(self):
        """Reward: move toward centroid, stay near, avoid collisions."""

        #Drone and herd info
        centroid = self.HerdCentroid()
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        pos = states[:, 0:3]
        vel = states[:, 10:13]
        dists = np.linalg.norm(pos - centroid, axis=1)

        #Directional reward (dot product of vel and direction)
        dir_to_centroid = centroid - pos
        dir_unit = np.where(dists[:, None] > 0, dir_to_centroid / dists[:, None], 0.0)
        approach_reward = np.mean(np.sum(vel * dir_unit, axis=1)) / (self.MAX_VEL + 1e-6)

        #Progress-based proximity reward
        if self.prev_dists is None:
            self.prev_dists = dists
        dist_change = self.prev_dists - dists   # positive if closer, negative if further
        progress_reward = np.mean(dist_change / (self.MAX_DIST + 1e-6))
        self.prev_dists = dists

        #Combine with weights
        r = (
            approach_reward * self.REWARD_WEIGHTS["approach"]
            + progress_reward * self.REWARD_WEIGHTS["proximity"]
            )

        return float(r)

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        ------- 
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        centroid = self.HerdCentroid()
        # Check if all drones are within 0.5 of the centroid
        if all(np.linalg.norm(centroid - states[i][0:3]) < 0.5 for i in range(self.NUM_DRONES)):
            return True
        
        return False

    ################################################################################
    
    def _computeTruncated(self):
        """Computes whether the current episode should be truncated due to unsafe drone states.

        Returns
        -------
        bool
            True if the episode should be truncated, False otherwise.
        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        # --- Episode timeout ---
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

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
        
    def HerdCentroid(self):
        """

        Calculates the center of the herd and updates the centorid marker to that location

        """
        
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
        cattle_positions = cattle_states[:, 0:3] 

        centroid_xy = np.mean(cattle_positions[:, :2], axis=0)
        centroid = np.array([centroid_xy[0], centroid_xy[1], self.DRONE_TARGET_ALTITUDE])

        p.resetBasePositionAndOrientation(self.CattleCentroidMarker,
                                            centroid,
                                            p.getQuaternionFromEuler([0, 0, 0]),
                                            physicsClientId=self.CLIENT)
        

        return centroid


    
    ################################################################################
    
    def InteractionForce(self, xi, xj, a, c, d):
        """
        Computes the interaction force between two 3D positions.

        Parameters
        ----------
        xi : np.ndarray
            Position of agent i, shape (3,)
        xj : np.ndarray
            Position of agent j, shape (3,)
        a : float
            Strength parameter
        c : float
            Distance scaling parameter
        d : float
            Desired distance between i and j
            must be less than animal reaction radius r for robot-cow interaction

        Returns
        -------
        np.ndarray
            Force vector acting on agent i due to agent j, shape (3,)
        """
        delta = xi - xj                     # vector from j to i
        distance = np.linalg.norm(delta)    # Euclidean distance
        exponent = -(distance - d) / c
        numerator = a * (1 - np.exp(exponent))
        denominator = 1 + distance
        force = (numerator / denominator) * delta
        return force
    