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
        self.EPISODE_LEN_SEC = 20
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
        self.MAX_VEL = 6
        self.MAX_DIST = 7
        self.prev_dists = None
        self.prev_cent_dists = None

        self.SPACING_A = 1.1 #pos amp coef
        self.SPACING_B = 2 #neg amp coef
        self.SPACING_C = 1.1 #width of pos coef
        self.SPACING_K = 0.3 #width of neg coef
        self.SPACING_D = 0.5 #pos peak offset
        self.SPACING_R0 = 2.5 #piecewise threshold
        self.SPACING_LAM = 0.2 #exp decay


        self.MAX_ALT_ERROR = self.DRONE_TARGET_ALTITUDE * 0.1
        self.REWARD_WEIGHTS = dict(proximity=0.5, approach=0.8, centroid=1, drone_spacing = 1)
    ################################################################################
    

    def _computeReward(self):
        """Reward: move toward centroid, stay near, avoid collisions."""

        #Drone and herd info
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        pos = states[:, 0:3]
        vel = states[:, 10:13]
        dists = np.linalg.norm(pos - cattle_centroid, axis=1)
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)

        #Directional reward (dot product of vel and direction)
        dir_to_centroid = cattle_centroid - pos
        dir_unit = np.where(dists[:, None] > 0, dir_to_centroid / dists[:, None], 0.0)
        approach_reward = np.mean(np.sum(vel * dir_unit, axis=1)) / (self.MAX_VEL + 1e-6)

        #Progress-based proximity reward
        if self.prev_dists is None:
            self.prev_dists = dists
        dist_change = self.prev_dists - dists   # positive if closer, negative if further
        progress_reward = np.mean(dist_change / (self.MAX_DIST + 1e-6))
        self.prev_dists = dists

        #Centroid proximity reward
        if self.prev_cent_dists is None:
            self.prev_cent_dists = cent_dist
        cent_dist_change = self.prev_cent_dists - cent_dist
        centroid_reward = np.mean(cent_dist_change / (self.MAX_DIST + 1e-6))
        self.prev_cent_dists = cent_dist

        #drone to drone spcaing reward
        drone_spacing_reward = 0
        for drone in range(self.NUM_DRONES):
            drone_pos = states[drone, 0:3]
            for other_drones in range(drone + 1, self.NUM_DRONES):
                if drone == other_drones:
                    continue
                other_drones_pos = states[other_drones, 0:3]
                dist = np.linalg.norm(drone_pos - other_drones_pos)
                drone_spacing_reward += self.SpacingRewardValue(dist)

        drone_spacing_reward /= self.NUM_DRONES

        #Combine with weights
        r = (
            # approach_reward * self.REWARD_WEIGHTS["approach"]
            # + progress_reward * self.REWARD_WEIGHTS["proximity"]
            # + centroid_reward * self.REWARD_WEIGHTS["centroid"]
            drone_spacing_reward * self.REWARD_WEIGHTS["drone_spacing"]
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
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)

        # Check if all drones are within 0.5 of the centroid
        if cent_dist < 0.1:
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
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)

        if cent_dist > self.MAX_DIST:
            return True
        
        for i in range(self.NUM_DRONES):
            roll = states[i][7]
            pitch = states[i][8]
            if abs(roll) > 0.5 or abs(pitch) > 0.5:
                return True
            
            z = states[i][2]
            if abs(z - self.DRONE_TARGET_ALTITUDE) > self.MAX_ALT_ERROR:
                return True
            
            vel = states[i][10:13]
            if np.linalg.norm(vel) > self.MAX_VEL:
                return True
    
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
    
    def InteractionForce(self, xi, xj, a, c, d):
        """
        Computes the interaction force between two 3D positions. Original Attraction-Repulsion Force - Not Used

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
    ################################################################################

    def SpacingRewardValue(self, r):
        """
        Compute Spacing Reward
            
        Returns:
        float
        """
        A = self.SPACING_A
        B = self.SPACING_B
        c = self.SPACING_C
        k = self.SPACING_K
        d = self.SPACING_D
        r0 = self.SPACING_R0
        lam = self.SPACING_LAM

        if r <= r0:
            return A * np.exp(-((r - d)**2) / (2 * c**2)) - B * np.exp(-(r**2) / (2 * k**2))
        else:
            fr0 = A * np.exp(-((r0 - d)**2) / (2 * c**2)) - B * np.exp(-(r0**2) / (2 * k**2))
            C = fr0 / np.exp(-lam * r0)
            return C * np.exp(-lam * r)