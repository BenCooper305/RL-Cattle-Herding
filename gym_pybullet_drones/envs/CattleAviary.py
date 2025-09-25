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
                 num_cattle: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 120,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.COKIN,
                 act: ActionType=ActionType.VEL
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
                         num_cattle=num_cattle,
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
        self.MAX_VEL = 2
        self.MAX_DIST = 12
        self.prev_dists = None
        self.prev_cent_dists = None

        self.SPACING_A = 1.2 #pos amp coef
        self.SPACING_B = 2.1 #neg amp coef
        self.SPACING_C = 3.3 #width of pos coef
        self.SPACING_K = 0.2 #width of neg coef
        self.SPACING_D = -1 #pos peak offset
        self.SPACING_R0 = 1.3 #piecewise threshold
        self.SPACING_LAM = 0.8 #exp decay

        self.MAX_ALT_ERROR = self.DRONE_TARGET_ALTITUDE * 0.4

        #Reward Paramaters
        self.REWARD_WEIGHTS = dict(drone_to_drone_spacing = 1,
                                   centroid_distance= 1, 
                                #    drone_to_cattle_spacing = 0.4
                                   )
        
        self.TERMINATION_CENTROID_THRESH = 0.25
    ################################################################################
    

    def _computeReward(self):
        #Data
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2]
        drone_vel = drone_states[:, 10:12]

        cattle_poses = cattle_states[:, :2]

        #Drone to Drone Spacing
        drone_to_drone_spacing_reward = 0.0
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]

            other_dists = np.linalg.norm(drones_poses[:] - pos_i, axis=1)
            other_dists[i] = np.inf  # ignore self

            nearest_two = np.partition(other_dists, 1)[:2]  # get two smallest distances
            for dist in nearest_two:
                rwd = self.SpacingRewardValue(dist)
                drone_to_drone_spacing_reward += rwd

        # # Average over all drones and terms
        # drone_spacing_reward /= self.NUM_DRONES * 3  # 2 nearest + 1 centroid per drone

        #Centroid Distance Reward
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
        if self.prev_cent_dists is None:
            self.prev_cent_dists = cent_dist
        cent_dist_change = self.prev_cent_dists - cent_dist
        centroid_distance_reward = cent_dist_change / (self.MAX_DIST + 1e-6)
        self.prev_cent_dists = cent_dist



        # dir_to_cattle = cattle_centroid[:2] - drones_poses[:, :2]
        # dists_to_cattle = np.linalg.norm(dir_to_cattle, axis=1)
        # dir_unit = np.where(dists_to_cattle[:, None] > 0, dir_to_cattle / dists_to_cattle[:, None], 0.0)
        # centroid_approach_reward = np.mean(np.sum(drone_vel[:, :2] * dir_unit, axis=1)) / (self.MAX_VEL + 1e-6)


        # --- Drone to Cattle Spacing (closest cattle only) ---
        # drone_cattle_spacing_reward = 0.0
        # for i in range(self.NUM_DRONES):
        #     pos_i = drones_poses[i]
        #     dists_to_cattle = np.linalg.norm(cattle_poses - pos_i, axis=1)
        #     closest_dist = np.min(dists_to_cattle)
        #     drone_cattle_spacing_reward += self.SpacingRewardValue(closest_dist)
        #     print(f"drones: {i}, drone distance to nearest cattle: {closest_dist}, drone reward: {drone_cattle_spacing_reward}")

        # # Average over drones
        # drone_cattle_spacing_reward /= self.NUM_DRONES

        #Combine Rewards
        r = (
             centroid_distance_reward * self.REWARD_WEIGHTS["centroid_distance"]
             + drone_to_drone_spacing_reward * self.REWARD_WEIGHTS["drone_to_drone_spacing"]
            #drone_cattle_spacing_reward * 1
        )

        #End of Episode Rewards
        done = self._computeTerminated() or self._computeTruncated()
        if done:

            #reward for drone centorid being near cattle centroid
            if cent_dist < self.TERMINATION_CENTROID_THRESH:
                r+= 100
            else:
                r -= cent_dist * 2

        #     effectiveness = self.eval_system.calculate_effectiveness(cattle_poses,drones_poses)
        #     if effectiveness == 100: #missive reward for herding all cattle
        #         r += 100
        #     elif effectiveness == 0: #negative reward for herding nothing
        #         r -= 25
        #     else:
        #         r +=  effectiveness/10 #bonus 1 - 9 points for number of cattle herded

        # print(f"reward given: {r}")
        # print(f"####################")
        return float(r)

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value."""

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2] 
        cattle_poses = cattle_states[:, :2]

        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)
        if cent_dist < self.TERMINATION_CENTROID_THRESH:
            return True

        # effectiveness = self.eval_system.calculate_effectiveness(cattle_poses,drones_poses)

        # if effectiveness > 80:
        #     dists = np.linalg.norm(drones_poses[:, None, :] - cattle_poses[None, :, :], axis=-1)
        #     min_dists = np.min(dists, axis=1) 
        #     if np.all(min_dists < 1.0):
        #         print("DRONE MISSION COMEPLTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #         return True

        return False


    ################################################################################
    
    def _computeTruncated(self):
        """Computes whether the current episode should be truncated due to unsafe drone states.

        Returns
        -------
        bool
            True if the episode should be truncated, False otherwise.
        """
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2] 
        cattle_poses = cattle_states[:, :2]

        #check to make sure drones dont lose altitude
        for i in range(self.NUM_DRONES):
            z = drone_states[i][2]
            if abs(z - self.DRONE_TARGET_ALTITUDE) > self.MAX_ALT_ERROR:
                if self.is_evaluating:
                    self.evaluation_episode_trigger()
                    print("Drone Altitude loss")
                return True    

        #check to make sure drones dont fly to far apart
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
            other_dists[i] = np.inf
            nearest_two = np.partition(other_dists, 1)[:2]
            if np.any(nearest_two > 8.0):
                print(f"Drone {i} distance limit exceeded with nearest drone at {nearest_two}")
                return True

        #check to make sure centroid doesnt move far away
        if cent_dist > self.MAX_DIST + 2:
                print(f"Centroid distance exceeded with distance: {cent_dist}")
                return True
        
        # --- Episode timeout ---self.step_counter
        elapsed_sec = self.step_counter / self.CTRL_FREQ
        if elapsed_sec > self.EPISODE_LEN_SEC:
            if self.is_evaluating:
                self.evaluation_episode_trigger()
                print("Episode Time limit reached!")
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
        