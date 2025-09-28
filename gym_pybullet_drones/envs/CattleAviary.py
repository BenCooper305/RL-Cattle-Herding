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
                 ctrl_freq: int = 60,
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
        self.EPISODE_LEN_SEC = 100
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

        self.MAX_ALT_ERROR = self.DRONE_TARGET_ALTITUDE * 0.6

        #Reward Paramaters
        self.REWARD_WEIGHTS = dict(drone_to_drone_spacing = 0.8,
                                   centroid_distance= 1, 
                                   drone_to_cattle_spacing = 0.5
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
        cattle_poses = cattle_states[:, :2]

        if self.is_evaluating:
            print("\n==== REWARD DEBUG INFO ====")

        # Drone to Drone Spacing
        drone_to_drone_spacing_reward = 0.0
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
            other_dists[i] = np.inf  # ignore self

            # get two smallest distances
            nearest_two = np.partition(other_dists, 1)[:2]

            # sum rewards for the two nearest neighbours
            for dist in nearest_two:
                rwd = self.SpacingRewardValue(dist)
                if self.is_evaluating:
                    print(f"d-2-d: drone id {i}, drone distance {dist:.6f} got reward {rwd:.6f}")
                drone_to_drone_spacing_reward += rwd

        # Average over drones and neighbours (2)
        drone_to_drone_spacing_reward /= (self.NUM_DRONES * 2.0)

        # Optional: clip to a sensible range
        drone_to_drone_spacing_reward = float(np.clip(drone_to_drone_spacing_reward, -2.0, 1.0))

        #Centroid Distance Reward
        centroid_distance_reward = 0
        max_step_distance = self.SPEED_LIMIT/self.CTRL_FREQ
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
        if self.prev_cent_dists is not None:
            cent_dist_change = self.prev_cent_dists - cent_dist
            centroid_distance_reward = np.clip((cent_dist_change / (max_step_distance + 1e-6)) * 5, -1.0, 1.0)
            if self.is_evaluating:
                print(f"ent. dist: raw distane {cent_dist_change / (max_step_distance + 1e-6)}, cliped {centroid_distance_reward}")
        self.prev_cent_dists = cent_dist
        

        #Drone to Cattle Spacing
        drone_to_cattle_spacing_reward = 0.0
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            dists_to_cattle = np.linalg.norm(cattle_poses - pos_i, axis=1)
            closest_dist = np.min(dists_to_cattle)
            drone_to_cattle_spacing_reward += self.old_SpacingRewardValue(closest_dist)

        #Average over drones
        drone_to_cattle_spacing_reward /= self.NUM_DRONES

        #penalty to encourage efficiency
        small_step_penalty = 0.01

        #Combine Rewards
        r = (
            centroid_distance_reward * self.REWARD_WEIGHTS["centroid_distance"]
            + drone_to_drone_spacing_reward * self.REWARD_WEIGHTS["drone_to_drone_spacing"]
            + drone_to_cattle_spacing_reward * self.REWARD_WEIGHTS["drone_to_cattle_spacing"]
            #  - small_step_penalty
        )

        if self.is_evaluating:
            print("\n==== REWARD Summary ====")
            print(f"Drone-to-Drone Spacing Reward: {drone_to_drone_spacing_reward:.4f}")
            # print(f"Centroid Distance: {cent_dist:.4f}")
            # print(f"Centroid Distance Reward: {centroid_distance_reward:.4f}")
            print(f"Initial Combined Reward (before end-episode bonuses/penalties): {r:.4f}")

        #End of Episode Rewards
        done = self._computeTerminated() or self._computeTruncated()
        if done:
            r += self._endOfEpisodeReward(cent_dist, drones_poses, cattle_poses)
           
        return float(r)
    ################################################################################

    def _endOfEpisodeReward(self, cent_dist, drones_poses, cattle_poses):
        """
        Calculates reward at end of episode
        """
        reward = 0

         #Reward for drone centorid being near cattle centroid
        if cent_dist < self.TERMINATION_CENTROID_THRESH:
            reward+= 40
        else:
            reward -= cent_dist * 1.5

        #Penalty for crashing
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
            other_dists[i] = np.inf
            nearest_two = np.sort(other_dists)[:2]
            if np.any(nearest_two < 0.1):
                reward -= 20
                print(f"Crash penalty applied!")

        #Reward for surviving longer
        elapsed_sec = self.step_counter / self.CTRL_FREQ
        if elapsed_sec >= self.EPISODE_LEN_SEC - 1:
            reward += 50
        elif elapsed_sec> self.EPISODE_LEN_SEC/4:
            reward += elapsed_sec *0.3

        print(f"epiosde length {elapsed_sec}")
        #Reward for effectivness
        # print(f"episode ended in {elapsed_sec}s, bonus reward: {rwd}")
        effectiveness = self.eval_system.calculate_effectiveness(cattle_poses,drones_poses)
        if effectiveness == 100: #missive reward for herding all cattle
            reward += 100
        elif effectiveness == 0: #negative reward for herding nothing
            reward -= 5
        else:
            reward +=  effectiveness/5 #bonus 2 - 18 points for number of cattle herded

        return reward


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

        effectiveness = self.eval_system.calculate_effectiveness(cattle_poses,drones_poses)

        if effectiveness > 80:
            dists = np.linalg.norm(drones_poses[:, None, :] - cattle_poses[None, :, :], axis=-1)
            min_dists = np.min(dists, axis=1) 
            if np.all(min_dists < 1.0):
                print("DRONE MISSION COMEPLTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                return True
            
        elapsed_sec = self.step_counter / self.CTRL_FREQ
        if elapsed_sec > self.EPISODE_LEN_SEC:
            if self.is_evaluating:
                self.evaluation_episode_trigger()
            print("Episode Time limit reached!")
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
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2] 
        cattle_poses = cattle_states[:, :2]

        #check to make sure drones dont hit each other
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
            other_dists[i] = np.inf
            nearest_two = np.sort(other_dists)[:2]
            if np.any(nearest_two < 0.1):
                print(f"Drone {i} collision with distance(s): {nearest_two}")
                return True
            
        #check to make sure drones dont lose altitude
        for i in range(self.NUM_DRONES):
            z = drone_states[i][2]
            if abs(z - self.DRONE_TARGET_ALTITUDE) > self.MAX_ALT_ERROR:
                print(f"Drone {i} Altitude loss with {z} alltidue with max error {self.MAX_ALT_ERROR}, from target {self.DRONE_TARGET_ALTITUDE}")
                return True    

        #check to make sure drones dont fly to far apart
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
            other_dists[i] = np.inf
            nearest_two = np.partition(other_dists, 1)[:2]
            if np.any(nearest_two > 8):
                print(f"Drone {i} distance limit exceeded with nearest drone at {nearest_two}")
                return True
            


        #check to make sure centroid doesnt move far away
        if cent_dist > self.MAX_DIST + 2:
                print(f"Centroid distance exceeded with distance: {cent_dist}")
                return True
        
        # --- Episode timeout ---self.step_counter
        # elapsed_sec = self.step_counter / self.CTRL_FREQ
        # if elapsed_sec > self.EPISODE_LEN_SEC:
        #     if self.is_evaluating:
        #         self.evaluation_episode_trigger()
        #         print("Episode Time limit reached!")
        #     return True

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

    def old_SpacingRewardValue(self, r):
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
        
    # def SpacingRewardValue(self, r):
    #     """
    #     Combined spacing reward:
    #     - gaussian peak at d_star (max ~1)
    #     - collision linear penalty for r < collision_thresh
    #     - linear attraction for r > long_range_start to pull drones inward
    #     """
    #     # params
    #     d_star = getattr(self, "SPACING_DSTAR", 0.8)
    #     sigma = getattr(self, "SPACING_SIGMA", 0.25)
    #     collision_thresh = getattr(self, "SPACING_COLLIDE", 0.15)
    #     collision_penalty = getattr(self, "SPACING_COLL_PEN", 1.0)

    #     # long-range attraction params
    #     long_range_start = getattr(self, "SPACING_LONG_RANGE", 1.5)  # distance where linear term starts
    #     long_range_slope = getattr(self, "SPACING_LONG_SLOPE", 0.25)  # how strongly to pull from far away
    #     max_range = getattr(self, "SPACING_MAX_RANGE", 5.0)  # for normalisation if desired

    #     # Gaussian bump (peak at 1.0)
    #     gauss = np.exp(-0.5 * ((r - d_star) / (sigma + 1e-9))**2)

    #     # Collision penalty (strong negative)
    #     if r < collision_thresh:
    #         coll_pen = -collision_penalty * (1.0 - (r / (collision_thresh + 1e-9)))
    #     else:
    #         coll_pen = 0.0

    #     # Long-range attraction: linear term that encourages decreasing r toward d_star
    #     # Positive if r > d_star and pulls reward up when r decreases.
    #     if r > long_range_start:
    #         # normalized distance above long_range_start
    #         pull = -long_range_slope * (r - long_range_start) / (max_range - long_range_start + 1e-9)
    #     else:
    #         pull = 0.0

    #     # Combine. Weighting chosen so gauss still dominates near d_star.
    #     reward = gauss + coll_pen + pull
    #     return float(reward)

    def SpacingRewardValue(self, r):
        """
        Combined spacing reward:
        - gaussian peak at d_star (max ~1)
        - collision linear penalty for r < collision_thresh
        - linear attraction for r > long_range_start to pull drones inward
        now scaled so that at r = max_range the term = -1
        """
        import numpy as np

        # params
        d_star = 0.8 #deisred distance
        sigma = 0.25
        collision_thresh = 0.3
        collision_penalty = 1.0

        # long-range attraction params
        long_range_start = 1.5 # distance where linear term starts
        max_range = 5.0  # maximum distance

        # Gaussian bump (peak at 1.0)
        gauss = np.exp(-0.5 * ((r - d_star) / (sigma + 1e-9))**2)

        # Collision penalty (strong negative)
        if r < collision_thresh:
            coll_pen = -collision_penalty * (1.0 - (r / (collision_thresh + 1e-9)))
        else:
            coll_pen = 0.0

        # Long-range attraction: linear term scaled so r = max_range -> -1
        if r > long_range_start:
            # Linear mapping: reward = 0 at long_range_start, -1 at max_range
            pull = - (r - long_range_start) / (max_range - long_range_start + 1e-9)
        else:
            pull = 0.0

        # Combine
        reward = gauss + coll_pen + pull
        return float(reward)