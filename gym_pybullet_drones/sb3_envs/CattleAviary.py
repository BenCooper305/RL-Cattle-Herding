import numpy as np
import pybullet as p

from gym_pybullet_drones.sb3_envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.utils.curriculum_learning import CurriculumLearning
from gym_pybullet_drones.utils.evaluation import evaluate_herding_effectiveness, evaluate_formation_quality

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
        #### Reward Weights ##################################
        self.curriculum_starting_level = 0
        self.curriculum = CurriculumLearning(self.curriculum_starting_level)
        self.drone_spacing_clock = 0 
        #### Episode Length ##################################
        self.EPISODE_LEN_SEC = self.curriculum.current_curriculum["episode_length"]
        

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         num_cattle=num_cattle,
                         max_num_drones=self.curriculum.current_curriculum["max_drones"],
                         min_num_drones=self.curriculum.current_curriculum["min_drones"],
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

        #### Episode Data ####################################
        self._last_dist_to_centroid = None
        self.prev_dists = None
        self.prev_cent_dists = None
        #### ETerminated Vars ################################
        self.EFFECTIVENESS_THRESHOLD = 85
        self.REQUIRED_CENTROID_DISTANCE = 0.5
        #### Truncated Vars ##################################
        self.MISSION_BOUNDARY = 15
        self.MAX_FORMATTION_DISTANCE = 8
        self.COLLISION_THRESHOLD = 0.2
        self.MAX_ALT_ERROR = self.DRONE_TARGET_ALTITUDE * 0.6
        #### Cattle Spacing Reward Vars ######################
        self.SPACING_A = 1.2            # pos amp coef
        self.SPACING_B = 2.1            # neg amp coef
        self.SPACING_C = 3.3            # width of pos coef
        self.SPACING_K = 0.2            # width of neg coef
        self.SPACING_D = -1             # pos peak offset
        self.SPACING_R0 = 1.3           # piecewise threshold
        self.SPACING_LAM = 0.8          # exp decay


    ################################################################################
    
    # def _computeReward(self):
    #     #Data
    #     cattle_centroid = self.HerdCentroid()
    #     drone_centroid = self.DroneCentroid()
    #     drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

    #     drones_poses = drone_states[:, :2]
    #     cattle_poses = cattle_states[:, :2]

    #     # REWARD 1: Drone-to-drone spacing reward
    #     drone_to_drone_spacing_reward_simple = 0.0
    #     drone_to_drone_spacing_reward_complex = 0.0
    #     for i in range(self.NUM_DRONES):
    #         pos_i = drones_poses[i]
    #         other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
    #         other_dists[i] = np.inf  # ignore self

    #         # get two smallest distances
    #         nearest_two = np.partition(other_dists, 1)[:2]

    #         # sum rewards for the two nearest neighbours
    #         for dist in nearest_two:
    #             drone_to_drone_spacing_reward_complex += self.DroneSpacingRewardFunction(dist)
    #             drone_to_drone_spacing_reward_simple += self.SimpleSpacingReward(dist)

    #     # Average over drones and neighbours (2)
    #     drone_to_drone_spacing_reward_complex /= (self.NUM_DRONES * 2.0)
    #     drone_to_drone_spacing_reward_simple /= (self.NUM_DRONES * 2.0)

    #     # REWARD 2: Drone-Survival Reward
    #     drone_survival_reward = 0.1

    #     # REWARD 3: Cattle Approach Reward
    #     cattle_approach_reward = 0
    #     max_step_distance = self.SPEED_LIMIT/self.CTRL_FREQ
    #     cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
    #     if self.prev_cent_dists is not None:
    #         cent_dist_change = self.prev_cent_dists - cent_dist
    #         cattle_approach_reward = np.clip((cent_dist_change / (max_step_distance + 1e-6)) * 5, -1.0, 1.0)
    #     self.prev_cent_dists = cent_dist
        
    #     # REWARD 4: Herding effectiveness reward (surrounding cattle)
    #     herding_effectiveness_reward = 0
    #     effectiveness = evaluate_herding_effectiveness(cattle_poses,drones_poses)
    #     herding_effectiveness_reward = effectiveness/100

    #     # REWARD 5: Drone-to-cattle spacing reward
    #     drone_to_cattle_spacing_reward = 0.0
    #     for i in range(self.NUM_DRONES):
    #         pos_i = drones_poses[i]
    #         dists_to_cattle = np.linalg.norm(cattle_poses - pos_i, axis=1)
    #         closest_dist = np.min(dists_to_cattle)
    #         drone_to_cattle_spacing_reward += self.CattleSpacingRewardFunction(closest_dist)
    
    #     #Average over drones
    #     drone_to_cattle_spacing_reward /= self.NUM_DRONES


    #     #Combine Rewards
    #     r = (
    #         drone_to_drone_spacing_reward_simple * self.curriculum.current_curriculum["drone_to_drone_spacing_simple"]
    #         + drone_to_drone_spacing_reward_complex * self.curriculum.current_curriculum["drone_to_drone_spacing_complex"]
    #         + drone_survival_reward * self.curriculum.current_curriculum['drone_survival']
    #         + cattle_approach_reward * self.curriculum.current_curriculum['cattle_approach']
    #         + herding_effectiveness_reward * self.curriculum.current_curriculum['effectiveness']
    #         + drone_to_cattle_spacing_reward * self.curriculum.current_curriculum['cattle_to_drone_spacing']

    #     )

    #     if self.is_evaluating:
    #         print("\n==== REWARD Summary ====")
    #         print(f"Curriculum Level: {self.curriculum.curriculum_level}")
    #         print(f"drone_to_drone_spacing Reward: {drone_to_drone_spacing_reward_simple:.4f}")
    #         print(f"drone_survival Reward: {drone_survival_reward:.4f}")
    #         print(f"cattle_approach: {cattle_approach_reward:.4f}")
    #         print(f"effectiveness Reward: {herding_effectiveness_reward:.4f}")
    #         print(f"cattle_to_drone_spacing Reward: {drone_to_cattle_spacing_reward:.4f}")
    #         print(f"Initial Combined Reward (before end-episode bonuses/penalties): {r:.4f}")

    #     # REWARD 6: End of episode reward
    #     term = self._computeTerminated()
    #     trun = self._computeTruncated()

    #     if term or trun:
    #         print("EPISODE ENDED!")

    #     if term:
    #         r += self._endOfEpisodeReward(cent_dist)
    #         self.curriculum.evaluate_curriculum_results(term)

    #     elif trun:
    #         r -= 50

    #     if term or trun:
    #         print("=== Current Curriculum Parameters ===")
    #         for key, value in self.curriculum.current_curriculum.items():
    #             print(f"{key:<30} : {value}")


           
    #     return float(r)

    def _computeReward(self):
        # Data
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2]
        cattle_poses = cattle_states[:, :2]

        # REWARD 1: Drone-to-drone spacing reward
        drone_to_drone_spacing_reward_simple = 0.0
        drone_to_drone_spacing_reward_complex = 0.0
        per_drone_spacing_rewards = np.zeros(self.NUM_DRONES)  ### DIFF-REWARD

        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
            other_dists[i] = np.inf  # ignore self

            # get two smallest distances
            nearest_two = np.partition(other_dists, 1)[:2]

            # sum rewards for the two nearest neighbours
            for dist in nearest_two:
                r_complex = self.DroneSpacingRewardFunction(dist)
                r_simple = self.SimpleSpacingReward(dist)
                drone_to_drone_spacing_reward_complex += r_complex
                drone_to_drone_spacing_reward_simple += r_simple
                per_drone_spacing_rewards[i] += (r_complex + r_simple) / 2.0  ### DIFF-REWARD

        # Average over drones and neighbours (2)
        drone_to_drone_spacing_reward_complex /= (self.NUM_DRONES * 2.0)
        drone_to_drone_spacing_reward_simple /= (self.NUM_DRONES * 2.0)

        # REWARD 2: Drone-Survival Reward
        drone_survival_reward = 0.1

        # REWARD 3: Cattle Approach Reward
        cattle_approach_reward = 0
        max_step_distance = self.SPEED_LIMIT / self.CTRL_FREQ
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
        if self.prev_cent_dists is not None:
            cent_dist_change = self.prev_cent_dists - cent_dist
            cattle_approach_reward = np.clip((cent_dist_change / (max_step_distance + 1e-6)) * 5, -1.0, 1.0)
        self.prev_cent_dists = cent_dist

        # REWARD 4: Herding effectiveness reward (global metric)
        effectiveness = evaluate_herding_effectiveness(cattle_poses, drones_poses)
        herding_effectiveness_reward = effectiveness / 100

        # REWARD 5: Drone-to-cattle spacing reward
        drone_to_cattle_spacing_reward = 0.0
        per_drone_cattle_rewards = np.zeros(self.NUM_DRONES)  ### DIFF-REWARD

        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            dists_to_cattle = np.linalg.norm(cattle_poses - pos_i, axis=1)
            closest_dist = np.min(dists_to_cattle)
            r_cattle = self.CattleSpacingRewardFunction(closest_dist)
            drone_to_cattle_spacing_reward += r_cattle
            per_drone_cattle_rewards[i] = r_cattle  ### DIFF-REWARD

        # Average over drones
        drone_to_cattle_spacing_reward /= self.NUM_DRONES

        # Combine global rewards
        r_global = (
            drone_to_drone_spacing_reward_simple * self.curriculum.current_curriculum["drone_to_drone_spacing_simple"]
            + drone_to_drone_spacing_reward_complex * self.curriculum.current_curriculum["drone_to_drone_spacing_complex"]
            + drone_survival_reward * self.curriculum.current_curriculum['drone_survival']
            + cattle_approach_reward * self.curriculum.current_curriculum['cattle_approach']
            + herding_effectiveness_reward * self.curriculum.current_curriculum['effectiveness']
            + drone_to_cattle_spacing_reward * self.curriculum.current_curriculum['cattle_to_drone_spacing']
        )

        ### ------------------ DIFF-REWARD ADDITION ------------------
        # Compute approximate per-drone difference rewards
        avg_other_spacing = np.mean(per_drone_spacing_rewards)
        avg_other_cattle = np.mean(per_drone_cattle_rewards)

        difference_rewards = (
            (per_drone_spacing_rewards - avg_other_spacing)
            + (per_drone_cattle_rewards - avg_other_cattle)
        )

        # Scale and mix with the global reward for stability
        per_drone_rewards = r_global + 0.5 * difference_rewards
        ### ---------------------------------------------------------

        if self.is_evaluating:
            print("\n==== REWARD Summary ====")
            print(f"Curriculum Level: {self.curriculum.curriculum_level}")
            print(f"drone_to_drone_spacing Reward: {drone_to_drone_spacing_reward_simple:.4f}")
            print(f"drone_survival Reward: {drone_survival_reward:.4f}")
            print(f"cattle_approach: {cattle_approach_reward:.4f}")
            print(f"effectiveness Reward: {herding_effectiveness_reward:.4f}")
            print(f"cattle_to_drone_spacing Reward: {drone_to_cattle_spacing_reward:.4f}")
            print(f"Initial Combined Reward (before end-episode bonuses/penalties): {r_global:.4f}")

        # REWARD 6: End of episode reward
        term = self._computeTerminated()
        trun = self._computeTruncated()

        if term or trun:
            print("EPISODE ENDED!")

        if term:
            r_global += self._endOfEpisodeReward(cent_dist)
            self.curriculum.evaluate_curriculum_results(term)
        elif trun:
            r_global -= 50

        if term or trun:
            print("=== Current Curriculum Parameters ===")
            for key, value in self.curriculum.current_curriculum.items():
                print(f"{key:<30} : {value}")

        ### Return per-drone rewards (for centralised multi-agent PPO)
        return float(np.mean(per_drone_rewards))


    ################################################################################

    def _endOfEpisodeReward(self, cent_dist):
        """
        Calculates reward at end of episode
        """


        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
        drones_poses = drone_states[:, :2] 
        cattle_poses = cattle_states[:, :2]

        if self.curriculum.curriculum_level in [0, 1]:
            satisfied_drones = 0
            total_reward = 0

            desired = self.curriculum.current_curriculum["drone_desired_distance"]
            tol = self.curriculum.current_curriculum["drone_spacing_tolerance"]
            upper_tolerance_range = desired + desired* tol
            lower_tolerance_range = desired - desired *  tol

            #for each drone in NUM_DRONE each drone that statsies:
            for i in range(self.NUM_DRONES):
                pos_i = drones_poses[i]
                other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
                other_dists[i] = np.inf  # ignore self

                # get two smallest distances
                nearest_two = np.partition(other_dists, 1)[:2]
                #if the distances are within the drone desired distance and tolerance give a reward euqapl to NUMDRONES/50, if all drones are satisfed retrun 100

                if np.all((nearest_two >= lower_tolerance_range) & (nearest_two <= upper_tolerance_range)):
                    total_reward += 50.0 / self.NUM_DRONES
                    satisfied_drones += 1
                    
            if satisfied_drones == self.NUM_DRONES:
                return 50.0
            
            return total_reward


        if self.curriculum.curriculum_level == 2 or self.curriculum.curriculum_level == 3:
            cattle_centroid = self.HerdCentroid()
            drone_centroid = self.DroneCentroid()
            cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
            if cent_dist < self.curriculum.current_curriculum["cattle_approach_min_distance"]:
                return 50

        if self.curriculum.curriculum_level == 4 or self.curriculum.curriculum_level == 6:
            herded_effectiveness = evaluate_herding_effectiveness(cattle_poses, drones_poses)
            if herded_effectiveness > self.curriculum.current_curriculum["min_effectiveness"]:
                return herded_effectiveness * 2
            
                
        if self.curriculum.curriculum_level == 5:
            herded_effectiveness = evaluate_herding_effectiveness(cattle_poses, drones_poses)
            if herded_effectiveness > self.curriculum.current_curriculum["min_effectiveness"]:
                satisfied_drones = 0
                total_reward = 0
                desired = self.curriculum.current_curriculum["cattle_desired_distance"]
                tol = self.curriculum.current_curriculum["cattle_spacing_tolerance"]
                upper_tolerance_range = desired + desired* tol
                lower_tolerance_range = desired - desired *  tol

                #for each drone in NUM_DRONE each drone that statsies:
                for i in range(self.NUM_DRONES):
                    pos_i = drones_poses[i]
                    other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
                    other_dists[i] = np.inf  # ignore self

                    # get two smallest distances
                    nearest_two = np.partition(other_dists, 1)[:2]
                    #if the distances are within the drone desired distance and tolerance give a reward euqapl to NUMDRONES/50, if all drones are satisfed retrun 100

                    if np.all((nearest_two >= lower_tolerance_range) & (nearest_two <= upper_tolerance_range)):
                        total_reward += self.NUM_DRONES / 50.0
                        satisfied_drones += 1
                        
                if satisfied_drones == self.NUM_DRONES:
                    return 50.0

        return 0


    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value."""

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2] 
        cattle_poses = cattle_states[:, :2]

        #SUCCESS CONDITION C1/2: Drones Spaced        
        #if the two cleost drones are spcaed within the tolerace win!!!! 
        if self.curriculum.curriculum_level == 0 or self.curriculum.curriculum_level == 1:
            desired = self.curriculum.current_curriculum["drone_desired_distance"]
            tol = self.curriculum.current_curriculum["drone_spacing_tolerance"]

            upper_tolerance_range = desired + desired * tol
            lower_tolerance_range = desired - desired * tol

            min_spacing = np.min([
                np.linalg.norm(d1 - d2)
                for i, d1 in enumerate(drones_poses)
                for j, d2 in enumerate(drones_poses)
                if i != j
            ])
            if min_spacing < upper_tolerance_range  and min_spacing >  lower_tolerance_range:
                self.drone_spacing_clock += 1/240
                if self.drone_spacing_clock >= self.curriculum.current_curriculum['drone_spacing_hold_timer']:
                    print("TERMINATED: Drones Spaced!")
                    return True
            else:
                self.drone_spacing_clock = 0
            
        #SUCCESS CONDITION C3/4: Cattle Approach
        #if the distacne to the cattle is wihtin toleracne win!!!!!!
        elif self.curriculum.curriculum_level == 2 or self.curriculum.curriculum_level == 3: 
            cattle_centroid = self.HerdCentroid()
            drone_centroid = self.DroneCentroid()
            cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)           
            if cent_dist < self.curriculum.current_curriculum["cattle_approach_min_distance"]:
                print("TERMINATED: Cattle Approached!")
                return True

        #SCUCCESS CONDITION C5: Surrounding Cattle
        elif self.curriculum.curriculum_level == 4 or self.curriculum.curriculum_level == 6:
            herded_effectiveness = evaluate_herding_effectiveness(cattle_poses, drones_poses)
            if herded_effectiveness > self.curriculum.current_curriculum["min_effectiveness"]:
                print("TERMINATED: Surrounded Cattle!")
                return True


        #SCUCCESS CONDITION C6: Cattle Spacing
        elif self.curriculum.curriculum_level == 5:
            herded_effectiveness = evaluate_herding_effectiveness(cattle_poses, drones_poses)
            if herded_effectiveness > self.curriculum.current_curriculum["min_effectiveness"]:
                desired = self.curriculum.current_curriculum["cattle_desired_distance"]
                tol = self.curriculum.current_curriculum["cattle_spacing_tolerance"]
                upper_tolerance_range = desired + desired * tol
                lower_tolerance_range = desired - desired * tol

                min_spacing = np.min([
                    np.linalg.norm(d1 - d2)
                    for i, d1 in enumerate(drones_poses)
                    for j, d2 in enumerate(drones_poses)
                    if i != j
                ])
                if min_spacing < upper_tolerance_range  and min_spacing >  lower_tolerance_range:
                    print("TERMINATED: Cattle Spacing!")
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
        drones_poses = drone_states[:, :2] 

        # FAILURE CONDITION 1: Altitude safety
        for i in range(self.NUM_DRONES):
            z = drone_states[i][2]
            if abs(z - self.DRONE_TARGET_ALTITUDE) > self.MAX_ALT_ERROR:
                print(f"TRUNCATED: Drone {i} Altitude loss with {z}")
                return True    

        # FAILURE CONDITION 2: Collision detection 
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                if dist < self.COLLISION_THRESHOLD:
                    print(f"TRUNCATED: Collision between drones {i} and {j}: {dist:.2f}m")
                    return True

        # FAILURE CONDITION 3: Formation breakdown
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
            other_dists[i] = np.inf  # Ignore self
            
            # Check if drone is isolated (too far from ALL others)
            if np.all(other_dists > self.MAX_FORMATTION_DISTANCE):
                print(f"TRUNCATED: Drone {i} isolated from formation, min distance: {np.min(other_dists):.2f}m")
                return True

        # FAILURE CONDITION 4: Mission area boundary
        if cent_dist > self.MISSION_BOUNDARY:
            print(f"TRUNCATED: Formation too far from herd: {cent_dist:.2f}m")
            return True
        
        # FAILURE CONDITION 5: Episode time limit
        elapsed_sec = self.step_counter / self.CTRL_FREQ
        if elapsed_sec > self.EPISODE_LEN_SEC:
            if self.is_evaluating:
                self.evaluation_episode_trigger()
                print("TRUNCATED: Episode time limit reached!")
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

    def CattleSpacingRewardFunction(self, r):
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

    def SimpleSpacingReward(self, r):
        """
        Reward function for drone spacing:
        - Reward = 1 when within target range (desired ± tolerance)
        - Smooth decay to -1 at distance = 0 or 7
        """
        import numpy as np

        desired = self.curriculum.current_curriculum['drone_desired_distance']  # e.g., 0.8
        tol_percent = self.curriculum.current_curriculum['drone_spacing_tolerance']  # e.g., 0.3

        # Convert tolerance from % to absolute distance
        tol = desired * tol_percent

        # Define the bounds of the "ideal" range
        lower_bound = desired - tol
        upper_bound = desired + tol

        # Case 1: within tolerance → full reward
        if lower_bound <= r <= upper_bound:
            return 1.0

        # Case 2: too close (r < lower_bound) → drop to -1 at r = 0
        elif r < lower_bound:
            return -1 + (r / lower_bound) * 2  # Linear scale from -1 (at 0) to +1 (at lower_bound)

        # Case 3: too far (r > upper_bound) → drop to -1 at r = 7
        elif r > upper_bound:
            max_range = 7.0
            return 1 - ((r - upper_bound) / (max_range - upper_bound)) * 2  # Linear scale down to -1

        # Safety default
        return -1.0



    def DroneSpacingRewardFunction(self, r, debug=False):
        """
        Combined spacing reward:
        - Gaussian peak at d_star (max ~1)
        - Collision linear penalty for r < collision_thresh
        - Linear attraction for r > long_range_start to pull drones inward
        Scaled so that at r = max_range, the long-range term = -1
        """
        import numpy as np

        # Parameters
        d_star = self.curriculum.current_curriculum['drone_desired_distance']
        sigma = 0.4
        collision_thresh = 0.3
        collision_penalty = 1.0

        # Long-range attraction parameters
        long_range_start = 1.5
        max_range = 5.0

        # Gaussian bump (peak near 1 at d_star)
        gauss = np.exp(-0.5 * ((r - d_star) / (sigma + 1e-9)) ** 2)

        # Collision penalty
        if r < collision_thresh:
            coll_pen = -collision_penalty * (1.0 - (r / (collision_thresh + 1e-9)))
        else:
            coll_pen = 0.0

        # Long-range attraction
        if r > long_range_start:
            pull = -0.3 * (r - long_range_start) / (max_range - long_range_start)
        else:
            pull = 0.0

        # Combined reward
        reward = gauss + coll_pen + pull
        reward += 0.1 * (1 - abs(r - d_star))


        if debug:
            print(f"[DroneSpacingRewardFunction DEBUG]")
            print(f"  r = {r:.3f}")
            print(f"  Desired (d_star) = {d_star:.3f}")
            print(f"  Gaussian term = {gauss:.3f}")
            print(f"  Collision penalty = {coll_pen:.3f} (applied? {'YES' if r < collision_thresh else 'NO'})")
            print(f"  Long-range pull = {pull:.3f} (applied? {'YES' if r > long_range_start else 'NO'})")
            print(f"  --> Total reward = {reward:.3f}")

        return float(reward)