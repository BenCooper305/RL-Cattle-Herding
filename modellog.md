rwd_exp v1-0
looking at cattle drone spaing

rwd_exp 1-[1-4]
looking at drone to drone

rwd_exp v1-5-1
spaced out drones more so they dont start so close together and are on the oposite side of the spacing function



rwd_exp-v1-5-6

        self.SPACING_A = 1.5 #pos amp coef
        self.SPACING_B = 2.4 #neg amp coef
        self.SPACING_C = 0.8 #width of pos coef
        self.SPACING_K = 0.3 #width of neg coef
        self.SPACING_D = 0.1 #pos peak offset
        self.SPACING_R0 = 1.25 #piecewise threshold
        self.SPACING_LAM = 0.7 #exp decay

        drones did not converg over 8.4million steps

rwd_exp-v1-5-7

        self.SPACING_A = 1.2 #pos amp coef
        self.SPACING_B = 2.1 #neg amp coef
        self.SPACING_C = 3.3 #width of pos coef
        self.SPACING_K = 0.2 #width of neg coef
        self.SPACING_D = -1 #pos peak offset
        self.SPACING_R0 = 1.3 #piecewise threshold
        self.SPACING_LAM = 0.8 #exp decay

        drones seem to converge better

rwd_exp-v1-5-8 extra training


Centroid distance reward
rwd_exp-v2-1-0
build on rwd_exp-v1-5-8 with centoird distacne reward, both with weithgin = 1

Added Terminate Reward for centroid reaching herd

rwd_exp-v3-1-0
BUil;d on v2 with varyiosu number of drones and 16 cattle
and uncomented this: drone_spacing_reward /= self.NUM_DRONES * 3  # 2 nearest + 1 centroid per drone

wieghts:
drone_to_drone_spacing = 0.8,
centroid_distance= 1, 



removed:
        # dir_to_cattle = cattle_centroid[:2] - drones_poses[:, :2]
        # dists_to_cattle = np.linalg.norm(dir_to_cattle, axis=1)
        # dir_unit = np.where(dists_to_cattle[:, None] > 0, dir_to_cattle / dists_to_cattle[:, None], 0.0)
        # centroid_approach_reward = np.mean(np.sum(drone_vel[:, :2] * dir_unit, axis=1)) / (self.MAX_VEL + 1e-6)

model v10-1
rewward:

    def _computeReward(self):
        #Data
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2]
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

        #Average over all drones and terms
        drone_to_drone_spacing_reward /= self.NUM_DRONES * 2  # 2 nearest

        #Centroid Distance Reward
        # cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
        # if self.prev_cent_dists is None:
        #     self.prev_cent_dists = cent_dist
        # cent_dist_change = self.prev_cent_dists - cent_dist
        # centroid_distance_reward = cent_dist_change / (0.2 + 1e-6)
        # self.prev_cent_dists = cent_dist

        #Drone to Cattle Spacing
        drone_to_cattle_spacing_reward = 0.0
        for i in range(self.NUM_DRONES):
            pos_i = drones_poses[i]
            dists_to_cattle = np.linalg.norm(cattle_poses - pos_i, axis=1)
            closest_dist = np.min(dists_to_cattle)
            drone_to_cattle_spacing_reward += self.SpacingRewardValue(closest_dist)

        #Average over drones
        drone_to_cattle_spacing_reward /= self.NUM_DRONES

        small_step_penalty = 0.01

        #Combine Rewards
        r = (
             centroid_distance_reward * self.REWARD_WEIGHTS["centroid_distance"]
             + drone_to_drone_spacing_reward * self.REWARD_WEIGHTS["drone_to_drone_spacing"]
             + drone_to_cattle_spacing_reward * self.REWARD_WEIGHTS["drone_to_cattle_spacing"]
             - small_step_penalty
        )

        #End of Episode Rewards
        done = self._computeTerminated() or self._computeTruncated()
        if done:

            #Reward for drone centorid being near cattle centroid
            if cent_dist < self.TERMINATION_CENTROID_THRESH:
                r+= 50
            else:
                r -= cent_dist * 1.5

            #Reward for effectivness
            effectiveness = self.eval_system.calculate_effectiveness(cattle_poses,drones_poses)
            if effectiveness == 100: #missive reward for herding all cattle
                r += 100
            elif effectiveness == 0: #negative reward for herding nothing
                r -= 25
            else:
                r +=  effectiveness/10 #bonus 1 - 9 points for number of cattle herded

        return float(r)

v10-2 Flocking disabled Reward:

