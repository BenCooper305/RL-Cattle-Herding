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