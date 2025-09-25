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