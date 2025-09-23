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
        self.MAX_VEL = 2
        self.MAX_DIST = 7
        self.prev_dists = None
        self.prev_cent_dists = None

        self.SPACING_A = 1.5 #pos amp coef
        self.SPACING_B = 2.4 #neg amp coef
        self.SPACING_C = 0.8 #width of pos coef
        self.SPACING_K = 0.3 #width of neg coef
        self.SPACING_D = 0.1 #pos peak offset
        self.SPACING_R0 = 1.25 #piecewise threshold
        self.SPACING_LAM = 0.7 #exp decay

        self.MAX_ALT_ERROR = self.DRONE_TARGET_ALTITUDE * 0.4

        self.REWARD_WEIGHTS = dict(drone_hull_distance=1, 
                                   drone_hull_approach=0.5, 
                                   centroid=1, 
                                   centroid_approach = 0.3,
                                   drone_spacing = 0.4
                                   )
    ################################################################################
    

    def _computeReward(self):
        """Reward: move toward hull points, maintain spacing, herd centroid."""

        # --- Data ---
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        pos = states[:, :3]        # drone positions
        vel = states[:, 10:13]     # drone velocities

        # --- Centroid rewards (smaller weight) ---
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
        if self.prev_cent_dists is None:
            self.prev_cent_dists = cent_dist
        cent_dist_change = self.prev_cent_dists - cent_dist
        centroid_distance_reward = cent_dist_change / (self.MAX_DIST + 1e-6)
        self.prev_cent_dists = cent_dist

        dir_to_cattle = cattle_centroid[:2] - pos[:, :2]
        dists_to_cattle = np.linalg.norm(dir_to_cattle, axis=1)
        dir_unit = np.where(dists_to_cattle[:, None] > 0, dir_to_cattle / dists_to_cattle[:, None], 0.0)
        centroid_approach_reward = np.mean(np.sum(vel[:, :2] * dir_unit, axis=1)) / (self.MAX_VEL + 1e-6)

        # --- Drone spacing ---
        drone_spacing_reward = 0.0
        for i in range(self.NUM_DRONES):
            pos_i = pos[i, :2]
            for j in range(i + 1, self.NUM_DRONES):
                pos_j = pos[j, :2]
                dist = np.linalg.norm(pos_i - pos_j)
                drone_spacing_reward += self.SpacingRewardValue(dist)
            dist_cent = np.linalg.norm(pos_i - drone_centroid[:2])
            drone_spacing_reward += self.SpacingRewardValue(dist_cent)
        num_pairs = self.NUM_DRONES * (self.NUM_DRONES - 1) / 2 + self.NUM_DRONES
        drone_spacing_reward /= num_pairs

        # # --- Hull point rewards (stop-and-hold) ---
        # drone_hull_distance_reward = 0.0
        # drone_hull_approach_reward = 0.0
        # hull_positions = [np.array(p[:2]) for p in self.hullMarkerPositions]

        # sigma = 0.2       # smaller sigma for sharper peak
        # vel_penalty_factor = 0.2  # penalise velocity near target

        # if hull_positions:
        #     for i in range(self.NUM_DRONES):
        #         drone_pos = pos[i, :2]

        #         # Assign nearest hull point
        #         hull_dists = np.array([np.linalg.norm(drone_pos - hp) for hp in hull_positions])
        #         target_idx = np.argmin(hull_dists)
        #         target_hull = hull_positions[target_idx]
        #         dist_to_hull = hull_dists[target_idx]

        #         # --- Proximity reward with sharp peak at hull ---
        #         # Max reward at hull point, decays quickly with distance
        #         proximity_reward = self.SpacingRewardValue(dist_to_hull)

        #         # --- Directional reward with stopping behaviour ---
        #         dir_vec = target_hull - drone_pos
        #         dir_unit = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
        #         vel_toward_target = np.dot(vel[i, :2], dir_unit)

        #         # Encourage moving toward target when far, penalise moving when close
        #         if dist_to_hull > sigma:  # far from target → encourage moving
        #             directional_reward = np.clip(vel_toward_target / self.MAX_VEL, -1, 1)
        #             vel_penalty = 0.0
        #         else:  # near target → penalise velocity to encourage stop
        #             directional_reward = 0.0
        #             vel_penalty = vel_penalty_factor * (np.linalg.norm(vel[i, :2]) / self.MAX_VEL)

        #         drone_hull_approach_reward += directional_reward
        #         drone_hull_distance_reward += proximity_reward - vel_penalty

        #     drone_hull_approach_reward /= self.NUM_DRONES
        #     drone_hull_distance_reward /= self.NUM_DRONES

        # --- Combine rewards ---
        r = (
            centroid_distance_reward * self.REWARD_WEIGHTS["centroid"]
            + centroid_approach_reward * self.REWARD_WEIGHTS["centroid_approach"]
            + drone_spacing_reward * self.REWARD_WEIGHTS["drone_spacing"]
            # + drone_hull_approach_reward * self.REWARD_WEIGHTS["drone_hull_approach"]
            # + drone_hull_distance_reward * self.REWARD_WEIGHTS["drone_hull_distance"]
        )

        return float(r)


    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value."""
        hull_points = np.array(self.get_hull_positions())
        if hull_points.size == 0:
            return False
        
        hull_points = hull_points[:, :2]  # (H,2)
        drone_positions = np.array([self._getDroneStateVector(i)[:2] for i in range(self.NUM_DRONES)])  # (N,2)

        # Compute distance matrix
        dist_matrix = np.linalg.norm(hull_points[:, None, :] - drone_positions[None, :, :], axis=-1)

        hull_covered = np.any(dist_matrix <= 0.7, axis=1)
        done = np.all(hull_covered)

        if done and self.is_evaluating:
            self.evaluation_episode_trigger()

        return done


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
            #print("FAIL 3!!!!!!!!!!!!!")
            if self.is_evaluating:
                self.evaluation_episode_trigger()
            return True
        
        for i in range(self.NUM_DRONES):
            z = states[i][2]
            if abs(z - self.DRONE_TARGET_ALTITUDE) > self.MAX_ALT_ERROR:
                # print("FAIL 2!!!!!!!!!!!!!")
                if self.is_evaluating:
                    self.evaluation_episode_trigger()
                return True       
        
        # --- Episode timeout ---self.step_counter
        elapsed_sec = self.step_counter / self.CTRL_FREQ
        if elapsed_sec > self.EPISODE_LEN_SEC:
            if self.is_evaluating:
                self.evaluation_episode_trigger()
            # print(f"FAIL: Episode exceeded {self.EPISODE_LEN_SEC} s")
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