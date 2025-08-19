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
        self.EPISODE_LEN_SEC = 8
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
        self.REWARD_WEIGHTS = dict(progress=1.0, proximity=0.5, control=-0.01)
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value with directional guidance and collision avoidance."""
        
        # Interaction force parameters
        # d2d_a, d2d_c, d2d_d = 1.0, 1.0, 1.0  # drone-drone
        # d2c_a, d2c_c, d2c_d = 1.0, 1.0, 1.0  # drone-cattle
        
        # states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # reward = 0.0
        
        # cattleCentroid = self.HerdCentroid()
        
        # for i in range(self.NUM_DRONES):
        #     drone_pos = states[i][0:3]
        #     drone_vel = states[i][7:10]

            
        #     # --- Vector toward herd centroid ---
        #     to_centroid_vec = cattleCentroid - drone_pos
        #     dist_to_centroid = np.linalg.norm(to_centroid_vec)
            
        #     if dist_to_centroid > 0:
        #         dir_to_centroid = to_centroid_vec / dist_to_centroid  # unit vector
        #     else:
        #         dir_to_centroid = np.zeros(3)

        #     #print(f"Drone: {i}, has pos x:{round(drone_pos[0],2)} y:{round(drone_pos[1],2)}, z:{round(drone_pos[2],2)} the vector to cenotird is: {dir_to_centroid}, distance is: {dist_to_centroid}, and has pos x:{round(cattleCentroid[0],2)} y:{round(cattleCentroid[1],2)}, z:{round(cattleCentroid[2],2)} ")
        #     # --- Reward for moving toward centroid ---
        #     reward += 10.0 / (1.0 + dist_to_centroid)          # proximity
        #     #reward += 1 * np.dot(dir_to_centroid, drone_vel) # directional bonus
        #    # print(f"reward given is {10.0 / (1.0 + dist_to_centroid) }")

                        
        #     # # --- Drone-drone repulsion / spacing ---
        #     # for j in range(self.NUM_DRONES):
        #     #     if i == j:
        #     #         continue
        #     #     other_pos = states[j][0:3]
        #     #     f_vec = self.InteractionForce(drone_pos, other_pos, d2d_a, d2d_c, d2d_d)
        #     #     reward -= 0.01 * np.linalg.norm(f_vec)  # small penalty if too close
            
        #     # # --- Drone-cattle interaction (avoid collisions) ---
        #     # for j in range(self.NUM_CATTLE):
        #     #     cow_pos = self._getCowStateVector(j)[0:3]
        #     #     f_vec = self.InteractionForce(drone_pos, cow_pos, d2c_a, d2c_c, d2c_d)
        #     #     reward -= 0.01 * np.linalg.norm(f_vec)  # small penalty if too close

        # return reward
    
        centroid = self.HerdCentroid()
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        pos = states[:, 0:3]
        vel = states[:, 10:13]  # actual velocity vector

        dists = np.linalg.norm(pos - centroid, axis=1)
        dir_to_centroid = centroid - pos
        dir_unit = np.where(dists[:, None] > 0, dir_to_centroid / dists[:, None], 0.0)

        # Reward for moving toward centroid
        approach_reward = np.mean(np.sum(vel * dir_unit, axis=1))

        # Dense proximity reward
        proximity_reward = np.mean(1.0 / (1.0 + dists))

        # Control penalty
        control_pen = 0.0
        if hasattr(self, "last_action"):
            control_pen = np.mean(np.sum(self.last_action**2, axis=1))

        r = (
            self.REWARD_WEIGHTS.get("proximity", 1.0) * proximity_reward
            + self.REWARD_WEIGHTS.get("approach", 1.0) * approach_reward
            - self.REWARD_WEIGHTS.get("control", 1.0) * control_pen
        )

        # Store for next step
        self._last_dist_to_centroid = dists.copy()

        # Debug info
        self.last_reward_info = {
            "proximity": float(proximity_reward),
            "approach": float(approach_reward),
            "control_pen": float(control_pen),
            "mean_dist": float(np.mean(dists)),
        }

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
        dist = 0
        centroid = self.HerdCentroid()
        for i in range(self.NUM_DRONES):
            dist += np.linalg.norm(centroid-states[i][0:3])
        if dist < .01:
            return True
        else:
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

        # for i in range(self.NUM_DRONES):
        #     #--- Tilt check (roll/pitch) ---
        #     roll = states[i][7]
        #     pitch = states[i][8]
        #     if abs(roll) > 0.8 or abs(pitch) > 0.8:
        #         #print("TRUNCATED!!!!!!!!! PITCH")
        #         return True  # too tilted

        #     #--- Altitude check ---
        #     z = states[i][2]
        #     if abs(z - self.DRONE_TARGET_ALTITUDE) > 0.25:
        #         #print("TRUNCATED!!!!!!!!! ALT")
        #         return True  # too far from target altitude

        #     # --- Velocity check ---
        #     vel = states[i][10:13]
        #     max_vel = 8.0  # tune based on your simulation
        #     if np.linalg.norm(vel) > max_vel:
        #         #print("TRUNCATED!!!!!!!!! VEL")
        #         return True  # too fast

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
    