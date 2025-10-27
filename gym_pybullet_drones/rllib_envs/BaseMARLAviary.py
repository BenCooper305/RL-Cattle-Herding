import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.rllib_envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class BaseMARLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 num_cattle: int =1,
                 max_num_drones: int = 12,
                 min_num_drones: int = 4,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.COKIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(0.15 * ctrl_freq)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.ACTION_SPACE = 4
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         num_cattle=num_cattle,
                         max_num_drones=max_num_drones,
                         min_num_drones=min_num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.3 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of a single drone (MARL-friendly).

        Returns
        -------
        gym.spaces.Box
            A Box of size 4, 3, or 1, depending on the action type.
        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseMARLAvairy._actionSpace()")
            exit()

        self.ACTION_SPACE = size

        # Initialize the action buffer for all drones
        self.action_buffer = [np.zeros((self.NUM_DRONES, size), dtype=np.float32)
                            for _ in range(self.ACTION_BUFFER_SIZE)]

        return spaces.Box(low=-1*np.ones(size), high=+1*np.ones(size), dtype=np.float32)


    ###############################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(self.NUM_DRONES):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                # Normalize horizontal component of velocity target
                horiz_vel = target[0:2]
                norm = np.linalg.norm(horiz_vel)
                if norm != 0:
                    v_unit_xy = horiz_vel / norm
                else:
                    v_unit_xy = np.zeros(2)

                # Construct full 3D target velocity, zero vertical component since altitude is fixed
                target_vel = np.array([
                    v_unit_xy[0],
                    v_unit_xy[1],
                    0.0
                ]) * (self.SPEED_LIMIT * abs(target[3]))

                # Define fixed target position: current x, y plus desired altitude
                target_pos = np.array([
                    state[0],
                    state[1],
                    self.DRONE_TARGET_ALTITUDE
                ])

                # Keep current yaw by using current yaw angle (state[9] assumed to be yaw)
                target_rpy = np.array([0.0, 0.0, state[9]])

                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=target_pos,
                    target_rpy=target_rpy,
                    target_vel=target_vel
                )
                rpm[k, :] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm


    ################################################################################
   
    def _observationSpace(self):
        """Returns the observation space of a single drone."""

        obs_len = 10 + self.MAX_NEIGHBORS*2 + self.MAX_NEARBY_CATTLE*2 + self.ACTION_BUFFER_SIZE * self.ACTION_SPACE
        lo = -np.inf * np.ones(obs_len, dtype=np.float32)
        hi = np.inf * np.ones(obs_len, dtype=np.float32)

        return spaces.Box(low=lo, high=hi, dtype=np.float32)


    ################################################################################

    def _computeObs(self, drone_id):
        """Returns the current observation for a single drone."""
        N = self.NUM_DRONES
        M = self.NUM_CATTLE
        obs_len_per_drone = 10 + self.MAX_NEIGHBORS*2 + self.MAX_NEARBY_CATTLE*2 + self.ACTION_BUFFER_SIZE * self.ACTION_SPACE

        obs_vec = self._getDroneStateVector(drone_id)
        drone_pos = obs_vec[0:3]

        # Own state: RPY + linear vel + angular vel
        obs_i = list(np.hstack([
            obs_vec[2],
            obs_vec[7:10],
            obs_vec[10:13],
            obs_vec[13:16],
        ]))

        # Relative positions of neighbors
        rel_neighbors = []
        for j in range(N):
            if drone_id == j:
                continue
            other_pos = self._getDroneStateVector(j)[0:2]
            rel_neighbors.append((other_pos - drone_pos[0:2], np.linalg.norm(other_pos - drone_pos[0:2])))
        rel_neighbors.sort(key=lambda x: x[1])
        rel_neighbors = [vec for vec, dist in rel_neighbors[:self.active_neighbors]]
        while len(rel_neighbors) < self.MAX_NEIGHBORS:
            rel_neighbors.append(np.zeros(2))
        obs_i.extend(np.array(rel_neighbors).flatten())

        # Relative positions of nearby cattle
        rel_cattle = []
        for j in range(M):
            cow_pos = self._getCowStateVector(j)[0:2]
            rel_cattle.append(cow_pos - drone_pos[0:2])
        while len(rel_cattle) < self.MAX_NEARBY_CATTLE:
            rel_cattle.append(np.zeros(2))
        obs_i.extend(np.array(rel_cattle[:self.MAX_NEARBY_CATTLE]).flatten())

        # Action buffer
        for k in range(self.ACTION_BUFFER_SIZE):
            obs_i.extend(self.action_buffer[k][drone_id, :])

        # Pad/truncate
        obs_i_array = np.array(obs_i, dtype=np.float32)
        if len(obs_i_array) < obs_len_per_drone:
            obs_i_array = np.pad(obs_i_array, (0, obs_len_per_drone - len(obs_i_array)), 'constant')
        else:
            obs_i_array = obs_i_array[:obs_len_per_drone]

        return obs_i_array


    ################################################################################  
    
    def HerdCentroid(self):
        """
        Calculates the center of the herd and updates the visual centroid marker to that location.
        """

        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
        cattle_positions = cattle_states[:, 0:3]

        # Compute XY centroid
        centroid_xy = np.mean(cattle_positions[:, :2], axis=0)
        centroid = np.array([centroid_xy[0], centroid_xy[1], self.DRONE_TARGET_ALTITUDE+0.5])

        # Update visual marker position
        p.resetBasePositionAndOrientation(
            self.CattleCentroidMarker,
            centroid,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )

        return centroid

    ################################################################################

    def DroneCentroid(self):
        """
        Calculates the center of the drones and updates the visual centroid marker to that location.
        """

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        drone_positions = drone_states[:, 0:3]

        # Compute XY centroid
        centroid_xy = np.mean(drone_positions[:, :2], axis=0)
        centroid = np.array([centroid_xy[0], centroid_xy[1], self.DRONE_TARGET_ALTITUDE+0.5])

        # Update visual marker position
        p.resetBasePositionAndOrientation(
            self.DroneCentroidMarker,
            centroid,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )

        return centroid

