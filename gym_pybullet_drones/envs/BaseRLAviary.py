import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 num_cattel: int =1,
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
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
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
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

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
        for k in range(action.shape[0]):
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
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
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
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.COKIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])

            ###########################################################
            ### Add drone–drone distances (NUM_DRONES - 1 per drone)
            ### Distances are always >= 0

            # Absolute positions of other drones (exclude self)
            drone_abs_lo = np.zeros((self.NUM_DRONES, (self.NUM_DRONES-1) * 3))
            drone_abs_hi = np.full((self.NUM_DRONES, (self.NUM_DRONES-1) * 3), hi)

            # Relative positions of other drones (exclude self)
            drone_rel_lo = np.zeros((self.NUM_DRONES, (self.NUM_DRONES-1) * 3))
            drone_rel_hi = np.full((self.NUM_DRONES, (self.NUM_DRONES-1) * 3), hi)

            # Combine absolute + relative
            obs_lower_bound = np.hstack([obs_lower_bound, drone_abs_lo, drone_rel_lo])
            obs_upper_bound = np.hstack([obs_upper_bound, drone_abs_hi, drone_rel_hi])

            ############################################################
            #### Add drone–cattle distances (NUM_CATTLE per drone)
            # Absolute cow positions
            cattle_abs_lo = -np.inf * np.ones((self.NUM_DRONES, self.NUM_CATTLE * 3))
            cattle_abs_hi = +np.inf * np.ones((self.NUM_DRONES, self.NUM_CATTLE * 3))

            # Relative cow positions (existing)
            cattle_rel_lo = np.zeros((self.NUM_DRONES, self.NUM_CATTLE * 3))
            cattle_rel_hi = np.full((self.NUM_DRONES, self.NUM_CATTLE * 3), hi)

            # Combine absolute + relative
            obs_lower_bound = np.hstack([obs_lower_bound, cattle_abs_lo, cattle_rel_lo])
            obs_upper_bound = np.hstack([obs_upper_bound, cattle_abs_hi, cattle_rel_hi])

            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])

            ############################################################  
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observations of all drones in the environment.

        Returns
        -------
        ndarray
            Observation array of shape (NUM_DRONES, obs_length), where obs_length depends on OBS_TYPE,
            including drone state, relative positions to other drones, absolute and relative positions to cows, 
            relative vector to centroid, and action buffer.
        """
        if self.OBS_TYPE != ObservationType.COKIN:
            print("[ERROR] in BaseRLAviary._computeObs()")
            return None

        N = self.NUM_DRONES
        M = self.NUM_CATTLE

        ret_list = []

        for i in range(N):
            #### Add drone state #######################
            obs_vec = self._getDroneStateVector(i)
            drone_pos = obs_vec[0:3]
            obs_i = list(np.hstack([
                drone_pos,            # x, y, z
                obs_vec[7:10],        # velocity
                obs_vec[10:13],       # angular velocity
                obs_vec[13:16]        # orientation / rotation
            ]))

            # #### Add drone to drone relative positions #######################
            for j in range(N):
                if i == j:
                    continue
                other_drone_pos = self._getDroneStateVector(j)[0:3]
                rel_pos = other_drone_pos - drone_pos
                obs_i.extend(other_drone_pos)
                obs_i.extend(rel_pos)

            #### Add absolute and relative cow positions #######################
            for j in range(M):
                cow_pos = self._getCowStateVector(j)[0:3]
                rel_pos = cow_pos - drone_pos
                obs_i.extend(cow_pos)   # absolute position
                obs_i.extend(rel_pos)   # relative to drone

            #### Add action buffer to observation #######################
            for k in range(self.ACTION_BUFFER_SIZE):
                obs_i.extend(self.action_buffer[k][i, :])

            ret_list.append(obs_i)

        ret = np.array(ret_list, dtype='float32')
        return ret
