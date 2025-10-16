import os
import pickle
import numpy as np

class evaluator():
    def __init__(self):
        # Episode-level
        self.total_drone_distances = []
        self.total_time_taken = []
        self.total_effectiveness = []
        self.total_number_of_drones = []
        self.total_drone_poses = []
        self.total_cattle_poses = []
        self.total_drone_vel = []
        self.total_cattle_vel = []

        # Timestep-level: store as list-of-lists per episode
        self.drone_distances_per_step = []
        self.effectiveness_per_step = []
        self.time_per_step = []
        self.drone_poses_per_step = []
        self.cattle_poses_per_step = []
        self.drone_vel_per_step = []
        self.cattle_vel_per_step = []

        # Temporary buffers for current episode
        self.curr_drone_poses = []
        self.curr_cattle_poses = []
        self.curr_drone_vel = []
        self.curr_cattle_vel = []
        self.curr_drone_distances = []
        self.curr_effectiveness = []
        self.curr_time = []


    # --- Called every timestep ---
    def append_timestep_data(self, drone_distances, timestep_time, effectiveness, drone_poses, cattle_poses, drone_vel, cattle_vel):
        self.curr_drone_distances.append(drone_distances)
        self.curr_time.append(timestep_time)
        self.curr_effectiveness.append(effectiveness)
        self.curr_drone_poses.append(drone_poses)
        self.curr_cattle_poses.append(cattle_poses)
        self.curr_drone_vel.append(drone_vel)
        self.curr_cattle_vel.append(cattle_vel)

    
    # --- Called at the end of an episode ---
    def append_episode_data(self, drone_distances, num_drones, time, effectiveness):
        # Save episode-level data
        self.total_drone_distances.append(drone_distances)
        self.total_number_of_drones.append(num_drones)
        self.total_time_taken.append(time)
        self.total_effectiveness.append(effectiveness)

        # Save timestep-level data for this episode
        self.drone_poses_per_step.append(self.curr_drone_poses)
        self.cattle_poses_per_step.append(self.curr_cattle_poses)
        self.drone_vel_per_step.append(self.curr_drone_vel)
        self.cattle_vel_per_step.append(self.curr_cattle_vel)
        self.drone_distances_per_step.append(self.curr_drone_distances)
        self.time_per_step.append(self.curr_time)
        self.effectiveness_per_step.append(self.curr_effectiveness)

        # Reset temporary buffers
        self.curr_drone_poses = []
        self.curr_cattle_poses = []
        self.curr_drone_vel = []
        self.curr_cattle_vel = []
        self.curr_drone_distances = []
        self.curr_effectiveness = []
        self.curr_time = []

    def save_evaluation_data(self, save_path="evaluation_data.pkl"):
        eval_data = {
            "distances": self.total_drone_distances,
            "num_drones": self.total_number_of_drones,
            "time_taken": self.total_time_taken,
            "effectiveness": self.total_effectiveness,

            "distances_per_step": self.drone_distances_per_step,
            "time_per_step": self.time_per_step,
            "effectiveness_per_step": self.effectiveness_per_step,

            "drone_poses_per_step": self.drone_poses_per_step,
            "cattle_poses_per_step": self.cattle_poses_per_step,
            "drone_vel_per_step": self.drone_vel_per_step,
            "cattle_vel_per_step": self.cattle_vel_per_step,

        }

        with open(save_path, "wb") as f:
            pickle.dump(eval_data, f)

        print(f"Evaluation data saved to {os.path.abspath(save_path)}")

########################################
#### ---- Evaluation Functions ---- ####
########################################

def evaluate_herding_effectiveness(cattle_poses, drones_poses):
    # Convert to numpy array
    polygon = np.array(drones_poses)
    cattle_poses = np.array(cattle_poses)

    # Ensure 2D shape: each row = [x, y]
    if polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)
    elif polygon.ndim == 3:  # sometimes hull returns shape (n,1,2)
        polygon = polygon.reshape(-1, 2)

    if cattle_poses.ndim == 1:
        cattle_poses = cattle_poses.reshape(-1, 2)
    elif cattle_poses.ndim == 3:
        cattle_poses = cattle_poses.reshape(-1, 2)

    total_herded_cattle = 0

    for cow_pose in cattle_poses:
        px, py = cow_pose
        wn = 0
        n = len(polygon)

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            if y1 <= py:
                if y2 > py and is_left((x1, y1), (x2, y2), (px, py)) > 0:
                    wn += 1
            else:
                if y2 <= py and is_left((x1, y1), (x2, y2), (px, py)) < 0:
                    wn -= 1

        if wn:
            total_herded_cattle += 1 

    effectiveness = total_herded_cattle / len(cattle_poses) * 100 if len(cattle_poses) > 0 else 0
    return effectiveness

def evaluate_formation_quality(drones_poses):
    """Evaluate overall formation quality (0-1 score)"""
    N = len(drones_poses)
    if N < 2:
        return 1.0
    
    # Check spacing consistency
    spacing_score = 0.0
    target_spacing = 1.75
    pair_count = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
            # Score based on how close to target spacing
            spacing_score += np.exp(-((dist - target_spacing) ** 2) / (2 * 0.5 ** 2))
            pair_count += 1
    
    spacing_score = spacing_score / pair_count if pair_count > 0 else 0.0
    
    # Check formation structure
    structure_score = max(
        self._evaluate_line_formation(drones_poses),
        self._evaluate_v_formation(drones_poses)
    )
    
    # Combine scores
    return (spacing_score * 0.6 + structure_score * 0.4)


def evaluate_line_formation(drones_poses):
        """Evaluate how well drones form a line"""
        N = len(drones_poses)
        if N < 3:
            return 0.0
        
        # Sort drones by x-coordinate to find line direction
        sorted_indices = np.argsort(drones_poses[:, 0])
        sorted_poses = drones_poses[sorted_indices]
        
        # Calculate line from first to last drone
        if N == 2:
            return 1.0  # Two drones always form a perfect line
        
        start_point = sorted_poses[0]
        end_point = sorted_poses[-1]
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        
        if line_length < 0.1:  # Drones too close together
            return 0.0
        
        line_unit = line_vector / line_length
        
        # Calculate deviation from line for each intermediate drone
        total_deviation = 0.0
        for i in range(1, N-1):
            point = sorted_poses[i]
            # Vector from start to this point
            to_point = point - start_point
            # Project onto line direction
            projection_length = np.dot(to_point, line_unit)
            projection_point = start_point + projection_length * line_unit
            # Distance from line
            deviation = np.linalg.norm(point - projection_point)
            total_deviation += deviation
        
        # Convert to reward (lower deviation = higher reward)
        avg_deviation = total_deviation / max(1, N-2)
        line_reward = np.exp(-avg_deviation / 0.5)  # Decay with deviation
        
        return line_reward
    
def evaluate_v_formation(self, drones_poses):
    """Evaluate how well drones form a V-formation"""
    N = len(drones_poses)
    if N < 3:
        return 0.0
    
    # Find the drone that could be the apex (furthest forward in y-direction)
    center_y = np.mean(drones_poses[:, 1])
    apex_candidates = []
    
    for i, pos in enumerate(drones_poses):
        if pos[1] > center_y - 0.5:  # Near or ahead of center
            apex_candidates.append((i, pos))
    
    if not apex_candidates:
        return 0.0
    
    best_v_reward = 0.0
    
    for apex_idx, apex_pos in apex_candidates:
        # Calculate V-formation score with this apex
        other_drones = [drones_poses[i] for i in range(N) if i != apex_idx]
        
        if len(other_drones) < 2:
            continue
        
        # Split other drones into left and right wings
        left_wing = []
        right_wing = []
        
        for pos in other_drones:
            relative_x = pos[0] - apex_pos[0]
            if relative_x < -0.2:  # Left wing
                left_wing.append(pos)
            elif relative_x > 0.2:  # Right wing
                right_wing.append(pos)
        
        if len(left_wing) == 0 or len(right_wing) == 0:
            continue
        
        # Check symmetry and alignment
        v_score = 0.0
        
        # Reward balanced wings
        wing_balance = 1.0 - abs(len(left_wing) - len(right_wing)) / max(len(left_wing), len(right_wing))
        v_score += wing_balance * 0.5
        
        # Reward proper V-angle (wings behind apex)
        left_behind = all(pos[1] < apex_pos[1] + 0.5 for pos in left_wing)
        right_behind = all(pos[1] < apex_pos[1] + 0.5 for pos in right_wing)
        
        if left_behind and right_behind:
            v_score += 0.5
        
        best_v_reward = max(best_v_reward, v_score)
    
    return best_v_reward

def is_left(p0, p1, p2):
    """Tests if p2 is left of the directed line p0 -> p1"""
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
            