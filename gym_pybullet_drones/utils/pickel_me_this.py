import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def load_eval_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def print_all_data(data):
    print("\n--- All Evaluation Data ---")
    for key, value in data.items():
        print(f"\n{key} ({type(value)}):")
        if isinstance(value, list):
            for i, v in enumerate(value):
                print(f"  Episode {i+1}: {v}")
        elif isinstance(value, np.ndarray):
            print(value)
        else:
            print(value)

def print_means(data):
    print("\n--- Mean values ---")
    for key, value in data.items():
        if isinstance(value, list):
            # Flatten list carefully
            flat_values = []
            for v in value:
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    # If sublist is array, convert to float and take mean per sublist
                    try:
                        flat_values.append(np.mean(np.array(v, dtype=float)))
                    except:
                        continue
                elif v is not None:
                    flat_values.append(float(v))
            if flat_values:
                print(f"{key}: {np.mean(flat_values):.3f}")
            else:
                print(f"{key}: No data")
        elif isinstance(value, np.ndarray):
            print(f"{key}: {np.mean(value):.3f}")
        else:
            try:
                print(f"{key}: {np.mean(value):.3f}")
            except:
                print(f"{key}: cannot compute mean")


def plot_eval_data(data):
    # Determine number of episodes
    n_episodes = max(len(data.get("time_taken", [])),
                     len(data.get("effectiveness", [])),
                     len(data.get("num_drones", [])),
                     len(data.get("distances", [])))
    episodes = list(range(1, n_episodes + 1))

    plt.figure(figsize=(14, 10))

    # --- Time Taken ---
    plt.subplot(2, 3, 1)
    time_taken = [abs(x) if x is not None else 0 for x in data.get("time_taken", [])]
    plt.plot(episodes, time_taken, marker="o")
    plt.title("Episode Time Taken")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")

    # --- Effectiveness ---
    plt.subplot(2, 3, 2)
    effectiveness = [x if x is not None else 0 for x in data.get("effectiveness", [])]
    plt.plot(episodes, effectiveness, marker="o", color="green")
    plt.title("Effectiveness (%)")
    plt.xlabel("Episode")
    plt.ylabel("Effectiveness")

    # --- Number of Drones ---
    plt.subplot(2, 3, 3)
    num_drones = [x if x is not None else 0 for x in data.get("num_drones", [])]
    plt.plot(episodes, num_drones, marker="o", color="orange")
    plt.title("Number of Drones")
    plt.xlabel("Episode")
    plt.ylabel("Drones")

    # --- Drone Distances (average per episode) ---
    avg_distances = []
    for episode in data.get("distances", []):
        if episode is None or len(episode) == 0:
            avg_distances.append(0)
        else:
            # Convert to numpy array safely
            try:
                arr = np.array(episode, dtype=float)
                avg_distances.append(np.mean(arr))
            except:
                avg_distances.append(0)

    plt.subplot(2, 3, 4)
    plt.plot(episodes, avg_distances, marker="o", color="red")
    plt.title("Avg Drone Distance per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Distance")

    # --- Effectiveness vs Number of Drones (overlap) ---
    ax1 = plt.subplot(2, 3, (5, 6))
    ax1.plot(episodes, effectiveness, marker="o", color="green", label="Effectiveness")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Effectiveness (%)", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    ax2 = ax1.twinx()
    ax2.plot(episodes, num_drones, marker="o", color="orange", label="Num Drones")
    ax2.set_ylabel("Number of Drones", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax1.set_title("Effectiveness vs Number of Drones")

    plt.tight_layout()
    plt.show()


def plot_episode_data(eval_data, episode_idx=0):
    """
    Plot all timestep-level evaluation data for a specific episode.
    Works with evaluator class where timestep data is grouped per episode.
    Shows:
        - Effectiveness over time
        - Drone distances over time
        - Final positions of drones and cattle
    """
    # Check episode index
    n_episodes = len(eval_data["distances_per_step"])
    if episode_idx < 0 or episode_idx >= n_episodes:
        print(f"Episode index {episode_idx} out of range.")
        return

    # Extract timestep data for this episode
    ts_distances = eval_data["distances_per_step"][episode_idx]      # list of lists: timesteps x drones
    ts_effectiveness = eval_data["effectiveness_per_step"][episode_idx]
    ts_time = eval_data["time_per_step"][episode_idx]
    ts_drone_poses = eval_data["drone_poses_per_step"][episode_idx]   # timesteps x drones x 2
    ts_cattle_poses = eval_data["cattle_poses_per_step"][episode_idx] # timesteps x cattle x 2

    # Convert to arrays
    ts_distances = np.array(ts_distances, dtype=float)                # shape: (timesteps, drones)
    ts_effectiveness = np.array(ts_effectiveness, dtype=float)        # shape: (timesteps,)
    ts_time = np.array(ts_time, dtype=float)
    ts_drone_poses = np.array(ts_drone_poses, dtype=float)            # shape: (timesteps, drones, 2)
    ts_cattle_poses = np.array(ts_cattle_poses, dtype=float)          # shape: (timesteps, cattle, 2)

    n_timesteps = ts_effectiveness.shape[0]
    timesteps = np.arange(1, n_timesteps + 1)

    plt.figure(figsize=(18, 12))

    # --- 1) Effectiveness over time ---
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, ts_effectiveness, marker="o", color="green")
    plt.xlabel("Timestep")
    plt.ylabel("Effectiveness (%)")
    plt.title("Effectiveness Over Time")

    # --- 2) Average Drone Distance over time ---
    avg_distances = ts_distances.mean(axis=1)  # average over drones
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, avg_distances, marker="o", color="red")
    plt.xlabel("Timestep")
    plt.ylabel("Average Drone Distance")
    plt.title("Average Drone Distance Over Time")

    # --- 3) Final positions of drones and cattle ---
    final_drone_positions = ts_drone_poses[-1]
    final_cattle_positions = ts_cattle_poses[-1]

    plt.subplot(2, 2, 3)
    plt.scatter(final_cattle_positions[:, 0], final_cattle_positions[:, 1], color="brown", label="Cattle", s=100)
    plt.scatter(final_drone_positions[:, 0], final_drone_positions[:, 1], color="blue", label="Drones", s=100)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Final Positions")
    plt.legend()
    plt.axis("equal")

    # --- 4) Drone distance per timestep per drone (optional detailed) ---
    plt.subplot(2, 2, 4)
    for i in range(ts_distances.shape[1]):
        plt.plot(timesteps, ts_distances[:, i], marker="o", label=f"Drone {i+1}")
    plt.xlabel("Timestep")
    plt.ylabel("Distance Travelled")
    plt.title("Drone Distances Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())

    data = load_eval_data("v11-1-1_evaluation_data.pkl")
    print("Keys in data:", list(data.keys()))
    print("Number of episodes:", len(data["time_taken"]))
    
    print_means(data)
    plot_eval_data(data)

    # Example: plot data for episode 0 (first episode)
    plot_episode_data(data, episode_idx=3)
