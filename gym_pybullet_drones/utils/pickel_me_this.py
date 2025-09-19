import pickle
import os
import matplotlib.pyplot as plt

def load_eval_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

import numpy as np

def plot_eval_data(data):
    episodes = list(range(1, len(data["time_taken"]) + 1))

    # --- Time Taken ---
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(episodes, data["time_taken"], marker="o")
    plt.title("Episode Time Taken")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")

    # --- Effectiveness ---
    plt.subplot(2, 2, 2)
    plt.plot(episodes, data["effectiveness"], marker="o", color="green")
    plt.title("Effectiveness (%)")
    plt.xlabel("Episode")
    plt.ylabel("Effectiveness")

    # --- Number of Drones ---
    plt.subplot(2, 2, 3)
    plt.plot(episodes, data["num_drones"], marker="o", color="orange")
    plt.title("Number of Drones")
    plt.xlabel("Episode")
    plt.ylabel("Drones")

    # --- Drone Distances (average per episode) ---
    avg_distances = []
    for episode in data["distances"]:
        # Flatten all arrays for this episode and take mean
        all_dists = np.concatenate(episode)  # combine all timestep arrays
        avg_distances.append(np.mean(all_dists) if len(all_dists) > 0 else 0)

    plt.subplot(2, 2, 4)
    plt.plot(episodes, avg_distances, marker="o", color="red")
    plt.title("Avg Drone Distance per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = load_eval_data("evaluation_data 5-10.pkl")
    print("Keys in data:", list(data.keys()))
    print("Number of episodes:", len(data["time_taken"]))

    plot_eval_data(data)
