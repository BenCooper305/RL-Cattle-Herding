import numpy as np
import yaml

# Parameters
num_simulations = 100
num_cows = 16
min_radius = 8.0
max_radius = 13.0
offset_range = (-2, 2)
z_fixed = 0.1
min_cow_distance = 0.8

output_file = "cattle_positions.yaml"

def is_far_enough(pos, positions, min_dist):
    return all(np.linalg.norm(pos - p) >= min_dist for p in positions)

data = {"simulations": []}

for sim in range(1, num_simulations + 1):
    theta = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(min_radius, max_radius)
    herd_center = np.array([r * np.cos(theta), r * np.sin(theta), z_fixed])

    cow_positions = []
    cows = []

    for cow in range(1, num_cows + 1):
        while True:
            offset = np.random.uniform(low=offset_range[0], high=offset_range[1], size=2)
            offset *= np.random.choice([-1, 1], size=2)
            pos = herd_center[:2] + offset

            if is_far_enough(pos, cow_positions, min_cow_distance):
                cow_positions.append(pos)
                cows.append({
                    "id": cow,
                    "x": round(float(pos[0]), 3),
                    "y": round(float(pos[1]), 3)
                })
                break
    
    data["simulations"].append({"id": sim, "cows": cows})

with open(output_file, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"Generated {num_simulations} simulations with {num_cows} cows each in {output_file}")
