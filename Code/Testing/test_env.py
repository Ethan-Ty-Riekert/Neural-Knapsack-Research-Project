import numpy as np
from Code.binpacking import BasicBinPackingEnv, Bin
import matplotlib.pyplot as plt

# Create random items (3D: CPU, RAM, Disk)
items = [np.random.randint(1, 10, size=3) for _ in range(10)]

# Capacity of each new bin
bin_capacity = np.array([20, 20, 20])

# Create environment
env = BasicBinPackingEnv(items, bin_capacity)

state = env.reset()
done = False

print("Initial state:")
print(state)

# Simple policy: always place in first bin, else create new bin
while not done:
    item = env.cur_item()

    # Try all existing bins
    placed = False
    for i in range(env.num_bins):
        if env.bins[i].check_fit(item):
            action = i
            placed = True
            break

    # If no bin fits → create a new bin
    if not placed:
        action = env.num_bins

    state, reward, done = env.step(action)
    print(f"Step: item={env.item_index}, reward={reward}, bins={env.num_bins}")


# Visualise bins
print("\nFinal bins:")
for i, b in enumerate(env.bins):
    print(f"Bin {i} remaining: {b.remaining}")

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_bins_capacity_usage_3d(bins):
    """
    Visualise each bin as:
      - Outer prism = total capacity
      - Inner prism = used capacity
    Bins are placed side-by-side along X.
    """
    if len(bins) == 0:
        print("No bins to plot.")
        return

    dim = len(bins[0].capacity)
    if dim != 3:
        raise ValueError("3D capacity plotting only works for 3D bins.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(bins)))
    global_offset = 0.0

    for idx, b in enumerate(bins):
        bx, by, bz = b.capacity
        ux, uy, uz = b.capacity - b.remaining  # used capacity

        # Outer bin (capacity)
        outer_vertices = np.array([
            [global_offset,     0, 0],
            [global_offset+bx,  0, 0],
            [global_offset+bx, by, 0],
            [global_offset,    by, 0],
            [global_offset,     0, bz],
            [global_offset+bx,  0, bz],
            [global_offset+bx, by, bz],
            [global_offset,    by, bz]
        ])

        outer_faces = [
            [outer_vertices[0], outer_vertices[1], outer_vertices[2], outer_vertices[3]],
            [outer_vertices[4], outer_vertices[5], outer_vertices[6], outer_vertices[7]],
            [outer_vertices[0], outer_vertices[1], outer_vertices[5], outer_vertices[4]],
            [outer_vertices[2], outer_vertices[3], outer_vertices[7], outer_vertices[6]],
            [outer_vertices[1], outer_vertices[2], outer_vertices[6], outer_vertices[5]],
            [outer_vertices[0], outer_vertices[3], outer_vertices[7], outer_vertices[4]]
        ]

        outer_poly = Poly3DCollection(outer_faces, alpha=0.12, facecolor=colors[idx])
        outer_poly.set_edgecolor('k')
        ax.add_collection3d(outer_poly)

        # Inner prism (used capacity)
        used_vertices = np.array([
            [global_offset,     0, 0],
            [global_offset+ux,  0, 0],
            [global_offset+ux, uy, 0],
            [global_offset,    uy, 0],
            [global_offset,     0, uz],
            [global_offset+ux,  0, uz],
            [global_offset+ux, uy, uz],
            [global_offset,    uy, uz]
        ])

        used_faces = [
            [used_vertices[0], used_vertices[1], used_vertices[2], used_vertices[3]],
            [used_vertices[4], used_vertices[5], used_vertices[6], used_vertices[7]],
            [used_vertices[0], used_vertices[1], used_vertices[5], used_vertices[4]],
            [used_vertices[2], used_vertices[3], used_vertices[7], used_vertices[6]],
            [used_vertices[1], used_vertices[2], used_vertices[6], used_vertices[5]],
            [used_vertices[0], used_vertices[3], used_vertices[7], used_vertices[4]]
        ]

        used_poly = Poly3DCollection(used_faces, alpha=0.55, facecolor=colors[idx])
        used_poly.set_edgecolor('k')
        ax.add_collection3d(used_poly)

        ax.text(global_offset, 0, 0, f"Bin {idx}", color='black')

        global_offset += bx + 5  # spacing between bins

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Bin Capacity vs Used")

    plt.show()


plot_bins_capacity_usage_3d(env.bins)