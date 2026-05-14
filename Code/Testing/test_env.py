"""Testing file made entirely with AI.
To run, go to parent directory Code/ and run python3 -m Testing.test_env"""
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scheduling import SchedulingEnv


# ============================================================
# 1. Simple heuristic for testing
# ============================================================
def heuristic_action(env: SchedulingEnv):
    """
    Simple heuristic:
    - Pick the job with earliest deadline among remaining jobs.
    - Assign it to the machine with the most remaining capacity at current time.
    """

    if len(env.remaining_jobs) == 0:
        return None, None

    remaining = list(env.remaining_jobs)
    job = min(remaining, key=lambda j: env.job_deadlines[j])

    t = env.time
    machine_caps = [np.sum(env.capacity[m, :, t]) for m in range(env.num_machines)]
    machine = int(np.argmax(machine_caps))

    return job, machine


# ============================================================
# 2. 3D time-stacked block plot (grows upward)
# ============================================================
def plot_capacity_over_time(env: SchedulingEnv, ax):
    """
    3D block plot where:
    - X axis = machines placed side-by-side (resource 1 footprint)
    - Y axis = resource 2 footprint
    - Z axis = time
    - At each time step t, a 3D block is drawn from z=t to z=t+1
      with top face shrinking according to remaining capacity.
    - Only draw blocks up to env.time (so the graph grows over time).
    """

    ax.clear()
    ax.set_title(f"Machine Capacity Over Time (t = {env.time})")
    ax.set_xlabel("Machine stacking (Resource 1)")
    ax.set_ylabel("Resource 2")
    ax.set_zlabel("Time")

    num_machines = env.num_machines
    max_t = env.time + 1  # draw up to current time

    # Initial capacities (constant footprint)
    init_cap = env.capacity[:, :, 0]  # shape (M, 2)

    # Precompute x-offsets for each machine
    x_offsets = [0]
    for m in range(1, num_machines):
        x_offsets.append(x_offsets[-1] + init_cap[m-1, 0])

    colors = plt.cm.tab10(np.linspace(0, 1, num_machines))

    # Draw blocks for each machine and each time step up to env.time
    for m in range(num_machines):
        x0 = x_offsets[m]

        for t in range(max_t):
            # Skip out of bounds error
            if m == horizon:
                break
            rem_r1 = env.capacity[m, 0, t]
            rem_r2 = env.capacity[m, 1, t]

            # Shrinking top face
            x1 = x0 + rem_r1
            y1 = rem_r2

            # Block spans from z=t to z=t+1
            z0 = t
            z1 = t + 1

            # 8 vertices of the block
            vertices = np.array([
                [x0, 0,  z0],
                [x1, 0,  z0],
                [x1, y1, z0],
                [x0, y1, z0],

                [x0, 0,  z1],
                [x1, 0,  z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ])

            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            ]

            poly = Poly3DCollection(faces, alpha=0.35, facecolor=colors[m])
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)

    # Axis limits
    ax.set_xlim(0, x_offsets[-1] + init_cap[-1, 0])
    ax.set_ylim(0, np.max(init_cap[:, 1]) * 1.1)
    ax.set_zlim(0, env.horizon)


# ============================================================
# 3. Run heuristic + visualise (60-second simulation)
# ============================================================
def run_test(env: SchedulingEnv, delay=1.0):
    """
    Run the heuristic policy on the environment for exactly 60 time steps.
    """

    rewards = []
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax_reward = fig.add_subplot(122)

    for _ in range(env.horizon):  # horizon = 60
        if len(env.remaining_jobs) > 0:
            job, machine = heuristic_action(env)
            state, reward, done = env.step((job, machine))
            rewards.append(float(reward))
        else:
            # No jobs left, but continue time progression
            env.time += 1
            rewards.append(0.0)

        # Update 3D plot
        plot_capacity_over_time(env, ax3d)

        # Update reward plot
        ax_reward.clear()
        ax_reward.plot(rewards, label="Reward per step")
        ax_reward.set_title("Reward Over Time")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        ax_reward.legend()

        plt.pause(0.01)
        time.sleep(delay)

    plt.show()


# ============================================================
# 4. Example usage — 60-second simulation, 5 machines
# ============================================================
if __name__ == "__main__":
    num_jobs = 40
    num_machines = 5
    horizon = 60  # 60 seconds

    # Random but reasonable job set
    job_durations = np.random.randint(1, 6, size=num_jobs)
    job_resources = np.random.randint(1, 6, size=(num_jobs, 2))
    job_deadlines = np.random.randint(10, 80, size=num_jobs)
    job_weights = np.ones(num_jobs)

    # Machine capacity (2D resources)
    machine_capacity = np.array([20, 20])

    env = SchedulingEnv(
        job_durations,
        job_resources,
        job_deadlines,
        job_weights,
        num_machines,
        machine_capacity,
        horizon,
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0
    )

    run_test(env, delay=1.0)  # 1 second per time step
