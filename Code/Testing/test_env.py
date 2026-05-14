"""Testing file made entirely with AI.
To run, go to parent directory Code/ and run python3 -m Testing.test_env"""
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scheduling_env import SchedulingEnv


# ============================================================
# 1. Simple heuristic for testing
# ============================================================
def heuristic_action(env: SchedulingEnv):
    if len(env.remaining_jobs) == 0:
        return None, None

    remaining = list(env.remaining_jobs)
    job = min(remaining, key=lambda j: env.job_deadlines[j])

    t = min(env.time, env.horizon - 1)
    machine_caps = [np.sum(env.capacity[m, :, t]) for m in range(env.num_machines)]
    machine = int(np.argmax(machine_caps))

    return job, machine


# ============================================================
# 2. 3D time-stacked block plot (grows upward)
# ============================================================
def plot_capacity_over_time(env: SchedulingEnv, ax):
    ax.clear()
    ax.set_title(f"Machine Capacity Over Time (t = {env.time})")
    ax.set_xlabel("Machine stacking (Resource 1)")
    ax.set_ylabel("Resource 2")
    ax.set_zlabel("Time")

    num_machines = env.num_machines
    horizon = env.horizon

    init_cap = env.capacity[:, :, 0]  # initial capacities

    # Precompute x-offsets for each machine
    x_offsets = [0]
    for m in range(1, num_machines):
        x_offsets.append(x_offsets[-1] + init_cap[m - 1, 0])

    colors = plt.cm.tab10(np.linspace(0, 1, num_machines))

    max_t = min(env.time, horizon - 1)

    for m in range(num_machines):
        x0 = x_offsets[m]

        for t in range(max_t + 1):
            rem_r1 = env.capacity[m, 0, t]
            rem_r2 = env.capacity[m, 1, t]

            x1 = x0 + rem_r1
            y1 = rem_r2

            z0 = t
            z1 = t + 1

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
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
            ]

            poly = Poly3DCollection(faces, alpha=0.35, facecolor=colors[m])
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)

    ax.set_xlim(0, x_offsets[-1] + init_cap[-1, 0])
    ax.set_ylim(0, np.max(init_cap[:, 1]) * 1.1)
    ax.set_zlim(0, horizon)


# ============================================================
# 3. Utilisation heatmap
# ============================================================
def update_utilisation_heatmap(env: SchedulingEnv, heatmap_data, ax):
    t = min(env.time, env.horizon - 1)
    init_cap = env.capacity[:, :, 0]

    for m in range(env.num_machines):
        used_r1 = 1 - env.capacity[m, 0, t] / init_cap[m, 0]
        used_r2 = 1 - env.capacity[m, 1, t] / init_cap[m, 1]
        heatmap_data[m, t] = max(used_r1, used_r2)

    ax.clear()
    ax.set_title("Machine Utilisation Heatmap")
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")

    im = ax.imshow(
        heatmap_data,
        aspect='auto',
        cmap='inferno',
        vmin=0,
        vmax=1,
        origin='lower'
    )
    # Avoid stacking multiple colorbars
    if not hasattr(ax, "_heatmap_cb"):
        cb = ax.figure.colorbar(im, ax=ax)
        ax._heatmap_cb = cb
    else:
        ax._heatmap_cb.update_normal(im)


# ============================================================
# 4. Gantt chart
# ============================================================
def update_gantt_chart(ax, gantt_data, horizon, num_machines):
    ax.clear()
    ax.set_title("Gantt Chart (Job Schedule)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")

    if len(gantt_data) == 0:
        ax.set_xlim(0, horizon)
        ax.set_ylim(-1, num_machines)
        return

    for (job, machine, start, duration) in gantt_data:
        if start is None or start < 0:
            continue
        ax.barh(
            y=machine,
            width=duration,
            left=start,
            height=0.4,
            align='center',
            color=plt.cm.tab20(job % 20)
        )
        ax.text(start + duration / 2, machine, f"J{job}",
                va='center', ha='center', color='white', fontsize=8)

    ax.set_yticks(range(num_machines))
    ax.set_ylim(-1, num_machines)
    ax.set_xlim(0, horizon)


# ============================================================
# 5. Run heuristic + visualise (60-step simulation)
# ============================================================
def run_test(env: SchedulingEnv, delay=1.0):
    rewards = []
    heatmap_data = np.zeros((env.num_machines, env.horizon))
    gantt_data = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_reward = axes[0, 1]
    ax_heatmap = axes[1, 0]
    ax_gantt = axes[1, 1]

    for _ in range(env.horizon):
        if len(env.remaining_jobs) > 0:
            job, machine = heuristic_action(env)
            if job is not None:
                state, reward, done = env.step((job, machine))
                rewards.append(float(reward))

                start_time = env.start_times[job]
                duration = env.job_durations[job]
                gantt_data.append((job, machine, start_time, duration))
            else:
                env.time = min(env.time + 1, env.horizon - 1)
                rewards.append(0.0)
        else:
            env.time = min(env.time + 1, env.horizon - 1)
            rewards.append(0.0)

        plot_capacity_over_time(env, ax3d)

        ax_reward.clear()
        ax_reward.plot(rewards)
        ax_reward.set_title("Reward Over Time")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")

        update_utilisation_heatmap(env, heatmap_data, ax_heatmap)
        update_gantt_chart(ax_gantt, gantt_data, env.horizon, env.num_machines)

        plt.pause(0.01)
        time.sleep(delay)

    plt.show()


# ============================================================
# 6. Example usage — 60-step simulation, 5 machines
# ============================================================
if __name__ == "__main__":
    num_jobs = 40
    num_machines = 5
    horizon = 60

    job_durations = np.random.randint(1, 6, size=num_jobs)
    job_resources = np.random.randint(1, 6, size=(num_jobs, 2))
    job_deadlines = np.random.randint(10, 80, size=num_jobs)
    job_weights = np.ones(num_jobs)

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

    run_test(env, delay=1.0)
