"""Generative AI made.
NOTE TO SELF: MAKE MY OWN VERSION WHEN TIME PERMITS"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from scheduling_env import SchedulingEnv
from gym_scheduling_wrapper import GymSchedulingEnv


# ============================================================
# Action mask function
# ============================================================
def mask_fn(env: GymSchedulingEnv):
    return env._get_action_mask()


# ============================================================
# Environment factory (same config as training)
# ============================================================
def make_env(seed: int = 0):
    rng = np.random.default_rng(seed)

    num_jobs = 40
    num_machines = 5
    horizon = 60
    num_resources = 3  # keep consistent with training

    job_durations = rng.integers(1, 6, size=num_jobs)
    job_resources = rng.integers(1, 6, size=(num_jobs, num_resources))
    job_deadlines = rng.integers(10, 80, size=num_jobs)
    job_weights = np.ones(num_jobs)

    machine_capacity = np.array([20] * num_resources)

    base_env = SchedulingEnv(
        job_durations=job_durations,
        job_resources=job_resources,
        job_deadlines=job_deadlines,
        job_weights=job_weights,
        num_machines=num_machines,
        machine_capacity=machine_capacity,
        horizon=horizon,
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0,
        invalid_penalty=5.0,
    )

    gym_env = GymSchedulingEnv(base_env)
    masked_env = ActionMasker(gym_env, mask_fn)

    return masked_env, base_env


# ============================================================
# Run one evaluation episode
# ============================================================
def run_evaluation(model_path: str):
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}.zip")

    # Load model
    model = MaskablePPO.load(model_path)

    # Create env
    env, base_env = make_env(seed=123)

    obs, info = env.reset()
    done = False
    truncated = False

    rewards = []
    jobs_completed_over_time = []
    utilisation_over_time = []

    initial_capacity = base_env.capacity[:, :, 0].copy()
    num_machines = base_env.num_machines
    horizon = base_env.horizon

    step_count = 0

    while not (done or truncated):
        action_mask = info.get("action_mask", None)
        action, _ = model.predict(
            obs,
            action_masks=action_mask,
            deterministic=True,
        )

        obs, reward, done, truncated, info = env.step(action)

        rewards.append(float(reward))

        # Jobs completed so far
        jobs_completed = base_env.num_jobs - len(base_env.remaining_jobs)
        jobs_completed_over_time.append(jobs_completed)

        # Machine utilisation at current time
        t_idx = min(base_env.time - 1, horizon - 1)
        used = initial_capacity - base_env.capacity[:, :, t_idx]
        utilisation = np.mean(used / (initial_capacity + 1e-8), axis=1)  # per machine
        utilisation_over_time.append(utilisation)

        step_count += 1
        if step_count > horizon * 2:
            # Safety break in case something goes weird
            break

    utilisation_over_time = np.array(utilisation_over_time)  # shape (T, M)

    # Final metrics
    total_reward = np.sum(rewards)
    tardiness = base_env.tardiness.copy()
    machine_active = base_env.machine_active.copy()
    theta = base_env.compute_theta()

    return {
        "rewards": np.array(rewards),
        "jobs_completed_over_time": np.array(jobs_completed_over_time),
        "utilisation_over_time": utilisation_over_time,
        "tardiness": tardiness,
        "machine_active": machine_active,
        "theta": theta,
        "total_reward": total_reward,
    }


# ============================================================
# Plot professional evaluation dashboard
# ============================================================
def plot_dashboard(results):
    rewards = results["rewards"]
    jobs_completed_over_time = results["jobs_completed_over_time"]
    utilisation_over_time = results["utilisation_over_time"]
    tardiness = results["tardiness"]
    machine_active = results["machine_active"]
    theta = results["theta"]
    total_reward = results["total_reward"]

    num_steps = len(rewards)
    num_machines = utilisation_over_time.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_reward = axes[0, 0]
    ax_jobs = axes[0, 1]
    ax_util = axes[1, 0]
    ax_tard = axes[1, 1]

    # 1) Reward per step
    ax_reward.plot(range(num_steps), rewards, color="tab:blue")
    ax_reward.set_title("Step-wise Reward")
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.3)

    # 2) Jobs completed over time
    ax_jobs.plot(range(num_steps), jobs_completed_over_time, color="tab:green")
    ax_jobs.set_title("Jobs Completed Over Time")
    ax_jobs.set_xlabel("Step")
    ax_jobs.set_ylabel("Completed Jobs")
    ax_jobs.grid(True, alpha=0.3)

    # 3) Machine utilisation over time
    for m in range(num_machines):
        ax_util.plot(
            range(num_steps),
            utilisation_over_time[:, m],
            label=f"Machine {m}",
        )
    ax_util.set_title("Machine Utilisation Over Time")
    ax_util.set_xlabel("Step")
    ax_util.set_ylabel("Utilisation (avg over resources)")
    ax_util.set_ylim(0, 1.05)
    ax_util.grid(True, alpha=0.3)
    ax_util.legend(loc="upper right", fontsize=8)

    # 4) Tardiness distribution + summary text
    nonzero_tardiness = tardiness[tardiness > 0]
    if len(nonzero_tardiness) > 0:
        ax_tard.hist(nonzero_tardiness, bins=10, color="tab:red", alpha=0.7)
    ax_tard.set_title("Tardiness Distribution (Non-zero Jobs)")
    ax_tard.set_xlabel("Tardiness")
    ax_tard.set_ylabel("Job Count")

    # Add summary text box
    text_lines = [
        f"Total reward: {total_reward:.2f}",
        f"Mean tardiness: {np.mean(tardiness):.2f}",
        f"Max tardiness: {np.max(tardiness):.2f}",
        f"Active machines: {int(np.sum(machine_active))}",
        f"Max utilisation θ: {theta:.3f}",
    ]
    ax_tard.text(
        0.95,
        0.95,
        "\n".join(text_lines),
        transform=ax_tard.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    MODEL_PATH = "./rl_training/models/ppo_scheduling"

    results = run_evaluation(MODEL_PATH)
    plot_dashboard(results)


if __name__ == "__main__":
    main()
