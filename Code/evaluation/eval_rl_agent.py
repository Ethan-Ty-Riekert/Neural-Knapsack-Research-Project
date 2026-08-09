"""eval_rl_agent.py - Generative AI made.
NOTE TO SELF:
- ALLOW FOR CHANGING THE CONFIGURATION WITHOUT HAVING TO RETRAIN THE ENTIRE AGENT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.env_config import generate_env_config
from Code.policies.a2c_policy import make_maskable_a2c
from Code.utils.plotting_utils import make_run_dir, save_and_show, EvalProgressPlotter
from Code.utils.paths import ENV_CONFIG_PATH, PPO_MODEL_PATH, A2C_MODEL_PATH, PLOTS_DIR
import torch



def mask_fn(env: GymSchedulingEnv):
    return env.get_action_mask()


# ============================================================
# Environment factory
# ============================================================
def make_env():
    data = np.load(ENV_CONFIG_PATH)
    config = {k: data[k] for k in data.files}

    base_env = SchedulingEnv(
        job_durations=config["job_durations"],
        job_resources=config["job_resources"],
        job_deadlines=config["job_deadlines"],
        job_weights=config["job_weights"],
        num_machines=int(config["num_machines"]),
        machine_capacity=config["machine_capacity"],
        horizon=int(config["horizon"]),
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0,
        invalid_penalty=5.0,
    )

    # max_jobs (padding capacity) must match what the model was trained with -- see
    # gym_scheduling_wrapper.py and train_rl_agent.py's curriculum for why.
    max_jobs = int(config["max_jobs"]) if "max_jobs" in config else None
    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs)
    masked_env = ActionMasker(gym_env, mask_fn)

    return masked_env


# ============================================================
# Run one PPO evaluation episode
# ============================================================
def run_ppo(model):
    print("Running PPO episode...")
    env = make_env()
    obs, info = env.reset()

    done = False
    truncated = False

    rewards = []
    utilisation_over_time = []

    base_env = env.env.env
    initial_capacity = base_env.capacity[:, :, 0].copy()
    horizon = base_env.horizon

    while not (done or truncated):
        action_mask = info["action_mask"]

        if hasattr(model, "predict"):
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        else:
            action = model.act(obs, action_mask)

        obs, reward, done, truncated, info = env.step(action)

        rewards.append(float(reward))

        t_idx = min(base_env.time - 1, horizon - 1)
        used = initial_capacity - base_env.capacity[:, :, t_idx]
        utilisation = np.mean(used / (initial_capacity + 1e-8), axis=1)
        utilisation_over_time.append(utilisation)

    utilisation_over_time = np.array(utilisation_over_time)

    return {
        "total_reward": np.sum(rewards),
        "tardiness": base_env.tardiness.copy(),
        "late_jobs": (base_env.tardiness > 0).sum(),
        "utilisation_over_time": utilisation_over_time,
    }



# ============================================================
# Run heuristic
# ============================================================
def run_heuristic(name):

    env = make_env()
    base_env = env.env.env

    obs, info = env.reset()
    done = False
    truncated = False

    rewards = []
    utilisation_over_time = []

    initial_capacity = base_env.capacity[:, :, 0].copy()
    horizon = base_env.horizon
    num_machines = base_env.num_machines

    idle_action = env.env.max_jobs * num_machines

    def decode(a):
        return a // num_machines, a % num_machines

    def choose_action():
        mask = env.env.get_action_mask()
        valid = [a for a, ok in enumerate(mask) if ok]
        if not valid:
            return None

        # The idle action doesn't decode to a (job, machine) pair, so it must be
        # excluded from the job-based heuristics below. Only fall back to idling
        # when no real job/machine placement is currently feasible.
        job_actions = [a for a in valid if a != idle_action]
        if not job_actions:
            return idle_action

        if name == "EDF":
            return min(job_actions, key=lambda a: base_env.job_deadlines[decode(a)[0]])
        if name == "SPT":
            return min(job_actions, key=lambda a: base_env.job_durations[decode(a)[0]])
        if name == "LST":
            return min(job_actions, key=lambda a:
                       base_env.job_deadlines[decode(a)[0]]
                       - base_env.job_durations[decode(a)[0]]
                       - base_env.time)
        return np.random.choice(job_actions)

    while not (done or truncated):
        action = choose_action()
        if action is None:
            break

        obs, reward, done, truncated, info = env.step(action)

        rewards.append(float(reward))

        t_idx = min(base_env.time - 1, horizon - 1)
        used = initial_capacity - base_env.capacity[:, :, t_idx]
        utilisation = np.mean(used / (initial_capacity + 1e-8), axis=1)
        utilisation_over_time.append(utilisation)

    utilisation_over_time = np.array(utilisation_over_time)

    return {
        "total_reward": np.sum(rewards),
        "tardiness": base_env.tardiness.copy(),
        "late_jobs": (base_env.tardiness > 0).sum(),
        "utilisation_over_time": utilisation_over_time,
    }


# ============================================================
# Aggregate over N runs
# ============================================================
def evaluate_multiple(model, heuristic_name, runs=50, run_dir=None):
    ppo_metrics = []
    heur_metrics = []

    progress_plotter = EvalProgressPlotter(heuristic_name)

    print("Evaluating PPO...")
    for _ in tqdm(range(runs)):
        result = run_ppo(model)
        ppo_metrics.append(result)
        progress_plotter.update(ppo_reward=result["total_reward"])

    print("Evaluating heuristic...")
    for _ in tqdm(range(runs)):
        result = run_heuristic(heuristic_name)
        heur_metrics.append(result)
        progress_plotter.update(heur_reward=result["total_reward"])

    if run_dir is not None:
        progress_plotter.save(run_dir)
    progress_plotter.close()

    return ppo_metrics, heur_metrics



# ============================================================
# Plot clean, informative graphs
# ============================================================
def plot_results(ppo_runs, heur_runs, heuristic_name, run_dir):
    # Episodes can terminate at different timesteps (depending on how many jobs get
    # scheduled before the horizon is reached), so utilisation_over_time isn't
    # uniform-length across runs. Truncate to the shortest common length across
    # BOTH sets before stacking, since they're plotted on one shared time axis.
    min_len = min(
        min(len(r["utilisation_over_time"]) for r in ppo_runs),
        min(len(r["utilisation_over_time"]) for r in heur_runs),
    )
    ppo_util = np.stack([r["utilisation_over_time"][:min_len] for r in ppo_runs])
    heur_util = np.stack([r["utilisation_over_time"][:min_len] for r in heur_runs])

    # Compute mean curves
    ppo_mean = ppo_util.mean(axis=0).mean(axis=1)
    ppo_std = ppo_util.mean(axis=2).std(axis=0)

    heur_mean = heur_util.mean(axis=0).mean(axis=1)
    heur_std = heur_util.mean(axis=2).std(axis=0)

    # Scalar metrics
    ppo_tard = np.array([r["tardiness"].sum() for r in ppo_runs])
    heur_tard = np.array([r["tardiness"].sum() for r in heur_runs])

    ppo_late = np.array([r["late_jobs"] for r in ppo_runs])
    heur_late = np.array([r["late_jobs"] for r in heur_runs])

    ppo_reward = np.array([r["total_reward"] for r in ppo_runs])
    heur_reward = np.array([r["total_reward"] for r in heur_runs])

    # ---------------------------------------------------------
    # 1. Mean utilisation curve
    # ---------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 5))
    steps = np.arange(len(ppo_mean))

    plt.plot(steps, ppo_mean, label="PPO", color="tab:blue")
    plt.fill_between(steps, ppo_mean - ppo_std, ppo_mean + ppo_std, alpha=0.2)

    plt.plot(steps, heur_mean, label="Fixed Heuristic", color="tab:orange")
    plt.fill_between(steps, heur_mean - heur_std, heur_mean + heur_std, alpha=0.2)

    plt.title("Mean Machine Utilisation Over Time (50 runs)")
    plt.xlabel("Step")
    plt.ylabel("Utilisation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_and_show(fig1, run_dir, "mean_utilisation.png")

    # ---------------------------------------------------------
    # 2. Total tardiness (bar chart)
    # ---------------------------------------------------------
    fig2 = plt.figure(figsize=(6, 5))
    plt.bar(["PPO", heuristic_name], [ppo_tard.mean(), heur_tard.mean()],
            yerr=[ppo_tard.std(), heur_tard.std()],
            color=["tab:blue", "tab:orange"])
    plt.title("Total Tardiness (mean ± std over 50 runs)")
    plt.ylabel("Tardiness")
    save_and_show(fig2, run_dir, "total_tardiness.png")

    # ---------------------------------------------------------
    # 3. Late jobs
    # ---------------------------------------------------------
    fig3 = plt.figure(figsize=(6, 5))
    plt.bar(["PPO", heuristic_name], [ppo_late.mean(), heur_late.mean()],
            yerr=[ppo_late.std(), heur_late.std()],
            color=["tab:blue", "tab:orange"])
    plt.title("Number of Late Jobs (mean ± std)")
    plt.ylabel("Late Jobs")
    save_and_show(fig3, run_dir, "late_jobs.png")

    # ---------------------------------------------------------
    # 4. Total reward
    # ---------------------------------------------------------
    fig4 = plt.figure(figsize=(6, 5))
    plt.bar(["PPO", heuristic_name], [ppo_reward.mean(), heur_reward.mean()],
            yerr=[ppo_reward.std(), heur_reward.std()],
            color=["tab:blue", "tab:orange"])
    plt.title("Total Reward (mean ± std)")
    plt.ylabel("Reward")
    save_and_show(fig4, run_dir, "total_reward.png")


# ============================================================
# Main
# ============================================================
def main():
    # -----------------------------------------
    # Parse command‑line argument: --algo ppo/a2c
    # -----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"])
    parser.add_argument("--policy-type", type=str, default="pointer", choices=["pointer", "flat"],
                         help="A2C only: must match the policy_type the checkpoint was trained with.")
    args = parser.parse_args()

    USE_PPO = (args.algo == "ppo")

    # -----------------------------------------
    # Load model depending on algorithm
    # -----------------------------------------
    if USE_PPO:
        model = MaskablePPO.load(PPO_MODEL_PATH)
    else:
        env = make_env()
        model = make_maskable_a2c(env, policy_type=args.policy_type)
        model.model.load_state_dict(torch.load(A2C_MODEL_PATH))

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------
    run_dir = make_run_dir(str(PLOTS_DIR / "eval"), args.algo)
    ppo_runs, heur_runs = evaluate_multiple(model, "EDF", runs=50, run_dir=run_dir)
    plot_results(ppo_runs, heur_runs, "EDF", run_dir)
    print(f"\nEvaluation plots saved to: {run_dir}\n")




if __name__ == "__main__":
    main()
