"""eval_reduced_budget.py - Reduced-budget report-prep evaluation (2026-09-04).

Compares the 3 RL checkpoints from train_reduced_budget_models.py against
EDF, LST, and PSO on the same 8-instance held-out set and a shared showcase
instance, producing everything the user's mathematical report needs: a
6-way comparison table + bar charts, a utilisation-over-time comparison, and
one Gantt chart per method. See
Results/reduced_budget_2026-09-04/README.md for the caveats on why this is a
reduced-scale/reduced-N pass, not the project's full-fidelity numbers.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Code.baselines.pso import _decode_priorities, optimize_and_run
from Code.baselines.registry import HEURISTICS
from Code.evaluation.eval_rl_agent import make_env, run_heuristic, run_model
from Code.policies.a2c_policy import make_maskable_a2c
from Code.training.train_reduced_budget_models import (
    HELDOUT_SEEDS, HORIZON, NUM_JOBS, NUM_MACHINES, build_config,
)

SHOWCASE_SEED = HELDOUT_SEEDS[0]  # 500000

METHOD_ORDER = ["EDF", "LST", "Pointer+shaping", "Pointer+RCPO", "Pointer+Pareto-knee", "PSO"]
METHOD_COLORS = {
    "EDF": "#4C72B0", "LST": "#DD8452", "Pointer+shaping": "#55A868",
    "Pointer+RCPO": "#C44E52", "Pointer+Pareto-knee": "#8172B2", "PSO": "#937860",
}


def load_rl_model(models_dir: Path, name: str):
    hp = json.load(open(models_dir / f"{name}_hyperparams.json"))
    template_env = make_env(config=build_config(0))
    model = make_maskable_a2c(
        template_env, policy_type="pointer",
        policy_kwargs=dict(embed_dim=hp["embed_dim"], hidden=hp["hidden"]),
    )
    model.model.load_state_dict(torch.load(models_dir / f"{name}.pt"))
    return model, hp


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Schedule capture (for Gantt charts) -- none of run_model/run_heuristic/
# pso._simulate record *which machine* a job landed on, only base_env's own
# start_times array (per job, no machine). Reimplement the same replay-loop
# shape here, additionally recording (job, machine, start_time, duration).
# ============================================================
def capture_episode_schedule(action_fn, config):
    env = make_env(config)
    obs, info = env.reset()
    base_env = env.env.env
    idle_action = env.env.max_jobs * base_env.num_machines

    done = truncated = False
    schedule = []
    rewards = []
    while not (done or truncated):
        action = action_fn(env, base_env, obs, info)
        if action != idle_action:
            job = action // base_env.num_machines
            machine = action % base_env.num_machines
            schedule.append((int(job), int(machine), int(base_env.time), int(base_env.job_durations[job])))
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))

    return schedule, {
        "total_reward": float(np.sum(rewards)),
        "tardiness": base_env.tardiness.copy(),
        "late_jobs": int((base_env.tardiness > 0).sum()),
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
    }


def heuristic_action_fn(name):
    def fn(env, base_env, obs, info):
        mask = env.env.get_action_mask()
        idle_action = env.env.max_jobs * base_env.num_machines
        valid = [a for a, ok in enumerate(mask) if ok]
        job_actions = [a for a in valid if a != idle_action]
        if not job_actions:
            return idle_action

        def decode(a):
            return a // base_env.num_machines, a % base_env.num_machines
        return HEURISTICS[name](base_env, job_actions, decode)
    return fn


def rl_action_fn(model):
    def fn(env, base_env, obs, info):
        return model.act(obs, info["action_mask"], deterministic=True)
    return fn


def pso_action_fn(position):
    job_rank, machine_rank = _decode_priorities(position, NUM_JOBS, NUM_MACHINES)

    def fn(env, base_env, obs, info):
        mask = env.env.get_action_mask()
        idle_action = env.env.max_jobs * base_env.num_machines
        valid = [a for a, ok in enumerate(mask) if ok]
        job_actions = [a for a in valid if a != idle_action]
        if not job_actions:
            return idle_action

        def decode(a):
            return a // base_env.num_machines, a % base_env.num_machines
        unique_jobs = {decode(a)[0] for a in job_actions}
        job = min(unique_jobs, key=lambda j: (job_rank[j], j))
        feasible_machines = sorted({decode(a)[1] for a in job_actions if decode(a)[0] == job})
        machine = min(feasible_machines, key=lambda m: (machine_rank[m], m))
        return job * base_env.num_machines + machine
    return fn


# ============================================================
# Gantt chart
# ============================================================
def plot_gantt(schedule, method_name, jobs_scheduled, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cmap = plt.get_cmap("tab20")
    for job, machine, start, duration in schedule:
        ax.barh(machine, duration, left=start, height=0.6,
                color=cmap(job % 20), edgecolor="black", linewidth=0.5)
        ax.text(start + duration / 2, machine, str(job), ha="center", va="center", fontsize=7)
    ax.set_yticks(range(NUM_MACHINES))
    ax.set_yticklabels([f"M{m}" for m in range(NUM_MACHINES)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"{method_name} -- schedule on showcase instance "
                 f"(seed={SHOWCASE_SEED}, {jobs_scheduled}/{NUM_JOBS} jobs scheduled)")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, out_path)


# ============================================================
# Comparison bar charts / utilisation / table
# ============================================================
def bar_chart(stats, metric_label, means_key, stds_key, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = METHOD_ORDER
    means = [stats[n][means_key] for n in names]
    stds = [stats[n][stds_key] for n in names]
    colors = [METHOD_COLORS[n] for n in names]
    ax.bar(names, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} (mean +/- std, {stats[names[0]]['n_episodes']} held-out instances)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, out_path)


def utilisation_chart(util_curves, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in METHOD_ORDER:
        curve = util_curves[name]
        ax.plot(np.arange(len(curve)), curve, label=name, color=METHOD_COLORS[name])
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean machine utilisation")
    ax.set_title("Mean machine utilisation over time (8 held-out instances)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)


def summary_table_image(stats, out_path: Path):
    cols = ["Method", "Reward", "Tardiness", "Late jobs", "Jobs scheduled"]
    rows = []
    for name in METHOD_ORDER:
        s = stats[name]
        rows.append([
            name,
            f"{s['reward_mean']:.2f} +/- {s['reward_std']:.2f}",
            f"{s['tardiness_mean']:.2f} +/- {s['tardiness_std']:.2f}",
            f"{s['late_jobs_mean']:.2f} +/- {s['late_jobs_std']:.2f}",
            f"{s['jobs_scheduled_mean']:.2f}/{NUM_JOBS} +/- {s['jobs_scheduled_std']:.2f}",
        ])
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j, name in enumerate(METHOD_ORDER):
        table[(j + 1, 0)].set_facecolor(METHOD_COLORS[name])
        table[(j + 1, 0)].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Summary: 8 held-out instances (seeds {HELDOUT_SEEDS[0]}-{HELDOUT_SEEDS[-1]}), "
                 f"{NUM_JOBS} jobs / {NUM_MACHINES} machines", pad=20)
    save_fig(fig, out_path)


def write_csv(stats, out_path: Path):
    fields = ["method", "reward_mean", "reward_std", "tardiness_mean", "tardiness_std",
              "late_jobs_mean", "late_jobs_std", "jobs_scheduled_mean", "jobs_scheduled_std", "n_episodes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in METHOD_ORDER:
            row = {"method": name, **{k: stats[name][k] for k in fields if k != "method"}}
            w.writerow(row)


def write_markdown(stats, out_path: Path):
    lines = [
        "# Reduced-budget results summary",
        "",
        f"8 held-out instances (seeds {HELDOUT_SEEDS[0]}-{HELDOUT_SEEDS[-1]}), "
        f"{NUM_JOBS} jobs / {NUM_MACHINES} machines / horizon {HORIZON}.",
        "",
        "| Method | Reward | Tardiness | Late jobs | Jobs scheduled |",
        "|---|---|---|---|---|",
    ]
    for name in METHOD_ORDER:
        s = stats[name]
        lines.append(
            f"| {name} | {s['reward_mean']:.2f} +/- {s['reward_std']:.2f} "
            f"| {s['tardiness_mean']:.2f} +/- {s['tardiness_std']:.2f} "
            f"| {s['late_jobs_mean']:.2f} +/- {s['late_jobs_std']:.2f} "
            f"| {s['jobs_scheduled_mean']:.2f}/{NUM_JOBS} +/- {s['jobs_scheduled_std']:.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stats_from_runs(runs):
    reward = np.array([r["total_reward"] for r in runs])
    tard = np.array([r["tardiness"].sum() for r in runs])
    late = np.array([r["late_jobs"] for r in runs])
    sched = np.array([r["jobs_scheduled"] for r in runs])
    return {
        "reward_mean": float(reward.mean()), "reward_std": float(reward.std()),
        "tardiness_mean": float(tard.mean()), "tardiness_std": float(tard.std()),
        "late_jobs_mean": float(late.mean()), "late_jobs_std": float(late.std()),
        "jobs_scheduled_mean": float(sched.mean()), "jobs_scheduled_std": float(sched.std()),
        "n_episodes": len(runs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pso-swarm-size", type=int, default=10)
    parser.add_argument("--pso-iterations", type=int, default=20)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    heldout_configs = [build_config(s) for s in HELDOUT_SEEDS]
    showcase_config = build_config(SHOWCASE_SEED)

    print("Loading RL checkpoints...")
    model_a, hp_a = load_rl_model(models_dir, "pointer_shaped")
    model_b, hp_b = load_rl_model(models_dir, "pointer_rcpo")
    model_c, hp_c = load_rl_model(models_dir, "pointer_paretoknee")

    stats = {}
    util_curves = {}
    schedules = {}

    def eval_method(name, run_fn):
        print(f"Evaluating {name} on {len(heldout_configs)} held-out instances...")
        runs = [run_fn(cfg) for cfg in heldout_configs]
        stats[name] = stats_from_runs(runs)
        min_len = min(len(r["utilisation_over_time"]) for r in runs)
        util = np.stack([r["utilisation_over_time"][:min_len] for r in runs])
        util_curves[name] = util.mean(axis=0).mean(axis=1)

    eval_method("EDF", lambda cfg: run_heuristic("EDF", config=cfg))
    eval_method("LST", lambda cfg: run_heuristic("LST", config=cfg))
    eval_method("Pointer+shaping", lambda cfg: run_model(model_a, config=cfg))
    eval_method("Pointer+RCPO", lambda cfg: run_model(model_b, config=cfg))
    eval_method("Pointer+Pareto-knee", lambda cfg: run_model(model_c, config=cfg))

    print(f"Running PSO search per held-out instance "
          f"(swarm={args.pso_swarm_size}, iterations={args.pso_iterations})...")
    assert HELDOUT_SEEDS[0] == SHOWCASE_SEED, "showcase instance must be the first held-out instance"
    pso_runs = []
    pso_showcase_position = None
    for i, (seed, cfg) in enumerate(zip(HELDOUT_SEEDS, heldout_configs)):
        r = optimize_and_run(cfg, num_jobs=NUM_JOBS, num_machines=NUM_MACHINES,
                              swarm_size=args.pso_swarm_size, iterations=args.pso_iterations,
                              seed=seed)
        pso_runs.append(r)
        if i == 0:  # showcase instance -- reuse this exact search's winning position for the Gantt capture
            pso_showcase_position = r["gbest_position"]
    stats["PSO"] = stats_from_runs(pso_runs)
    min_len = min(len(r["utilisation_over_time"]) for r in pso_runs)
    util = np.stack([r["utilisation_over_time"][:min_len] for r in pso_runs])
    util_curves["PSO"] = util.mean(axis=0).mean(axis=1)

    print("Capturing showcase-instance schedules for Gantt charts...")
    schedules["EDF"], gantt_stats_edf = capture_episode_schedule(heuristic_action_fn("EDF"), showcase_config)
    schedules["LST"], gantt_stats_lst = capture_episode_schedule(heuristic_action_fn("LST"), showcase_config)
    schedules["Pointer+shaping"], gantt_stats_a = capture_episode_schedule(rl_action_fn(model_a), showcase_config)
    schedules["Pointer+RCPO"], gantt_stats_b = capture_episode_schedule(rl_action_fn(model_b), showcase_config)
    schedules["Pointer+Pareto-knee"], gantt_stats_c = capture_episode_schedule(rl_action_fn(model_c), showcase_config)
    schedules["PSO"], gantt_stats_pso = capture_episode_schedule(pso_action_fn(np.array(pso_showcase_position)), showcase_config)
    gantt_stats = {
        "EDF": gantt_stats_edf, "LST": gantt_stats_lst, "Pointer+shaping": gantt_stats_a,
        "Pointer+RCPO": gantt_stats_b, "Pointer+Pareto-knee": gantt_stats_c, "PSO": gantt_stats_pso,
    }

    print("Writing figures/tables...")
    bar_chart(stats, "Reward", "reward_mean", "reward_std", out_dir / "figures" / "comparison_reward.png")
    bar_chart(stats, "Tardiness", "tardiness_mean", "tardiness_std", out_dir / "figures" / "comparison_tardiness.png")
    bar_chart(stats, "Late jobs", "late_jobs_mean", "late_jobs_std", out_dir / "figures" / "comparison_late_jobs.png")
    bar_chart(stats, "Jobs scheduled", "jobs_scheduled_mean", "jobs_scheduled_std",
              out_dir / "figures" / "comparison_jobs_scheduled.png")
    utilisation_chart(util_curves, out_dir / "figures" / "comparison_utilisation.png")
    summary_table_image(stats, out_dir / "figures" / "summary_table.png")

    slug = {"EDF": "edf", "LST": "lst", "Pointer+shaping": "pointer_shaped",
            "Pointer+RCPO": "pointer_rcpo", "Pointer+Pareto-knee": "pointer_paretoknee", "PSO": "pso"}
    for name in METHOD_ORDER:
        plot_gantt(schedules[name], name, gantt_stats[name]["jobs_scheduled"],
                   out_dir / "figures" / f"gantt_{slug[name]}.png")

    write_csv(stats, out_dir / "raw_eval_results.csv")
    write_markdown(stats, out_dir / "results_summary.md")

    print(f"\nDone. Outputs in: {out_dir}")
    for name in METHOD_ORDER:
        s = stats[name]
        print(f"  {name:22s} reward={s['reward_mean']:8.2f} tardiness={s['tardiness_mean']:8.2f} "
              f"late={s['late_jobs_mean']:6.2f} sched={s['jobs_scheduled_mean']:5.2f}/{NUM_JOBS}")


if __name__ == "__main__":
    main()
