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
from Code.utils.paths import ENV_CONFIG_PATH, PPO_MODEL_PATH, A2C_MODEL_PATH, PLOTS_DIR, EVAL_RESULTS_CSV
from Code.utils.results_log import append_eval_result
from Code.baselines.registry import HEURISTICS, DEFAULT_HEURISTICS, ALL_HEURISTICS
import torch



def mask_fn(env: GymSchedulingEnv):
    return env.get_action_mask()


# ============================================================
# Environment factory
# ============================================================
def make_env(config=None):
    """config=None (default): load the single saved instance from
    ENV_CONFIG_PATH, matching every prior eval in this project (fixed
    seed=0 instance). Pass an explicit config dict (from generate_env_config())
    to evaluate on a different instance instead -- used by --randomized-eval
    below to build N distinct held-out instances rather than reusing one.
    """
    if config is None:
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
# Run one model evaluation episode (PPO or A2C, flat or pointer)
# ============================================================
def run_model(model, config=None):
    # BUG FIX (this session): this was named run_ppo() and unconditionally
    # printed "Running PPO episode..." even when evaluating A2C (flat or
    # pointer) -- the function has always handled both via the
    # hasattr(model, "predict") branch below, so the name/print were
    # misleading leftovers from when it was PPO-only. Caused user confusion
    # (mistook an A2C flat eval run for a PPO run with bad tardiness).
    env = make_env(config)
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
            # BUG FIX (this session): model.act() previously had no
            # deterministic/greedy mode, so A2C evaluation always sampled from the
            # masked Categorical distribution while PPO's evaluation (above) is
            # already greedy via deterministic=True -- an eval-protocol asymmetry
            # that made A2C's reported numbers noisier/less representative of its
            # best behaviour. See a2c_policy.py::select_action()'s docstring.
            action = model.act(obs, action_mask, deterministic=True)

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
        # jobs_scheduled: added 2026-08-28 -- see results_log.py's
        # jobs_scheduled_* field comment for why this is tracked by default
        # now (a hidden completion-rate gap explained two separate
        # misleading-looking results this session).
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
        "utilisation_over_time": utilisation_over_time,
    }



# ============================================================
# Run heuristic
# ============================================================
def run_heuristic(name, config=None):

    env = make_env(config)
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

        # Heuristic dispatch delegated to Code.baselines.registry -- see
        # Future/research/2026-08-28-classical-heuristic-baselines.md. Unknown
        # names fall back to Random, matching this function's original
        # behaviour (any name other than EDF/SPT/LST used to fall through to
        # np.random.choice(job_actions)).
        heuristic_fn = HEURISTICS.get(name, HEURISTICS["Random"])
        return heuristic_fn(base_env, job_actions, decode)

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
        # jobs_scheduled: added 2026-08-28 -- see results_log.py's
        # jobs_scheduled_* field comment for why this is tracked by default
        # now (a hidden completion-rate gap explained two separate
        # misleading-looking results this session).
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
        "utilisation_over_time": utilisation_over_time,
    }


# ============================================================
# Aggregate over N runs
# ============================================================
def evaluate_multiple(model, heuristic_name, runs=50, run_dir=None, configs=None, model_runs=None):
    """configs=None (default): every episode re-evaluates the single fixed
    ENV_CONFIG_PATH instance (previous behaviour -- reward_std is always
    exactly 0.0 as a result, since a deterministic model on a fixed instance
    is fully deterministic). Pass a list of `runs` distinct config dicts
    (from generate_env_config()) to evaluate one held-out instance per
    episode instead -- --randomized-eval below, where std reflects genuine
    cross-instance performance variance rather than being zero by construction.

    model_runs=None (default): evaluate the model fresh (previous behaviour).
    Pass a precomputed list of run_model() results (same configs) to skip
    re-running the model -- main() below now compares one model against
    several heuristics per invocation, and the model's own performance on a
    fixed set of configs doesn't change between heuristics, so recomputing
    it once per heuristic would be a pure waste of episodes.
    """
    heur_metrics = []

    if model_runs is not None:
        ppo_metrics = model_runs
    else:
        ppo_metrics = []
        print("Evaluating model...")
        for i in tqdm(range(runs)):
            result = run_model(model, config=configs[i] if configs is not None else None)
            ppo_metrics.append(result)

    progress_plotter = EvalProgressPlotter(heuristic_name)
    for r in ppo_metrics:
        progress_plotter.update(ppo_reward=r["total_reward"])

    print(f"Evaluating heuristic ({heuristic_name})...")
    for i in tqdm(range(runs)):
        result = run_heuristic(heuristic_name, config=configs[i] if configs is not None else None)
        heur_metrics.append(result)
        progress_plotter.update(heur_reward=result["total_reward"])

    if run_dir is not None:
        progress_plotter.save(run_dir)
    progress_plotter.close()

    return ppo_metrics, heur_metrics



# ============================================================
# Plot clean, informative graphs
# ============================================================
def plot_results(ppo_runs, heur_runs, heuristic_name, run_dir, model_label="PPO"):
    # BUG FIX (this session): every plot below used to hardcode the literal
    # string "PPO" for the model's legend/bar label regardless of which
    # algorithm/policy_type was actually evaluated -- so an A2C (flat or
    # pointer) eval run produced plots indistinguishable from a real PPO run
    # by anything except the output directory name. `model_label` now carries
    # the real identity through from main() (e.g. "A2C (flat)").
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

    plt.plot(steps, ppo_mean, label=model_label, color="tab:blue")
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
    plt.bar([model_label, heuristic_name], [ppo_tard.mean(), heur_tard.mean()],
            yerr=[ppo_tard.std(), heur_tard.std()],
            color=["tab:blue", "tab:orange"])
    plt.title("Total Tardiness (mean ± std over 50 runs)")
    plt.ylabel("Tardiness")
    save_and_show(fig2, run_dir, "total_tardiness.png")

    # ---------------------------------------------------------
    # 3. Late jobs
    # ---------------------------------------------------------
    fig3 = plt.figure(figsize=(6, 5))
    plt.bar([model_label, heuristic_name], [ppo_late.mean(), heur_late.mean()],
            yerr=[ppo_late.std(), heur_late.std()],
            color=["tab:blue", "tab:orange"])
    plt.title("Number of Late Jobs (mean ± std)")
    plt.ylabel("Late Jobs")
    save_and_show(fig3, run_dir, "late_jobs.png")

    # ---------------------------------------------------------
    # 4. Total reward
    # ---------------------------------------------------------
    fig4 = plt.figure(figsize=(6, 5))
    plt.bar([model_label, heuristic_name], [ppo_reward.mean(), heur_reward.mean()],
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
    parser.add_argument("--model-path", type=str, default=None,
                         help="Override the checkpoint path (default: the fixed PPO_MODEL_PATH/A2C_MODEL_PATH "
                              "from Code/utils/paths.py). Needed e.g. to evaluate a "
                              "train_optimized.py checkpoint, which is saved under a "
                              "differently-named path (a2c_{policy_type}_scheduling_optimized.pt).")
    parser.add_argument("--run-tag", type=str, default=None,
                         help="Optional suffix for the eval-plot run directory, to tell runs apart.")
    parser.add_argument("--embed-dim", type=int, default=None,
                         help="Pointer policy only: must match the embed_dim the checkpoint was trained "
                              "with (e.g. a train_optimized.py run using Optuna-found params). Defaults to "
                              "PointerActorCritic's own default (128) if not given.")
    parser.add_argument("--hidden", type=int, default=None,
                         help="Pointer policy only: must match the hidden size the checkpoint was trained "
                              "with. Defaults to PointerActorCritic's own default (64) if not given.")
    parser.add_argument("--randomized-eval", action="store_true",
                         help="Experiment 2: evaluate on N distinct held-out random instances (seeds "
                              ">= RANDOM_INSTANCE_SEED_CEILING, disjoint from any training seed) instead "
                              "of the single fixed ENV_CONFIG_PATH instance every prior eval used. "
                              "Instance dimensions (num_jobs/num_machines/horizon/max_jobs) are still "
                              "read from ENV_CONFIG_PATH -- only which specific job set is used changes.")
    parser.add_argument("--eval-seeds", type=int, default=50,
                         help="Number of held-out instances for --randomized-eval (default 50, matching "
                              "the default n_episodes of the fixed-instance eval).")
    parser.add_argument("--heuristics", type=str, nargs="+", default=None,
                         help="Named baselines (Code.baselines.registry.HEURISTICS) to compare the model "
                              "against, one full evaluate_multiple()+plot_results()+append_eval_result() "
                              "pass per name. Defaults to DEFAULT_HEURISTICS (a curated subset covering "
                              "every priority and placement rule). Pass 'all' to run every registered "
                              "combo instead.")
    args = parser.parse_args()

    if args.heuristics is None:
        heuristic_names = DEFAULT_HEURISTICS
    elif args.heuristics == ["all"]:
        heuristic_names = ALL_HEURISTICS
    else:
        unknown = [n for n in args.heuristics if n not in HEURISTICS]
        if unknown:
            raise ValueError(f"Unknown heuristic name(s) {unknown}; registered names: {ALL_HEURISTICS}")
        heuristic_names = args.heuristics

    USE_PPO = (args.algo == "ppo")

    # -----------------------------------------
    # Load model depending on algorithm
    # -----------------------------------------
    if USE_PPO:
        model = MaskablePPO.load(args.model_path or PPO_MODEL_PATH)
    else:
        env = make_env()
        # BUG FIX (this session): building the pointer network with its bare
        # defaults (embed_dim=128, hidden=64) crashes load_state_dict with a
        # shape mismatch against any checkpoint trained with different
        # architecture params (e.g. Optuna-tuned hidden=32) -- there was
        # previously no way to tell eval which architecture size to build.
        policy_kwargs = None
        if args.policy_type == "pointer" and (args.embed_dim is not None or args.hidden is not None):
            policy_kwargs = {}
            if args.embed_dim is not None:
                policy_kwargs["embed_dim"] = args.embed_dim
            if args.hidden is not None:
                policy_kwargs["hidden"] = args.hidden
        model = make_maskable_a2c(env, policy_type=args.policy_type, policy_kwargs=policy_kwargs)
        model.model.load_state_dict(torch.load(args.model_path or A2C_MODEL_PATH))

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------
    run_tag = args.algo if not args.run_tag else f"{args.algo}_{args.run_tag}"
    run_dir = make_run_dir(str(PLOTS_DIR / "eval"), run_tag)
    model_label = "PPO" if USE_PPO else f"A2C ({args.policy_type})"

    configs = None
    if args.randomized_eval:
        # Experiment 2: build N held-out instances at the same dimensions as
        # the (last-trained) fixed instance, but with seeds >= the ceiling
        # training draws from (Code/training/train_optimized.py::
        # RANDOM_INSTANCE_SEED_CEILING = 500_000) -- guarantees no overlap with
        # any seed the model could have trained on, by construction.
        from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING
        dims = np.load(ENV_CONFIG_PATH)
        num_jobs = int(dims["num_jobs"])
        num_machines = int(dims["num_machines"])
        horizon = int(dims["horizon"])
        max_jobs = int(dims["max_jobs"]) if "max_jobs" in dims else None
        configs = []
        for i in range(args.eval_seeds):
            cfg = generate_env_config(
                seed=RANDOM_INSTANCE_SEED_CEILING + i,
                num_jobs=num_jobs, num_machines=num_machines, horizon=horizon,
            )
            if max_jobs is not None:
                cfg["max_jobs"] = max_jobs
            configs.append(cfg)
        print(f"\n--randomized-eval: evaluating on {len(configs)} held-out instances "
              f"(seeds {RANDOM_INSTANCE_SEED_CEILING}..{RANDOM_INSTANCE_SEED_CEILING + len(configs) - 1})\n")

    n_episodes = len(configs) if configs is not None else 50

    # The model's own performance on this fixed set of configs is identical
    # across every heuristic comparison below -- evaluate it once and reuse,
    # rather than re-running n_episodes model rollouts per heuristic name.
    model_runs = None

    for heuristic_name in heuristic_names:
        print(f"\n=== Comparing against: {heuristic_name} ===")
        ppo_runs, heur_runs = evaluate_multiple(
            model, heuristic_name, runs=n_episodes, run_dir=run_dir,
            configs=configs, model_runs=model_runs,
        )
        model_runs = ppo_runs  # reuse for every subsequent heuristic this run

        plot_results(ppo_runs, heur_runs, heuristic_name, run_dir, model_label=model_label)

        # Persist the summary stats to a running CSV (rl_training/results/eval_results.csv)
        # so they accumulate across runs instead of only living in stdout/PNGs -- see
        # Code/utils/results_log.py. One row per heuristic name.
        model_rewards = np.array([r["total_reward"] for r in ppo_runs])
        model_tardiness = np.array([r["tardiness"].sum() for r in ppo_runs])
        model_late = np.array([r["late_jobs"] for r in ppo_runs])
        model_scheduled = np.array([r["jobs_scheduled"] for r in ppo_runs])
        heur_rewards = np.array([r["total_reward"] for r in heur_runs])
        heur_tardiness = np.array([r["tardiness"].sum() for r in heur_runs])
        heur_late = np.array([r["late_jobs"] for r in heur_runs])
        heur_scheduled = np.array([r["jobs_scheduled"] for r in heur_runs])

        append_eval_result({
            "algo": args.algo,
            "policy_type": args.policy_type if not USE_PPO else "",
            "tag": (args.run_tag or "") + ("_randomized_eval" if args.randomized_eval else ""),
            "model_path": str(args.model_path or (PPO_MODEL_PATH if USE_PPO else A2C_MODEL_PATH)),
            "reward_mean": model_rewards.mean(), "reward_std": model_rewards.std(),
            "tardiness_mean": model_tardiness.mean(), "tardiness_std": model_tardiness.std(),
            "late_jobs_mean": model_late.mean(), "late_jobs_std": model_late.std(),
            "jobs_scheduled_mean": model_scheduled.mean(), "jobs_scheduled_std": model_scheduled.std(),
            "heuristic_name": heuristic_name,
            "heuristic_reward_mean": heur_rewards.mean(), "heuristic_reward_std": heur_rewards.std(),
            "heuristic_tardiness_mean": heur_tardiness.mean(), "heuristic_tardiness_std": heur_tardiness.std(),
            "heuristic_late_jobs_mean": heur_late.mean(), "heuristic_late_jobs_std": heur_late.std(),
            "heuristic_jobs_scheduled_mean": heur_scheduled.mean(), "heuristic_jobs_scheduled_std": heur_scheduled.std(),
            "n_episodes": n_episodes,
        })

    print(f"\nEvaluation plots saved to: {run_dir}\n")
    print(f"Eval summary appended to: {EVAL_RESULTS_CSV}\n")




if __name__ == "__main__":
    main()
