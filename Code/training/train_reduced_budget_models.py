"""train_reduced_budget_models.py - Reduced-budget report-prep run (2026-09-04).

v2 (same session, same date): the first pass used a single combined
optimize_for="pareto" search and picked models A/B's hyperparameters from its
max-reward trial. That trial turned out to have lambda_2=0.505 -- essentially
no tardiness penalty -- reproducing, at small scale, exactly the reward/
tardiness misalignment this project's own training log repeatedly documents
(Stage A/B, the Pareto investigation). This version fixes that by running TWO
searches, matching the project's real methodology instead of economizing it
away: optimize_for="tardiness" (a composite mean_reward - 50*tardiness_norm
score, built specifically to avoid this trap -- see optuna_tune.py's
objective_a2c docstring) for models A/B, and a separate optimize_for="pareto"
search for model C's genuine knee. Also scaled the instance up (more jobs/
machines, tighter capacity) so the problem actually stresses resource-packing
rather than being solvable by deadline-ordering alone, which was letting
EDF/LST win almost by construction at the smaller v1 scale.

See Results/<out-dir>/README.md for the full list of deviations from the
project's real, full-fidelity methodology.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from Code.env.env_config import generate_env_config
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.scheduling_env import SchedulingEnv
from Code.policies.a2c_policy import MaskableA2C
from Code.training.optuna_tune import run_optimization

# Overwritten from CLI args in main() before any other function runs.
NUM_JOBS = 45
NUM_MACHINES = 6
HORIZON = 50
DEADLINE_RANGE = (10, 50)
JOB_DURATION_RANGE = (1, 8)
TRAIN_SEED = 0
HELDOUT_SEEDS = list(range(500000, 500008))  # 8 held-out instances


def mask_fn(env: GymSchedulingEnv):
    return env.get_action_mask()


def build_config(seed: int) -> dict:
    return generate_env_config(
        seed=seed,
        num_jobs=NUM_JOBS,
        num_machines=NUM_MACHINES,
        horizon=HORIZON,
        job_duration_range=JOB_DURATION_RANGE,
        deadline_range=DEADLINE_RANGE,
    )


def build_env(config: dict, lambda_1, lambda_2, lambda_3, idle_penalty,
              invalid_penalty, use_potential_shaping, shaping_gamma):
    base_env = SchedulingEnv(
        job_durations=config["job_durations"],
        job_resources=config["job_resources"],
        job_deadlines=config["job_deadlines"],
        job_weights=config["job_weights"],
        num_machines=config["num_machines"],
        machine_capacity=config["machine_capacity"],
        horizon=config["horizon"],
        lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3,
        invalid_penalty=invalid_penalty, idle_penalty=idle_penalty,
        use_potential_shaping=use_potential_shaping, shaping_gamma=shaping_gamma,
    )
    gym_env = GymSchedulingEnv(base_env, max_jobs=NUM_JOBS)
    return Monitor(ActionMasker(gym_env, mask_fn))


class PrintProgress:
    """Minimal stand-in for LiveTrainingPlotter -- prints periodically instead
    of live-plotting, to avoid matplotlib overhead during a time-boxed run."""

    def __init__(self, label: str, every: int = 25):
        self.label = label
        self.every = every
        self.count = 0
        self.rewards = []

    def notify_episode(self, reward, timestep):
        self.count += 1
        self.rewards.append(reward)
        if self.count % self.every == 0 or self.count == 1:
            recent = np.mean(self.rewards[-self.every:])
            print(f"  [{self.label}] episode {self.count}, t={timestep}, "
                  f"mean reward (last {min(self.every, self.count)}): {recent:.2f}")


def select_knee_trial(study):
    """Pick a genuine non-dominated knee trial off a pareto-mode study's
    front -- excludes the max-reward extreme, picks whichever remaining trial
    buys the most tardiness reduction per unit of reward given up (avoids a
    degenerate near-zero-reward/zero-tardiness extreme, which the front can
    also contain -- see v1's training-log entry)."""
    trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
    print(f"\nPareto front: {len(trials)} non-dominated trial(s)")
    for t in trials:
        print(f"  trial {t.number}: reward={t.values[0]:.2f} tardiness_norm={t.values[1]:.4f}")

    best_reward_trial = trials[0]
    if len(trials) == 1:
        return best_reward_trial, True
    rewards = np.array([t.values[0] for t in trials])
    tards = np.array([t.values[1] for t in trials])
    r_span = max(rewards.max() - rewards.min(), 1e-8)
    t_span = max(tards.max() - tards.min(), 1e-8)
    rest = trials[1:]
    knee_trial = max(
        rest,
        key=lambda t: (best_reward_trial.values[1] - t.values[1]) / t_span
        - (best_reward_trial.values[0] - t.values[0]) / r_span,
    )
    return knee_trial, False


def train_one_model(name, params, out_dir: Path, timesteps: int,
                     use_rcpo=False, rcpo_alpha=0.0, rcpo_lambda_lr=0.05,
                     rcpo_update_every=3, trial_number=None):
    print(f"\n{'='*70}\nTraining model: {name} ({timesteps} timesteps)\n{'='*70}")
    config = build_config(TRAIN_SEED)
    env = build_env(
        config,
        lambda_1=params["lambda_1"], lambda_2=params["lambda_2"], lambda_3=params["lambda_3"],
        idle_penalty=params["idle_penalty"], invalid_penalty=params["invalid_penalty"],
        use_potential_shaping=True, shaping_gamma=params["gamma"],
    )
    policy_kwargs = dict(embed_dim=params["embed_dim"], hidden=params["hidden"])
    rcpo_kwargs = dict(
        use_rcpo=True, rcpo_alpha=rcpo_alpha, rcpo_lambda_init=params["lambda_2"],
        rcpo_lambda_lr=rcpo_lambda_lr, rcpo_lambda_max=50.0,
        rcpo_update_every_episodes=rcpo_update_every,
    ) if use_rcpo else {}

    agent = MaskableA2C(env, device="cpu", policy_type="pointer",
                         policy_kwargs=policy_kwargs, **rcpo_kwargs)
    agent.n_steps = params["n_steps"]
    agent.gamma = params["gamma"]
    agent.lam = params["gae_lambda"]
    agent.ent_coef = params["ent_coef"]
    agent.value_coef = params["value_coef"]
    agent.max_grad_norm = params["max_grad_norm"]
    agent.lr = params["learning_rate"]
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.lr)

    progress = PrintProgress(name)
    agent.train(total_timesteps=timesteps, plotter=progress)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{name}.pt"
    torch.save(agent.model.state_dict(), ckpt_path)

    hp = dict(params)
    hp.update({
        "model_name": name, "source_trial": trial_number, "timesteps": timesteps,
        "use_rcpo": use_rcpo, "rcpo_alpha": rcpo_alpha, "rcpo_lambda_lr": rcpo_lambda_lr,
        "rcpo_update_every_episodes": rcpo_update_every,
        "num_jobs": NUM_JOBS, "num_machines": NUM_MACHINES, "horizon": HORIZON,
        "deadline_range": list(DEADLINE_RANGE), "job_duration_range": list(JOB_DURATION_RANGE),
        "use_potential_shaping": True, "policy_type": "pointer",
    })
    with open(out_dir / f"{name}_hyperparams.json", "w") as f:
        json.dump(hp, f, indent=2, default=float)
    print(f"  Saved: {ckpt_path}")
    print(f"  Saved: {out_dir / f'{name}_hyperparams.json'}")
    return agent, ckpt_path


def derive_rcpo_alpha(agent) -> float:
    """Mean held-out episode_cost C(tau) for `agent`, on the held-out set --
    used as model B's rcpo_alpha (the achievable-alpha methodology,
    re-derived from THIS run rather than reusing any historical value)."""
    costs = []
    for seed in HELDOUT_SEEDS:
        config = build_config(seed)
        env = build_env(
            config,
            lambda_1=1.0, lambda_2=1.0, lambda_3=1.0,  # fixed eval rubric, matches eval_rl_agent.py
            idle_penalty=0.5, invalid_penalty=5.0,
            use_potential_shaping=False, shaping_gamma=0.99,
        )
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            mask = info["action_mask"]
            action = agent.act(obs, mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        costs.append(info["episode_cost"])
    alpha = float(np.mean(costs))
    print(f"\nDerived rcpo_alpha from model A's held-out mean episode_cost: {alpha:.4f} "
          f"(over {len(costs)} instances, costs={[round(c,3) for c in costs]})")
    return alpha


def main():
    global NUM_JOBS, NUM_MACHINES, HORIZON, DEADLINE_RANGE, JOB_DURATION_RANGE

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-jobs", type=int, default=NUM_JOBS)
    parser.add_argument("--num-machines", type=int, default=NUM_MACHINES)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--deadline-range", type=int, nargs=2, default=list(DEADLINE_RANGE))
    parser.add_argument("--job-duration-range", type=int, nargs=2, default=list(JOB_DURATION_RANGE))
    parser.add_argument("--search-trials", type=int, default=15)
    parser.add_argument("--search-trial-timesteps", type=int, default=6000)
    parser.add_argument("--final-timesteps", type=int, default=35000)
    parser.add_argument("--rcpo-lambda-lr", type=float, default=0.05)
    parser.add_argument("--rcpo-update-every", type=int, default=3)
    args = parser.parse_args()

    NUM_JOBS = args.num_jobs
    NUM_MACHINES = args.num_machines
    HORIZON = args.horizon
    DEADLINE_RANGE = tuple(args.deadline_range)
    JOB_DURATION_RANGE = tuple(args.job_duration_range)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}\n# Instance: {NUM_JOBS} jobs / {NUM_MACHINES} machines / horizon {HORIZON}, "
          f"deadline_range={DEADLINE_RANGE}, job_duration_range={JOB_DURATION_RANGE}\n{'#'*70}")

    print(f"\n{'#'*70}\n# Step 1a: Tardiness-aware Optuna search (optimize_for=tardiness), "
          f"for models A/B ({args.search_trials} trials x {args.search_trial_timesteps} timesteps)\n{'#'*70}")
    tardiness_study = run_optimization(
        algorithm="a2c", policy_type="pointer", n_trials=args.search_trials, n_jobs=1,
        optimize_for="tardiness", use_potential_shaping=True, randomize_instances=False,
        tuning_num_jobs=NUM_JOBS, tuning_num_machines=NUM_MACHINES, tuning_horizon=HORIZON,
        trial_timesteps=args.search_trial_timesteps, deadline_range=DEADLINE_RANGE,
    )
    ab_trial = tardiness_study.best_trial
    print(f"\nBest composite-score trial: {ab_trial.number} "
          f"(composite={ab_trial.value:.2f}, mean_reward={ab_trial.user_attrs.get('mean_reward'):.2f}, "
          f"mean_tardiness={ab_trial.user_attrs.get('mean_tardiness'):.2f}, "
          f"lambda_2={ab_trial.params['lambda_2']:.3f})")

    print(f"\n{'#'*70}\n# Step 1b: Multi-objective Optuna search (optimize_for=pareto), "
          f"for model C's knee ({args.search_trials} trials x {args.search_trial_timesteps} timesteps)\n{'#'*70}")
    pareto_study = run_optimization(
        algorithm="a2c", policy_type="pointer", n_trials=args.search_trials, n_jobs=1,
        optimize_for="pareto", use_potential_shaping=True, randomize_instances=False,
        tuning_num_jobs=NUM_JOBS, tuning_num_machines=NUM_MACHINES, tuning_horizon=HORIZON,
        trial_timesteps=args.search_trial_timesteps, deadline_range=DEADLINE_RANGE,
    )
    knee_trial, degenerate = select_knee_trial(pareto_study)

    search_summary = {
        "instance": {"num_jobs": NUM_JOBS, "num_machines": NUM_MACHINES, "horizon": HORIZON,
                     "deadline_range": list(DEADLINE_RANGE), "job_duration_range": list(JOB_DURATION_RANGE)},
        "n_trials": args.search_trials, "trial_timesteps": args.search_trial_timesteps,
        "ab_trial_tardiness_mode": {
            "number": ab_trial.number, "composite_score": ab_trial.value,
            "mean_reward": ab_trial.user_attrs.get("mean_reward"),
            "mean_tardiness": ab_trial.user_attrs.get("mean_tardiness"),
            "params": ab_trial.params,
        },
        "degenerate_pareto_front": degenerate,
        "knee_trial_pareto_mode": {"number": knee_trial.number, "values": list(knee_trial.values),
                                    "params": knee_trial.params},
    }
    with open(out_dir / "optuna_search_summary.json", "w") as f:
        json.dump(search_summary, f, indent=2, default=float)

    print(f"\n{'#'*70}\n# Step 2: Train model A (Pointer + shaping)\n{'#'*70}")
    agent_a, _ = train_one_model(
        "pointer_shaped", ab_trial.params, out_dir, args.final_timesteps,
        trial_number=ab_trial.number,
    )

    print(f"\n{'#'*70}\n# Step 3: Derive rcpo_alpha from model A's held-out eval\n{'#'*70}")
    rcpo_alpha = derive_rcpo_alpha(agent_a)

    print(f"\n{'#'*70}\n# Step 4: Train model B (Pointer + RCPO fixed)\n{'#'*70}")
    train_one_model(
        "pointer_rcpo", ab_trial.params, out_dir, args.final_timesteps,
        use_rcpo=True, rcpo_alpha=rcpo_alpha, rcpo_lambda_lr=args.rcpo_lambda_lr,
        rcpo_update_every=args.rcpo_update_every, trial_number=ab_trial.number,
    )

    print(f"\n{'#'*70}\n# Step 5: Train model C (Pointer + Pareto-knee)\n{'#'*70}")
    train_one_model(
        "pointer_paretoknee", knee_trial.params, out_dir, args.final_timesteps,
        trial_number=knee_trial.number,
    )

    print(f"\nAll models trained. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
