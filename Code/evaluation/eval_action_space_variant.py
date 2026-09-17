"""eval_action_space_variant.py - Evaluate a trained action-space-reduction
checkpoint (Options 1/2/3, see Code/training/train_action_space_variant.py)
against the classical heuristic baselines: offline (fixed deployed
instance), offline randomized (50-held-out-instance protocol, matching
eval_rl_agent.py's --randomized-eval), or online (Poisson arrivals,
optionally heavy-tailed -- see Code/env/arrival_process.py). Mirrors
Code/evaluation/eval_rl_agent.py's run_model()/run_heuristic() output shape
(tardiness/late_jobs/jobs_scheduled) so numbers are directly comparable to
every other result in eval_results.csv.

Run from the repo root:
    python -m Code.evaluation.eval_action_space_variant --option 1
    python -m Code.evaluation.eval_action_space_variant --option 1 --randomized-eval --checkpoint-tag offline_randomized
    python -m Code.evaluation.eval_action_space_variant --option 1 --online \\
        --arrival-rate 3 --online-horizon 100 --online-max-jobs 400 \\
        --job-size-distribution lognormal --checkpoint-tag online_lognormal
"""
import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.training.train_action_space_variant import make_base_gym_env, make_online_base_gym_env, mask_fn
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv
from Code.env.priority_only_gym_wrapper import PriorityOnlyGymSchedulingEnv
from Code.policies.priority_pointer_ppo_policy import PriorityPointerMaskableActorCriticPolicy
from Code.env.env_config import generate_env_config
from Code.env.arrival_process import generate_poisson_arrivals
from Code.evaluation.eval_rl_agent import run_heuristic
from Code.baselines.registry import DEFAULT_HEURISTICS
from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING
from Code.utils.paths import MODELS_DIR


def build_eval_env(option: str, full_gym_env):
    if option == "1":
        env = RuleSelectionGymSchedulingEnv(full_gym_env)
    elif option in ("2", "3"):
        env = PriorityOnlyGymSchedulingEnv(full_gym_env, use_atc=(option == "3"))
    else:
        raise ValueError(f"Unknown option {option!r}")
    return ActionMasker(env, mask_fn)


def load_model(option: str, template_env, checkpoint_tag=None):
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    checkpoint = MODELS_DIR / f"action_space_option{option}_ppo{tag_suffix}.zip"
    custom_objects = {"policy_class": PriorityPointerMaskableActorCriticPolicy} if option in ("2", "3") else None
    return MaskablePPO.load(str(checkpoint), env=template_env, custom_objects=custom_objects)


def run_episode(model, env):
    """Run one deterministic episode of an already-loaded model against an
    already-built env -- model.predict() only needs matching obs/action-mask
    shapes, not the exact env instance it was loaded against, so the SAME
    model can be reused across many distinct held-out instances without
    reloading (load_model() is the expensive step, this isn't)."""
    obs, info = env.reset()
    done = truncated = False
    rewards = []
    base_env = env.env.env

    while not (done or truncated):
        action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))

    return {
        "total_reward": sum(rewards),
        "tardiness": base_env.tardiness.copy(),
        "late_jobs": int((base_env.tardiness > 0).sum()),
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
        "truncated": truncated,
    }


def _print_row(tag, result, denom, trunc_note=""):
    print(f"{tag:22s} reward={result['total_reward']:9.2f}  tardiness={result['tardiness'].sum():9.2f}  "
          f"late={result['late_jobs']:4d}  scheduled={result['jobs_scheduled']:4d}/{denom}{trunc_note}")


def _print_aggregate_row(tag, tardiness_vals, late_vals, scheduled_vals, denom):
    print(f"{tag:22s} tardiness={np.mean(tardiness_vals):8.2f}+/-{np.std(tardiness_vals):6.2f}  "
          f"late={np.mean(late_vals):5.2f}  scheduled={np.mean(scheduled_vals):5.2f}/{denom}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", choices=["1", "2", "3"], required=True)
    parser.add_argument("--heuristics", nargs="*", default=DEFAULT_HEURISTICS)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--online-horizon", type=int, default=100)
    parser.add_argument("--online-max-jobs", type=int, default=400)
    parser.add_argument("--job-size-distribution", choices=["uniform", "lognormal"], default="uniform")
    parser.add_argument("--eval-seed", type=int, default=0,
                         help="Fixed held-out instance seed for the single-instance offline/online "
                              "comparison (NOT part of training's resampler seed stream, which draws "
                              "from 1..RANDOM_INSTANCE_SEED_CEILING -- seed=0 is never drawn there).")
    parser.add_argument("--randomized-eval", action="store_true",
                         help="Offline only: evaluate on --eval-runs held-out instances (seeds "
                              "RANDOM_INSTANCE_SEED_CEILING..+N-1, matching eval_rl_agent.py's "
                              "--randomized-eval convention exactly) instead of the single fixed "
                              "instance -- the right comparison for a --randomize-instances-trained "
                              "checkpoint.")
    parser.add_argument("--eval-runs", type=int, default=50,
                         help="--randomized-eval only: number of held-out instances (default 50, "
                              "matching eval_rl_agent.py's own default).")
    parser.add_argument("--checkpoint-tag", type=str, default=None,
                         help="Must match --save-tag used when training this checkpoint "
                              "(train_action_space_variant.py), e.g. 'online_lognormal'.")
    parser.add_argument("--job-weight-min", type=int, default=None,
                         help="2026-09-18 follow-up: evaluate on weighted instances -- must match "
                              "what the checkpoint was TRAINED on (or deliberately not, to test "
                              "transfer). See train_action_space_variant.py's matching flag.")
    parser.add_argument("--job-weight-max", type=int, default=None)
    args = parser.parse_args()

    if args.online and args.arrival_rate is None:
        parser.error("--online requires --arrival-rate")
    if (args.job_weight_min is None) != (args.job_weight_max is None):
        parser.error("--job-weight-min and --job-weight-max must be given together")
    job_weight_range = (
        (args.job_weight_min, args.job_weight_max) if args.job_weight_min is not None else None
    )

    if args.randomized_eval:
        label = "ONLINE" if args.online else "OFFLINE"
        print(f"{label} RANDOMIZED-EVAL: {args.eval_runs} held-out instances "
              f"(seeds {RANDOM_INSTANCE_SEED_CEILING}..{RANDOM_INSTANCE_SEED_CEILING + args.eval_runs - 1})\n")

        if args.online:
            template_env = build_eval_env(args.option, make_online_base_gym_env(
                args.arrival_rate, args.online_horizon, args.online_max_jobs, args.job_size_distribution,
                seed=RANDOM_INSTANCE_SEED_CEILING, use_resampler=False, job_weight_range=job_weight_range,
            ))
        else:
            template_env = build_eval_env(args.option, make_base_gym_env(
                seed=RANDOM_INSTANCE_SEED_CEILING, job_weight_range=job_weight_range,
            ))
        model = load_model(args.option, template_env, checkpoint_tag=args.checkpoint_tag)

        variant_tardiness, variant_late, variant_scheduled, denoms = [], [], [], []
        for i in range(args.eval_runs):
            seed = RANDOM_INSTANCE_SEED_CEILING + i
            if args.online:
                cfg = generate_poisson_arrivals(seed=seed, arrival_rate=args.arrival_rate,
                                                 horizon=args.online_horizon, max_jobs=args.online_max_jobs,
                                                 job_size_distribution=args.job_size_distribution,
                                                 job_weight_range=job_weight_range)
                denoms.append(int((cfg["job_arrival_times"] <= args.online_horizon).sum()))
                env = build_eval_env(args.option, make_online_base_gym_env(
                    args.arrival_rate, args.online_horizon, args.online_max_jobs, args.job_size_distribution,
                    seed=seed, use_resampler=False, job_weight_range=job_weight_range,
                ))
            else:
                denoms.append(100)
                env = build_eval_env(args.option, make_base_gym_env(seed=seed, job_weight_range=job_weight_range))
            r = run_episode(model, env)
            variant_tardiness.append(r["tardiness"].sum())
            variant_late.append(r["late_jobs"])
            variant_scheduled.append(r["jobs_scheduled"])
        avg_denom = np.mean(denoms)
        _print_aggregate_row(f"Option {args.option}", variant_tardiness, variant_late, variant_scheduled, f"{avg_denom:.0f}")

        for name in args.heuristics:
            h_tardiness, h_late, h_scheduled = [], [], []
            for i in range(args.eval_runs):
                seed = RANDOM_INSTANCE_SEED_CEILING + i
                if args.online:
                    cfg = generate_poisson_arrivals(seed=seed, arrival_rate=args.arrival_rate,
                                                     horizon=args.online_horizon, max_jobs=args.online_max_jobs,
                                                     job_size_distribution=args.job_size_distribution,
                                                     job_weight_range=job_weight_range)
                else:
                    cfg = generate_env_config(seed=seed, num_jobs=100, num_machines=10, horizon=100,
                                               job_weight_range=job_weight_range)
                h = run_heuristic(name, config=cfg)
                h_tardiness.append(h["tardiness"].sum())
                h_late.append(h["late_jobs"])
                h_scheduled.append(h["jobs_scheduled"])
            _print_aggregate_row(name, h_tardiness, h_late, h_scheduled, f"{avg_denom:.0f}")
        return

    if args.online:
        eval_config = generate_poisson_arrivals(
            seed=args.eval_seed, arrival_rate=args.arrival_rate, horizon=args.online_horizon,
            max_jobs=args.online_max_jobs, job_size_distribution=args.job_size_distribution,
            job_weight_range=job_weight_range,
        )
        n_realized = int((eval_config["job_arrival_times"] <= args.online_horizon).sum())
        denom = n_realized
        variant_env = make_online_base_gym_env(
            args.arrival_rate, args.online_horizon, args.online_max_jobs,
            args.job_size_distribution, seed=args.eval_seed, use_resampler=False,
            job_weight_range=job_weight_range,
        )
        print(f"ONLINE eval: seed={args.eval_seed}, arrival_rate={args.arrival_rate}, "
              f"horizon={args.online_horizon}, job_size_distribution={args.job_size_distribution}, "
              f"{n_realized} realized arrivals\n")
    else:
        eval_config = generate_env_config(seed=0, num_jobs=100, num_machines=10, horizon=100,
                                           job_weight_range=job_weight_range) if job_weight_range else None
        denom = 100
        variant_env = make_base_gym_env(job_weight_range=job_weight_range)

    env = build_eval_env(args.option, variant_env)
    model = load_model(args.option, env, checkpoint_tag=args.checkpoint_tag)
    result = run_episode(model, env)
    trunc_note = " [TRUNCATED]" if result["truncated"] else ""
    _print_row(f"Option {args.option}", result, denom, trunc_note)

    for name in args.heuristics:
        h = run_heuristic(name, config=eval_config)
        _print_row(name, h, denom)


if __name__ == "__main__":
    main()
