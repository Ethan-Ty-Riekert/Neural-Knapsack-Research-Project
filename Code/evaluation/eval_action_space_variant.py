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
from Code.env.windowed_priority_gym_wrapper import WindowedPriorityGymSchedulingEnv
from Code.env.action_branching_gym_wrapper import ActionBranchingGymSchedulingEnv
from Code.policies.priority_pointer_ppo_policy import PriorityPointerMaskableActorCriticPolicy
from Code.policies.windowed_priority_pointer_ppo_policy import WindowedPriorityPointerMaskableActorCriticPolicy
from Code.policies.action_branching_ppo_policy import ActionBranchingMaskableActorCriticPolicy
from Code.env.env_config import generate_env_config
from Code.env.arrival_process import generate_poisson_arrivals
from Code.evaluation.eval_rl_agent import run_heuristic
from Code.baselines.registry import DEFAULT_HEURISTICS
from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING
from Code.utils.paths import MODELS_DIR
from Code.utils.results_log import append_eval_result


def build_eval_env(option: str, full_gym_env, window_size=None, window_order="edf",
                    use_atc_feature=False):
    """window_size (2026-09-18 follow-up): mirrors
    train_action_space_variant.py::build_env_and_policy's windowed branch --
    must be passed here too or a windowed checkpoint's obs/action-space shape
    won't match what MaskablePPO.load() expects. window_order (2026-09-21
    follow-up): doesn't affect obs/action-space SHAPE, but must still match
    the checkpoint's TRAINING window_order for correct evaluation semantics
    -- a mismatched order silently evaluates the model against differently-
    selected candidates than it learned to interpret, not a crash.

    use_atc_feature (2026-09-23 follow-up): mirrors build_env_and_policy's
    same-named param -- must be passed here too, or an Option 1 checkpoint
    trained with the extra ATC feature has an obs shape mismatch against the
    template env used to load it. Only meaningful for option == '1'."""
    if option == "1":
        if window_size is not None:
            raise ValueError("--window-size only applies to --option 2/3.")
        env = RuleSelectionGymSchedulingEnv(full_gym_env, use_atc_feature=use_atc_feature)
    elif option in ("2", "3"):
        if use_atc_feature:
            raise ValueError("--use-atc-feature only applies to --option 1.")
        use_atc = option == "3"
        if window_size is not None:
            env = WindowedPriorityGymSchedulingEnv(full_gym_env, window_size=window_size, use_atc=use_atc,
                                                     window_order=window_order)
        else:
            env = PriorityOnlyGymSchedulingEnv(full_gym_env, use_atc=use_atc)
    elif option == "4":
        if window_size is not None:
            raise ValueError("--window-size only applies to --option 2/3.")
        if use_atc_feature:
            raise ValueError("--use-atc-feature only applies to --option 1.")
        env = ActionBranchingGymSchedulingEnv(full_gym_env)
    else:
        raise ValueError(f"Unknown option {option!r}")
    return ActionMasker(env, mask_fn)


def load_model(option: str, template_env, checkpoint_tag=None, window_size=None):
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    checkpoint = MODELS_DIR / f"action_space_option{option}_ppo{tag_suffix}.zip"
    if window_size is not None:
        custom_objects = {"policy_class": WindowedPriorityPointerMaskableActorCriticPolicy}
    elif option in ("2", "3"):
        custom_objects = {"policy_class": PriorityPointerMaskableActorCriticPolicy}
    elif option == "4":
        custom_objects = {"policy_class": ActionBranchingMaskableActorCriticPolicy}
    else:
        custom_objects = None
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
        "job_weights": base_env.job_weights.copy(),
        "late_jobs": int((base_env.tardiness > 0).sum()),
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
        "truncated": truncated,
    }


def _weighted_tardiness(result):
    """SchedulingEnv.tardiness stores RAW, unweighted T_j -- no weight
    multiplication anywhere in the env (2026-09-18, S2W10 finding). Every
    result before this fix reported raw tardiness even once real job
    weights were introduced, which is NOT the objective the reward function
    (lambda_2 * sum(w_j*T_j)) or the RL training signal actually optimizes
    -- see training-log.md's matching entry. Compute it explicitly here so
    both are always reported and this can't silently recur. Equal to raw
    tardiness when every weight is 1.0 (the unweighted default)."""
    return float((result["tardiness"] * result["job_weights"]).sum())


def _print_row(tag, result, denom, trunc_note=""):
    print(f"{tag:22s} reward={result['total_reward']:9.2f}  tardiness={result['tardiness'].sum():9.2f}  "
          f"weighted_tardiness={_weighted_tardiness(result):9.2f}  "
          f"late={result['late_jobs']:4d}  scheduled={result['jobs_scheduled']:4d}/{denom}{trunc_note}")


def _print_aggregate_row(tag, tardiness_vals, weighted_tardiness_vals, late_vals, scheduled_vals, denom):
    print(f"{tag:22s} tardiness={np.mean(tardiness_vals):8.2f}+/-{np.std(tardiness_vals):6.2f}  "
          f"weighted_tardiness={np.mean(weighted_tardiness_vals):8.2f}+/-{np.std(weighted_tardiness_vals):6.2f}  "
          f"late={np.mean(late_vals):5.2f}  scheduled={np.mean(scheduled_vals):5.2f}/{denom}")


_POLICY_TYPE = {"1": "rule_selection", "2": "priority_pointer", "3": "priority_pointer_atc",
                 "4": "action_branching"}


def _log_result(args, model_path, heuristic_name,
                 variant_reward, variant_tardiness, variant_weighted, variant_late, variant_scheduled,
                 heur_reward, heur_tardiness, heur_weighted, heur_late, heur_scheduled, n_episodes):
    """Append one row to rl_training/eval_results.csv (Code/utils/results_log.py),
    matching eval_rl_agent.py's own append_eval_result convention exactly (one row
    per heuristic compared, --no-log to opt out). weighted_tardiness_mean/std are
    logged alongside raw tardiness_mean/std per the 2026-09-18 metric-bug fix --
    see EVAL_RESULT_FIELDS' matching comment in results_log.py."""
    if args.no_log:
        return
    policy_type = _POLICY_TYPE[args.option]
    if args.window_size is not None:
        policy_type = f"windowed{args.window_size}_{policy_type}"
    tag = ((args.checkpoint_tag or "") + ("_randomized_eval" if args.randomized_eval else "")
           + ("_online_eval" if args.online else ""))
    variant_reward = np.atleast_1d(variant_reward).astype(float)
    variant_tardiness = np.atleast_1d(variant_tardiness).astype(float)
    variant_weighted = np.atleast_1d(variant_weighted).astype(float)
    variant_late = np.atleast_1d(variant_late).astype(float)
    variant_scheduled = np.atleast_1d(variant_scheduled).astype(float)
    heur_reward = np.atleast_1d(heur_reward).astype(float)
    heur_tardiness = np.atleast_1d(heur_tardiness).astype(float)
    heur_weighted = np.atleast_1d(heur_weighted).astype(float)
    heur_late = np.atleast_1d(heur_late).astype(float)
    heur_scheduled = np.atleast_1d(heur_scheduled).astype(float)
    append_eval_result({
        "algo": f"action_space_option{args.option}",
        "policy_type": policy_type,
        "tag": tag,
        "model_path": str(model_path),
        "reward_mean": variant_reward.mean(), "reward_std": variant_reward.std(),
        "tardiness_mean": variant_tardiness.mean(), "tardiness_std": variant_tardiness.std(),
        "weighted_tardiness_mean": variant_weighted.mean(), "weighted_tardiness_std": variant_weighted.std(),
        "late_jobs_mean": variant_late.mean(), "late_jobs_std": variant_late.std(),
        "jobs_scheduled_mean": variant_scheduled.mean(), "jobs_scheduled_std": variant_scheduled.std(),
        "heuristic_name": heuristic_name,
        "heuristic_reward_mean": heur_reward.mean(), "heuristic_reward_std": heur_reward.std(),
        "heuristic_tardiness_mean": heur_tardiness.mean(), "heuristic_tardiness_std": heur_tardiness.std(),
        "heuristic_weighted_tardiness_mean": heur_weighted.mean(), "heuristic_weighted_tardiness_std": heur_weighted.std(),
        "heuristic_late_jobs_mean": heur_late.mean(), "heuristic_late_jobs_std": heur_late.std(),
        "heuristic_jobs_scheduled_mean": heur_scheduled.mean(), "heuristic_jobs_scheduled_std": heur_scheduled.std(),
        "n_episodes": n_episodes,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", choices=["1", "2", "3", "4"], required=True)
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
    parser.add_argument("--window-size", type=int, default=None,
                         help="Must match --window-size the checkpoint was TRAINED with "
                              "(train_action_space_variant.py) -- option 2/3 only.")
    parser.add_argument("--window-order", choices=["edf", "fifo"], default="edf",
                         help="Must match --window-order the checkpoint was TRAINED with.")
    parser.add_argument("--use-atc-feature", action="store_true",
                         help="Must match --use-atc-feature the checkpoint was TRAINED with "
                              "(train_action_space_variant.py) -- option 1 only.")
    parser.add_argument("--no-log", action="store_true",
                         help="Skip appending this run's aggregate result to "
                              "rl_training/eval_results.csv (Code/utils/results_log.py). "
                              "Logging is on by default so every eval run this script does "
                              "is queryable later, not just printed to stdout.")
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
            ), window_size=args.window_size, window_order=args.window_order,
               use_atc_feature=args.use_atc_feature)
        else:
            template_env = build_eval_env(args.option, make_base_gym_env(
                seed=RANDOM_INSTANCE_SEED_CEILING, job_weight_range=job_weight_range,
            ), window_size=args.window_size, window_order=args.window_order,
               use_atc_feature=args.use_atc_feature)
        model = load_model(args.option, template_env, checkpoint_tag=args.checkpoint_tag,
                            window_size=args.window_size)
        tag_suffix = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
        model_path = MODELS_DIR / f"action_space_option{args.option}_ppo{tag_suffix}.zip"

        variant_reward, variant_tardiness, variant_weighted, variant_late, variant_scheduled, denoms = \
            [], [], [], [], [], []
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
                ), window_size=args.window_size, window_order=args.window_order,
                   use_atc_feature=args.use_atc_feature)
            else:
                denoms.append(100)
                env = build_eval_env(args.option, make_base_gym_env(seed=seed, job_weight_range=job_weight_range),
                                      window_size=args.window_size, window_order=args.window_order,
                                      use_atc_feature=args.use_atc_feature)
            r = run_episode(model, env)
            variant_reward.append(r["total_reward"])
            variant_tardiness.append(r["tardiness"].sum())
            variant_weighted.append(_weighted_tardiness(r))
            variant_late.append(r["late_jobs"])
            variant_scheduled.append(r["jobs_scheduled"])
        avg_denom = np.mean(denoms)
        _print_aggregate_row(f"Option {args.option}", variant_tardiness, variant_weighted, variant_late,
                              variant_scheduled, f"{avg_denom:.0f}")

        for name in args.heuristics:
            h_reward, h_tardiness, h_weighted, h_late, h_scheduled = [], [], [], [], []
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
                h_reward.append(h["total_reward"])
                h_tardiness.append(h["tardiness"].sum())
                h_weighted.append(float((h["tardiness"] * cfg["job_weights"]).sum()))
                h_late.append(h["late_jobs"])
                h_scheduled.append(h["jobs_scheduled"])
            _print_aggregate_row(name, h_tardiness, h_weighted, h_late, h_scheduled, f"{avg_denom:.0f}")
            _log_result(args, model_path, name,
                        variant_reward, variant_tardiness, variant_weighted, variant_late, variant_scheduled,
                        h_reward, h_tardiness, h_weighted, h_late, h_scheduled, args.eval_runs)
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
                                           job_weight_range=job_weight_range)
        denom = 100
        variant_env = make_base_gym_env(job_weight_range=job_weight_range)

    env = build_eval_env(args.option, variant_env, window_size=args.window_size, window_order=args.window_order,
                          use_atc_feature=args.use_atc_feature)
    model = load_model(args.option, env, checkpoint_tag=args.checkpoint_tag, window_size=args.window_size)
    tag_suffix = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
    model_path = MODELS_DIR / f"action_space_option{args.option}_ppo{tag_suffix}.zip"
    result = run_episode(model, env)
    trunc_note = " [TRUNCATED]" if result["truncated"] else ""
    _print_row(f"Option {args.option}", result, denom, trunc_note)

    for name in args.heuristics:
        h = run_heuristic(name, config=eval_config)
        h["job_weights"] = eval_config["job_weights"]
        _print_row(name, h, denom)
        _log_result(args, model_path, name,
                    result["total_reward"], result["tardiness"].sum(), _weighted_tardiness(result),
                    result["late_jobs"], result["jobs_scheduled"],
                    h["total_reward"], h["tardiness"].sum(), _weighted_tardiness(h),
                    h["late_jobs"], h["jobs_scheduled"], 1)


if __name__ == "__main__":
    main()
