"""diagnose_rule_choice_congestion.py - One-off diagnostic script (2026-09-22,
S2W9), testing a specific hypothesis about the still-open online "more training
hurts" mystery from Future/research/training-log.md's 2026-09-21/22 entries.

Hypothesis (grounded in Smith 1956's flow-time-optimality result for SPT, and
the empirical WSPT/ATC-beat-SPT ordering already found): the online SPT-
collapsed policy converged onto "clear the queue fast" (a real congestion-
management strategy) rather than a fully weight-aware one. If that's right, rule
choice should correlate with CONGESTION (more remaining jobs = more SPT-like
urgency) rather than being uniformly state-independent -- and the policy should
be systematically weight-BLIND (not preferring higher-weight jobs among its
feasible options) even under high congestion, which is the specific gap between
SPT and WSPT/ATC.

Not part of the permanent test suite -- a targeted analysis script for this one
investigation, matching this project's established "rule-choice logging" pattern
(training-log.md's 2026-09-18 entries) but built as a reusable script instead of
discarded ad-hoc code this time.

Run from the repo root:
    python -m Code.evaluation.diagnose_rule_choice_congestion
"""
import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.training.train_action_space_variant import make_online_base_gym_env, mask_fn
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv, RULE_NAMES
from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING
from Code.utils.paths import MODELS_DIR

DEFAULT_CHECKPOINT_TAG = "online_lognormal_rho075_dense_weighted_diagnostics"
N_EPISODES = 10
ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST = 9, 100, 1300, "lognormal"
JOB_WEIGHT_RANGE = (1, 6)


def run_episode_with_logging(model, env):
    """Same rollout mechanics as eval_action_space_variant.py::run_episode(),
    plus per-step logging of: chosen rule, congestion level (|remaining_jobs|),
    and whether the placed job's weight is above/below the median weight among
    jobs CURRENTLY FEASIBLE (i.e. did the policy actually favour a heavier job
    when it had the option, or pick blind to weight)."""
    obs, info = env.reset()
    done = truncated = False
    base_env = env.env.env  # RuleSelectionGymSchedulingEnv.env == raw OnlineSchedulingEnv

    records = []
    while not (done or truncated):
        action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        action = int(action)
        rule_name = RULE_NAMES[action] if action < len(RULE_NAMES) else "idle"

        congestion = len(base_env.remaining_jobs)
        feasible_weights = [base_env.job_weights[j] for j in base_env.remaining_jobs]
        median_feasible_weight = float(np.median(feasible_weights)) if feasible_weights else float("nan")

        job_before = set(base_env.remaining_jobs)
        obs, reward, done, truncated, info = env.step(action)
        job_after = set(base_env.remaining_jobs)
        placed = job_before - job_after  # the one job removed this step, if any (empty if idle/no-op)

        placed_weight = base_env.job_weights[next(iter(placed))] if placed else None
        records.append({
            "rule": rule_name,
            "congestion": congestion,
            "median_feasible_weight": median_feasible_weight,
            "placed_weight": placed_weight,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-tag", type=str, default=DEFAULT_CHECKPOINT_TAG,
                         help="Must match --save-tag used when training the checkpoint "
                              "(train_action_space_variant.py, Option 1 online only).")
    args = parser.parse_args()
    checkpoint = MODELS_DIR / f"action_space_option1_ppo_{args.checkpoint_tag}.zip"
    print(f"Loading checkpoint: {checkpoint}")

    all_records = []
    template_full = make_online_base_gym_env(
        ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST, seed=RANDOM_INSTANCE_SEED_CEILING,
        use_resampler=False, job_weight_range=JOB_WEIGHT_RANGE,
    )
    template_env = ActionMasker(RuleSelectionGymSchedulingEnv(template_full), mask_fn)
    model = MaskablePPO.load(str(checkpoint), env=template_env)

    for i in range(N_EPISODES):
        seed = RANDOM_INSTANCE_SEED_CEILING + i
        full = make_online_base_gym_env(
            ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST, seed=seed,
            use_resampler=False, job_weight_range=JOB_WEIGHT_RANGE,
        )
        env = ActionMasker(RuleSelectionGymSchedulingEnv(full), mask_fn)
        all_records.extend(run_episode_with_logging(model, env))

    total = len(all_records)
    rule_counts = {}
    for r in all_records:
        rule_counts[r["rule"]] = rule_counts.get(r["rule"], 0) + 1
    print(f"=== Rule frequency across {N_EPISODES} episodes ({total} decisions) ===")
    for rule, count in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {rule:8s}: {count:5d} ({100*count/total:.1f}%)")

    congestion_vals = [r["congestion"] for r in all_records]
    print(f"\n=== Congestion (|remaining_jobs|) level ===")
    print(f"  mean={np.mean(congestion_vals):.1f}, median={np.median(congestion_vals):.1f}, "
          f"min={min(congestion_vals)}, max={max(congestion_vals)}")

    # Test 1: does rule choice correlate with congestion? Compare mean
    # congestion at SPT-chosen steps vs. all other (non-idle) rule choices.
    spt_congestion = [r["congestion"] for r in all_records if r["rule"] == "SPT"]
    other_congestion = [r["congestion"] for r in all_records if r["rule"] not in ("SPT", "idle")]
    print(f"\n=== Test 1: does SPT get chosen more under higher congestion? ===")
    if spt_congestion:
        print(f"  mean congestion when SPT chosen:        {np.mean(spt_congestion):.1f}  (n={len(spt_congestion)})")
    if other_congestion:
        print(f"  mean congestion when OTHER rule chosen: {np.mean(other_congestion):.1f}  (n={len(other_congestion)})")

    # Test 2: among steps where a job was actually placed, is the placed job's
    # weight systematically different from the median feasible weight at that
    # moment? A weight-BLIND policy should place jobs at ~50th percentile on
    # average (matching the median by definition, modulo tie noise); a
    # weight-AWARE one should systematically place jobs at/above the median.
    placements = [r for r in all_records if r["placed_weight"] is not None and not np.isnan(r["median_feasible_weight"])]
    above_median = sum(1 for r in placements if r["placed_weight"] >= r["median_feasible_weight"])
    print(f"\n=== Test 2: is the policy weight-aware when it DOES place a job? ===")
    print(f"  placed job's weight >= the feasible median: {above_median}/{len(placements)} "
          f"({100*above_median/max(1,len(placements)):.1f}%) -- 50% would indicate a weight-blind policy")

    print("\nDIAGNOSTIC COMPLETE (not a pass/fail test -- see printed values and training-log.md's matching entry)")


if __name__ == "__main__":
    main()
