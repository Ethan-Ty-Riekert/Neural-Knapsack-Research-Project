"""diagnose_credit_assignment_lag.py - One-off diagnostic script (2026-09-26,
S2W10), direct empirical test of the credit-assignment-lag HYPOTHESIS from
Future/research/training-log.md's matching 2026-09-26 entry: does the
dense_tardiness reward's per-tick tardiness charge fire further in the
future, relative to the decision tick, in states where SPT and ATC would
choose differently, than in states where they agree?

Background: SchedulingEnv._dense_tardiness_tick_charge() (Code/env/
scheduling_env.py:300-346) only charges a non-zero cost for a job once it
is BOTH incomplete AND already past its deadline. So the reward signal for
a rule choice made while a job still has slack is delayed until that job
(if ever) actually goes late -- a real temporal gap between the decision and
its credit. The hypothesis: ATC's benefit over SPT/WSPT is concentrated in
exactly the shrinking-slack-but-not-yet-late regime (see atc_priority()'s
exp(-slack/(k*mean_p)) term), so ATC-relevant decisions may systematically
sit further from their own reward signal than SPT/WSPT-relevant ones --
making them harder for PPO's local policy-gradient updates (via GAE/the
value function) to credit correctly.

Method: roll out a trained Option 1 checkpoint deterministically. At every
decision step, record (a) the SPT-vs-ATC disagreement label (reusing
diagnose_observation_informativeness.py's exact logic, not reimplemented)
and (b) the number of ticks until the next tick, anywhere later in this SAME
episode, at which >=1 job is late-and-incomplete (i.e. the tick-charge
mechanism would fire a non-zero cost) -- computed by replaying each job's
known final start_time/duration/deadline after the episode completes (the
env is deterministic given the actions actually taken, so this is exact, not
an estimate). Compares the mean/median lag between disagree=1 and
disagree=0 states.

Not part of the permanent test suite -- a targeted analysis script, same
convention as diagnose_rule_choice_congestion.py and
diagnose_observation_informativeness.py.

Run from the repo root:
    python -m Code.evaluation.diagnose_credit_assignment_lag
"""
import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.evaluation.diagnose_observation_informativeness import spt_vs_atc_disagreement
from Code.training.train_action_space_variant import make_online_base_gym_env, mask_fn
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv
from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING
from Code.utils.paths import MODELS_DIR

DEFAULT_CHECKPOINT_TAG = "online_lognormal_rho075_dense_weighted_atcfeature_300k"
N_EPISODES = 20
ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST = 9, 100, 1300, "lognormal"
JOB_WEIGHT_RANGE = (1, 6)


def _charged_ticks(base_env):
    """Reconstructs, from the FINISHED episode's final job state, the exact
    set of ticks at which >=1 job is late-and-incomplete -- i.e. the ticks
    SchedulingEnv._dense_tardiness_tick_charge() charged a non-zero cost.
    Read-only reimplementation of that method's own per-job late-window
    logic (not called directly, to avoid its side effect on episode_cost).
    """
    charged = set()
    horizon = base_env.horizon
    for j in range(base_env.num_jobs):
        deadline = base_env.job_deadlines[j]
        if base_env.start_times[j] != -1:
            completion = base_env.start_times[j] + base_env.job_durations[j]
        else:
            completion = horizon + 1  # never scheduled -- late every tick through the end
        for t in range(int(deadline), min(int(completion), horizon + 1)):
            charged.add(t)
    return charged


def run_episode_with_logging(model, env, use_atc_feature):
    obs, info = env.reset()
    done = truncated = False
    rule_env = env.env
    base_env = rule_env.env

    per_step = []  # (time, disagree_label)
    while not (done or truncated):
        job_actions = rule_env._job_actions()
        if job_actions:
            unique_jobs = {rule_env._decode(a)[0] for a in job_actions}
            label = spt_vs_atc_disagreement(base_env, unique_jobs)
            per_step.append((base_env.time, label))

        action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))

    charged = _charged_ticks(base_env)
    if not charged:
        return []
    max_charged = max(charged)

    records = []
    for t, label in per_step:
        future = [c for c in charged if c >= t]
        if not future:
            continue
        lag = min(future) - t
        records.append({"time": t, "disagree": label, "lag": lag})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-tag", type=str, default=DEFAULT_CHECKPOINT_TAG,
                         help="Must match --save-tag used when training the checkpoint "
                              "(train_action_space_variant.py, Option 1 online only).")
    parser.add_argument("--use-atc-feature", action="store_true", default=True,
                         help="Must match the checkpoint's own --use-atc-feature setting.")
    args = parser.parse_args()
    checkpoint = MODELS_DIR / f"action_space_option1_ppo_{args.checkpoint_tag}.zip"
    print(f"Loading checkpoint: {checkpoint}")

    template_full = make_online_base_gym_env(
        ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST, seed=RANDOM_INSTANCE_SEED_CEILING,
        use_resampler=False, job_weight_range=JOB_WEIGHT_RANGE,
    )
    template_env = ActionMasker(
        RuleSelectionGymSchedulingEnv(template_full, use_atc_feature=args.use_atc_feature), mask_fn,
    )
    model = MaskablePPO.load(str(checkpoint), env=template_env)

    all_records = []
    for i in range(N_EPISODES):
        seed = RANDOM_INSTANCE_SEED_CEILING + i
        full = make_online_base_gym_env(
            ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST, seed=seed,
            use_resampler=False, job_weight_range=JOB_WEIGHT_RANGE,
        )
        env = ActionMasker(
            RuleSelectionGymSchedulingEnv(full, use_atc_feature=args.use_atc_feature), mask_fn,
        )
        all_records.extend(run_episode_with_logging(model, env, args.use_atc_feature))

    n = len(all_records)
    lags = np.array([r["lag"] for r in all_records])
    labels = np.array([r["disagree"] for r in all_records])
    disagree_lags = lags[labels == 1]
    agree_lags = lags[labels == 0]

    print(f"\n=== Credit-assignment-lag diagnostic: {n} decisions across {N_EPISODES} episodes ===")
    print(f"  disagree (SPT!=ATC) states: n={len(disagree_lags)}, "
          f"mean_lag={disagree_lags.mean():.2f}, median_lag={np.median(disagree_lags):.1f}")
    print(f"  agree    (SPT==ATC) states: n={len(agree_lags)}, "
          f"mean_lag={agree_lags.mean():.2f}, median_lag={np.median(agree_lags):.1f}")

    diff = disagree_lags.mean() - agree_lags.mean()
    print(f"\n  mean lag difference (disagree - agree): {diff:+.2f} ticks")
    if diff > 1.0:
        verdict = ("Disagree states sit measurably FURTHER from their reward signal -- "
                    "supports the credit-assignment-lag hypothesis as a plausible contributor "
                    "to PPO's SPT/WSPT-over-ATC bias.")
    elif diff < -1.0:
        verdict = ("Disagree states sit CLOSER to their reward signal, the opposite of the "
                    "hypothesis -- does not support it as stated; the bias likely has a "
                    "different or additional cause.")
    else:
        verdict = ("No meaningful difference in lag between agree/disagree states -- does not "
                    "support the credit-assignment-lag hypothesis as a distinguishing factor "
                    "here, though a per-tick lag is a coarse proxy and doesn't rule out subtler "
                    "credit-assignment effects (e.g. lag variance, or GAE's discounting itself, "
                    "rather than raw tick-distance).")
    print(f"\n  Interpretation: {verdict}")
    print("\nDIAGNOSTIC COMPLETE (not a pass/fail test -- see printed values and training-log.md's matching entry)")


if __name__ == "__main__":
    main()
