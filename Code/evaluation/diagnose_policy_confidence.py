"""diagnose_policy_confidence.py - One-off diagnostic script (2026-09-22,
S2W9), direct follow-up to diagnose_rule_choice_congestion.py's negative
result (congestion does NOT explain the online SPT-collapse -- SPT dominates
essentially unconditionally). That script only looked at the deterministic
ARGMAX action; this one looks at the full probability distribution/value
function underneath it, to test a sharper question: is the policy's
CONFIDENCE in SPT itself state-independent (true, uniform collapse -- the
policy "stopped looking" at the state for this decision), or does the
underlying distribution still vary with state even though the argmax always
resolves to SPT (a softer, still-differentiated policy whose confident-choice
just happens to always be SPT)?

Extracts, at every step of several deterministic online rollouts: the full
softmax probability vector over all 8 actions (masked), SPT's probability,
the margin between SPT and the second-highest probability action, the value
estimate V(s), and the congestion level -- then reports how much each of
these VARIES across states. Low variance in SPT's probability/margin across
a wide range of congestion levels would indicate genuine state-blindness at
the distribution level, not just at the argmax level.

Not part of the permanent test suite -- a targeted analysis script, same
convention as diagnose_rule_choice_congestion.py.

Run from the repo root:
    python -m Code.evaluation.diagnose_policy_confidence
"""
import numpy as np
import argparse

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.training.train_action_space_variant import make_online_base_gym_env, mask_fn
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv, RULE_NAMES
from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING
from Code.utils.paths import MODELS_DIR

DEFAULT_CHECKPOINT_TAG = "online_lognormal_rho075_dense_weighted_diagnostics"
N_EPISODES = 6
ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST = 9, 100, 1300, "lognormal"
JOB_WEIGHT_RANGE = (1, 6)
SPT_INDEX = RULE_NAMES.index("SPT")


def run_episode_with_logging(model, env):
    obs, info = env.reset()
    done = truncated = False
    base_env = env.env.env

    records = []
    while not (done or truncated):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask = np.asarray(info["action_mask"]).reshape(1, -1)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor, action_masks=mask)
            probs = dist.distribution.probs.squeeze(0).numpy()
            value = model.policy.predict_values(obs_tensor).item()

        sorted_probs = np.sort(probs)[::-1]
        top1_prob, top2_prob = sorted_probs[0], sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top1_prob - top2_prob

        congestion = len(base_env.remaining_jobs)
        records.append({
            "spt_prob": float(probs[SPT_INDEX]),
            "top1_prob": float(top1_prob),
            "margin": float(margin),
            "value": float(value),
            "congestion": congestion,
        })

        action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))

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

    n = len(all_records)
    spt_probs = np.array([r["spt_prob"] for r in all_records])
    margins = np.array([r["margin"] for r in all_records])
    values = np.array([r["value"] for r in all_records])
    congestion = np.array([r["congestion"] for r in all_records])

    print(f"=== SPT probability across {n} decisions ({N_EPISODES} episodes) ===")
    print(f"  mean={spt_probs.mean():.4f}, std={spt_probs.std():.4f}, "
          f"min={spt_probs.min():.4f}, max={spt_probs.max():.4f}")
    print(f"  (near-1.0 mean AND near-0 std would mean the DISTRIBUTION itself is "
          f"state-blind, not just the argmax)")

    print(f"\n=== Margin (top1 - top2 probability) across {n} decisions ===")
    print(f"  mean={margins.mean():.4f}, std={margins.std():.4f}, "
          f"min={margins.min():.4f}, max={margins.max():.4f}")

    print(f"\n=== Value estimate V(s) across {n} decisions ===")
    print(f"  mean={values.mean():.2f}, std={values.std():.2f}, "
          f"min={values.min():.2f}, max={values.max():.2f}")
    print(f"  (near-zero std would mean the CRITIC also can't distinguish states)")

    # Correlation: does SPT-probability or the margin vary systematically with
    # congestion, even weakly? (Pearson r, not causal, but a real signal-or-
    # noise check that the previous script's coarser bucket comparison couldn't give.)
    if spt_probs.std() > 1e-9 and congestion.std() > 1e-9:
        r_spt_congestion = np.corrcoef(spt_probs, congestion)[0, 1]
        r_value_congestion = np.corrcoef(values, congestion)[0, 1]
        print(f"\n=== Correlation with congestion ===")
        print(f"  corr(SPT probability, congestion)  = {r_spt_congestion:+.3f}")
        print(f"  corr(value estimate, congestion)    = {r_value_congestion:+.3f}")
    else:
        print(f"\n=== Correlation with congestion ===")
        print(f"  SPT probability has ~zero variance (std={spt_probs.std():.6f}) -- "
              f"correlation is undefined/meaningless, this IS the finding.")

    print("\nDIAGNOSTIC COMPLETE (not a pass/fail test -- see printed values and training-log.md's matching entry)")


if __name__ == "__main__":
    main()
