"""diagnose_observation_informativeness.py - One-off diagnostic script
(2026-09-23, S2W9), direct follow-up to diagnose_policy_confidence.py's
result: fixing the entropy/logit-saturation mechanism (ent_coef=0.1,
training-log.md 2026-09-23 entry) left weighted tardiness essentially
unchanged (798.74 vs. the collapsed policy's 798.46), both far behind ATC's
648.16. That decoupling re-scoped the open "why does online RL underperform
ATC" mystery away from an exploration/optimization problem and toward a
capability/representation-limit hypothesis -- flagged as the natural next
diagnostic but not yet built, until now.

Question this script answers: does the online observation vector
(OnlineGymSchedulingEnv's _get_obs(), passed through
RuleSelectionGymSchedulingEnv unchanged) actually CONTAIN enough
information to distinguish states where ATC's job choice would differ from
SPT's -- i.e. states where "just clear the queue fast" (SPT) and "respect
weight/urgency" (ATC) genuinely disagree, which is exactly where an
SPT-collapsed policy pays for its collapse?

Method: roll out the trained checkpoint deterministically. At every decision
step with >=1 feasible job, compute the job SPT would pick and the job ATC
would pick among the SAME currently-feasible candidate set (reusing
Code/baselines/priority_rules.py's key functions directly -- not
reimplemented), and label the state disagree=1 if they differ, else 0.
Collect the (obs, disagree) pairs, then fit two probes on a held-out test
split to predict disagree from obs alone: a linear probe (logistic
regression) and a small nonlinear probe (one-hidden-layer MLP). Reporting
both -- rather than the linear probe alone -- lets a "weak but nonzero"
linear result (AUC in the 0.6-0.75 range) be disambiguated: if the MLP
scores meaningfully higher, the information is present but not linearly
accessible (representation is fine, a linear readout specifically is not);
if the MLP scores about the same as the linear probe, the ceiling looks
closer to a genuine limit on what these raw features encode, independent
of probe capacity.

Interpretation:
  - If the probe scores well above the majority-class baseline (and above
    chance AUC=0.5), the observation DOES carry the ATC-vs-SPT-relevant
    signal -- the online policy's failure to act on it is an
    optimization/exploration problem, not an information problem.
  - If the probe is indistinguishable from the majority-class baseline, the
    observation genuinely lacks (or heavily obscures) the signal needed --
    a representation/feature-engineering problem, motivating richer
    observation features (e.g. an explicit slack or weighted-urgency
    summary) rather than more training or entropy tuning.

A linear probe is a deliberately weak/conservative test: it establishes a
LOWER BOUND on how much information is linearly decodable, not an upper
bound on what a nonlinear policy network could in principle extract. A
strong linear-probe result is still strong evidence of informativeness; a
weak one is suggestive but not proof the information is entirely absent
(only that it isn't linearly accessible).

Not part of the permanent test suite -- a targeted analysis script, same
convention as diagnose_rule_choice_congestion.py and
diagnose_policy_confidence.py.

Run from the repo root:
    python -m Code.evaluation.diagnose_observation_informativeness
"""
import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from Code.baselines.priority_rules import atc_key, spt_key, _atc_mean_p
from Code.training.train_action_space_variant import make_online_base_gym_env, mask_fn
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv
from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING
from Code.utils.paths import MODELS_DIR

DEFAULT_CHECKPOINT_TAG = "online_lognormal_rho075_dense_weighted_diagnostics"
N_EPISODES = 20
ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST = 9, 100, 1300, "lognormal"
JOB_WEIGHT_RANGE = (1, 6)
TEST_FRACTION = 0.25
RANDOM_STATE = 0


def spt_vs_atc_disagreement(base_env, unique_jobs):
    """Among the SAME feasible-job candidate set a real rollout step would
    choose from, does SPT's argmin differ from ATC's argmin? Reuses the
    registry.py::choose() tie-break convention (secondary sort on job index)
    so this exactly matches what a real SPT/ATC rollout would pick, not an
    approximation of it."""
    mean_p = _atc_mean_p(base_env)
    spt_job = min(unique_jobs, key=lambda j: (spt_key(base_env, j), j))
    atc_job = min(unique_jobs, key=lambda j: (atc_key(base_env, j, mean_p=mean_p), j))
    return int(spt_job != atc_job)


def run_episode_with_logging(model, env):
    obs, info = env.reset()
    done = truncated = False
    rule_env = env.env  # ActionMasker.env == RuleSelectionGymSchedulingEnv
    base_env = rule_env.env  # RuleSelectionGymSchedulingEnv.env == raw OnlineSchedulingEnv

    records = []
    while not (done or truncated):
        job_actions = rule_env._job_actions()
        if job_actions:
            unique_jobs = {rule_env._decode(a)[0] for a in job_actions}
            label = spt_vs_atc_disagreement(base_env, unique_jobs)
            records.append({"obs": np.array(obs, dtype=np.float64), "label": label})

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

    template_full = make_online_base_gym_env(
        ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST, seed=RANDOM_INSTANCE_SEED_CEILING,
        use_resampler=False, job_weight_range=JOB_WEIGHT_RANGE,
    )
    template_env = ActionMasker(RuleSelectionGymSchedulingEnv(template_full), mask_fn)
    model = MaskablePPO.load(str(checkpoint), env=template_env)

    all_records = []
    for i in range(N_EPISODES):
        seed = RANDOM_INSTANCE_SEED_CEILING + i
        full = make_online_base_gym_env(
            ARRIVAL_RATE, HORIZON, MAX_JOBS, DIST, seed=seed,
            use_resampler=False, job_weight_range=JOB_WEIGHT_RANGE,
        )
        env = ActionMasker(RuleSelectionGymSchedulingEnv(full), mask_fn)
        all_records.extend(run_episode_with_logging(model, env))

    n = len(all_records)
    X = np.stack([r["obs"] for r in all_records])
    y = np.array([r["label"] for r in all_records])
    disagree_rate = y.mean()
    print(f"\n=== Dataset: {n} decision states across {N_EPISODES} episodes ===")
    print(f"  SPT-vs-ATC disagreement rate: {disagree_rate:.3f} "
          f"(majority-class baseline accuracy = {max(disagree_rate, 1 - disagree_rate):.3f})")

    if disagree_rate < 1e-6 or disagree_rate > 1 - 1e-6:
        print("\n  Degenerate label distribution (SPT and ATC never/always disagree on this "
              "rollout set) -- cannot fit or evaluate a probe meaningfully. Stopping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE, stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    probe = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    probe.fit(X_train_s, y_train)

    y_pred = probe.predict(X_test_s)
    y_prob = probe.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n=== Linear probe: predict SPT-vs-ATC disagreement from raw observation ===")
    print(f"  test raw accuracy:        {acc:.3f}  (majority-class baseline: {baseline_acc:.3f} -- "
          f"NOTE: class_weight='balanced' trades raw accuracy for balanced recall, so a raw "
          f"accuracy below this baseline is an expected side-effect of that choice, not itself "
          f"evidence of no signal -- ROC-AUC below is the metric that's insensitive to it)")
    print(f"  test balanced accuracy:  {bal_acc:.3f}  (0.500 = chance under class imbalance)")
    print(f"  test ROC-AUC:             {auc:.3f}  (0.5 = chance, 1.0 = perfect, threshold-independent)")

    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=RANDOM_STATE,
                         early_stopping=True, n_iter_no_change=15)
    mlp.fit(X_train_s, y_train)
    mlp_prob = mlp.predict_proba(X_test_s)[:, 1]
    mlp_auc = roc_auc_score(y_test, mlp_prob)
    print(f"\n=== Nonlinear probe (1-hidden-layer MLP, same train/test split) ===")
    print(f"  test ROC-AUC:             {mlp_auc:.3f}")

    gap = mlp_auc - auc
    if auc >= 0.75:
        verdict = ("Linear-probe AUC is clearly above chance: the observation linearly encodes "
                    "real ATC-vs-SPT-relevant signal. Points at an optimization/exploration "
                    "gap, not a missing-information one.")
    elif auc >= 0.60 and gap < 0.05:
        verdict = (f"Linear AUC ({auc:.3f}) is modestly above chance, and the nonlinear probe "
                    f"scores about the same (MLP AUC={mlp_auc:.3f}, gap={gap:+.3f}) -- the "
                    f"weak-but-real signal is NOT a linear-readout artifact, it looks like a "
                    f"genuine ceiling on what these raw features encode about ATC-vs-SPT "
                    f"disagreement. Suggests the observation is only partially informative for "
                    f"this distinction -- motivates richer features (e.g. an explicit "
                    f"slack/weighted-urgency summary term) rather than more training.")
    elif auc >= 0.60 and gap >= 0.05:
        verdict = (f"Linear AUC ({auc:.3f}) is modestly above chance, and the nonlinear probe "
                    f"does meaningfully better (MLP AUC={mlp_auc:.3f}, gap={gap:+.3f}) -- the "
                    f"information is present in the observation but not linearly accessible. "
                    f"Since the policy network itself is nonlinear, this leans toward an "
                    f"optimization/exploration gap rather than a hard representation limit.")
    else:
        verdict = (f"Linear AUC ({auc:.3f}) is close to chance; nonlinear MLP AUC ({mlp_auc:.3f}) "
                    f"{'is also near chance' if mlp_auc < 0.60 else 'recovers some signal the linear probe missed'}. "
                    f"{'Consistent with (not proof of) a genuine representation limit -- motivates richer observation features over further training or entropy tuning.' if mlp_auc < 0.60 else 'Since even a small nonlinear probe finds signal the linear one cannot, this leans toward the information being present but hard to extract -- an optimization/exploration angle, not purely missing information.'}")
    print(f"\n  Interpretation: {verdict}")

    print("\nDIAGNOSTIC COMPLETE (not a pass/fail test -- see printed values and training-log.md's matching entry)")


if __name__ == "__main__":
    main()
