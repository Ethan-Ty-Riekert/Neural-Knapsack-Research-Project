"""test_weighted_tardiness_metric.py - Regression check for the 2026-09-18
weighted-tardiness metric fix in eval_action_space_variant.py.

Finding: SchedulingEnv.tardiness[job] stores RAW, unweighted T_j -- no
weight multiplication anywhere in the env. Every result reported after job
weights were introduced (same day) used raw tardiness, not the actual
objective the reward function optimizes (lambda_2 * sum(w_j*T_j)). This
matters specifically for WSPT/ATC, which deliberately trade a low-weight
job's timeliness for a high-weight job's -- exactly what raw tardiness
can't see. See training-log.md's matching entry for the full story
(the offline picture genuinely changed: Option 3 beats LST under the
correct metric, not just approaches it).

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite).

Run from the repo root: python -m tests.test_weighted_tardiness_metric
"""
import numpy as np

from Code.evaluation.eval_action_space_variant import _weighted_tardiness

# ============================================================
# Check 1: weighted_tardiness == raw tardiness when every weight is 1.0
# (the unweighted default) -- must be a strict generalization, not a
# behaviour change for existing unweighted results.
# ============================================================
print("=== Check 1: weighted_tardiness == raw tardiness when weights are all 1.0 ===")
result = {"tardiness": np.array([0.0, 5.0, 3.0, 0.0]), "job_weights": np.ones(4)}
assert _weighted_tardiness(result) == result["tardiness"].sum()
print(f"  weighted_tardiness={_weighted_tardiness(result)} == raw sum={result['tardiness'].sum()}")

# ============================================================
# Check 2: weighted_tardiness correctly reflects non-uniform weights --
# a late high-weight job should dominate a late low-weight job's contribution.
# ============================================================
print("=== Check 2: weighted_tardiness correctly weights each job's contribution ===")
result = {"tardiness": np.array([10.0, 2.0]), "job_weights": np.array([1.0, 5.0])}
expected = 10.0 * 1.0 + 2.0 * 5.0  # = 20.0
assert _weighted_tardiness(result) == expected, f"expected {expected}, got {_weighted_tardiness(result)}"
print(f"  weighted_tardiness={_weighted_tardiness(result)} == expected {expected} "
      f"(low-weight-but-very-late job and high-weight-but-barely-late job contribute comparably)")

# ============================================================
# Check 3: end-to-end sanity check against the known offline fixed-instance
# numbers found the day this fix was made (seed=0, job_weight_range=(1,6)).
# ============================================================
print("=== Check 3: end-to-end sanity check against known values ===")
from Code.env.env_config import generate_env_config
from Code.evaluation.eval_rl_agent import run_heuristic

cfg = generate_env_config(seed=0, num_jobs=100, num_machines=10, horizon=100, job_weight_range=(1, 6))
h = run_heuristic("EDF", config=cfg)
h["job_weights"] = cfg["job_weights"]
weighted = _weighted_tardiness(h)
assert h["tardiness"].sum() == 16.0, f"expected EDF raw tardiness=16.00, got {h['tardiness'].sum()}"
assert weighted == 46.0, f"expected EDF weighted tardiness=46.00 (found 2026-09-18), got {weighted}"
print(f"  EDF: raw={h['tardiness'].sum()}, weighted={weighted} (matches the 2026-09-18 finding exactly)")

print("\nALL WEIGHTED-TARDINESS-METRIC CHECKS PASSED")
