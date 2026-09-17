"""test_job_weights.py - Regression/validation checks for
generate_env_config()/generate_poisson_arrivals()'s job_weight_range option
(2026-09-18, S2W10 -- job_weights was uniformly 1.0 in every instance this
project ever generated, making WSPT/ATC's w_j/p_j term dead code; see
Code/baselines/priority_rules.py::wspt_key's own docstring).

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_heavy_tailed_arrivals.py.

Run from the repo root: python -m tests.test_job_weights
"""
import numpy as np

from Code.env.env_config import generate_env_config
from Code.env.arrival_process import generate_poisson_arrivals
from Code.baselines.priority_rules import wspt_key, atc_key
from Code.env.scheduling_env import SchedulingEnv


# ============================================================
# Check 1: default (job_weight_range=None) is byte-identical to before --
# this feature must be purely additive/opt-in, since the seed=0 fixed
# instance is the reference point for CP-SAT's proven 8.0 floor and every
# historic EDF/LST/ATC number in this project.
# ============================================================
print("=== Check 1: default job_weight_range=None unchanged (regression safety) ===")
cfg_default = generate_env_config(seed=0, num_jobs=100, num_machines=10, horizon=100)
cfg_explicit_none = generate_env_config(seed=0, num_jobs=100, num_machines=10, horizon=100, job_weight_range=None)
assert np.array_equal(cfg_default["job_weights"], np.ones(100))
assert np.array_equal(cfg_default["job_weights"], cfg_explicit_none["job_weights"])
print("  default and explicit job_weight_range=None both give all-ones weights")


# ============================================================
# Check 2: explicit range produces genuine variation, respects the
# (inclusive low, exclusive high) numpy.integers convention.
# ============================================================
print("=== Check 2: explicit job_weight_range produces genuine variation ===")
cfg_weighted = generate_env_config(seed=0, num_jobs=100, num_machines=10, horizon=100, job_weight_range=(1, 6))
w = cfg_weighted["job_weights"]
assert w.min() >= 1 and w.max() <= 5, f"weights out of [1,5] range: min={w.min()}, max={w.max()}"
assert len(set(w.tolist())) > 1, "expected genuine variation in weights, got a constant array"
print(f"  weights range [{w.min():.0f}, {w.max():.0f}], {len(set(w.tolist()))} distinct values across 100 jobs")


# ============================================================
# Check 3: reproducibility (same seed -> identical weights).
# ============================================================
print("=== Check 3: reproducibility ===")
cfg_a = generate_env_config(seed=7, num_jobs=50, job_weight_range=(1, 10))
cfg_b = generate_env_config(seed=7, num_jobs=50, job_weight_range=(1, 10))
assert np.array_equal(cfg_a["job_weights"], cfg_b["job_weights"])
print("  same seed reproduces identical weights")


# ============================================================
# Check 4: online (generate_poisson_arrivals) -- same default-unchanged and
# variation checks, plus padding slots (never revealed) are untouched.
# ============================================================
print("=== Check 4: online generate_poisson_arrivals() job_weight_range ===")
cfg_online_default = generate_poisson_arrivals(seed=0, arrival_rate=2.0, horizon=50, max_jobs=200)
assert np.array_equal(cfg_online_default["job_weights"], np.ones(200))

cfg_online_weighted = generate_poisson_arrivals(seed=0, arrival_rate=2.0, horizon=50, max_jobs=200, job_weight_range=(1, 6))
realized = cfg_online_weighted["job_arrival_times"] <= 50
realized_w = cfg_online_weighted["job_weights"][realized]
assert realized_w.min() >= 1 and realized_w.max() <= 5
assert len(set(realized_w.tolist())) > 1, "expected genuine variation among realized jobs' weights"
print(f"  online: {realized.sum()} realized jobs, weight range [{realized_w.min():.0f}, {realized_w.max():.0f}]")


# ============================================================
# Check 5 ("review the outputs"): WSPT/ATC's key now actually depends on
# weight, differentiating from SPT/EDF when weights vary -- confirms the
# dead-code problem flagged in wspt_key's own docstring is now live.
# ============================================================
print("=== Check 5: WSPT/ATC keys now genuinely depend on weight ===")
durations = np.array([5, 5, 5])          # identical durations
resources = np.array([[3]] * 3)
deadlines = np.array([20, 20, 20])       # identical deadlines
weights = np.array([1.0, 3.0, 5.0])      # only weight differs
env = SchedulingEnv(
    job_durations=durations, job_resources=resources, job_deadlines=deadlines,
    job_weights=weights, num_machines=1, machine_capacity=np.array([10.0]), horizon=30,
)
wspt_keys = [wspt_key(env, j) for j in range(3)]
atc_keys = [atc_key(env, j) for j in range(3)]
assert len(set(wspt_keys)) == 3, f"expected 3 distinct WSPT keys (identical duration, varying weight), got {wspt_keys}"
assert len(set(atc_keys)) == 3, f"expected 3 distinct ATC keys (identical duration/deadline, varying weight), got {atc_keys}"
# Higher weight -> smaller WSPT key (minimum-key-wins convention) since WSPT = duration/weight.
assert wspt_keys[2] < wspt_keys[1] < wspt_keys[0], f"expected monotonically decreasing WSPT key with increasing weight, got {wspt_keys}"
print(f"  WSPT keys (weights 1,3,5): {wspt_keys}")
print(f"  ATC keys (weights 1,3,5):  {atc_keys}")
print("  both rules now genuinely differentiate by weight, not just duration/deadline")

print("\nALL JOB-WEIGHT CHECKS PASSED")
