"""test_efficiency_fixes.py - Regression checks for the 2026-09-21 (S2W9)
free/zero-risk performance fixes from Future/research/2026-09-20-
optimisation-and-efficiency-critique.md Sections 1.2-1.5 (Section 1.1,
get_state(), already has its own tests/test_compute_theta_bugfix.py-style
coverage via the earlier get_state() fix commit). Every fix here was
verified against the pre-fix implementation via a direct old-vs-new
equivalence script before being committed (not just reasoned about) --
these checks lock in the resulting invariants going forward, since the
"old" code itself won't stay around to diff against in a future session.

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_bugfixes.py.

Run from the repo root: python -m tests.test_efficiency_fixes
"""
import numpy as np

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.baselines.priority_rules import atc_key, atc_priority, _atc_mean_p


def make_env(num_jobs=5, num_machines=2, horizon=20, capacity=10.0):
    durations = np.array([2, 3, 1, 4, 2])[:num_jobs]
    resources = np.array([[3], [4], [2], [5], [3]])[:num_jobs]
    deadlines = np.array([8, 9, 6, 15, 10])[:num_jobs]
    weights = np.array([1.0, 2.0, 3.0, 1.0, 4.0])[:num_jobs]
    base = SchedulingEnv(
        job_durations=durations, job_resources=resources, job_deadlines=deadlines,
        job_weights=weights, num_machines=num_machines,
        machine_capacity=np.array([capacity]), horizon=horizon,
    )
    return base


# ============================================================
# Check 1 (Section 1.2): __init__/reset() vectorized capacity broadcast
# matches the pristine machine_capacity at every (machine, time) cell.
# ============================================================
print("=== Check 1: vectorized capacity broadcast is correct at construction and reset ===")
base = make_env(num_machines=3, capacity=7.0)
assert np.allclose(base.capacity, 7.0), "capacity should be uniformly 7.0 everywhere before any placement"
assert base.capacity.shape == (3, 1, 20)
base.capacity[0, 0, 5] = 2.0  # simulate a placement's effect
base.reset()
assert np.allclose(base.capacity, 7.0), "reset() must restore capacity to the pristine machine_capacity everywhere"
print("  capacity correctly broadcast at construction and restored to pristine on reset()")

# ============================================================
# Check 2 (Section 1.2): step()'s vectorized slice-subtraction touches
# EXACTLY the occupied tick range [time, time+duration) and no other.
# ============================================================
print("=== Check 2: vectorized resource subtraction touches exactly the occupied ticks ===")
base2 = make_env(num_machines=1, capacity=10.0)
base2.step((0, 0))  # job 0: duration=2, resource=3, placed at t=0
assert np.allclose(base2.capacity[0, 0, 0:2], 7.0), "occupied ticks [0,2) should be reduced by 3"
assert np.allclose(base2.capacity[0, 0, 2:], 10.0), "ticks from 2 onward must be untouched"
print("  capacity reduced exactly on ticks [0,2), untouched elsewhere")

# ============================================================
# Check 3 (Section 1.3/1.4): full GymSchedulingEnv obs/mask stay internally
# consistent after the vectorization -- obs's capacity block and the mask's
# feasibility agree with is_feasible() directly, on a real multi-step
# trajectory (not just at reset).
# ============================================================
print("=== Check 3: vectorized _get_obs()/get_action_mask() stay internally consistent ===")
base3 = make_env(num_jobs=5, num_machines=2, capacity=10.0)
env3 = GymSchedulingEnv(base3, max_jobs=5)
obs, info = env3.reset()
assert np.isfinite(obs).all(), "obs must be finite"
rng = np.random.RandomState(0)
for _ in range(15):
    mask = env3.get_action_mask()
    legal = np.where(mask)[0]
    action = int(rng.choice(legal))
    obs, reward, term, trunc, info = env3.step(action)
    assert np.isfinite(obs).all(), "obs must stay finite through a real trajectory"
    # Cross-check every mask entry directly against ground truth -- the
    # vectorized mask must agree with is_feasible() AND remaining_jobs
    # membership (is_feasible() alone says nothing about whether a job has
    # already been scheduled -- get_action_mask() only ever considers jobs
    # still in remaining_jobs, by construction).
    mask = env3.get_action_mask()
    for a in range(env3.max_jobs * env3.num_machines):
        j, m = a // env3.num_machines, a % env3.num_machines
        expected = 1 if (j in base3.remaining_jobs and base3.is_feasible(j, m, base3.time)) else 0
        assert mask[a] == expected, f"mask[{a}] (job={j}, machine={m}) disagrees with ground truth"
    if term or trunc:
        break
print("  obs stayed finite and mask matched is_feasible() exactly across a real trajectory")

# ============================================================
# Check 4 (Section 1.5): ATC's precomputed-mean_p path gives IDENTICAL
# results to the default (recompute-per-call) path, for every job.
# ============================================================
print("=== Check 4: ATC precomputed mean_p matches the default recompute-per-call path ===")
base4 = make_env(num_jobs=5)
base4.time = 3
mean_p = _atc_mean_p(base4)
expected_mean_p = sum(base4.job_durations) / len(base4.job_durations)
assert abs(mean_p - expected_mean_p) < 1e-9, f"expected mean_p={expected_mean_p}, got {mean_p}"
for j in range(5):
    default_val = atc_key(base4, j)
    precomputed_val = atc_key(base4, j, mean_p=mean_p)
    assert abs(default_val - precomputed_val) < 1e-9, (
        f"job {j}: default atc_key={default_val} != precomputed-mean_p atc_key={precomputed_val}"
    )
    default_p = atc_priority(base4, j)
    precomputed_p = atc_priority(base4, j, mean_p=mean_p)
    assert abs(default_p - precomputed_p) < 1e-9
print(f"  _atc_mean_p={mean_p:.4f} matches manual computation; all 5 jobs' atc_key/atc_priority "
      f"identical with vs. without precomputed mean_p")

# ============================================================
# Check 5 (Section 1.5): the online-adapted mean_p (restricted to
# revealed_jobs) still works correctly with the extracted helper.
# ============================================================
print("=== Check 5: _atc_mean_p respects revealed_jobs when present (online-adapted) ===")


class _FakeOnlineEnv:
    def __init__(self, job_durations, revealed_jobs):
        self.job_durations = job_durations
        self.revealed_jobs = revealed_jobs


fake_env = _FakeOnlineEnv(job_durations=np.array([2, 4, 6, 8]), revealed_jobs={0, 1})
mean_p_online = _atc_mean_p(fake_env)
assert abs(mean_p_online - 3.0) < 1e-9, f"expected mean_p=3.0 (mean of [2,4] only), got {mean_p_online}"
print(f"  mean_p={mean_p_online} correctly restricted to revealed_jobs {{0,1}} (durations [2,4]), not all 4 jobs")

print("\nALL EFFICIENCY-FIX CHECKS PASSED")
