"""test_compute_theta_bugfix.py - Regression check for the 2026-09-20 (S2W9)
compute_theta() bug fix in Code/env/scheduling_env.py.

Finding: compute_theta() used to read self.capacity[:, :, 0] as its "original
capacity" reference -- the LIVE, mutable capacity array at time-slot 0, not a
fixed baseline. The instant any job occupies time-slot 0 (true for nearly
every real episode, since self.time starts at 0), that reference silently
corrupts for the rest of the episode: every later call computes utilisation
against whatever got consumed at slot 0, not the true original capacity. Net
effect: the hotspot penalty (-lambda3*delta_theta) was systematically WEAKER
than intended for this project's entire history, not absent. See the matching
training-log.md entry (user asked "is our machine utilization part of the
objective correct?") for the full derivation and discovery context.

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see tests/test_bugfixes.py.

Run from the repo root: python -m tests.test_compute_theta_bugfix
"""
import numpy as np

from Code.env.scheduling_env import SchedulingEnv

# ============================================================
# Check 1: theta before anything is scheduled must be 0 (no usage anywhere).
# ============================================================
print("=== Check 1: theta is 0.0 before any job is scheduled ===")
env = SchedulingEnv(
    job_durations=np.array([1, 1]),
    job_resources=np.array([[5.0], [8.0]]),
    job_deadlines=np.array([10, 10]),
    job_weights=np.array([1.0, 1.0]),
    num_machines=1,
    horizon=5,
    machine_capacity=np.array([10.0]),
)
assert env.compute_theta() == 0.0, f"expected 0.0, got {env.compute_theta()}"
print(f"  theta={env.compute_theta()} (correct)")

# ============================================================
# Check 2: the exact repro from the bug's discovery -- a 50%-used slot-0 job
# followed by an 80%-used later job must report 0.8 (the true value), not 0.6
# (the pre-fix value, computed against the corrupted slot-0 baseline).
# ============================================================
print("=== Check 2: theta reflects TRUE utilisation, not a slot-0-anchored one ===")
env.step((0, 0))                # 5/10 = 50% at t=0
env.step_idle()
env.step_idle()
env.step((1, 0))                # 8/10 = 80% at t=3
theta = env.compute_theta()
assert abs(theta - 0.8) < 1e-6, f"expected 0.8 (true utilisation), got {theta} -- bug regressed"
print(f"  theta={theta} == true utilisation 0.8 (pre-fix value would have been 0.6)")

# ============================================================
# Check 3: slot 0's OWN utilisation must be able to read as non-zero. The old
# formula self-cancelled at slot 0 by construction (capacity[:,:,0] minus
# itself is always exactly 0), silently hiding a fully-loaded slot 0. Use a
# fresh env, load slot 0 heavily, and confirm theta picks it up.
# ============================================================
print("=== Check 3: slot 0's own utilisation is no longer hidden ===")
env2 = SchedulingEnv(
    job_durations=np.array([1]),
    job_resources=np.array([[9.0]]),
    job_deadlines=np.array([10]),
    job_weights=np.array([1.0]),
    num_machines=1,
    horizon=5,
    machine_capacity=np.array([10.0]),
)
env2.step((0, 0))               # 9/10 = 90% at t=0, the ONLY thing ever scheduled
theta2 = env2.compute_theta()
assert abs(theta2 - 0.9) < 1e-6, f"expected 0.9 (slot 0's own utilisation), got {theta2}"
print(f"  theta={theta2} == 0.9 (pre-fix value would have been exactly 0.0 for this case)")

print("\nALL COMPUTE_THETA BUGFIX CHECKS PASSED")
