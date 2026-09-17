"""test_dense_tardiness_reward.py - Regression checks for reward_mode=
"dense_tardiness" (Code/env/scheduling_env.py): verifies the per-tick
decomposition is algebraically exact against the legacy lump-sum charge, that
the flat completion bonuses are actually gone, and that legacy mode is
completely unaffected by the new code paths.

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_bugfixes.py.

Run from the repo root: python -m tests.test_dense_tardiness_reward
"""
import numpy as np

from Code.env.scheduling_env import SchedulingEnv


def make_env(reward_mode):
    # 3 jobs, 1 machine, capacity EXACTLY matching one job's demand (3) so no
    # two jobs can ever run concurrently -- forces strict sequential
    # scheduling, keeping the timeline easy to reason about by hand.
    return SchedulingEnv(
        job_durations=np.array([2, 2, 2]),
        job_resources=np.array([[3], [3], [3]]),
        job_deadlines=np.array([1, 20, 3]),
        job_weights=np.array([1.0, 2.0, 1.5]),
        num_machines=1,
        machine_capacity=np.array([3.0]),
        horizon=20,
        lambda_1=1.0,
        lambda_2=2.0,
        lambda_3=1.0,
        invalid_penalty=5.0,
        idle_penalty=0.5,
        reward_mode=reward_mode,
    )


def run_scripted_episode(env):
    """Capacity forces strict sequencing, and the decision clock (1 tick per
    action, regardless of job duration) means an explicit idle is needed to
    let each job's duration actually elapse before the machine frees up:
    t=0: schedule job 0 (occupies ticks 0,1, finishes t=2, deadline=1 -> T_0=1, late)
    t=1: idle (waiting for job 0 to finish occupying the machine)
    t=2: schedule job 1 (occupies ticks 2,3, finishes t=4, deadline=20 -> T_1=0, on time)
    t=3: idle (waiting for job 1)
    t=4: schedule job 2 (occupies ticks 4,5, finishes t=6, deadline=3 -> T_2=3, late)
    -- this also empties remaining_jobs, so the episode ends here (offline
    early-finish semantics), before ever reaching the horizon.
    """
    total_reward = 0.0
    script = [("step", (0, 0)), ("idle", None), ("step", (1, 0)), ("idle", None), ("step", (2, 0))]
    done = False
    for kind, action in script:
        assert not done, "episode ended earlier than the script expects"
        if kind == "step":
            _, r, done = env.step(action)
        else:
            _, r, done = env.step_idle()
        total_reward += r
    assert done, "episode should be done once all 3 jobs are scheduled (offline early-finish semantics)"
    return total_reward


print("=== Setup: 3 jobs, strictly sequential (capacity forces no overlap) ===")
env_legacy = make_env("legacy")
reward_legacy = run_scripted_episode(env_legacy)
tardiness_legacy = env_legacy.tardiness.copy()
cost_legacy = env_legacy.episode_cost

env_dense = make_env("dense_tardiness")
reward_dense = run_scripted_episode(env_dense)
tardiness_dense = env_dense.tardiness.copy()
cost_dense = env_dense.episode_cost

print(f"  tardiness (legacy): {tardiness_legacy}, (dense): {tardiness_dense}")
assert np.allclose(tardiness_legacy, tardiness_dense), "tardiness computation itself must not change with reward_mode"
assert np.array_equal(tardiness_legacy, [1, 0, 3]), f"scripted sequential schedule should give T=[1,0,3], got {tardiness_legacy}"

# ============================================================
# Check 1: the dense per-tick decomposition is algebraically EXACT --
# episode_cost (the un-lambda2-weighted running total) must match exactly.
# ============================================================
print("=== Check 1: dense decomposition reproduces legacy's episode_cost exactly ===")
weights = np.array([1.0, 2.0, 1.5])
expected_cost = float(np.sum(weights * tardiness_legacy / env_legacy.horizon))
print(f"  expected (w_j*T_j/horizon summed): {expected_cost}")
print(f"  legacy episode_cost: {cost_legacy}, dense episode_cost: {cost_dense}")
assert abs(cost_legacy - expected_cost) < 1e-9, "legacy episode_cost should equal the direct formula"
assert abs(cost_dense - expected_cost) < 1e-9, (
    f"dense_tardiness episode_cost ({cost_dense}) does not match the exact expected total "
    f"({expected_cost}) -- the per-tick decomposition is not algebraically exact"
)
print("  PASS: dense per-tick accrual reproduces the exact same total as the legacy lump sum")

# ============================================================
# Check 2: flat completion bonuses are present in legacy, absent in dense --
# the ENTIRE reward gap between the two runs must equal exactly the bonus
# total (3 valid placements * 3.0, plus 50 for finishing all 3 jobs), since
# every other term (activation, tardiness magnitude via Check 1, hotspot,
# idling) is identical between the two runs given the identical action sequence.
# ============================================================
print("=== Check 2: flat +3.0/+50 bonuses removed under dense_tardiness ===")
expected_bonus_total = 3.0 * 3 + 50
diff = reward_legacy - reward_dense
print(f"  reward_legacy={reward_legacy:.4f}, reward_dense={reward_dense:.4f}, diff={diff:.4f}, expected_bonus_total={expected_bonus_total}")
assert abs(diff - expected_bonus_total) < 1e-9, (
    f"reward difference ({diff}) does not equal the expected flat-bonus total ({expected_bonus_total}) -- "
    "dense_tardiness may still be granting some form of the removed bonuses, or legacy lost one"
)
print("  PASS: the entire reward gap between modes is exactly the removed flat bonuses")

# ============================================================
# Check 3: legacy mode's own numeric behaviour is completely unchanged
# (regression guard -- adding reward_mode must not perturb the default path).
# Isolated to a single fresh action so the hand derivation stays simple.
# ============================================================
print("=== Check 3: legacy mode unchanged (spot check against hand-derived reward) ===")
# lambda_3=0 here specifically to sidestep compute_theta()'s own pre-existing
# behaviour (unrelated to this reward_mode change, out of scope to touch) of
# comparing utilisation against capacity[:,:,0] -- which a job placed AT t=0
# itself already mutates before compute_theta() reads it, making the hotspot
# term's value for a first-action-at-t=0 scenario non-obvious by hand. Zeroing
# lambda_3 isolates exactly the terms this reward_mode change actually touches
# (activation, tardiness, the flat bonus) without that unrelated confound.
env_check = SchedulingEnv(
    job_durations=np.array([2, 2, 2]),
    job_resources=np.array([[3], [3], [3]]),
    job_deadlines=np.array([1, 20, 3]),
    job_weights=np.array([1.0, 2.0, 1.5]),
    num_machines=1,
    machine_capacity=np.array([3.0]),
    horizon=20,
    lambda_1=1.0,
    lambda_2=2.0,
    lambda_3=0.0,
    invalid_penalty=5.0,
    idle_penalty=0.5,
    reward_mode="legacy",
)
_, r0, _ = env_check.step((0, 0))
# job 0: duration 2, scheduled at t=0 -> tardiness = max(0, 0+2-1) = 1.
# lambda_2=2.0, weight=1.0, horizon=20 -> tardiness_cost = 1*(1/20) = 0.05
# machine activation: lambda_1=1.0 (machine 0 was inactive) -> -1.0
# hotspot: lambda_3=0 -> 0 regardless of delta_theta
# total: -1.0 (activation) - 2.0*0.05 (tardiness) - 0 (hotspot) + 3.0 (bonus) = 1.9
expected_r0 = -1.0 - 2.0 * 0.05 - 0.0 + 3.0
print(f"  first-step reward: got {r0:.4f}, hand-derived expected {expected_r0:.4f}")
assert abs(r0 - expected_r0) < 1e-9, "legacy mode's first-step reward no longer matches hand derivation -- regression"
print("  PASS: legacy mode reproduces its exact pre-existing formula")

print("\nALL DENSE-TARDINESS-REWARD CHECKS PASSED")
