"""test_online_env.py - Regression checks for the online (dynamic-arrival) case:
OnlineSchedulingEnv, OnlineGymSchedulingEnv, and generate_poisson_arrivals().

Follows this project's existing tests/ convention (runnable script with assert
invariants, not a pytest suite) -- see test_bugfixes.py.

Run from the repo root: python -m tests.test_online_env
"""
import numpy as np

from Code.env.arrival_process import generate_poisson_arrivals
from Code.env.online_scheduling_env import OnlineSchedulingEnv
from Code.env.online_gym_wrapper import OnlineGymSchedulingEnv


def make_env(arrival_times, num_jobs=4, num_machines=2, horizon=10):
    durations = np.array([2] * num_jobs)
    resources = np.array([[3]] * num_jobs)
    deadlines = np.array([arrival_times[j] + 5 for j in range(num_jobs)])
    weights = np.array([1.0] * num_jobs)
    base = OnlineSchedulingEnv(
        job_durations=durations,
        job_resources=resources,
        job_deadlines=deadlines,
        job_weights=weights,
        num_machines=num_machines,
        machine_capacity=np.array([10.0]),
        horizon=horizon,
        job_arrival_times=np.array(arrival_times),
    )
    return OnlineGymSchedulingEnv(base, max_jobs=num_jobs)


# ============================================================
# Check 1: action mask flips from infeasible to feasible exactly at arrival
# ============================================================
print("=== Check 1: action mask reflects arrival tick exactly ===")
env = make_env(arrival_times=[0, 3, 3, 100])
obs, info = env.reset()

job1_actions = list(range(1 * env.num_machines, 1 * env.num_machines + env.num_machines))
for t in range(3):
    mask = env.get_action_mask()
    assert not any(mask[a] for a in job1_actions), (
        f"job 1 (arrives at t=3) shows feasible at t={env.env.time}, before arrival"
    )
    obs, reward, term, trunc, info = env.step(env.max_jobs * env.num_machines)  # idle

mask = env.get_action_mask()
assert any(mask[a] for a in job1_actions), "job 1 should be feasible immediately at its arrival tick (t=3)"
print("  job 1 correctly masked-infeasible before t=3, feasible at t=3")


# ============================================================
# Check 2: observation for a not-yet-arrived job is all-zero (the leak-bug guard)
# ============================================================
print("=== Check 2: not-yet-arrived job's observation slot is all-zero ===")
env2 = make_env(arrival_times=[0, 7, 100, 100])
obs, info = env2.reset()
job_block_start = 1 + env2.num_machines * env2.num_resources
per_job = env2.num_resources + 4
job1_slot = obs[job_block_start + 1 * per_job: job_block_start + 2 * per_job]
assert np.allclose(job1_slot[:-1], 0.0), f"job 1 (not yet arrived) leaked real features: {job1_slot}"
assert job1_slot[-1] == 1.0, "not-yet-arrived job should read as 'scheduled' (padding convention)"
print(f"  job 1's pre-arrival observation slot is all-zero + scheduled-flag=1: {job1_slot}")


# ============================================================
# Check 3: a scheduled job never reappears in remaining_jobs
# ============================================================
print("=== Check 3: scheduled job never reappears in remaining_jobs ===")
env3 = make_env(arrival_times=[0, 0, 0, 0])
obs, info = env3.reset()
assert 0 in env3.env.remaining_jobs
action_id = 0 * env3.num_machines + 0
obs, reward, term, trunc, info = env3.step(action_id)
assert 0 not in env3.env.remaining_jobs, "scheduled job 0 still in remaining_jobs"
for t in range(5):
    obs, reward, term, trunc, info = env3.step(env3.max_jobs * env3.num_machines)
    assert 0 not in env3.env.remaining_jobs, "job 0 reappeared in remaining_jobs after further ticks"
print("  job 0 stays out of remaining_jobs across further ticks")


# ============================================================
# Check 4: a padding job (arrival_time = horizon+1) never appears / never
# contributes to the terminal penalty
# ============================================================
print("=== Check 4: padding job excluded from remaining_jobs and terminal penalty ===")
horizon = 6
env4 = make_env(arrival_times=[0, horizon + 1, horizon + 1, horizon + 1], horizon=horizon)
obs, info = env4.reset()
done = False
steps = 0
while not done and steps < 50:
    obs, reward, term, trunc, info = env4.step(env4.max_jobs * env4.num_machines)  # idle throughout
    done = term or trunc
    steps += 1
assert 1 not in env4.env.revealed_jobs and 2 not in env4.env.revealed_jobs and 3 not in env4.env.revealed_jobs, (
    "a padding job (arrival_time = horizon+1) was incorrectly revealed"
)
# Only job 0 (arrived at t=0, never scheduled here) should have been charged;
# padding jobs must not appear in remaining_jobs at all, so _finalize_unscheduled_job_cost
# (which iterates remaining_jobs) can never charge them.
assert env4.env.remaining_jobs == {0}, f"expected only job 0 unscheduled at end, got {env4.env.remaining_jobs}"
print(f"  episode ended after {steps} idle ticks; padding jobs never revealed; only job 0 unscheduled")


# ============================================================
# Check 5 & 6: generate_poisson_arrivals -- fixed array size, reproducibility,
# inter-arrival-mean sanity
# ============================================================
print("=== Check 5/6: generate_poisson_arrivals ===")
max_jobs = 200
horizon = 100
rate = 1.0  # well below max_jobs so padding is actually exercised, not just capped away
cfg_a = generate_poisson_arrivals(seed=0, arrival_rate=rate, horizon=horizon, max_jobs=max_jobs)
cfg_b = generate_poisson_arrivals(seed=0, arrival_rate=rate, horizon=horizon, max_jobs=max_jobs)

assert len(cfg_a["job_durations"]) == max_jobs, "job_durations length must equal max_jobs regardless of realized arrivals"
assert len(cfg_a["job_arrival_times"]) == max_jobs
assert np.array_equal(cfg_a["job_arrival_times"], cfg_b["job_arrival_times"]), "same seed must reproduce identical arrivals"
assert np.array_equal(cfg_a["job_durations"], cfg_b["job_durations"]), "same seed must reproduce identical job content"

realized = cfg_a["job_arrival_times"][cfg_a["job_arrival_times"] <= horizon]
expected_mean = rate * horizon
assert abs(len(realized) - expected_mean) < 5 * np.sqrt(expected_mean), (
    f"realized arrival count {len(realized)} far from expected mean {expected_mean} "
    "(Poisson std should keep this well within a handful of standard deviations)"
)
padding_count = int((cfg_a["job_arrival_times"] > horizon).sum())
print(f"  seed=0: {len(realized)} realized arrivals (expected ~{expected_mean:.0f}), "
      f"{padding_count} padding slots, seed reproducibility OK")


# ============================================================
# Regression guard: multiple valid same-tick placements must NOT be
# misclassified as invalid actions (the base GymSchedulingEnv.step()'s
# "time unchanged -> invalid" proxy breaks under Option 2's relaxed
# tick-advance rule -- see online_gym_wrapper.py's module docstring).
# ============================================================
print("=== Regression guard: same-tick multi-placement not flagged invalid ===")
env5 = make_env(arrival_times=[0, 0, 0, 0], num_jobs=4, num_machines=4, horizon=20)
obs, info = env5.reset()
for j in range(4):
    action_id = j * env5.num_machines + j
    obs, reward, term, trunc, info = env5.step(action_id)
    assert reward > 0, f"job {j}'s valid same-tick placement got a non-positive reward ({reward}) -- looks misclassified as invalid"
assert env5._invalid_action_count == 0, (
    f"_invalid_action_count={env5._invalid_action_count} after 4 valid same-tick placements -- "
    "the invalid-action proxy is misclassifying valid actions"
)
assert env5.env.time == 0, "time should not have advanced across 4 placement-only steps"
print("  4 valid same-tick placements: _invalid_action_count stayed 0, time stayed at 0")

print("\nALL ONLINE-ENV CHECKS PASSED")
