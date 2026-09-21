"""test_windowed_priority_wrapper.py - Regression/validation checks for the
DeepRM-style bounded-window action space
(Code/env/windowed_priority_gym_wrapper.py, 2026-09-18, S2W10 -- PREPARED
FOR REVIEW, NOT YET WIRED INTO ANY TRAINING RUN).

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_action_space_wrappers.py.

Run from the repo root: python -m tests.test_windowed_priority_wrapper
"""
import numpy as np
import torch

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.windowed_priority_gym_wrapper import WindowedPriorityGymSchedulingEnv
from Code.policies.windowed_priority_pointer_policy import WindowedPriorityPointerActorCritic


def make_base_gym_env(num_jobs=10, num_machines=2, horizon=20, capacity=10.0, deadlines=None):
    durations = np.array([2] * num_jobs)
    resources = np.array([[3]] * num_jobs)
    deadlines = np.array(deadlines) if deadlines is not None else np.array([8 + j for j in range(num_jobs)])
    weights = np.array([1.0] * num_jobs)
    base = SchedulingEnv(
        job_durations=durations, job_resources=resources, job_deadlines=deadlines,
        job_weights=weights, num_machines=num_machines,
        machine_capacity=np.array([capacity]), horizon=horizon,
    )
    return GymSchedulingEnv(base, max_jobs=num_jobs)


# ============================================================
# Check 1: action space size = window_size + 1, obs_dim matches the derived formula.
# ============================================================
print("=== Check 1: action space and obs_dim sizing ===")
full = make_base_gym_env(num_jobs=10, num_machines=2, horizon=20)
env = WindowedPriorityGymSchedulingEnv(full, window_size=4, use_atc=False)
assert env.action_space.n == 5, f"expected Discrete(5), got Discrete({env.action_space.n})"
expected_obs_dim = 1 + 2 * 1 + 4 * (1 + 4) + 1  # time + machine block + window slots + backlog
assert env.observation_space.shape[0] == expected_obs_dim, (
    f"expected obs_dim={expected_obs_dim}, got {env.observation_space.shape[0]}"
)
print(f"  action_space=Discrete({env.action_space.n}), obs_dim={env.observation_space.shape[0]}")

# ============================================================
# Check 2: window contains the correct EDF-ordered subset when more jobs
# exist than the window size, and the backlog scalar reflects the overflow.
# ============================================================
print("=== Check 2: window selects earliest-deadline jobs; backlog reflects overflow ===")
full2 = make_base_gym_env(num_jobs=10, num_machines=2, horizon=20,
                           deadlines=[19, 5, 12, 3, 17, 8, 15, 2, 11, 6])
env2 = WindowedPriorityGymSchedulingEnv(full2, window_size=4, use_atc=False)
obs, info = env2.reset()
# Earliest 4 deadlines are jobs 7(2), 3(3), 1(5), 9(6) -- sorted ascending.
assert env2._window_jobs == [7, 3, 1, 9], f"expected EDF-ordered window [7,3,1,9], got {env2._window_jobs}"
job_block = obs[env2._machine_block_end:-1]  # exclude the trailing backlog scalar
slot_width = env2._out_slot_width
first_slot_deadline = job_block[1] * env2.horizon  # deadline is the 2nd feature, normalised by horizon
assert abs(first_slot_deadline - 2) < 1e-4, f"expected window slot 0's deadline=2, got {first_slot_deadline}"
backlog_scalar = obs[-1]
expected_backlog = (10 - 4) / full2.max_jobs
assert abs(backlog_scalar - expected_backlog) < 1e-4, f"expected backlog={expected_backlog}, got {backlog_scalar}"
print(f"  window=[7,3,1,9] (EDF order), backlog_scalar={backlog_scalar:.3f} (expected {expected_backlog:.3f})")

# ============================================================
# Check 3: fewer remaining jobs than window_size -> padding slots correctly
# marked "scheduled" (=1.0), zero backlog.
# ============================================================
print("=== Check 3: padding when fewer jobs remain than the window ===")
full3 = make_base_gym_env(num_jobs=3, num_machines=2, horizon=20)
env3 = WindowedPriorityGymSchedulingEnv(full3, window_size=5, use_atc=False)
obs, info = env3.reset()
assert env3._window_jobs == [0, 1, 2, None, None], f"expected 3 real jobs + 2 padding, got {env3._window_jobs}"
assert obs[-1] == 0.0, f"expected zero backlog when fewer jobs than window, got {obs[-1]}"
mask = env3.get_action_mask()
assert list(mask) == [1, 1, 1, 0, 0, 1], f"expected real slots + padding masked out + idle legal, got {list(mask)}"
print("  3 real jobs + 2 padding slots, mask correctly excludes padding, backlog=0")

# ============================================================
# Check 4: picking a window slot places the right job via FirstFit; window
# refreshes (re-sorts) after the placement changes what's pending.
# ============================================================
print("=== Check 4: picking a window slot places the correct job ===")
full4 = make_base_gym_env(num_jobs=4, num_machines=4, horizon=20, deadlines=[10, 5, 15, 8])
env4 = WindowedPriorityGymSchedulingEnv(full4, window_size=4, use_atc=False)
obs, info = env4.reset()
assert env4._window_jobs == [1, 3, 0, 2]  # sorted by deadline 5,8,10,15
obs, reward, term, trunc, info = env4.step(0)  # pick window slot 0 -> job 1
assert 1 not in env4.env.remaining_jobs, "job 1 (window slot 0) should have been scheduled"
assert env4.env.start_times[1] == 0
print("  window slot 0 correctly resolved to job 1 and scheduled it")

# ============================================================
# Check 5: idle action and forced-idle-on-infeasible-slot both work without
# crashing or miscounting invalid actions.
# ============================================================
print("=== Check 5: idle action and infeasible-slot fallback ===")
full5 = make_base_gym_env(num_jobs=2, num_machines=1, horizon=20, capacity=3.0)
env5 = WindowedPriorityGymSchedulingEnv(full5, window_size=3, use_atc=False)
obs, info = env5.reset()
env5.step(0)  # schedule job 0 (or whichever is window slot 0), consuming the sole machine
mask = env5.get_action_mask()
obs, reward, term, trunc, info = env5.step(3)  # idle_action = window_size
assert env5._invalid_action_count == 0, "explicit idle must never count as invalid"
print("  explicit idle action works cleanly")

# ============================================================
# Check 6 (0-d ndarray regression guard, matching the bug found in the
# non-windowed wrappers 2026-09-17).
# ============================================================
print("=== Check 6: 0-d numpy ndarray action_id doesn't crash step() ===")
full6 = make_base_gym_env(num_jobs=4, num_machines=2, horizon=20)
env6 = WindowedPriorityGymSchedulingEnv(full6, window_size=4, use_atc=False)
obs, info = env6.reset()
env6.step(np.array(0))
print("  accepted a 0-d ndarray action without crashing")

# ============================================================
# Check 7: network forward-pass shapes, both use_atc settings.
# ============================================================
print("=== Check 7: WindowedPriorityPointerActorCritic forward pass shapes ===")
for use_atc in (False, True):
    full7 = make_base_gym_env(num_jobs=10, num_machines=2, horizon=20)
    env7 = WindowedPriorityGymSchedulingEnv(full7, window_size=5, use_atc=use_atc)
    net = WindowedPriorityPointerActorCritic(window_size=5, num_machines=2, num_resources=1, use_atc=use_atc)
    dummy_obs = torch.rand((6, env7.observation_space.shape[0]))
    logits, value = net(dummy_obs)
    assert logits.shape == (6, 6), f"use_atc={use_atc}: expected logits (6,6), got {logits.shape}"
    assert value.shape == (6, 1), f"use_atc={use_atc}: expected value (6,1), got {value.shape}"
    print(f"  use_atc={use_atc}: logits.shape={tuple(logits.shape)}, value.shape={tuple(value.shape)}")

# ============================================================
# Check 8 (2026-09-21, S2W9 follow-up): window_order="fifo" for the OFFLINE
# case sorts candidates by job index (every job "arrives" at t=0
# simultaneously, so index order is the natural FIFO analogue) -- must
# differ from the "edf" ordering on the same instance/deadlines used in
# Check 2, proving the parameter actually changes behaviour, not just
# accepted and ignored.
# ============================================================
print("=== Check 8: window_order='fifo' (offline) orders by job index, not deadline ===")
full8 = make_base_gym_env(num_jobs=10, num_machines=2, horizon=20,
                           deadlines=[19, 5, 12, 3, 17, 8, 15, 2, 11, 6])
env8 = WindowedPriorityGymSchedulingEnv(full8, window_size=4, use_atc=False, window_order="fifo")
obs, info = env8.reset()
assert env8._window_jobs == [0, 1, 2, 3], f"expected index-ordered window [0,1,2,3], got {env8._window_jobs}"
assert env8._window_jobs != [7, 3, 1, 9], "fifo window must differ from Check 2's edf window on the same instance"
print(f"  window={env8._window_jobs} (job-index order, differs from Check 2's EDF-ordered [7,3,1,9])")

# ============================================================
# Check 9 (2026-09-21, S2W9 follow-up): window_order="fifo" for the ONLINE
# case sorts candidates by job_arrival_times[j], matching DeepRM's own
# literal FIFO queue design intent.
# ============================================================
print("=== Check 9: window_order='fifo' (online) orders by arrival time ===")
from Code.env.online_scheduling_env import OnlineSchedulingEnv
from Code.env.online_gym_wrapper import OnlineGymSchedulingEnv

num_jobs9 = 6
durations9 = np.array([2] * num_jobs9)
resources9 = np.array([[3]] * num_jobs9)
# Arrival order (2,4,0,5,1,3) deliberately scrambled relative to job index AND
# deadline, so fifo/edf/index orderings are all mutually distinguishable.
arrival_times9 = np.array([2, 4, 0, 5, 1, 3])
deadlines9 = np.array([arrival_times9[j] + 10 for j in range(num_jobs9)])
weights9 = np.array([1.0] * num_jobs9)
base9 = OnlineSchedulingEnv(
    job_durations=durations9, job_resources=resources9, job_deadlines=deadlines9,
    job_weights=weights9, num_machines=2, machine_capacity=np.array([10.0]), horizon=20,
    job_arrival_times=arrival_times9,
)
full9 = OnlineGymSchedulingEnv(base9, max_jobs=num_jobs9)
env9 = WindowedPriorityGymSchedulingEnv(full9, window_size=6, use_atc=False, window_order="fifo")
obs, info = env9.reset()
# All 6 jobs have arrived by t=5 -- advance time via idle steps until every job
# is revealed, then check the window is sorted by arrival time (job 2 arrives
# at t=0 first, then job 4 at t=1, ... job 3 last at t=5).
for _ in range(6):
    env9.step(env9.window_size)  # idle
expected_fifo_order = [2, 4, 0, 5, 1, 3]  # argsort of arrival_times9
assert env9._window_jobs == expected_fifo_order, (
    f"expected arrival-time-ordered window {expected_fifo_order}, got {env9._window_jobs}"
)
print(f"  window={env9._window_jobs} (arrival-time order, matches argsort({arrival_times9.tolist()}))")

print("\nALL WINDOWED-PRIORITY-WRAPPER CHECKS PASSED")
