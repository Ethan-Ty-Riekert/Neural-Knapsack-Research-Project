"""test_action_space_wrappers.py - Regression checks for the 2026-09-17
action-space-reduction wrappers: RuleSelectionGymSchedulingEnv (Option 1)
and PriorityOnlyGymSchedulingEnv (Options 2/3), plus a network-shape smoke
test for PriorityPointerActorCritic.

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_online_env.py.

Run from the repo root: python -m tests.test_action_space_wrappers
"""
import numpy as np
import torch

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv, RULE_NAMES
from Code.env.priority_only_gym_wrapper import PriorityOnlyGymSchedulingEnv
from Code.policies.priority_pointer_policy import PriorityPointerActorCritic


def make_base_gym_env(num_jobs=4, num_machines=2, horizon=10, capacity=10.0):
    durations = np.array([2] * num_jobs)
    resources = np.array([[3]] * num_jobs)
    deadlines = np.array([8] * num_jobs)
    weights = np.array([1.0] * num_jobs)
    base = SchedulingEnv(
        job_durations=durations,
        job_resources=resources,
        job_deadlines=deadlines,
        job_weights=weights,
        num_machines=num_machines,
        machine_capacity=np.array([capacity]),
        horizon=horizon,
    )
    return GymSchedulingEnv(base, max_jobs=num_jobs)


# ============================================================
# Option 1: action space size, feasible decode, idle fallback
# ============================================================
print("=== Option 1 (rule selection): action space size ===")
full1 = make_base_gym_env(num_jobs=4, num_machines=2, horizon=10)
env1 = RuleSelectionGymSchedulingEnv(full1)
assert env1.action_space.n == len(RULE_NAMES) + 1 == 8, (
    f"expected Discrete(8), got Discrete({env1.action_space.n})"
)
print(f"  action_space = Discrete({env1.action_space.n}) (7 rules + idle)")

print("=== Option 1: a chosen rule decodes to a real, feasible placement ===")
obs, info = env1.reset()
assert env1.env.remaining_jobs == set(range(4))
edf_idx = RULE_NAMES.index("EDF")
obs, reward, term, trunc, info = env1.step(edf_idx)
assert len(env1.env.remaining_jobs) == 3, (
    f"EDF rule selection should have scheduled exactly one job, remaining={env1.env.remaining_jobs}"
)
assert reward != 0.0, "a real placement should produce a non-zero reward"
print(f"  EDF selection scheduled one job; remaining_jobs={env1.env.remaining_jobs}")

print("=== Option 1: idle action advances no jobs, and mask disables rules once nothing fits ===")
full1b = make_base_gym_env(num_jobs=2, num_machines=1, horizon=10, capacity=3.0)
env1b = RuleSelectionGymSchedulingEnv(full1b)
obs, info = env1b.reset()
# Both jobs need 3 capacity on the only machine (capacity=3) -- scheduling
# job 0 fully consumes it, so no rule can place job 1 until job 0 finishes.
edf_idx = RULE_NAMES.index("EDF")
obs, reward, term, trunc, info = env1b.step(edf_idx)
mask = env1b.get_action_mask()
assert not mask[:env1b.num_rules].any(), (
    f"rules should be masked out once no (job, machine) placement fits, mask={mask}"
)
assert mask[env1b.num_rules] == 1, "idle must stay legal even when every rule is masked out"
print("  rules correctly masked out once the sole machine is full; idle stays legal")

idle_id = env1b.num_rules
obs, reward, term, trunc, info = env1b.step(idle_id)
assert 1 in env1b.env.remaining_jobs, "idle step must not have scheduled job 1"
print("  idle action left job 1 unscheduled, as expected")


# ============================================================
# Options 2/3: action space size, obs dim, FirstFit placement, ATC feature
# ============================================================
print("=== Option 2 (priority-only, raw features): action space + obs dim ===")
full2 = make_base_gym_env(num_jobs=4, num_machines=2, horizon=10)
env2 = PriorityOnlyGymSchedulingEnv(full2, use_atc=False)
assert env2.action_space.n == 4 + 1 == 5, f"expected Discrete(5), got Discrete({env2.action_space.n})"
assert env2.observation_space.shape[0] == full2.observation_space.shape[0], (
    "Option 2 must not change the observation dimension"
)
print(f"  action_space=Discrete({env2.action_space.n}), obs_dim unchanged at {env2.observation_space.shape[0]}")

print("=== Option 2: picking a job slot places it via FirstFit on a feasible machine ===")
obs, info = env2.reset()
obs, reward, term, trunc, info = env2.step(0)  # pick job 0
assert 0 not in env2.env.remaining_jobs, "job 0 should have been scheduled"
assert env2.env.start_times[0] != -1 and env2.env.start_times[0] == 0
print(f"  job 0 scheduled on machine={np.argmax(env2.env.capacity[:, 0, 0] < 10.0)}, start_time={env2.env.start_times[0]}")

print("=== Option 3 (ATC-primed): obs dim grows by exactly max_jobs, ATC feature is finite/in-range ===")
full3 = make_base_gym_env(num_jobs=4, num_machines=2, horizon=10)
env3 = PriorityOnlyGymSchedulingEnv(full3, use_atc=True)
assert env3.observation_space.shape[0] == full3.observation_space.shape[0] + full3.max_jobs, (
    f"Option 3 obs_dim should be base+max_jobs, got base={full3.observation_space.shape[0]}, "
    f"option3={env3.observation_space.shape[0]}"
)
obs3, info = env3.reset()
job_slot_width = env3.num_resources + 5  # +1 over Option 2's R+4 for the appended ATC feature
job_block = obs3[env3._machine_block_end:]
atc_values = job_block.reshape(env3.max_jobs, job_slot_width)[:, -1]
assert np.all((atc_values >= 0.0) & (atc_values <= 1.0)), f"ATC feature out of [0,1] range: {atc_values}"
assert np.any(atc_values > 0.0), "expected at least one real (non-padding) job's ATC feature to be > 0"
print(f"  obs_dim grew by exactly max_jobs ({full3.max_jobs}); ATC features in range: {atc_values}")

print("=== Options 2/3: idle fallback when a chosen job has no feasible machine ===")
full2b = make_base_gym_env(num_jobs=2, num_machines=1, horizon=10, capacity=3.0)
env2b = PriorityOnlyGymSchedulingEnv(full2b, use_atc=False)
obs, info = env2b.reset()
env2b.step(0)  # schedule job 0, consuming the sole machine's capacity
mask = env2b.get_action_mask()
assert mask[1] == 0, "job 1 should be masked infeasible once the sole machine is full"
assert mask[env2b.max_jobs] == 1, "idle must stay legal"
obs, reward, term, trunc, info = env2b.step(1)  # RL "incorrectly" picks job 1 anyway
assert 1 in env2b.env.remaining_jobs, "job 1 must remain unscheduled (fell through to idle, not forced invalid)"
assert env2b._invalid_action_count == 0, "the idle fallback path must not count as an invalid action"
print("  picking an infeasible job slot fell through to idle without counting as invalid")


# ============================================================
# Network shape smoke test: PriorityPointerActorCritic forward pass
# ============================================================
print("=== PriorityPointerActorCritic: forward pass shapes (Option 2 and Option 3) ===")
for use_atc in (False, True):
    max_jobs, num_machines, num_resources = full2.max_jobs, full2.num_machines, full2.num_resources
    net = PriorityPointerActorCritic(max_jobs, num_machines, num_resources, use_atc=use_atc)
    obs_dim = env2.observation_space.shape[0] if not use_atc else env3.observation_space.shape[0]
    dummy_obs = torch.rand((5, obs_dim))
    logits, value = net(dummy_obs)
    assert logits.shape == (5, max_jobs + 1), f"use_atc={use_atc}: expected logits (5, {max_jobs+1}), got {logits.shape}"
    assert value.shape == (5, 1), f"use_atc={use_atc}: expected value (5, 1), got {value.shape}"
    print(f"  use_atc={use_atc}: logits.shape={tuple(logits.shape)}, value.shape={tuple(value.shape)}")


# ============================================================
# Regression guard: a 0-d numpy ndarray action (what model.predict() hands
# back when called directly on a single non-batched obs, e.g.
# eval_action_space_variant.py) must not crash step()'s set-membership
# checks. numpy 0-d arrays tolerate == and list-indexing (why this wasn't
# caught by hand-testing with plain python ints) but are NOT hashable,
# which broke `job in self.env.remaining_jobs` in
# PriorityOnlyGymSchedulingEnv.step() until action_id is cast to int() up
# front.
# ============================================================
print("=== Regression guard: 0-d numpy ndarray action_id doesn't crash step() ===")
full1c = make_base_gym_env(num_jobs=4, num_machines=2, horizon=10)
env1c = RuleSelectionGymSchedulingEnv(full1c)
obs, info = env1c.reset()
env1c.step(np.array(RULE_NAMES.index("EDF")))  # 0-d ndarray, not a python int
print("  RuleSelectionGymSchedulingEnv.step() accepted a 0-d ndarray action")

full2c = make_base_gym_env(num_jobs=4, num_machines=2, horizon=10)
env2c = PriorityOnlyGymSchedulingEnv(full2c, use_atc=False)
obs, info = env2c.reset()
env2c.step(np.array(0))  # 0-d ndarray, not a python int
print("  PriorityOnlyGymSchedulingEnv.step() accepted a 0-d ndarray action")

print("\nALL ACTION-SPACE-WRAPPER CHECKS PASSED")
