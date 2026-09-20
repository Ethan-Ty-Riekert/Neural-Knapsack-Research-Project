"""test_action_branching_wrapper.py - Regression/validation checks for
Option 4 (action-branching, learned placement,
Code/env/action_branching_gym_wrapper.py, 2026-09-20, S2W9).

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_windowed_priority_wrapper.py.

Run from the repo root: python -m tests.test_action_branching_wrapper
"""
import numpy as np
import torch

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.action_branching_gym_wrapper import ActionBranchingGymSchedulingEnv
from Code.policies.action_branching_policy import ActionBranchingActorCritic
from Code.policies.action_branching_ppo_policy import ActionBranchingMaskableActorCriticPolicy


def make_base_gym_env(num_jobs=4, num_machines=2, horizon=20, capacity=10.0,
                       deadlines=None, job_resources=None):
    durations = np.array([2] * num_jobs)
    resources = np.array(job_resources) if job_resources is not None else np.array([[3]] * num_jobs)
    deadlines = np.array(deadlines) if deadlines is not None else np.array([8 + j for j in range(num_jobs)])
    weights = np.array([1.0] * num_jobs)
    base = SchedulingEnv(
        job_durations=durations, job_resources=resources, job_deadlines=deadlines,
        job_weights=weights, num_machines=num_machines,
        machine_capacity=np.array([capacity]), horizon=horizon,
    )
    return GymSchedulingEnv(base, max_jobs=num_jobs)


# ============================================================
# Check 1: action space is MultiDiscrete([max_jobs+1, num_machines]);
# observation space is UNCHANGED from the wrapped base env (no ATC variant
# in this first pass).
# ============================================================
print("=== Check 1: action space and observation space sizing ===")
full = make_base_gym_env(num_jobs=10, num_machines=3, horizon=20)
env = ActionBranchingGymSchedulingEnv(full)
assert list(env.action_space.nvec) == [11, 3], f"expected MultiDiscrete([11,3]), got {list(env.action_space.nvec)}"
assert env.observation_space.shape == full.observation_space.shape, (
    f"expected unchanged obs_dim={full.observation_space.shape}, got {env.observation_space.shape}"
)
print(f"  action_space.nvec={list(env.action_space.nvec)}, obs_dim={env.observation_space.shape[0]}")

# ============================================================
# Check 2: get_action_mask() shape and job-part correctness -- flat
# [job/idle mask][machine mask], idle always legal, both machines feasible
# on a freshly-reset (full-capacity) env.
# ============================================================
print("=== Check 2: get_action_mask() shape and initial feasibility ===")
full2 = make_base_gym_env(num_jobs=4, num_machines=2, horizon=20)
env2 = ActionBranchingGymSchedulingEnv(full2)
obs, info = env2.reset()
mask = env2.get_action_mask()
assert mask.shape == (5 + 2,), f"expected flat mask length 7, got {mask.shape}"
assert mask[4] == 1, "idle must always be legal"
assert mask[-2:].tolist() == [1, 1], f"expected both machines feasible at full capacity, got {mask[-2:]}"
print(f"  mask shape={mask.shape}, idle={mask[4]}, machine part={mask[-2:].tolist()}")

# ============================================================
# Check 3: machine-mask UNION correctness -- a machine infeasible for EVERY
# remaining job must be masked out; a machine feasible for at least one
# remaining job must NOT be masked out (even if infeasible for others --
# see Check 6 for the resulting mask-legal-but-infeasible case this implies).
# ============================================================
print("=== Check 3: machine-mask is the union of feasible machines across ALL remaining jobs ===")
full3 = make_base_gym_env(num_jobs=3, num_machines=2, horizon=20, capacity=10.0,
                           job_resources=[[4], [5], [6]])
env3 = ActionBranchingGymSchedulingEnv(full3)
obs, info = env3.reset()
env3.env.capacity[1, :, :] = 0.5  # machine 1 can't fit ANY remaining job (min demand=4)
mask3 = env3.get_action_mask()
assert mask3[-2:].tolist() == [1, 0], (
    f"expected machine0=feasible(for someone), machine1=infeasible-for-everyone, got {mask3[-2:]}"
)
print(f"  machine mask={mask3[-2:].tolist()} (machine1 correctly excluded -- fits nothing)")

# ============================================================
# Check 4: all-empty-union safety net -- if NO machine fits ANY remaining
# job, fall back to all-ones rather than a degenerate all-zero mask (which
# would make the machine branch unsampleable).
# ============================================================
print("=== Check 4: all-ones fallback when the feasible-machine union is empty ===")
full4 = make_base_gym_env(num_jobs=2, num_machines=2, horizon=20, capacity=10.0,
                           job_resources=[[4], [5]])
env4 = ActionBranchingGymSchedulingEnv(full4)
obs, info = env4.reset()
env4.env.capacity[:, :, :] = 0.5  # NO machine fits ANY remaining job
mask4 = env4.get_action_mask()
assert mask4[-2:].tolist() == [1, 1], f"expected all-ones fallback, got {mask4[-2:]}"
print(f"  machine mask={mask4[-2:].tolist()} (correctly falls back to all-ones, not all-zero)")

# ============================================================
# Check 5: a genuinely feasible (job, machine) step schedules correctly.
# ============================================================
print("=== Check 5: a genuinely feasible (job, machine) step places the job ===")
full5 = make_base_gym_env(num_jobs=3, num_machines=2, horizon=20)
env5 = ActionBranchingGymSchedulingEnv(full5)
obs, info = env5.reset()
obs, reward, term, trunc, info = env5.step((0, 1))  # job 0 on machine 1, both feasible
assert 0 not in env5.env.remaining_jobs, "job 0 should have been scheduled"
assert env5.env.start_times[0] == 0
assert info["mask_mismatch"] is False, "a genuinely feasible placement must not be flagged as a mismatch"
print("  job 0 correctly scheduled on machine 1, mask_mismatch=False")

# ============================================================
# Check 6 (the most important check given the parallel-branch design's
# approximation, see the wrapper's module docstring): a mask-LEGAL but
# actually-infeasible (job, machine) pair -- machine 1 fits job 0 (so the
# union mask allows it) but NOT job 1 -- must fall through to idle
# gracefully, not crash or silently misplace the job, and must be flagged
# via info["mask_mismatch"]=True.
# ============================================================
print("=== Check 6: mask-legal-but-infeasible pair -> graceful idle fallback + mask_mismatch=True ===")
full6 = make_base_gym_env(num_jobs=2, num_machines=2, horizon=20, capacity=10.0,
                           job_resources=[[4], [9]])
env6 = ActionBranchingGymSchedulingEnv(full6)
obs, info = env6.reset()
env6.env.capacity[1, :, :] = 5.0  # machine 1 fits job 0 (4<=5) but NOT job 1 (9>5)
mask6 = env6.get_action_mask()
assert mask6[-2:].tolist() == [1, 1], f"expected union to include machine1 (fits job0), got {mask6[-2:]}"
obs, reward, term, trunc, info = env6.step((1, 1))  # job 1 (needs 9) on machine 1 (cap 5) -- infeasible!
assert info["mask_mismatch"] is True, "mask-legal-but-infeasible placement must be flagged"
assert 1 in env6.env.remaining_jobs, "job 1 must remain pending (fell through to idle, not misplaced)"
assert term is False and trunc is False, "a single mismatch must not terminate/truncate the episode"
print("  job 1 -> machine 1 (mask-legal, actually infeasible) fell through to idle, mask_mismatch=True")

# ============================================================
# Check 7 (0-d ndarray / list / tuple action regression guard, matching the
# bug class already fixed in the other action-space-variant wrappers --
# extended here to BOTH action components).
# ============================================================
print("=== Check 7: 0-d ndarray / list / tuple actions don't crash step() ===")
for action in (
    np.array([0, 1]),
    [0, 1],
    (0, 1),
    [np.array(0), np.array(1)],
):
    full7 = make_base_gym_env(num_jobs=3, num_machines=2, horizon=20)
    env7 = ActionBranchingGymSchedulingEnv(full7)
    env7.reset()
    env7.step(action)
print("  accepted np.ndarray, list, tuple, and 0-d-ndarray-component actions without crashing")

# ============================================================
# Check 8: ActionBranchingActorCritic forward-pass shapes.
# ============================================================
print("=== Check 8: ActionBranchingActorCritic forward pass shapes ===")
full8 = make_base_gym_env(num_jobs=10, num_machines=3, horizon=20)
env8 = ActionBranchingGymSchedulingEnv(full8)
net = ActionBranchingActorCritic(max_jobs=10, num_machines=3, num_resources=1)
dummy_obs = torch.rand((6, env8.observation_space.shape[0]))
logits, value = net(dummy_obs)
expected_logit_dim = (10 + 1) + 3
assert logits.shape == (6, expected_logit_dim), f"expected logits (6,{expected_logit_dim}), got {logits.shape}"
assert value.shape == (6, 1), f"expected value (6,1), got {value.shape}"
print(f"  logits.shape={tuple(logits.shape)}, value.shape={tuple(value.shape)}")

# ============================================================
# Check 9 (validates the A1 design decision directly, not just trusting the
# library): build the full SB3 policy, apply a KNOWN mask with specific
# entries excluded in EACH branch independently, and confirm masked-out
# choices get ~0 probability in that branch without affecting the other
# branch -- i.e. MaskableMultiCategoricalDistribution really does split and
# mask each branch independently, as the whole design assumes.
# ============================================================
print("=== Check 9: distribution-level masking is applied independently per branch ===")
full9 = make_base_gym_env(num_jobs=4, num_machines=3, horizon=20)
env9 = ActionBranchingGymSchedulingEnv(full9)
policy = ActionBranchingMaskableActorCriticPolicy(
    observation_space=env9.observation_space,
    action_space=env9.action_space,
    lr_schedule=lambda _: 3e-4,
    max_jobs=4, num_machines=3, num_resources=1,
)
policy.eval()
dummy_obs = torch.rand((1, env9.observation_space.shape[0]))
# job branch: mask out everything except index 2; machine branch: mask out
# everything except index 0. If branches are independently masked, job
# probs must concentrate on index 2 regardless of the machine branch's mask.
job_mask = np.zeros(5, dtype=bool); job_mask[2] = True
machine_mask = np.zeros(3, dtype=bool); machine_mask[0] = True
flat_mask = np.concatenate([job_mask, machine_mask]).reshape(1, -1)
with torch.no_grad():
    dist = policy.get_distribution(dummy_obs, action_masks=flat_mask)
job_probs = dist.distributions[0].probs.squeeze(0)
machine_probs = dist.distributions[1].probs.squeeze(0)
assert torch.allclose(job_probs, torch.tensor([0., 0., 1., 0., 0.]), atol=1e-4), (
    f"expected all probability mass on job index 2, got {job_probs.tolist()}"
)
assert torch.allclose(machine_probs, torch.tensor([1., 0., 0.]), atol=1e-4), (
    f"expected all probability mass on machine index 0, got {machine_probs.tolist()}"
)
print(f"  job branch probs={[round(p, 3) for p in job_probs.tolist()]} (mass on index 2)")
print(f"  machine branch probs={[round(p, 3) for p in machine_probs.tolist()]} (mass on index 0)")

# ============================================================
# Check 10 (2026-09-20 fix validation, user-prompted -- "maybe we need to
# do some more research into the decoupling"): _pool_job_context() must
# concentrate on the job the job branch actually favours, must reduce to
# the OLD flat-mean behaviour when the job branch is uniform (regression-
# equivalence, not a silent behaviour change in the common early-training
# case), and must not produce NaN when no job is active.
# ============================================================
print("=== Check 10: _pool_job_context weights by the job branch's own distribution ===")
net10 = ActionBranchingActorCritic(max_jobs=4, num_machines=2, num_resources=1)
job_emb10 = torch.stack([
    torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
    torch.tensor([10.0, 10.0]), torch.tensor([-5.0, -5.0]),
]).unsqueeze(0)  # (1, 4, 2) -- 4 distinguishable "embeddings"

# Peaked: job branch overwhelmingly favours index 2 -> context should be
# (near-)exactly that job's embedding, not a blend.
peaked_logits = torch.tensor([[-100.0, -100.0, 100.0, -100.0]])
active_all = torch.ones((1, 4, 1))
ctx_peaked = net10._pool_job_context(job_emb10, peaked_logits, active_all)
assert torch.allclose(ctx_peaked.squeeze(0), job_emb10[0, 2], atol=1e-3), (
    f"expected context ~= job 2's embedding {job_emb10[0,2].tolist()}, got {ctx_peaked.tolist()}"
)
print(f"  peaked job branch -> context={ctx_peaked.squeeze(0).tolist()} ~= job 2's embedding (correct)")

# Uniform: job branch has no preference -> must reduce to the flat mean
# (the OLD pre-fix behaviour) over active jobs, not an arbitrary weighting.
uniform_logits = torch.zeros((1, 4))
ctx_uniform = net10._pool_job_context(job_emb10, uniform_logits, active_all)
expected_flat_mean = job_emb10.mean(dim=1)
assert torch.allclose(ctx_uniform, expected_flat_mean, atol=1e-4), (
    f"expected flat mean {expected_flat_mean.tolist()}, got {ctx_uniform.tolist()}"
)
print(f"  uniform job branch -> context={ctx_uniform.squeeze(0).tolist()} == flat mean (regression-safe)")

# No active jobs at all (edge case, e.g. right at episode end): must not
# produce NaN.
active_none = torch.zeros((1, 4, 1))
ctx_none = net10._pool_job_context(job_emb10, peaked_logits, active_none)
assert torch.isfinite(ctx_none).all(), f"expected finite output with no active jobs, got {ctx_none.tolist()}"
print(f"  no active jobs -> context={ctx_none.squeeze(0).tolist()} (finite, no NaN)")

print("\nALL ACTION-BRANCHING WRAPPER CHECKS PASSED")
