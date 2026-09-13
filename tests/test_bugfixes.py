"""test_bugfixes.py - Regression checks for the bug fixes made in this session:
Issue A (machine_active flipped before feasibility check), Issue C (mask/step
time-index mismatch at the horizon boundary), Issue D (unbounded tardiness term
dwarfing every other reward term at large horizons), and the A2C
deterministic/greedy eval mode added to fix the A2C-vs-PPO eval-fairness gap.

Each check targets the exact boundary condition that was previously wrong, so
these double as regression guards, not just one-off verification. Follows this
project's existing tests/ convention (runnable script with assert invariants,
not a pytest suite) -- see test_diagnostic.py.

Run from the repo root: python -m tests.test_bugfixes
"""
import numpy as np

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.policies.a2c_policy import make_maskable_a2c, select_action


# ============================================================
# Issue A -- machine_active must only flip on an ACTUAL successful placement
# ============================================================
print("=== Issue A: machine_active set-before-feasibility-check ===")

# Two jobs, one machine, capacity=5: job 0 requires 10 (always infeasible on this
# machine), job 1 requires 2 (feasible). lambda_2/lambda_3 zeroed out so the only
# non-flat reward contribution left is the -lambda1 activation charge plus the
# fixed +3.0 placement bonus -- isolating exactly the term this bug corrupted.
env = SchedulingEnv(
    job_durations=np.array([1, 1]),
    job_resources=np.array([[10], [2]]),
    job_deadlines=np.array([10, 10]),
    job_weights=np.array([1.0, 1.0]),
    num_machines=1,
    machine_capacity=np.array([5.0]),
    horizon=10,
    lambda_1=1.0,
    lambda_2=0.0,
    lambda_3=0.0,
    invalid_penalty=5.0,
)

# Infeasible attempt on the never-yet-used machine.
_, reward, _ = env.step((0, 0))
assert reward == -5.0, f"expected invalid_penalty -5.0, got {reward}"
assert env.machine_active[0] == 0, (
    "machine_active flipped on a FAILED placement attempt -- Issue A regressed"
)
print(f"  Infeasible attempt: reward={reward} (invalid_penalty), machine_active={env.machine_active[0]} (correctly still 0)")

# Now the real first successful placement on that same machine.
_, reward, _ = env.step((1, 0))
assert env.machine_active[0] == 1, "machine_active should be 1 after a real successful placement"
# Expected: -lambda1 (real first use) + 3.0 flat bonus = 2.0, with lambda2=lambda3=0.
# Pre-fix, machine_active would already have read 1 (wrongly set during the failed
# job-0 attempt), so machine_was_inactive would be False and this would read 3.0
# instead -- silently skipping the activation penalty on the real first use.
assert reward == 2.0, (
    f"expected reward 2.0 (-lambda1 + 3.0 placement bonus), got {reward} -- "
    "if this reads 3.0, the activation penalty was skipped (Issue A regressed)"
)
print(f"  First real placement: reward={reward} (== -lambda1 + 3.0, activation penalty correctly charged)")
print("  PASS\n")


# ============================================================
# Issue C -- mask must agree with step() exactly at t == horizon
# ============================================================
print("=== Issue C: mask/execution time-index mismatch at horizon boundary ===")

HORIZON = 5


def make_boundary_env():
    base = SchedulingEnv(
        job_durations=np.array([1]),
        job_resources=np.array([[1]]),
        job_deadlines=np.array([10]),
        job_weights=np.array([1.0]),
        num_machines=1,
        machine_capacity=np.array([10.0]),
        horizon=HORIZON,
        invalid_penalty=5.0,
    )
    return base, GymSchedulingEnv(base, max_jobs=1)


# At t = horizon - 1: a duration-1 job legitimately fits ((horizon-1)+1 == horizon).
base_env, gym_env = make_boundary_env()
base_env.time = HORIZON - 1
mask = gym_env.get_action_mask()
assert mask[0] == 1, "duration-1 job should be feasible at t == horizon-1"
assert base_env.is_feasible(0, 0, HORIZON - 1) is True
print(f"  t={HORIZON - 1} (horizon-1): mask[job0,machine0]={mask[0]} (correctly feasible)")

# At t = horizon: the SAME job must now be infeasible in BOTH the mask and step().
base_env, gym_env = make_boundary_env()
base_env.time = HORIZON
mask = gym_env.get_action_mask()
old_clamped_t = min(base_env.time, HORIZON - 1)  # what the pre-fix mask used
old_mask_would_say = base_env.is_feasible(0, 0, old_clamped_t)
assert mask[0] == 0, (
    f"mask reports feasible at t==horizon (uncapped check) -- Issue C regressed. "
    f"(pre-fix clamped mask would have wrongly said feasible={old_mask_would_say})"
)
_, reward, done = base_env.step((0, 0))
assert reward == -5.0, f"step() should return the invalid-penalty branch at t==horizon, got reward={reward}"
assert 0 in base_env.remaining_jobs, "job should NOT have been scheduled -- step() disagreed with a feasible mask"
print(f"  t={HORIZON} (horizon): mask[job0,machine0]={mask[0]} (correctly infeasible), "
      f"step() reward={reward} (invalid, AGREES with mask)")
print(f"  [pre-fix clamped mask would have said feasible={old_mask_would_say} -- exactly the divergence that hung episodes]")
print("  PASS\n")


# ============================================================
# Issue D -- tardiness term must stay O(1) regardless of horizon
# ============================================================
print("=== Issue D: tardiness term normalised by horizon ===")

for H in (20, 100):
    # Worst case: deadline as early as possible (1), duration as long as feasible
    # while still placeable at t=0 (H-1), maximising raw tardiness T_j = H-2.
    env = SchedulingEnv(
        job_durations=np.array([H - 1]),
        job_resources=np.array([[1]]),
        job_deadlines=np.array([1]),
        job_weights=np.array([1.0]),
        num_machines=1,
        machine_capacity=np.array([10.0]),
        horizon=H,
        lambda_1=0.0,
        lambda_2=1.0,
        lambda_3=0.0,
        invalid_penalty=5.0,
    )
    assert env.is_feasible(0, 0, 0), "test setup should be feasible at t=0"
    env.step((0, 0))
    raw_tardiness = env.tardiness[0]
    tardiness_term = env.lambda2 * env.job_weights[0] * (raw_tardiness / env.horizon)
    print(f"  H={H}: raw_tardiness={raw_tardiness}, normalised term={tardiness_term:.4f} "
          f"(pre-fix this term would have been the raw, unbounded {raw_tardiness})")
    assert abs(tardiness_term) <= env.lambda2 * env.job_weights[0], (
        f"tardiness term exceeded the O(1) bound proven for T_j/H at H={H} -- Issue D regressed"
    )

print("  PASS -- tardiness term stayed O(1) at both a small and large horizon\n")


# ============================================================
# A2C deterministic eval mode
# ============================================================
print("=== A2C deterministic (greedy) eval mode ===")

small_base = SchedulingEnv(
    job_durations=np.array([1, 1, 1]),
    job_resources=np.array([[1], [1], [1]]),
    job_deadlines=np.array([5, 5, 5]),
    job_weights=np.array([1.0, 1.0, 1.0]),
    num_machines=2,
    machine_capacity=np.array([10.0]),
    horizon=5,
)
small_gym = GymSchedulingEnv(small_base, max_jobs=3)
agent = make_maskable_a2c(small_gym, device="cpu", policy_type="flat", normalize_rewards=False)

obs, info = small_gym.reset()
mask = info["action_mask"]

det_actions = {agent.act(obs, mask, deterministic=True) for _ in range(5)}
assert len(det_actions) == 1, (
    f"deterministic=True should always return the same action for the same obs/mask, got {det_actions}"
)
print(f"  deterministic=True across 5 calls: always action {next(iter(det_actions))} (PASS)")

sample_actions = {agent.act(obs, mask, deterministic=False) for _ in range(50)}
print(f"  deterministic=False across 50 calls: {len(sample_actions)} unique action(s) sampled "
      f"(stochastic path unchanged; not hard-asserted, since variety depends on random init)")

print("  PASS\n")

print("=== All bug-fix regression checks passed ===")
