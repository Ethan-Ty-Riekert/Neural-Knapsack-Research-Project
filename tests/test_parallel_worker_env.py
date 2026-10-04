"""test_parallel_worker_env.py - Regression checks for train_action_space_variant.py's --n-envs > 1
worker envs (2026-10-04, S2W11, found while merging autonomous-overnight-2026-08-28 into main).

The parallel-rollout workers were written before the v2 objective program existed, and built their
envs from their own argument list -- so with --n-envs > 1 they silently dropped --reward-mode
objective's ObjectiveConfig, --difficulty, the extended horizon and --gamma's shaping_gamma, and
trained on a different reward than the single-env path. Both paths now build through
make_full_gym_env() from one base_env_kwargs dict. Checks:

  1. A worker env gets the same objective / extend_horizon as the single-env path (v2, online).
  2. Workers' difficulty resamplers are seeded per worker (independent instance streams), while
     worker 0 reproduces the single-env path's stream exactly (rng_seed = seed = 0).
  3. Offline fixed instance: every worker sees the IDENTICAL instance (no per-worker seed).

Run from the repo root: python -m tests.test_parallel_worker_env
"""
import numpy as np

from Code.core.difficulty import DIFFICULTIES
from Code.methods.rl.training.train_action_space_variant import make_full_gym_env, make_worker_env
from Code.variants.v2_objectives import objective_config


def base_env(env):
    """Walk the wrapper chain (ActionMasker -> action-space wrapper -> gym wrapper -> raw env)."""
    while not hasattr(env, "reward_mode"):
        env = env.env
    return env


def gym_env(env):
    """The GymSchedulingEnv/OnlineGymSchedulingEnv layer, which owns job_resampler (the Option 1
    wrapper keeps it as `_full`; its `.env` skips straight to the raw env)."""
    while not hasattr(env, "job_resampler"):
        env = env._full if hasattr(env, "_full") else env.env
    return env


# 1. v2 objective + difficulty reach the workers.
objective = objective_config(("tardiness_sq",), drop_shaping=True)
kwargs = dict(reward_mode="objective", use_potential_shaping=False, shaping_gamma=0.97,
              job_weight_range=None, objective=objective, difficulty=DIFFICULTIES["on_rho075"],
              extend_horizon=True, arrival_rate=1.0, horizon=100, max_jobs=1300,
              job_size_distribution="uniform")
single = make_full_gym_env(True, kwargs)
workers = [make_worker_env(i, "1", True, kwargs, None, "edf", False, 0) for i in range(3)]
for w in workers:
    raw = base_env(w)
    assert raw.objective is not None, "worker env lost the v2 ObjectiveConfig"
    assert raw.extend_horizon == base_env(single).extend_horizon is True, "worker env lost extend_horizon"
print("  1. workers carry the v2 objective and extended horizon")

# 2. Per-worker difficulty resampler streams.
draws = [gym_env(w).job_resampler()["job_durations"] for w in workers]
assert not np.array_equal(draws[1], draws[2]), "workers 1 and 2 drew the same difficulty instance"
assert np.array_equal(gym_env(single).job_resampler()["job_durations"], draws[0]), \
    "worker 0 should reproduce the single-env resampler stream (both rng_seed=0)"
print("  2. difficulty resamplers are independent per worker; worker 0 matches the single env")

# 3. Offline fixed instance is identical across workers.
off_kwargs = dict(reward_mode="legacy", use_potential_shaping=False, shaping_gamma=0.99,
                  job_weight_range=None, objective=None, difficulty=None, extend_horizon=False,
                  randomize_instances=False)
off = [base_env(make_worker_env(i, "1", False, off_kwargs, None, "edf", False, 0)) for i in range(3)]
for raw in off[1:]:
    assert np.array_equal(raw.job_durations, off[0].job_durations), "fixed instance differs across workers"
    assert np.array_equal(raw.job_deadlines, off[0].job_deadlines), "fixed instance differs across workers"
print("  3. offline fixed instance is identical across workers")

print("test_parallel_worker_env: all checks passed")
