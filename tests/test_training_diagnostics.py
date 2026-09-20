"""test_training_diagnostics.py - Regression/validation checks for
Code/utils/training_diagnostics.py (2026-09-20, S2W9) -- the training-time
observability tooling added to diagnose the still-open "more online
training hurts" mystery and give an early-warning signal for whether a
long training run is trending toward success or failure.

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_windowed_priority_wrapper.py.

These are BaseCallback subclasses, so the cleanest way to unit-test them
without a full model.learn() loop is to instantiate directly and manually
set the attributes SB3 normally injects (logger, locals, num_timesteps),
call _on_step() directly, then assert against logger.name_to_value
(populated by .record() calls).

Run from the repo root: python -m tests.test_training_diagnostics
"""
import numpy as np
import torch

from stable_baselines3.common.logger import configure

from Code.utils.training_diagnostics import (
    ActionDistributionCallback, TardinessEvalCallback, build_diagnostics_callbacks, _entropy_normalized,
)
from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv, RULE_NAMES
from sb3_contrib.common.wrappers import ActionMasker
from Code.training.train_action_space_variant import mask_fn


class _LoggerOnlyModel:
    """BaseCallback.logger is a read-only property that reads self.model.logger
    -- this is the minimal stand-in needed to set a logger on a callback
    without a real SB3 model/training loop."""
    def __init__(self, logger):
        self.logger = logger


def _feed(callback, actions_list, infos_list=None, num_timesteps=100):
    """Drive a callback through len(actions_list) steps, mimicking what
    SB3's OnPolicyAlgorithm.collect_rollouts() does each step (set
    self.locals["actions"]/["infos"], call _on_step())."""
    callback.model = _LoggerOnlyModel(configure(folder=None, format_strings=[]))
    callback.num_timesteps = num_timesteps
    for i, actions in enumerate(actions_list):
        infos = infos_list[i] if infos_list is not None else [{}] * len(actions)
        callback.locals = {"actions": np.array(actions), "infos": infos}
        callback._on_step()
    return callback


# ============================================================
# Check 1: mode="rule" -- all-idle action stream -> idle=1.0, entropy=0.0.
# ============================================================
print("=== Check 1: mode='rule', all-idle stream -> entropy=0.0 ===")
cb1 = ActionDistributionCallback(mode="rule", num_job_choices=8, rule_names=RULE_NAMES, log_interval=10)
_feed(cb1, [[7]] * 10)  # idle index = num_rules = 7
assert cb1.logger.name_to_value["action_dist/idle"] == 1.0, cb1.logger.name_to_value
assert abs(cb1.logger.name_to_value["action_dist/entropy_normalized"] - 0.0) < 1e-6
print(f"  idle={cb1.logger.name_to_value['action_dist/idle']}, "
      f"entropy_normalized={cb1.logger.name_to_value['action_dist/entropy_normalized']}")

# ============================================================
# Check 2: mode="rule" -- uniform-over-8-choices stream -> entropy ~= 1.0.
# ============================================================
print("=== Check 2: mode='rule', uniform stream -> entropy~=1.0 ===")
cb2 = ActionDistributionCallback(mode="rule", num_job_choices=8, rule_names=RULE_NAMES, log_interval=80)
_feed(cb2, [[i % 8] for i in range(80)])
assert abs(cb2.logger.name_to_value["action_dist/entropy_normalized"] - 1.0) < 1e-3, cb2.logger.name_to_value
for name in RULE_NAMES:
    assert abs(cb2.logger.name_to_value[f"action_dist/{name}"] - 0.125) < 1e-6
print(f"  entropy_normalized={cb2.logger.name_to_value['action_dist/entropy_normalized']:.4f} (~1.0)")

# ============================================================
# Check 3: mode="job" -- idle_frac/top1_frac/entropy on an engineered
# histogram (known composition: 50% idle, 30% job 0, 20% spread over jobs
# 1-4 uniformly).
# ============================================================
print("=== Check 3: mode='job' -- idle_frac/top1_frac from a known histogram ===")
max_jobs = 10
actions = [[max_jobs]] * 50 + [[0]] * 30 + [[1], [2], [3], [4]] * 5  # 50+30+20 = 100
cb3 = ActionDistributionCallback(mode="job", num_job_choices=max_jobs + 1, log_interval=len(actions))
_feed(cb3, actions)
assert abs(cb3.logger.name_to_value["action_dist/idle_frac"] - 0.5) < 1e-6, cb3.logger.name_to_value
assert abs(cb3.logger.name_to_value["action_dist/top1_frac"] - 0.3) < 1e-6, cb3.logger.name_to_value
print(f"  idle_frac={cb3.logger.name_to_value['action_dist/idle_frac']}, "
      f"top1_frac={cb3.logger.name_to_value['action_dist/top1_frac']}")

# ============================================================
# Check 4: mode="branching" -- job stats as above, PLUS
# machine_entropy_normalized/machine_top1_frac/machine_mask_mismatch_frac.
# ============================================================
print("=== Check 4: mode='branching' -- machine-branch + mask_mismatch stats ===")
num_machines = 4
# Each step's actions batch is (n_envs=1, 2) -- [[job, machine]] -- matching
# real SB3 self.locals["actions"] shape for a MultiDiscrete action space.
actions4 = [[[0, 0]]] * 25 + [[[0, 1]]] * 25 + [[[0, 2]]] * 25 + [[[0, 3]]] * 25  # uniform over 4 machines
infos4 = [[{"mask_mismatch": (i % 10 == 0)}] for i in range(100)]  # 10% mismatch rate
cb4 = ActionDistributionCallback(mode="branching", num_job_choices=max_jobs + 1, num_machines=num_machines,
                                  log_interval=len(actions4))
_feed(cb4, actions4, infos_list=infos4)
assert abs(cb4.logger.name_to_value["action_dist/machine_entropy_normalized"] - 1.0) < 1e-3, cb4.logger.name_to_value
assert abs(cb4.logger.name_to_value["action_dist/machine_top1_frac"] - 0.25) < 1e-6
assert abs(cb4.logger.name_to_value["action_dist/machine_mask_mismatch_frac"] - 0.1) < 1e-6
print(f"  machine_entropy_normalized={cb4.logger.name_to_value['action_dist/machine_entropy_normalized']:.4f}, "
      f"machine_mask_mismatch_frac={cb4.logger.name_to_value['action_dist/machine_mask_mismatch_frac']}")

# ============================================================
# Check 5: _entropy_normalized closed-form sanity (single-bin degenerate
# case must not divide by zero / NaN).
# ============================================================
print("=== Check 5: _entropy_normalized handles a single-bin distribution ===")
assert _entropy_normalized(np.array([5])) == 0.0
assert _entropy_normalized(np.array([0, 0, 0])) == 0.0  # all-zero counts -> defined as 0, not NaN
print("  single-bin and all-zero-count edge cases both return 0.0, no NaN")

# ============================================================
# Check 6: TardinessEvalCallback -- a small deterministic env + a stub
# .predict() (no real trained model needed, keeps this fast and
# deterministic), assert eval_tardiness/mean_tardiness is recorded, finite,
# and non-negative after one _on_step() at the eval boundary.
# ============================================================
print("=== Check 6: TardinessEvalCallback logs a finite, non-negative tardiness ===")


class _StubModel:
    """Always picks rule index 0 (EDF) -- deterministic, no training needed."""
    def predict(self, obs, action_masks=None, deterministic=True):
        return np.array(0), None


def _make_held_out_env():
    num_jobs = 6
    durations = np.array([2] * num_jobs)
    resources = np.array([[3]] * num_jobs)
    deadlines = np.array([8 + j for j in range(num_jobs)])
    weights = np.array([1.0] * num_jobs)
    base = SchedulingEnv(
        job_durations=durations, job_resources=resources, job_deadlines=deadlines,
        job_weights=weights, num_machines=2, machine_capacity=np.array([10.0]), horizon=20,
    )
    full = GymSchedulingEnv(base, max_jobs=num_jobs)
    return ActionMasker(RuleSelectionGymSchedulingEnv(full), mask_fn)


held_out_envs = [_make_held_out_env() for _ in range(2)]
cb6 = TardinessEvalCallback(held_out_envs=held_out_envs, eval_freq=10)
cb6.model = _StubModel()
cb6.model.logger = configure(folder=None, format_strings=[])
cb6.num_timesteps = 10
cb6._on_step()
mean_tardiness = cb6.logger.name_to_value["eval_tardiness/mean_tardiness"]
mean_weighted = cb6.logger.name_to_value["eval_tardiness/mean_weighted_tardiness"]
assert np.isfinite(mean_tardiness) and mean_tardiness >= 0, f"expected finite, non-negative, got {mean_tardiness}"
assert np.isfinite(mean_weighted) and mean_weighted >= 0
print(f"  eval_tardiness/mean_tardiness={mean_tardiness}, mean_weighted_tardiness={mean_weighted}")

# ============================================================
# Check 7: build_diagnostics_callbacks -- construction-only smoke test that
# it returns the right callback TYPES for every option, without running
# any training.
# ============================================================
print("=== Check 7: build_diagnostics_callbacks returns the right callback types per option ===")
from Code.utils.training_diagnostics import MaskableEvalCallback

held_out_smoke = [_make_held_out_env() for _ in range(2)]
for option in ("1", "2", "3", "4"):
    callbacks = build_diagnostics_callbacks(
        option, held_out_envs=held_out_smoke, diagnostics_interval=1000,
        rule_names=RULE_NAMES, max_jobs=6, num_machines=2,
    )
    assert len(callbacks) == 3, f"option {option}: expected 3 callbacks, got {len(callbacks)}"
    types = [type(c) for c in callbacks]
    assert ActionDistributionCallback in types, f"option {option}: missing ActionDistributionCallback"
    assert TardinessEvalCallback in types, f"option {option}: missing TardinessEvalCallback"
    assert MaskableEvalCallback in types, f"option {option}: missing MaskableEvalCallback"
    print(f"  option {option}: {[t.__name__ for t in types]}")

print("\nALL TRAINING-DIAGNOSTICS CHECKS PASSED")
