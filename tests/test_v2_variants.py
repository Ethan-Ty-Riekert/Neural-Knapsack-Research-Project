"""test_v2_variants.py - Checks for the v2 model-roster variants added 2026-10-05 (S2W12):

  1. Option 1 per-tick decisions: one action fills the tick with the chosen rule, then time advances;
     the transition reward equals the sum of the per-placement rewards over the same tick.
  2. Offline, per-tick and per-placement Option 1 produce identical schedules (each placement
     already advances time there).
  3. Option 0 (full action space) builds with both the flat and the pointer architecture.
  4. --algo a2c resolves to A2C-as-PPO (one epoch, one full batch, RMSprop, no advantage
     normalisation); --algo ppo keeps the exact previous defaults.

Run from the repo root: python -m tests.test_v2_variants
"""
import types

import numpy as np

from Code.methods.heuristics.registry import HEURISTICS
from Code.methods.rl.action_spaces.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv
from Code.methods.rl.training.train_action_space_variant import (
    build_env_and_policy, make_base_gym_env, make_online_base_gym_env, resolve_algo_kwargs,
)

# 1. per-tick, online
full = make_online_base_gym_env(9, 30, 400, "lognormal", seed=3, use_resampler=False)
env = RuleSelectionGymSchedulingEnv(full, decision_epoch="tick")
env.reset()
for _ in range(5):  # advance into the episode so several jobs are waiting
    env.step(env.num_rules)
t0, waiting = env.env.time, len(env._job_actions())
assert waiting > 1, "test setup: need several feasible placements in one tick"
lst = env.rule_names.index("LST")
_, r_tick, _, _, _ = env.step(lst)
assert env.env.time == t0 + 1, ("per-tick action must end the tick", t0, env.env.time)

# same state, per-placement: apply LST until the tick is full, then idle; rewards must sum the same
full2 = make_online_base_gym_env(9, 30, 400, "lognormal", seed=3, use_resampler=False)
env2 = RuleSelectionGymSchedulingEnv(full2, decision_epoch="placement")
env2.reset()
for _ in range(5):
    env2.step(env2.num_rules)
total = 0.0
while env2._job_actions() and env2.env.time == t0:
    total += env2.step(lst)[1]
if env2.env.time == t0:
    total += env2.step(env2.num_rules)[1]
assert np.isclose(total, r_tick), (total, r_tick)
assert np.array_equal(np.asarray(env.env.start_times), np.asarray(env2.env.start_times))
print(f"  1. per-tick: one action placed {waiting}-feasible tick and advanced time; reward = summed placements")

# 2. offline equivalence
starts = []
for mode in ("placement", "tick"):
    e = RuleSelectionGymSchedulingEnv(make_base_gym_env(seed=5), decision_epoch=mode)
    e.reset()
    done, k = False, 0
    while not done and k < 500:
        _, _, done, trunc, _ = e.step(e.rule_names.index("EDF"))
        done, k = done or trunc, k + 1
    starts.append(np.asarray(e.env.start_times).copy())
assert np.array_equal(*starts), "offline per-tick must equal per-placement"
print("  2. offline: per-tick and per-placement schedules are identical")

# 3. Option 0 both architectures
for arch in ("flat", "pointer"):
    env0, policy, kw = build_env_and_policy("0", full_gym_env=make_base_gym_env(seed=1), policy_arch=arch)
    assert env0.action_space.n == make_base_gym_env(seed=1).action_space.n, "Option 0 must keep the full action space"
    assert (policy == "MlpPolicy") == (arch == "flat"), (arch, policy)
print("  3. Option 0 builds with flat and pointer architectures over the full action space")

# 4. algorithm settings
def ns(**k):
    base = dict(algo="ppo", learning_rate=None, rollout_size=None, batch_size=None, n_epochs=None,
                clip_range=None, gae_lambda=None, vf_coef=None, max_grad_norm=None)
    base.update(k)
    return types.SimpleNamespace(**base)

ppo, _ = resolve_algo_kwargs(ns(), 4)
assert (ppo["n_steps"], ppo["batch_size"], ppo["n_epochs"], ppo["learning_rate"], ppo["gae_lambda"]) == (512, 64, 10, 3e-4, 0.95), ppo
a2c, spec = resolve_algo_kwargs(ns(algo="a2c"), 4)
import torch  # noqa: E402
assert a2c["n_steps"] == 5 and a2c["batch_size"] == 20 and a2c["n_epochs"] == 1, a2c
assert a2c["normalize_advantage"] is False and a2c["gae_lambda"] == 1.0 and a2c["learning_rate"] == 7e-4
assert a2c["policy_kwargs_update"]["optimizer_class"] is torch.optim.RMSprop
over, _ = resolve_algo_kwargs(ns(learning_rate=1e-4, n_epochs=4), 4)
assert over["learning_rate"] == 1e-4 and over["n_epochs"] == 4, "explicit flags must override defaults"
print("  4. ppo defaults unchanged; a2c = 1 epoch x 1 full batch, RMSprop, lambda 1, no adv. normalisation")

print("test_v2_variants: all checks passed")
