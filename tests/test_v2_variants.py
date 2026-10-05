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

# 5. vectorised feasibility / ATC feature == the original Python loops, on many real states
from Code.methods.heuristics.priority_rules import atc_priority, atc_priorities, _atc_mean_p  # noqa: E402
from Code.methods.rl.action_spaces.priority_only_gym_wrapper import PriorityOnlyGymSchedulingEnv  # noqa: E402
from Code.methods.rl.action_spaces.windowed_priority_gym_wrapper import WindowedPriorityGymSchedulingEnv  # noqa: E402
from Code.methods.rl.action_spaces.action_branching_gym_wrapper import ActionBranchingGymSchedulingEnv  # noqa: E402


def loop_feasible(base, j):
    return [m for m in range(base.num_machines) if base.is_feasible(j, m, base.time)]


checked = 0
for online in (False, True):
    make = (lambda: make_online_base_gym_env(9, 30, 400, "lognormal", seed=11, use_resampler=False)) if online \
        else (lambda: make_base_gym_env(seed=11))
    for cls, kw in ((PriorityOnlyGymSchedulingEnv, dict(use_atc=True)),
                    (WindowedPriorityGymSchedulingEnv, dict(window_size=20, use_atc=True)),
                    (ActionBranchingGymSchedulingEnv, {})):
        w = cls(make(), **kw)
        w.reset()
        rng = np.random.default_rng(1)
        for step in range(120):
            base = w.env
            rem = list(base.remaining_jobs)
            if cls is PriorityOnlyGymSchedulingEnv:
                ref = np.zeros(w.max_jobs + 1, dtype=np.int8)
                for j in rem:
                    ref[j] = 1 if loop_feasible(base, j) else 0
                ref[w.max_jobs] = 1
                assert np.array_equal(w.get_action_mask(), ref), (online, step)
                for j in rem[:5]:
                    assert w._feasible_machines(j) == loop_feasible(base, j)
                obs = w._get_obs()  # ATC feature: last column of each job slot
                slots = obs[w._machine_block_end:].reshape(w.max_jobs, w._job_slot_width + 1)
                revealed = getattr(base, "revealed_jobs", None)
                mp = _atc_mean_p(base)
                for j in range(w.max_jobs):
                    real = (j < base.num_jobs) if revealed is None else (j in revealed)
                    expect = np.float32(np.clip(atc_priority(base, j, mean_p=mp), 0, 1)) if real else 0.0
                    assert slots[j, -1] == expect, (online, step, j)
            elif cls is ActionBranchingGymSchedulingEnv:
                m = w.get_action_mask()
                jm, mm = m[:w.max_jobs + 1], m[w.max_jobs + 1:]
                ref_j = np.array([1 if (j in base.remaining_jobs and loop_feasible(base, j)) else 0
                                  for j in range(w.max_jobs)] + [1], dtype=np.int8)
                ref_m = np.zeros(w.num_machines, dtype=np.int8)
                for j in rem:
                    for mach in loop_feasible(base, j):
                        ref_m[mach] = 1
                if not ref_m.any():
                    ref_m[:] = 1
                assert np.array_equal(jm, ref_j) and np.array_equal(mm, ref_m), (online, step)
            if rem:
                jobs = np.array(rem)
                assert np.allclose(atc_priorities(base, jobs), [atc_priority(base, j) for j in rem], rtol=1e-12, atol=0)
            mask = w.get_action_mask()
            a = w.action_space.sample() if cls is ActionBranchingGymSchedulingEnv else int(rng.choice(np.flatnonzero(mask)))
            _, _, done, trunc, _ = w.step(a)
            checked += 1
            if done or trunc:
                w.reset()
print(f"  5. vectorised masks / feasible machines / ATC feature equal the original loops on {checked} states")

# 6. work-conserving (non-delay) mode: idle masked iff some job is placeable; default unchanged
for online in (False, True):
    for cls, kw, idle_of in (
            (PriorityOnlyGymSchedulingEnv, dict(use_atc=True), lambda w, m: (m[:w.max_jobs], m[w.max_jobs])),
            (WindowedPriorityGymSchedulingEnv, dict(window_size=20, use_atc=True), lambda w, m: (m[:w.window_size], m[w.window_size])),
            (ActionBranchingGymSchedulingEnv, {}, lambda w, m: (m[:w.max_jobs], m[w.max_jobs])),
            (RuleSelectionGymSchedulingEnv, {}, lambda w, m: (m[:w.num_rules], m[w.num_rules]))):
        for restrict in (False, True):
            full = (make_online_base_gym_env(9, 30, 400, "lognormal", seed=2, use_resampler=False) if online
                    else make_base_gym_env(seed=2))
            full.restrict_idle = restrict
            w = cls(full, **kw)
            w.reset()
            saw_nothing_fits = False
            for _ in range(60):
                jobs, idle = idle_of(w, w.get_action_mask())
                expect_idle = 1 if (not restrict or not jobs.any()) else 0
                assert idle == expect_idle, (cls.__name__, online, restrict, jobs.any(), idle)
                saw_nothing_fits |= not jobs.any()
                m = w.get_action_mask()
                a = w.action_space.sample() if cls is ActionBranchingGymSchedulingEnv else int(np.flatnonzero(m)[0])
                if w.step(a)[2]:
                    break
print("  6. work-conserving mode masks idle exactly when a job is placeable (all wrappers); default unchanged")

# 7. Option 4 placement repair: a mismatched (job, machine) places the job by FirstFit; off = idle fallback
from Code.methods.heuristics.placement_rules import first_fit  # noqa: E402
for repair in (False, True):
    full = make_online_base_gym_env(9, 30, 400, "lognormal", seed=4, use_resampler=False)
    full.repair_placement = repair
    w = ActionBranchingGymSchedulingEnv(full)
    w.reset()
    found = False
    for _ in range(200):
        F = w._full.feasibility_matrix()
        # find a remaining job that fits somewhere but NOT on some machine m
        cand = [(j, m) for j in w.env.remaining_jobs for m in range(w.num_machines) if F[j].any() and not F[j, m]]
        if cand:
            j, m = cand[0]
            t0, expect_m = w.env.time, first_fit(w.env, j, np.flatnonzero(F[j]).tolist(), w.env.time)
            _, _, _, _, info = w.step(np.array([j, m]))
            assert info["mask_mismatch"]
            if repair:
                assert j not in w.env.remaining_jobs and w.env.time == t0, "repair must place the job, not idle"
                assert np.asarray(w.env.start_times)[j] == t0
            else:
                assert j in w.env.remaining_jobs and w.env.time == t0 + 1, "default must keep the idle fallback"
            found = True
            break
        fits = np.argwhere(F)
        if len(fits):  # load the machines (place a job on its first fitting machine) until one is full
            w.step(np.array([int(fits[0][0]), int(fits[0][1])]))
        else:
            w.step(np.array([w.max_jobs, 0]))  # nothing fits: idle so time and arrivals advance
    assert found, "test setup: no mismatch case found"
print("  7. Option 4 placement repair places a mismatched job by FirstFit; default keeps the idle fallback")

# 8. weight-aware rules (hand-computed), Option 1 menu unchanged, unweighted presets = same jobs with w=1
import dataclasses  # noqa: E402
from Code.methods.heuristics.priority_rules import wmdd_key, covert_key, PRIORITY_RULES  # noqa: E402
from Code.core.difficulty import DIFFICULTIES, generate  # noqa: E402
env_k = types.SimpleNamespace(job_durations=np.array([4, 2, 3]), job_deadlines=np.array([10, 5, 30]),
                              job_weights=np.array([2.0, 1.0, 5.0]), time=3)
# WMDD = max(p, d - t) / w:  job0 max(4,7)/2 = 3.5 ; job1 max(2,2)/1 = 2 ; job2 max(3,27)/5 = 5.4
assert [wmdd_key(env_k, j) for j in range(3)] == [3.5, 2.0, 5.4]
# COVERT priority = (w/p) * max(0, 1 - max(0, slack)/(2p)), slack = d - p - t:
#   job0 slack 3 -> (2/4)*(1 - 3/8) = 0.3125 ; job1 slack 0 -> (1/2)*1 = 0.5 ; job2 slack 24 -> 0
assert [covert_key(env_k, j)[0] for j in range(3)] == [-0.3125, -0.5, -0.0]
assert list(PRIORITY_RULES) == ["EDF", "SPT", "LST", "FCFS", "LPT", "WSPT", "ATC"], "Option 1 menu must not change"
for s in range(500000, 500005):
    a, b = generate(DIFFICULTIES["off_tf05"], s), generate(DIFFICULTIES["off_tf05_w1"], s)
    for key in ("job_durations", "job_resources", "job_deadlines"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert np.all(np.asarray(b["job_weights"]) == 1) and len(np.unique(a["job_weights"])) > 1
print("  8. WMDD/COVERT match hand-computed values; Option 1 menu unchanged; *_w1 presets = same jobs, w=1")

print("test_v2_variants: all checks passed")
