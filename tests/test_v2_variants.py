"""test_v2_variants.py - Checks for the v2 model-roster variants added 2026-10-05 (S2W12):

  1. Option 1 per-tick decisions: one action fills the tick with the chosen rule, then time advances;
     the transition reward equals the sum of the per-placement rewards over the same tick.
  2. Offline, per-tick and per-placement Option 1 produce identical schedules (each placement
     already advances time there).
  3. Option 0 (full action space) builds with both the flat and the pointer architecture.
  4. --algo a2c resolves to A2C-as-PPO (one epoch, one full batch, RMSprop, no advantage
     normalisation); --algo ppo keeps the exact previous defaults.

  (5-9: see the numbered sections below; 10: the uniform v2 protocol of tune_optuna_v2.)

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

# 9. Full-state observation (the report's MDP state): every job row carries (s_j, m_j) exactly as the
#    environment records them, every machine its y_m; online, unrevealed rows leak nothing; every action
#    design and its network runs on it (Code/core/obs_layout.py)
import torch as _torch  # noqa: E402
for online in (False, True):
    full = (make_online_base_gym_env(9, 30, 400, "lognormal", seed=6, use_resampler=False) if online
            else make_base_gym_env(seed=6))
    full.set_markov_obs()
    full.reset()
    L = full.obs_layout
    R, M = full.num_resources, full.num_machines
    assert L.machine_feat_dim == R + 1 and L.job_slot_width == R + 6
    assert full.observation_space.shape[0] == L.dim
    rng = np.random.default_rng(0)
    for step in range(60):
        obs = full._get_obs()
        base = full.env
        mach = obs[1:L.machine_block_end].reshape(M, R + 1)
        t_idx = min(base.time, base.horizon - 1)
        assert np.allclose(mach[:, :R], base.capacity[:, :, t_idx] / (full.initial_capacity + 1e-8), atol=1e-6)
        assert np.array_equal(mach[:, R], base.machine_active), "y_m must equal the env's activation vector"
        slots = obs[L.machine_block_end:].reshape(L.max_jobs, L.job_slot_width)
        visible = set(range(base.num_jobs)) if not online else set(base.revealed_jobs)
        for j in range(L.max_jobs):
            if j in visible and base.start_times[j] >= 0:
                assert np.isclose(slots[j, R + 4], base.start_times[j] / base.preferred_horizon, atol=1e-6)
                assert np.isclose(slots[j, R + 5], (base.job_machines[j] + 1) / M, atol=1e-6)
                assert 1 <= round(slots[j, R + 5] * M) <= M
            else:
                assert slots[j, R + 4] == 0 and slots[j, R + 5] == 0, "unstarted / unrevealed jobs use the 0 sentinel"
            if online and j not in visible:
                assert not slots[j, :R + 3].any(), "an unrevealed job must not leak its features"
        m = full.get_action_mask()
        full.step(int(rng.choice(np.flatnonzero(m))))
    assert (np.asarray(full.env.start_times) >= 0).any(), "test setup: some jobs must have started"
    for opt, kw in (("0", dict(policy_arch="flat")), ("0", dict(policy_arch="pointer")), ("1", {}),
                    ("2", {}), ("3", {}), ("3", dict(window_size=20)), ("4", {})):
        f2 = (make_online_base_gym_env(9, 30, 400, "lognormal", seed=6, use_resampler=False) if online
              else make_base_gym_env(seed=6))
        f2.set_markov_obs()
        env_o, policy, pkw = build_env_and_policy(opt, full_gym_env=f2, **kw)
        o, _ = env_o.reset()
        assert o.shape == env_o.observation_space.shape
        if policy != "MlpPolicy":
            from sb3_contrib import MaskablePPO
            model = MaskablePPO(policy, env_o, policy_kwargs=pkw, n_steps=8, batch_size=8, verbose=0)
            with _torch.no_grad():
                model.policy.predict(o, action_masks=env_o.env.action_masks())  # Monitor -> ActionMasker
print("  9. Full-state obs: s_j, m_j, y_m exact, no online leak; all designs and networks run on it")

# 10. Uniform v2 protocol (tune_optuna_v2): every option's trial / final run reaches the training script
#     with the full state, work-conserving mode and its own design flags; final tags parse as campaign tags
import run as _run  # noqa: E402
from Code.methods.rl.training import tune_optuna_v2 as tov  # noqa: E402
from Code.methods.rl.training import train_action_space_variant as tasv  # noqa: E402
from tools.campaign.common import TAG  # noqa: E402

for design in tov.OPTIONS:
    opt = tov.OPTIONS[design]["option"]
    for algo in tov.ALGOS:
        hp = {k: (v[1] if isinstance(v, tuple) else v[0]) for k, v in tov.SPACE[algo].items()}
        for preset in tov.FINAL_PRESETS:
            ns = _run.build_parser().parse_args(["--checkpoint-tag", "x",
                                                 *tov.run_args(design, algo, preset, 0, tov.TRIAL_STEPS, hp)])
            a = tasv.build_parser().parse_args(_run.rl_command(ns.variant, ns.preset, ns.method, ns)[3:])
            spec = tasv.env_spec_from_args(a)
            assert a.option == opt and a.algo == algo and a.timesteps == tov.TRIAL_STEPS and a.difficulty == preset
            assert spec["markov_obs"] and spec["work_conserving"] and a.learning_rate == hp["learning-rate"]
            assert spec["repair_placement"] == (opt == "4")
            assert spec["window_size"] == (20 if design == "3w" else None)
            assert spec["policy_arch"] == ("pointer" if opt == "0" else None)
            assert spec["placements"] == (["FirstFit", "Consolidate"] if opt == "1" else ["FirstFit"])
            for line in tov.final_job_lines(design, algo, preset, hp):
                m = TAG.match(line.split()[0])
                assert m and m["opt"] == opt and m["hp"] == "_tuned" and bool(m["algo"]) == (algo == "a2c")
                assert "m" in m["mods"] and "n" in m["mods"] and m["preset"] in tov.FINAL_PRESETS[preset]
print(" 10. Uniform protocol: every option x algorithm x preset carries its exact design flags")

# 11. Observation fixes and the critic-only input (2026-10-06)
#  a) the capacity look-ahead equals C_r - sum_{j in P_t, m_j = m, s_j <= t' < s_j + p_j} A_jr, computed from
#     F_t alone (so it adds no information); b) fixed scaling gives A_jr / C_r and p_j / H; c) the critic
#     block summarises only jobs not yet arrived (zero offline) and no design's actor depends on it.
from Code.core.critic_input import critic_input_dim, future_arrival_summary  # noqa: E402
from Code.core.difficulty import max_job_duration  # noqa: E402
from Code.methods.rl.training.train_action_space_variant import make_full_gym_env  # noqa: E402
from Code.variants.v2_objectives import objective_config  # noqa: E402

_obj = objective_config(("tardiness_sq",), None, 1.0)
for preset in ("on_rho095", "off_tf05"):
    d = DIFFICULTIES[preset]
    online = d.case == "online"
    K = max_job_duration(d)
    kw = dict(reward_mode="objective", objective=_obj, difficulty=d, extend_horizon=True, markov_obs=True,
              work_conserving=True, lookahead=K, fixed_scaling=True, critic_arrivals=True)
    kw.update(dict(arrival_rate=1.0, horizon=100, max_jobs=500, job_size_distribution="lognormal") if online
              else dict(randomize_instances=True))
    full = make_full_gym_env(online, kw, seed=4)
    rng = np.random.default_rng(0)
    full.reset()
    L, base = full.obs_layout, full.env
    M, R = full.num_machines, full.num_resources
    C = base.machine_capacity
    for step in range(60):
        obs = full._get_obs()
        t = base.time
        mblock = obs[1:L.machine_block_end].reshape(M, L.machine_feat_dim)
        running = np.flatnonzero((base.start_times >= 0) & (base.start_times + base.job_durations > t))
        for k in range(K + 1):
            expect = np.tile(C, (M, 1)).astype(float)
            for j in running:
                if base.start_times[j] <= t + k < base.start_times[j] + base.job_durations[j]:
                    expect[base.job_machines[j]] -= base.job_resources[j]
            got = mblock[:, :R] if k == 0 else mblock[:, R:R + R * K].reshape(M, R, K)[:, :, k - 1]
            assert np.allclose(got, expect / C, atol=1e-6), (preset, step, k)
        slots = obs[L.machine_block_end:].reshape(L.max_jobs, L.job_slot_width)
        vis = sorted(base.revealed_jobs) if online else range(full.num_jobs)
        for j in list(vis)[:5]:
            assert np.allclose(slots[j, 3:3 + R], base.job_resources[j] / C, atol=1e-6)
            assert np.isclose(slots[j, 0], base.job_durations[j] / base.preferred_horizon, atol=1e-6)
        z = future_arrival_summary(full)
        assert z.shape == (critic_input_dim(R),)
        if online:  # job count in the summary = jobs still to arrive within the arrival horizon
            n_future = ((base.arrival_times > t) & (base.arrival_times <= base.preferred_horizon)).sum()
            assert np.isclose(z.reshape(-1, 3 + R)[:, 0].sum() * M * 5, n_future, atol=1e-3)
        else:
            assert not z.any(), "offline: every job is known at t = 0, nothing is critic-only"
        m = full.get_action_mask()
        full.step(int(rng.choice(np.flatnonzero(m))))
    assert len(running) > 0 or step > 0
print(" 11. Look-ahead = capacity formula from F_t; fixed scaling; critic block = future arrivals only")

# 12. The unbiasedness precondition (Code/core/critic_input.py): pi(a | obs) must not depend on the critic-only
#     block for any design, while the value estimate does (otherwise the critic would not use it).
from sb3_contrib import MaskablePPO as _MPPO  # noqa: E402

d = DIFFICULTIES["on_rho095"]
kw = dict(reward_mode="objective", objective=_obj, difficulty=d, extend_horizon=True, markov_obs=True,
          work_conserving=True, lookahead=max_job_duration(d), fixed_scaling=True, critic_arrivals=True,
          arrival_rate=1.0, horizon=100, max_jobs=300, job_size_distribution="lognormal")
for opt, extra in (("0", dict(policy_arch="pointer")), ("1", {}), ("2", {}), ("3", {}), ("3", dict(window_size=20)),
                   ("4", {})):
    full = make_full_gym_env(True, kw, seed=2)
    env_o, policy, pkw = build_env_and_policy(opt, full_gym_env=full, **extra)
    model = _MPPO(policy, env_o, policy_kwargs=pkw, n_steps=8, batch_size=8, verbose=0)
    o, info = env_o.reset()
    for _ in range(20):
        o, _, _, _, info = env_o.step(model.predict(o, action_masks=info["action_mask"])[0])
    C = critic_input_dim(full.num_resources)
    x = _torch.as_tensor(o[None])
    y = x.clone()
    y[:, -C:] = _torch.randn(1, C) * 5
    dist = model.policy.get_distribution

    def logits(z):
        dd = dist(z)
        return (_torch.cat([q.logits for q in dd.distributions], -1) if hasattr(dd, "distributions")
                else dd.distribution.logits)
    with _torch.no_grad():
        assert _torch.equal(logits(x), logits(y)), f"Option {opt}: the actor reads the critic-only block"
        assert not _torch.allclose(model.policy.predict_values(x), model.policy.predict_values(y)), opt
print(" 12. Every design's actor ignores the critic-only block; every critic uses it")

# 13. Lateness shaping (Code/core/objectives.py). (a) Policy invariance (Ng et al. 1999): with gamma = 1 the
#     shaped return satisfies  sum_t r_t * c + J = -Phi(s_0)  for EVERY policy, the same constant, so the
#     shaping cannot prefer any policy. (b) An idle tick offline gives exactly
#     -sum_{waiting j} w_j [(t+1+p_j-d_j)_+^2 - (t+p_j-d_j)_+^2]: the extra lateness the delay causes, and
#     nothing for running jobs or for waiting jobs that can still finish on time.
_shaped = objective_config(("tardiness_sq",), None, 1.0, lateness_shaping=True)
_shaped.shaping_gamma = 1.0
for preset in ("off_tf05", "on_rho095"):
    d = DIFFICULTIES[preset]
    online = d.case == "online"
    kw = dict(reward_mode="objective", objective=_shaped, difficulty=d, extend_horizon=True, markov_obs=True)
    kw.update(dict(arrival_rate=1.0, horizon=100, max_jobs=300, job_size_distribution="lognormal") if online
              else dict(randomize_instances=False))
    constants, n_idle_checked = [], 0
    for policy_seed, p_idle in ((0, 0.05), (1, 0.3), (2, 0.6)):
        full = make_full_gym_env(online, kw, seed=7)
        rng = np.random.default_rng(policy_seed)
        full.reset()
        base, ob = full.env, full.env.objective
        phi0, total, done = ob.phi, 0.0, False
        idle = full.max_jobs * full.num_machines
        while not done:
            m = full.get_action_mask()
            placeable = np.flatnonzero(m[:idle])
            take_idle = (not placeable.size) or rng.random() < p_idle
            if take_idle and not online:
                t = base.time
                P, dd, w = (np.asarray(x, float) for x in (base.job_durations, base.job_deadlines, base.job_weights))
                wait = np.zeros(len(P), bool)
                wait[list(base.remaining_jobs)] = True
                expect = -(w * (np.maximum(0, t + 1 + P - dd) ** 2 - np.maximum(0, t + P - dd) ** 2))[wait].sum()
            _, r, done, _, _ = full.step(idle if take_idle else int(rng.choice(placeable)))
            total += r
            if take_idle and not online and not done:
                assert np.isclose(r * ob.c, expect, rtol=1e-9, atol=1e-6), (r * ob.c, expect)
                n_idle_checked += 1
        constants.append((total * ob.c + ob.objective_value(), -phi0))
    for got, want in constants:
        assert np.isclose(got, want, rtol=1e-9, atol=1e-4), (preset, got, want)
    assert np.isclose(constants[0][1], constants[1][1]) and np.isclose(constants[0][1], constants[2][1])
    assert online or n_idle_checked > 10
print(" 13. Lateness shaping: shaped return = -J/c - Phi(s0)/c for every policy; idle penalty = delay's lateness")

# 14. The 2026-10-07 fixes. (a) Two-stage greedy rule: idle only if pi(idle) >= total probability of starting a
#     job; otherwise the most probable job action -- the 2026-10-07 collapse case (idle 0.003 vs 0.002 per action)
#     no longer idles, confident policies are unchanged. (b) Reward scaling divides rewards by a running return std
#     without clipping: the scaled rewards are the raw rewards times one positive factor per step.
from Code.methods.rl.evaluation.greedy_decoding import two_stage_choice  # noqa: E402
from Code.methods.rl.training.train_action_space_variant import scale_rewards  # noqa: E402

collapse = np.full(1001, (1 - 0.003) / 1000)
collapse[-1] = 0.003
assert np.argmax(collapse) == 1000 and two_stage_choice(collapse) != 1000
confident = np.array([0.01, 0.9, 0.04, 0.05])
assert two_stage_choice(confident) == int(np.argmax(confident)) == 1
waiting = np.array([0.1, 0.2, 0.7])
assert two_stage_choice(waiting) == 2 and two_stage_choice(np.array([1.0])) == 0
full = make_full_gym_env(False, dict(reward_mode="objective", objective=_obj, difficulty=DIFFICULTIES["off_tf05"],
                                     extend_horizon=True, markov_obs=True, randomize_instances=False), seed=3)
env_o, _, _ = build_env_and_policy("2", full_gym_env=full)
venv = scale_rewards(env_o, 0.99)
assert venv.clip_reward == np.inf and not venv.norm_obs
venv.reset()
raw, scaled = [], []
for _ in range(40):
    m = venv.env_method("action_masks")[0]
    _, r, _, info = venv.step(np.array([int(np.flatnonzero(m)[0])]))
    scaled.append(float(r[0]))
    raw.append(float(venv.get_original_reward()[0]))
ratios = [s / q for s, q in zip(scaled, raw) if abs(q) > 1e-12]
assert ratios and all(x > 0 for x in ratios), "scaling must keep every reward's sign (a positive factor)"
print(" 14. Two-stage greedy rule fixes the idle collapse, keeps confident choices; reward scaling unclipped, positive")

# 15. Event-driven idling (GymSchedulingEnv.idle_step). A policy that idles WHENEVER idle is offered still finishes,
#     every idle decision ends exactly at the next event (a completion or an arrival), idle is never offered when no
#     event is left, and the lateness shaping stays exactly potential-based per decision even when one idle spans
#     several ticks: reward * c = -(J increment) + gamma * Phi(s') - Phi(s), here with gamma = 0.9.
import copy as _copy  # noqa: E402

_sh = objective_config(("tardiness_sq",), None, 1.0, lateness_shaping=True)
_sh.shaping_gamma = 0.9
for preset in ("off_tf05", "on_rho095"):
    d = DIFFICULTIES[preset]
    online = d.case == "online"
    kw = dict(reward_mode="objective", objective=_copy.deepcopy(_sh), difficulty=d, extend_horizon=True,
              markov_obs=True, event_idle=True)
    kw.update(dict(arrival_rate=1.0, horizon=100, max_jobs=300, job_size_distribution="lognormal") if online
              else dict(randomize_instances=False))
    full = make_full_gym_env(online, kw, seed=5)
    full.reset()
    base, ob = full.env, full.env.objective
    idle, rng, done, n_idle, multi = full.max_jobs * full.num_machines, np.random.default_rng(0), False, 0, 0
    while not done:
        m = full.get_action_mask()
        nxt = full.next_event_time()
        if base.remaining_jobs and nxt is None:
            assert not m[idle], "idle offered although no event can change the state"
        if m[idle]:
            t0, phi0, J0 = base.time, ob.phi, ob.objective_value()
            _, r, done, _, _ = full.step(idle)
            n_idle += 1
            if not done:
                assert base.time == (nxt if nxt is not None else t0 + 1), (base.time, nxt)
                multi += base.time - t0 > 1
                assert np.isclose(r * ob.c, -(ob.objective_value() - J0) + 0.9 * ob.phi - phi0, rtol=1e-9, atol=1e-6)
        else:
            _, r, done, _, _ = full.step(int(rng.choice(np.flatnonzero(m[:idle]))))
        assert n_idle < 10 * full.max_jobs, "idle decisions must be bounded by the number of events"
    assert multi > 0, "test setup: some idle decision should span several ticks"
print(" 15. Event-driven idling: waits end at the next event, never loop; shaping exact per multi-tick decision")

# 16. Weight-aware variants of the best rules (2026-10-07), hand-computed at t = 3:
#     job   p   d   w   slack d-p-t   d-t   lateness if started now t+p-d
#      0    4  10   2        3          7         -3
#      1    2   5   1        0          2          0
#      2    3  30   5       24         27        -24
#      3    2   4   3       -1          1          1   (already late: heavier must mean MORE urgent)
from Code.methods.heuristics.priority_rules import wlst_key, wedf_key, mdc_key, ALL_PRIORITY_RULES  # noqa: E402
from Code.methods.heuristics.registry import HEURISTICS  # noqa: E402

env_w = types.SimpleNamespace(job_durations=np.array([4, 2, 3, 2]), job_deadlines=np.array([10, 5, 30, 4]),
                              job_weights=np.array([2.0, 1.0, 5.0, 3.0]), time=3)
# WLST: slack / w while >= 0, slack * w once negative -> 1.5, 0, 4.8, -3
assert [wlst_key(env_w, j)[0] for j in range(4)] == [1.5, 0.0, 4.8, -3.0]
# WEDF: (d - t) / w -> 3.5, 2.0, 5.4, 1/3
assert np.allclose([wedf_key(env_w, j)[0] for j in range(4)], [3.5, 2.0, 5.4, 1 / 3])
# MDC: -w[(late+1)_+^2 - late_+^2] / p -> job1 -1/2, job3 -3(4-1)/2 = -4.5, jobs 0 and 2 zero (tie -> least slack)
assert [mdc_key(env_w, j)[0] for j in range(4)] == [-0.0, -0.5, -0.0, -4.5]
for key in (wlst_key, wedf_key, mdc_key):
    assert sorted(range(4), key=lambda j: key(env_w, j))[0] == 3, key.__name__  # the late heavy job first
assert all(f"{r}+{pl}" in HEURISTICS for r in ("WLST", "WEDF", "MDC") for pl in ("FirstFit", "Consolidate"))
assert list(PRIORITY_RULES) == ["EDF", "SPT", "LST", "FCFS", "LPT", "WSPT", "ATC"] and "MDC" in ALL_PRIORITY_RULES
print(" 16. WLST / WEDF / MDC match hand-computed keys; a late heavy job ranks first; registry has them")

print("test_v2_variants: all checks passed")
