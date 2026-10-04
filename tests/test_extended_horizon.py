"""test_extended_horizon.py - Checks for the extended-horizon mechanic (2026-09-29, S2W11; user
decision in Future/research/2026-09-28-objective-redesign-discussion.md sec. 4a): unfinished jobs
run past the preferred horizon H and pay their true lateness, instead of being dropped.

  1. Random policies (random valid actions incl. voluntary idles), offline and online: every real
     job is finished (no drops), the charged tardiness total equals sum_j w_j T_j over ALL jobs,
     reward == -J/c exactly (shaping is off under extension), and the within/past-horizon split
     of the metrics sums to the total.
  2. Heuristics: no drops on off_c_15; LST/EDF J equal their weighted tardiness.
  3. CP-SAT over the same extended window: optimum == replayed J (offline).
  4. Fixed-window mode and the legacy reward are untouched (EDF seed 500000: legacy 286.00).

Run from the repo root: python -m tests.test_extended_horizon
"""
import numpy as np

from Code.core.env_config import generate_env_config
from Code.core.arrival_process import generate_poisson_arrivals
from Code.core.objectives import ObjectiveConfig
from Code.core.scheduling_env import extension_needed
from Code.methods.rl.evaluation.eval_rl_agent import make_env, run_heuristic
from Code.methods.exact.exact_solver import solve, replay_schedule

EXT = {"reward_mode": "objective", "objective": ObjectiveConfig(), "extend_horizon": True}
TOL = 1e-6
rng = np.random.default_rng(7)

for seed in range(4):
    off = generate_env_config(seed=700 + seed, num_jobs=40, num_machines=2, horizon=30, job_weight_range=(1, 6))
    on = generate_poisson_arrivals(seed=700 + seed, arrival_rate=3.0, horizon=30, max_jobs=150,
                                   num_machines=2, job_weight_range=(1, 6))
    for label, cfg in (("offline", off), ("online", on)):
        env = make_env(cfg, EXT)
        base = env.env.env
        obs, info = env.reset()
        idle = env.env.max_jobs * base.num_machines
        total, done, truncated = 0.0, False, False
        while not (done or truncated):
            valid = np.flatnonzero(info["action_mask"])
            jobs = valid[valid != idle]
            a = idle if (len(jobs) == 0 or rng.random() < 0.3) else rng.choice(jobs)
            obs, r, done, truncated, info = env.step(a)
            total += r
        obj = base.objective
        start = np.asarray(base.start_times)
        real = obj.real
        assert not truncated, label
        assert np.all(start[real] != -1), f"{label} seed {seed}: an arrived job was never started"
        assert not obj.dropped.any(), label
        T = np.maximum(0, start + obj.P - obj.d)
        exp_T = float((obj.w * T)[real].sum())
        assert abs(obj.totals["weighted_tardiness"] - exp_T) < TOL, (label, obj.totals, exp_T)
        assert abs(total - (-exp_T / obj.c)) < TOL, (label, total, -exp_T / obj.c)
        from Code.core.metrics import schedule_metrics
        m = schedule_metrics(base)
        assert abs(m["weighted_tardiness_within_horizon"] + m["weighted_tardiness_past_horizon"]
                   - m["weighted_tardiness"]) < TOL, label
        assert m["dropped"] == 0 and m["completion_rate"] == 1.0, label
        assert base.horizon == base.preferred_horizon + extension_needed(cfg["job_durations"]), label
    print(f"  seed {seed}: offline+online -- all jobs finished, J exact "
          f"(online: {m['completed_past_horizon']} completed past H)")

cfg15 = generate_env_config(seed=500000, num_jobs=100, num_machines=10, horizon=100)
for h in ("LST", "EDF"):
    r = run_heuristic(h, config=cfg15, env_kwargs=EXT)
    assert r["metrics"]["dropped"] == 0, h
    assert abs(r["metrics"]["objective_J"] - r["metrics"]["weighted_tardiness"]) < TOL, h
print("  heuristics on off_c_15 seed 500000: no drops, J == weighted tardiness")

small = generate_env_config(seed=500001, num_jobs=20, num_machines=2, horizon=15)
H_ext = 15 + extension_needed(small["job_durations"])
res = solve(small, time_limit_seconds=30, num_search_workers=8, horizon_override=H_ext)
rep = replay_schedule(small, res["schedule"], EXT)
assert res["status"] in ("OPTIMAL", "FEASIBLE")
assert abs(res["objective"] - rep["metrics"]["objective_J"]) < TOL, (res["objective"], rep["metrics"]["objective_J"])
assert rep["metrics"]["dropped"] == 0
print(f"  CP-SAT extended ({res['status']}): optimum {res['objective']:.0f} == replayed J, "
      f"{rep['metrics']['completed_past_horizon']} jobs past H")

legacy = run_heuristic("EDF", config=cfg15)
assert abs(legacy["total_reward"] - 286.00) < 1e-6
fixed = run_heuristic("EDF", config=cfg15, env_kwargs={"reward_mode": "objective",
                                                      "objective": ObjectiveConfig(drop_shaping=False)})
assert fixed["metrics"]["dropped"] == 3, fixed["metrics"]["dropped"]
print("  legacy reward and fixed-window v2 unchanged (EDF: 286.00; 3 drops)")
print("PASS test_extended_horizon")
