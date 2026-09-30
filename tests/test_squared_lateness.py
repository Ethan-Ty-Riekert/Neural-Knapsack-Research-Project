"""test_squared_lateness.py - Checks for the squared-lateness objective (2026-09-30, S2W11; the v2
default since the user chose it: a few very late jobs should cost more than many slightly late ones).

  1. Exactness of the dense decomposition: charging w_j (2k+1) on a job's k-th overdue tick sums to
     w_j T_j^2 -- checked on random policies, extended horizon and fixed window (with drops, where a
     dropped job counts as K_sq_j = w_j (max(0,H-d_j)+B)^2), offline and online; reward == -J/c
     (+ exact gamma=1 shaping in fixed-window mode).
  2. CP-SAT with lambda_squared: its optimum equals the replayed J (extended and fixed window).
  3. The objective now separates schedules with equal total lateness: two hand-built timelines with
     the same sum of T_j but different spread get different squared costs.

Run from the repo root: python -m tests.test_squared_lateness
"""
import numpy as np

from Code.core.env_config import generate_env_config
from Code.core.arrival_process import generate_poisson_arrivals
from Code.core.objectives import ObjectiveConfig
from Code.core.scheduling_env import SchedulingEnv, extension_needed
from Code.methods.rl.evaluation.eval_rl_agent import make_env
from Code.methods.exact.exact_solver import solve, replay_schedule

TOL = 1e-6
rng = np.random.default_rng(11)

for seed in range(4):
    for extend in (True, False):
        cfg_obj = ObjectiveConfig(tardiness=float(rng.choice([0.0, 1.0])), tardiness_sq=1.0,
                                  drop_shaping=not extend, shaping_gamma=1.0)
        off = generate_env_config(seed=800 + seed, num_jobs=30, num_machines=2, horizon=30, job_weight_range=(1, 6))
        on = generate_poisson_arrivals(seed=800 + seed, arrival_rate=2.5, horizon=30, max_jobs=120,
                                       num_machines=2, job_weight_range=(1, 6))
        for label, cfg in (("offline", off), ("online", on)):
            env = make_env(cfg, {"reward_mode": "objective", "objective": cfg_obj, "extend_horizon": extend})
            base = env.env.env
            obs, info = env.reset()
            phi0 = base.objective.phi
            idle = env.env.max_jobs * base.num_machines
            total, done, truncated = 0.0, False, False
            while not (done or truncated):
                valid = np.flatnonzero(info["action_mask"])
                jobs = valid[valid != idle]
                a = idle if (len(jobs) == 0 or rng.random() < 0.25) else rng.choice(jobs)
                obs, r, done, truncated, info = env.step(a)
                total += r
            o = base.objective
            start = np.asarray(base.start_times)
            fin, drop = o.real & (start != -1), o.real & (start == -1)
            T = np.maximum(0.0, start + o.P - o.d)
            exp_lin, exp_sq = float((o.w * T)[fin].sum()), float((o.w * T ** 2)[fin].sum())
            exp_D, exp_Dsq = float(o.K[drop].sum()), float(o.K_sq[drop].sum())
            tag = f"{label} seed {seed} extend={extend}"
            assert abs(o.totals["weighted_sq_tardiness"] - exp_sq) < TOL, (tag, o.totals, exp_sq)
            assert abs(o.totals["drop_cost_sq"] - exp_Dsq) < TOL, tag
            assert abs(o.totals["weighted_tardiness"] - exp_lin) < TOL, tag
            J = cfg_obj.tardiness * exp_lin + cfg_obj.tardiness_sq * (exp_sq + exp_Dsq) + cfg_obj.drops * exp_D
            shaping = cfg_obj.drops * (-phi0) / o.c if (cfg_obj.drop_shaping and not extend) else 0.0
            assert abs(total - (-J / o.c + shaping)) < 1e-6, (tag, total, -J / o.c + shaping)
            assert abs(o.objective_value() - J) < 1e-6, tag
    print(f"  seed {seed}: squared lateness exact (extended + fixed window, offline + online)")

# CP-SAT with squared lateness == replayed J
sq_only = ObjectiveConfig(tardiness=0.0, tardiness_sq=1.0, drops=0.0, drop_shaping=False)  # as v2 builds it
small = generate_env_config(seed=500001, num_jobs=16, num_machines=2, horizon=15)
H_ext = 15 + extension_needed(small["job_durations"])
res = solve(small, time_limit_seconds=30, num_search_workers=8, horizon_override=H_ext,
            lambda_linear=0, lambda_squared=1)
rep = replay_schedule(small, res["schedule"], {"reward_mode": "objective", "objective": sq_only,
                                               "extend_horizon": True})
assert abs(res["objective"] - rep["metrics"]["objective_J"]) < TOL, (res["objective"], rep["metrics"]["objective_J"])
res_fw = solve(small, time_limit_seconds=30, num_search_workers=8, drop_surcharge=15,
               lambda_linear=0, lambda_squared=1)
rep_fw = replay_schedule(small, res_fw["schedule"], {"reward_mode": "objective", "objective": sq_only})
assert abs(res_fw["objective"] - rep_fw["metrics"]["objective_J"]) < TOL, (res_fw["objective"],
                                                                            rep_fw["metrics"]["objective_J"])
print(f"  CP-SAT squared: extended {res['status']} {res['objective']:.0f} == replay; "
      f"fixed window {res_fw['status']} {res_fw['objective']:.0f} == replay")

# Same total lateness, different spread -> different squared cost (the point of the change)
def timeline(deadlines):
    env = SchedulingEnv(job_durations=np.array([2, 2]), job_resources=np.array([[3], [3]]),
                        job_deadlines=np.array(deadlines), job_weights=np.array([1.0, 1.0]),
                        num_machines=1, machine_capacity=np.array([3.0]), horizon=10,
                        reward_mode="objective", objective=ObjectiveConfig(tardiness=0.0, tardiness_sq=1.0,
                                                                           drop_shaping=False))
    total = env.step((0, 0))[1] + env.step_idle()[1]   # job 0: ticks 0-1, C = 2
    _, r, done = env.step((1, 0))                       # job 1: ticks 2-3, C = 4 -- last job, episode ends
    total += r
    assert done
    J = env.objective.totals["weighted_sq_tardiness"]
    assert abs(total - (-J / 2)) < 1e-9                  # reward = -J / c with c = 2 jobs
    return J, sum(np.maximum(0, np.array([2, 4]) - np.array(deadlines)))
spread = timeline([1, 3])     # T = (1, 1): total 2, squared 2
concentrated = timeline([2, 2])  # T = (0, 2): total 2, squared 4
assert spread[1] == concentrated[1] == 2 and (spread[0], concentrated[0]) == (2.0, 4.0), (spread, concentrated)
print("  equal total lateness (2 = 2): squared cost 2 (spread) vs 4 (one job very late)")
print("PASS test_squared_lateness")
