"""test_schedule_metrics.py - Regression checks for Code/core/metrics.py::schedule_metrics
(2026-09-28, S2W11): every reported metric is checked against a hand-computed timeline,
plus consistency checks against the legacy stats on a real held-out instance.

Follows this project's tests/ convention (runnable script with asserts, not pytest).
Run from the repo root: python -m tests.test_schedule_metrics
"""
import numpy as np

from Code.core.scheduling_env import SchedulingEnv
from Code.core.metrics import schedule_metrics
from Code.core.env_config import generate_env_config
from Code.methods.rl.evaluation.eval_rl_agent import run_heuristic


def hand_checked_episode():
    # 3 jobs, 2 machines, capacity 3 per machine and each job needs 3 -> one job per machine at a time.
    env = SchedulingEnv(
        job_durations=np.array([2, 3, 2]),
        job_resources=np.array([[3], [3], [3]]),
        job_deadlines=np.array([2, 2, 20]),
        job_weights=np.array([1.0, 2.0, 1.0]),
        num_machines=2,
        machine_capacity=np.array([3.0]),
        horizon=6,
    )
    env.step((0, 0))  # t=0: job 0 on m0, runs ticks 0-1, C=2, d=2 -> on time
    env.step_idle()   # t=1: idle
    env.step((1, 1))  # t=2: job 1 on m1, runs ticks 2-4, C=5, d=2 -> T=3 (weighted 6)
    # job 2 (P=2) never started: dropped. Episode ends at the horizon.
    while env.time <= env.horizon:
        env.step_idle()
    return schedule_metrics(env)


m = hand_checked_episode()
expected = {
    "jobs_total": 3, "jobs_scheduled": 2, "dropped": 1,
    "completion_rate": 2 / 3, "on_time_rate": 1 / 3, "late_jobs": 1,
    "tardiness": 3.0, "weighted_tardiness": 6.0, "max_tardiness": 3.0,
    # dropped job 2: lower bound max(0, H + P - d) = max(0, 6 + 2 - 20) = 0
    "tardiness_with_dropped_lb": 3.0,
    "mean_wait": (0 + 2) / 2,            # start - arrival (offline arrival = 0)
    "mean_flow_time": (2 + 5) / 2,       # completion - arrival
    "mean_slowdown": (2 / 2 + 5 / 3) / 2,
    "makespan": 5.0,
    "active_machine_ticks": 2 + 3,       # m0 busy ticks 0-1, m1 busy ticks 2-4
    "mean_active_utilisation": 1.0,      # every active tick is at full capacity
}
for k, v in expected.items():
    assert abs(m[k] - v) < 1e-9, f"{k}: expected {v}, got {m[k]}"
print("  hand-checked timeline: all", len(expected), "metrics match")

# Consistency with the legacy stats on a real held-out instance (off_c_15 seed 500000, H=100).
cfg = generate_env_config(seed=500000, num_jobs=100, num_machines=10, horizon=100)
r = run_heuristic("EDF", config=cfg)
mm = r["metrics"]
assert mm["tardiness"] == float(r["tardiness"].sum())
assert mm["late_jobs"] == int(r["late_jobs"])
assert mm["jobs_scheduled"] == int(r["jobs_scheduled"])
assert mm["jobs_scheduled"] + mm["dropped"] == mm["jobs_total"] == 100
assert mm["mean_flow_time"] >= mm["mean_wait"] + 1   # flow = wait + duration, duration >= 1
assert mm["mean_slowdown"] >= 1.0
assert 0 < mm["active_machine_ticks"] <= 10 * 100
print(f"  EDF seed 500000: metrics consistent with legacy stats "
      f"(scheduled {mm['jobs_scheduled']}, dropped {mm['dropped']}, tardiness {mm['tardiness']:.0f})")
print("PASS test_schedule_metrics")
