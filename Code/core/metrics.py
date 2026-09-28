"""metrics.py - reward-independent schedule metrics, computed from an episode's final env state.

Every method (heuristic, PSO, CP-SAT replay, RL) reports these, so methods can be compared
on what the project actually cares about regardless of which reward (v1 legacy, v2 objectives)
they were run under. Metric groups follow the evaluation criteria used in the fog/cloud
energy-scheduling literature -- energy consumption, latency, task completion time, quality of
service (Nagabushnam, Choi & Kim 2025, Cluster Computing 28:375) -- mapped onto this model:

  QoS / SLA     completion_rate, dropped, on_time_rate, late_jobs, tardiness (+ weighted, max, P95)
  latency       waiting time = start - arrival (this model has no network, so latency is pure
                queueing delay; fog transmission delay is out of scope until a topology exists)
  completion    flow time = completion - arrival (a.k.a. job completion time / response time),
                slowdown = flow time / duration (>= 1; normalises for job length -- DeepRM's
                objective, Mao et al. 2016), makespan
  energy        active machine-ticks (a machine is active at tick t if it runs >= 1 job): equal to
                energy up to constants under the linear power model P(u) = P_idle + (P_max-P_idle)u,
                since total work is fixed (Fan, Weber & Barroso 2007); mean utilisation of active
                machines (consolidation quality)

Conventions (see Future/research/2026-09-28-objective-redesign-discussion.md):
  - "Jobs" = jobs that arrived by the horizon (offline: all jobs).
  - Tardiness / waiting / flow statistics are over SCHEDULED jobs; dropped jobs are counted
    separately and also folded into on_time_rate (a dropped job is not on time) and into
    tardiness_with_dropped_lb, which charges each dropped job the lower bound
    max(0, H + P_j - d_j) (the tardiness it would have had if started at the horizon).
"""
import numpy as np


def schedule_metrics(env) -> dict:
    """Compute reward-independent metrics from a finished SchedulingEnv/OnlineSchedulingEnv."""
    H = int(env.horizon)
    start = np.asarray(env.start_times)
    dur = np.asarray(env.job_durations)
    dl = np.asarray(env.job_deadlines)
    w = np.asarray(env.job_weights, dtype=float)
    arrival = np.asarray(getattr(env, "arrival_times", np.zeros(len(start))))

    arrived = arrival <= H
    sched = (start != -1) & arrived
    dropped = arrived & (start == -1)
    n = int(arrived.sum())

    completion = np.where(sched, start + dur, -1)
    tard = np.where(sched, np.maximum(0, completion - dl), 0)
    drop_lb = np.where(dropped, np.maximum(0, H + dur - dl), 0)
    on_time = sched & (completion <= dl)

    wait = (start - arrival)[sched]
    flow = (completion - arrival)[sched]
    t_sched = tard[sched]

    used = env.machine_capacity[None, :, None] - env.capacity  # (M, R, H) resource-units in use
    active = (used > 1e-9).any(axis=1)                          # (M, H)
    util = (used / env.machine_capacity[None, :, None]).max(axis=1)  # bottleneck-resource utilisation

    def _stat(fn, x, default=0.0):
        return float(fn(x)) if len(x) else default

    return {
        # QoS / SLA
        "jobs_total": n,
        "jobs_scheduled": int(sched.sum()),
        "dropped": int(dropped.sum()),
        "completion_rate": _stat(np.mean, sched[arrived].astype(float), 1.0),
        "on_time_rate": _stat(np.mean, on_time[arrived].astype(float), 1.0),
        "late_jobs": int((t_sched > 0).sum()),
        "tardiness": float(t_sched.sum()),
        "weighted_tardiness": float((tard * w)[sched].sum()),
        "max_tardiness": _stat(np.max, t_sched),
        "p95_tardiness": _stat(lambda x: np.percentile(x, 95), t_sched),
        "tardiness_with_dropped_lb": float(t_sched.sum() + drop_lb.sum()),
        # latency (queueing delay)
        "mean_wait": _stat(np.mean, wait),
        "p95_wait": _stat(lambda x: np.percentile(x, 95), wait),
        # task completion time
        "mean_flow_time": _stat(np.mean, flow),
        "mean_slowdown": _stat(np.mean, flow / dur[sched]),
        "makespan": _stat(np.max, completion[sched]),
        # energy / consolidation
        "active_machine_ticks": int(active.sum()),
        "mean_active_utilisation": _stat(np.mean, util[active]),
    }
