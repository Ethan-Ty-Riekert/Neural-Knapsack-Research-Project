"""exact_solver.py - Exact (provably optimal, given a time limit) baseline
via OR-Tools CP-SAT, for small instances only.

Grounded in:
- Google OR-Tools CP-SAT solver (Perron, L., & Furnon, V. "OR-Tools",
  https://developers.google.com/optimization/cp/cp_solver) -- interval
  variables + AddCumulative map directly onto this environment's per-
  machine, per-resource-dimension capacity constraints.
- This project's own literature review (NotesForAI/ResearchProjectAsOf_23-
  07-2026.pdf, p.5, p.40) names MILP/exact solvers as the classical-optimal
  reference point for VM/resource-allocation scheduling.
- Lenstra, J. K., Rinnooy Kan, A. H. G., & Brucker, P. (1977). "Complexity
  of machine scheduling problems." Annals of Discrete Mathematics, 1,
  343-362 -- resource-constrained scheduling with tardiness objectives is
  strongly NP-hard, which is *why* this baseline is restricted to small
  instances (not merely a convenience choice).

IMPORTANT MODELLING DERIVATION (2026-08-28, S2W6) -- this environment does
NOT model truly parallel machines with independent clocks, despite the
"vector bin packing across M machines" framing used everywhere else in this
project. Re-reading `SchedulingEnv.step()`/`step_idle()`
(Code/env/scheduling_env.py) shows `self.time += 1` fires unconditionally
after EVERY accepted action (a placement or an idle step) -- there is a
single global decision clock, and at most one job can be *started* per
tick, system-wide, regardless of how many machines are idle at that tick.
A machine's per-resource capacity is still consumed for a job's full
duration starting at that tick (so `num_machines` still matters for how
many jobs can be *in flight* at once), but two jobs can never begin in the
same tick on two different machines. This directly implies **this
environment can schedule at most `horizon` jobs in a single episode**,
independent of `num_machines` -- the deployed fixed instance (100 jobs,
horizon=100) sits exactly at that ceiling, with zero slack for any idle
tick if every job is to be scheduled. Every heuristic/RL policy compared in
this project is already bound by this same constraint (they're all driven
through the same env); this note exists because the CP-SAT model below must
bake it in explicitly (`AddAllDifferent` on start times) to be a *fair*,
directly-replayable comparison rather than a mathematically-looser "true
parallel machines" relaxation whose optimum this environment could not
actually realise.

Because every job is required to be scheduled (a boolean "leave it
unscheduled" choice is not modelled here -- see the "Scope/limitations"
note in the dated doc), this baseline should be read as "the best possible
schedule IF every job must be completed," an oracle upper bound distinct
from the RL/heuristic protocol which permits abandoning jobs.
"""
import numpy as np
from ortools.sat.python import cp_model

from Code.evaluation.eval_rl_agent import make_env


def solve(config, time_limit_seconds=60):
    """Build and solve the CP-SAT model for one instance. Returns a dict
    with solver status, objective value, and (if any solution was found)
    a schedule: list of (job, machine, start_time) sorted by start_time,
    ready to replay through the real env via replay_schedule().
    """
    job_durations = np.asarray(config["job_durations"])
    job_resources = np.asarray(config["job_resources"])
    job_deadlines = np.asarray(config["job_deadlines"])
    job_weights = np.asarray(config["job_weights"])
    machine_capacity = np.asarray(config["machine_capacity"])
    num_machines = int(config["num_machines"])
    horizon = int(config["horizon"])
    num_jobs = len(job_durations)
    num_resources = job_resources.shape[1]

    model = cp_model.CpModel()

    start = []
    assign = {}
    intervals_by_machine = [[] for _ in range(num_machines)]
    demands_by_machine = [[[] for _ in range(num_resources)] for _ in range(num_machines)]

    for j in range(num_jobs):
        dur = int(job_durations[j])
        latest_start = horizon - dur
        if latest_start < 0:
            # This job cannot possibly complete within the horizon at all --
            # matches SchedulingEnv.is_feasible()'s own t+duration<=H check.
            raise ValueError(f"Job {j} (duration={dur}) cannot fit within horizon={horizon}")
        s = model.NewIntVar(0, latest_start, f"start_{j}")
        start.append(s)

        assign_j = []
        for m in range(num_machines):
            a = model.NewBoolVar(f"assign_{j}_{m}")
            assign_j.append(a)
            interval = model.NewOptionalIntervalVar(s, dur, s + dur, a, f"interval_{j}_{m}")
            intervals_by_machine[m].append(interval)
            for r in range(num_resources):
                demands_by_machine[m][r].append(int(job_resources[j, r]))
        model.Add(sum(assign_j) == 1)  # every job scheduled on exactly one machine
        assign[j] = assign_j

    # Single global decision clock (see module docstring): no two jobs may
    # start in the same tick, system-wide -- this is what makes the CP-SAT
    # solution directly replayable through the real, sequential env.
    model.AddAllDifferent(start)

    for m in range(num_machines):
        for r in range(num_resources):
            model.AddCumulative(intervals_by_machine[m], demands_by_machine[m][r], int(machine_capacity[r]))

    tardiness_vars = []
    for j in range(num_jobs):
        dur = int(job_durations[j])
        completion = start[j] + dur
        t = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.Add(t >= completion - int(job_deadlines[j]))
        model.Add(t >= 0)
        tardiness_vars.append(t)

    # job_weights are always 1.0 in every instance this project generates
    # (Code/env/env_config.py::generate_env_config) -- CP-SAT requires
    # integer objective coefficients, so weights are rounded to the nearest
    # integer here. Stated explicitly per CLAUDE.md rather than silently:
    # this would need revisiting (e.g. scaling by 1000 and dividing back)
    # if a future instance ever uses non-uniform, non-integer weights.
    weights = np.round(job_weights).astype(int)
    model.Minimize(sum(int(weights[j]) * tardiness_vars[j] for j in range(num_jobs)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    result = {"status": status_name, "objective": None, "schedule": None, "wall_clock_seconds": solver.WallTime()}

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result["objective"] = solver.ObjectiveValue()
        schedule = []
        for j in range(num_jobs):
            s = solver.Value(start[j])
            m = next(m for m in range(num_machines) if solver.Value(assign[j][m]))
            schedule.append((j, m, s))
        schedule.sort(key=lambda x: x[2])
        result["schedule"] = schedule

    return result


def replay_schedule(config, schedule):
    """Replay a CP-SAT schedule (list of (job, machine, start_time), sorted
    by start_time) through the real SchedulingEnv/GymSchedulingEnv, so its
    reward/tardiness/late-jobs are computed the exact same way as every
    other baseline in this comparison, and any accidental infeasibility in
    the CP-SAT model is caught for real rather than trusted blindly.
    """
    env = make_env(config)
    obs, info = env.reset()
    base_env = env.env.env
    num_machines = base_env.num_machines
    horizon = base_env.horizon
    idle_action = env.env.max_jobs * num_machines

    rewards = []
    utilisation_over_time = []
    initial_capacity = base_env.capacity[:, :, 0].copy()

    def _step(action):
        nonlocal obs, info
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))
        t_idx = min(base_env.time - 1, horizon - 1)
        used = initial_capacity - base_env.capacity[:, :, t_idx]
        utilisation_over_time.append(np.mean(used / (initial_capacity + 1e-8), axis=1))
        if truncated:
            raise RuntimeError("Replay truncated (too many invalid actions) -- CP-SAT schedule was not feasible.")

    for job, machine, planned_start in schedule:
        # AddAllDifferent only guarantees DISTINCT start times, not
        # consecutive ones -- CP-SAT has no reason to avoid gaps since idle
        # ticks are free in its objective. The real env has no "skip ahead"
        # action, so any gap must be filled with explicit idle steps to
        # advance base_env.time up to this job's planned start.
        while base_env.time < planned_start:
            _step(idle_action)
        if base_env.time != planned_start:
            raise RuntimeError(
                f"Replay desynced: env.time={base_env.time} overshot job {job}'s planned "
                f"start {planned_start}. AddAllDifferent should make every start time "
                f"unique -- this indicates a bug in the CP-SAT model, not an expected outcome."
            )
        _step(job * num_machines + machine)

    return {
        "total_reward": float(np.sum(rewards)),
        "tardiness": base_env.tardiness.copy(),
        "late_jobs": int((base_env.tardiness > 0).sum()),
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
        "utilisation_over_time": np.array(utilisation_over_time),
    }


def _main():
    """Run the CP-SAT baseline on a handful of small, freshly-generated
    instances (NOT the deployed 100-job/horizon=100 fixed instance -- see
    module docstring for why exact solving doesn't scale there), alongside
    EDF and LST (the two strongest classical heuristics found in Stage A)
    on the same instances for a direct small-scale comparison. Does not
    touch the shared ENV_CONFIG_PATH file, so this is safe to run even
    while a training run is concurrently active (see training-log.md's
    2026-08-28 hazard entry) -- unlike eval_rl_agent.py/pso.py's _main().
    """
    import argparse
    from Code.env.env_config import generate_env_config
    from Code.evaluation.eval_rl_agent import run_heuristic
    from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=5)
    parser.add_argument("--num-jobs", type=int, default=10)
    parser.add_argument("--num-machines", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--time-limit", type=float, default=60.0)
    args = parser.parse_args()

    print(f"CP-SAT baseline: {args.num_instances} instances, num_jobs={args.num_jobs}, "
          f"num_machines={args.num_machines}, horizon={args.horizon}, time_limit={args.time_limit}s\n")

    for i in range(args.num_instances):
        seed = RANDOM_INSTANCE_SEED_CEILING + i
        config = generate_env_config(seed=seed, num_jobs=args.num_jobs, num_machines=args.num_machines,
                                      horizon=args.horizon)
        config["max_jobs"] = args.num_jobs

        result = solve(config, time_limit_seconds=args.time_limit)
        if result["schedule"] is not None:
            replayed = replay_schedule(config, result["schedule"])
            cpsat_str = (f"reward={replayed['total_reward']:.2f} tardiness={replayed['tardiness'].sum():.2f} "
                         f"late={replayed['late_jobs']} (objective={result['objective']}, "
                         f"solve_time={result['wall_clock_seconds']:.2f}s, status={result['status']})")
        else:
            cpsat_str = f"NO SOLUTION FOUND (status={result['status']}, time_limit={args.time_limit}s)"

        edf = run_heuristic("EDF", config=config)
        lst = run_heuristic("LST", config=config)
        print(f"[seed {seed}] CP-SAT: {cpsat_str}")
        print(f"           EDF:    reward={edf['total_reward']:.2f} tardiness={edf['tardiness'].sum():.2f} late={edf['late_jobs']}")
        print(f"           LST:    reward={lst['total_reward']:.2f} tardiness={lst['tardiness'].sum():.2f} late={lst['late_jobs']}\n")


if __name__ == "__main__":
    _main()
