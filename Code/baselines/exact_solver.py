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


def solve(config, time_limit_seconds=60, num_search_workers=1, earliest_start=None,
          enforce_single_start_per_tick=True):
    """Build and solve the CP-SAT model for one instance. Returns a dict
    with solver status, objective value, a best-known lower bound, and (if
    any solution was found) a schedule: list of (job, machine, start_time)
    sorted by start_time, ready to replay through the real env via
    replay_schedule().

    best_bound (added 2026-09-14, S2W9): CP-SAT's branch-and-bound search
    maintains a provably-valid lower bound on the true optimum throughout
    the search, independent of whether it manages to prove optimality or
    find any feasible solution at all within time_limit_seconds --
    `solver.BestObjectiveBound()` exposes this. This matters for instances
    too large to solve to proven optimality (e.g. the deployed 100-job fixed
    instance -- this project's own 2026-08-28 finding restricted proven-
    optimal CP-SAT solving to ~10-job instances, since resource-constrained
    tardiness scheduling is strongly NP-hard, Lenstra/Rinnooy Kan/Brucker
    1977): even an UNKNOWN-status run (no feasible solution found in time,
    or found-but-not-proven-optimal) still yields a real, citable "true
    tardiness is at least this much" floor to compare a policy's result
    against, rather than only ever comparing against other heuristics/RL
    checkpoints with no absolute reference point.

    earliest_start (added 2026-09-17, S2W10, for solve_retrospective() below):
    optional per-job array of earliest-legal start ticks -- every job's
    start[j] domain becomes [earliest_start[j], horizon-duration] instead of
    [0, horizon-duration]. None (default, unchanged behaviour) means every
    job is available at t=0, matching every existing offline caller of this
    function exactly. This is the retrospective online case's causality
    constraint (x_jmt=0 for t < tau_j) applied with hindsight: even knowing
    the whole arrival sequence in advance, a job still cannot be scheduled
    before it has actually arrived.

    enforce_single_start_per_tick (added 2026-09-17, S2W10, for
    solve_retrospective()): True (default, unchanged for every existing
    offline caller) adds AddAllDifferent(start) -- correct there because the
    offline SchedulingEnv.step() unconditionally advances self.time by 1
    after every action, a single global clock (see module docstring). The
    ONLINE case's OnlineSchedulingEnv deliberately does NOT do this (Option
    2's tick-advance relaxation, mathformulation.tex): step() never advances
    time, only step_idle() does, specifically so multiple placements CAN
    share a tick. Keeping AddAllDifferent for the online retrospective
    oracle would impose an artificial one-job-per-tick ceiling the real
    online env never enforces -- confirmed empirically this session: at
    realistic online arrival scale (200-300 realized jobs over horizon=100)
    it makes the CP-SAT model PROVABLY INFEASIBLE by pigeonhole (more jobs
    than distinct tick values), while EDF/ATC, run through the real env,
    schedule every one of them with zero tardiness. Pass False for the
    online case; machine capacity (AddCumulative, below) remains the only
    real per-tick constraint, matching OnlineSchedulingEnv exactly.
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
        earliest = int(earliest_start[j]) if earliest_start is not None else 0
        if latest_start < earliest:
            # This job cannot possibly complete within the horizon at all
            # given when it arrives -- matches SchedulingEnv.is_feasible()'s
            # own t+duration<=H check, extended with the arrival-time floor.
            raise ValueError(
                f"Job {j} (duration={dur}, earliest_start={earliest}) cannot fit "
                f"within horizon={horizon}"
            )
        s = model.NewIntVar(earliest, latest_start, f"start_{j}")
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

    # Single global decision clock (see module docstring / this function's
    # enforce_single_start_per_tick docstring): no two jobs may start in the
    # same tick, system-wide, for the OFFLINE case only -- the ONLINE case's
    # OnlineSchedulingEnv permits concurrent same-tick placements (Option 2),
    # so this must be skipped there or the model becomes artificially
    # infeasible at realistic job counts.
    if enforce_single_start_per_tick:
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
    solver.parameters.num_search_workers = num_search_workers
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    result = {
        "status": status_name,
        "objective": None,
        "best_bound": solver.BestObjectiveBound(),
        "schedule": None,
        "wall_clock_seconds": solver.WallTime(),
    }

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
        # schedule is sorted by start_time, so this loop only ever needs to
        # move time FORWARD. The real env has no "skip ahead" action, so any
        # gap must be filled with explicit idle steps to advance
        # base_env.time up to this job's planned start. For an offline
        # replay (solve()'s default enforce_single_start_per_tick=True),
        # AddAllDifferent guarantees every start time is distinct, so this
        # loop runs at least once per job. For an online retrospective
        # replay (solve_retrospective(), enforce_single_start_per_tick=
        # False), several jobs can legitimately share one planned_start --
        # the online env's step() doesn't advance time on a placement
        # (Option 2), so the loop condition is already false for the 2nd+
        # job at that tick and it falls straight through to placing it,
        # exactly matching OnlineSchedulingEnv's own semantics.
        while base_env.time < planned_start:
            _step(idle_action)
        if base_env.time != planned_start:
            raise RuntimeError(
                f"Replay desynced: env.time={base_env.time} overshot job {job}'s planned "
                f"start {planned_start} -- this indicates a bug in the CP-SAT model or its "
                f"time-advance assumptions, not an expected outcome."
            )
        _step(job * num_machines + machine)

    return {
        "total_reward": float(np.sum(rewards)),
        "tardiness": base_env.tardiness.copy(),
        "late_jobs": int((base_env.tardiness > 0).sum()),
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
        "utilisation_over_time": np.array(utilisation_over_time),
    }


def solve_retrospective(config, time_limit_seconds=60, num_search_workers=1):
    """Retrospective CP-SAT oracle for the ONLINE (dynamic-arrival) case
    (2026-09-17, S2W10 -- see training-log.md's matching entry and
    Future/research/2026-09-17-action-space-reduction.md's "Added scope"
    section for the motivation). CP-SAT cannot run *live* online (it needs
    the whole problem upfront); this instead re-solves *after* an episode's
    arrival sequence is fully realized -- exactly the offline solve() above,
    except every job also carries the new earliest_start causality
    constraint (start[j] >= arrival_time[j]), since even with hindsight a
    job cannot be scheduled before it has actually arrived. This answers
    "what is the best any scheduler could have done, told the whole arrival
    sequence in advance, while still respecting causality" -- a genuine
    oracle upper bound for the online case, directly comparable to the
    offline case's solve()/best_bound role, and to what real online
    heuristics/RL (which must react without foreknowledge) actually achieve
    on the identical instance.

    config: a dict from Code.env.arrival_process.generate_poisson_arrivals()
    (or any dict with the same keys, including "job_arrival_times"). Padding
    job slots (arrival_time > horizon, never actually revealed -- see
    arrival_process.py's "phantom padding" convention) are filtered out
    before building the model; only realized arrivals participate.

    Returns the same result shape as solve(), plus "included_jobs": the
    original job-slot indices (into config's full max_jobs-sized arrays)
    that the returned schedule's job indices 0..n-1 refer to -- needed to
    interpret the schedule against the original config, and passed straight
    through to make_retrospective_config() for replay via replay_schedule()
    (which already auto-detects online configs via make_env(), so no
    separate online replay function is needed) -- and
    "unschedulable_by_horizon": count of realized-but-arrived-too-late jobs
    excluded for the reason explained next.

    A second, DIFFERENT exclusion from padding: a job that arrives at
    tau_j with duration p_j such that tau_j + p_j > horizon cannot be
    completed by ANY scheduler, no matter how it's sequenced -- the online
    arrival process (arrival_process.py's deadline_slack_range is relative
    to arrival, not bounded by the horizon) can and does generate these.
    solve()'s underlying model requires every included job to be scheduled
    (see its module docstring's "Scope/limitations" note -- already true,
    unchanged, for the offline case), which would make such a job's
    presence make the WHOLE model infeasible, not just that job abandoned.
    These are filtered out here before building the model, matching exactly
    how padding is filtered -- every real scheduler (heuristic, RL, or this
    oracle) is equally unable to complete them, so excluding them from the
    "must-schedule" requirement does not advantage the oracle; it only
    avoids modelling a choice ("abandon this job") that isn't actually a
    choice for anyone.
    """
    arrival_times = np.asarray(config["job_arrival_times"])
    durations = np.asarray(config["job_durations"])
    horizon = int(config["horizon"])
    arrived = arrival_times <= horizon
    completable = (arrival_times + durations) <= horizon
    included = np.where(arrived & completable)[0]
    unschedulable_by_horizon = int((arrived & ~completable).sum())

    if len(included) == 0:
        return {
            "status": "TRIVIAL_NO_ARRIVALS", "objective": 0.0, "best_bound": 0.0,
            "schedule": [], "included_jobs": included,
            "unschedulable_by_horizon": unschedulable_by_horizon, "wall_clock_seconds": 0.0,
        }

    sub_config, sub_arrival_times = make_retrospective_config(config, included)
    result = solve(sub_config, time_limit_seconds=time_limit_seconds,
                    num_search_workers=num_search_workers, earliest_start=sub_arrival_times,
                    enforce_single_start_per_tick=False)
    result["included_jobs"] = included
    result["unschedulable_by_horizon"] = unschedulable_by_horizon
    return result


def make_retrospective_config(config, included):
    """Build the filtered, realized-arrivals-only online config
    solve_retrospective() solves, and return the matching earliest_start
    array alongside it -- factored out so the exact same filtered instance
    can also be handed to replay_schedule() (via make_env()'s existing
    online auto-detection) for a real, replayable comparison against
    whatever online heuristic/RL result is being benchmarked against it.
    """
    arrival_times = np.asarray(config["job_arrival_times"])
    sub_config = {
        "job_durations": np.asarray(config["job_durations"])[included],
        "job_resources": np.asarray(config["job_resources"])[included],
        "job_deadlines": np.asarray(config["job_deadlines"])[included],
        "job_weights": np.asarray(config["job_weights"])[included],
        "job_arrival_times": arrival_times[included],
        "machine_capacity": np.asarray(config["machine_capacity"]),
        "num_machines": int(config["num_machines"]),
        "horizon": int(config["horizon"]),
        "max_jobs": len(included),
    }
    return sub_config, arrival_times[included]


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
    parser.add_argument(
        "--num-search-workers", type=int, default=1,
        help="CP-SAT parallel search workers (solver.parameters.num_search_workers). "
             "Default 1 (unchanged prior behaviour); pass e.g. os.cpu_count()-1 for a "
             "large instance (--fixed-instance) when the machine is otherwise idle."
    )
    parser.add_argument(
        "--fixed-instance", action="store_true",
        help="Target the real deployed fixed instance (generate_env_config(seed=0), "
             "num_jobs=100, num_machines=10, horizon=100 -- the exact instance every "
             "fixed-instance training/eval run in this project uses) instead of small "
             "freshly-generated held-out instances. Overrides --num-instances/--num-jobs/"
             "--num-machines/--horizon. Not expected to reach proven OPTIMAL status "
             "(NP-hard, this project's own finding capped proven-optimal solving at "
             "~10 jobs) -- read result['best_bound'] as a valid lower bound regardless "
             "of status, per solve()'s docstring."
    )
    parser.add_argument(
        "--online", action="store_true",
        help="Retrospective CP-SAT oracle for the ONLINE (dynamic-arrival) case "
             "(2026-09-17, S2W10) instead of the offline baseline above -- see "
             "solve_retrospective()'s docstring. Generates online instance(s) via "
             "Code.env.arrival_process.generate_poisson_arrivals and compares the "
             "oracle against EDF/ATC (the online-adapted heuristics) on the SAME "
             "realized instance. Overrides every offline-only flag above."
    )
    parser.add_argument(
        "--arrival-rate", type=float, default=8.0,
        help="--online only: mean Poisson arrival rate (jobs/tick). Implied utilisation "
             "rho = arrival_rate/12 under this project's default machine/resource "
             "parameters (num_machines=10, machine_capacity=30, num_resources=4, "
             "mean job duration/resource demand ~5.5 -- see mathformulation.tex's "
             "arrival-rate derivation). Default 8.0 -> rho~0.67 (comfortably below 1, "
             "a sanity-check default); pass e.g. --arrival-rate 14 for rho~1.17 "
             "(genuine overload -- see the 2026-09-17 plan's rho>1 testing scope)."
    )
    parser.add_argument(
        "--online-max-jobs", type=int, default=400,
        help="--online only: job-slot array capacity (arrival_process.py's fixed-array "
             "padding convention). Must comfortably exceed the expected arrival count "
             "(arrival_rate * horizon) or arrivals silently truncate -- sized generously "
             "here since --arrival-rate can be pushed above the rho<1 default."
    )
    parser.add_argument(
        "--online-horizon", type=int, default=100,
        help="--online only: matches the offline fixed instance's horizon=100 by default "
             "so tardiness numbers stay on a comparable scale."
    )
    parser.add_argument(
        "--online-seed", type=int, default=0,
        help="--online only: seed for generate_poisson_arrivals (one realized instance "
             "per seed)."
    )
    parser.add_argument(
        "--job-size-distribution", choices=["uniform", "lognormal"], default="uniform",
        help="--online only: 'lognormal' for the 2026-09-17 heavy-tailed job-size option "
             "(see Code/env/arrival_process.py's docstring / "
             "Future/research/2026-09-17-heavy-tailed-arrivals.md)."
    )
    args = parser.parse_args()

    if args.online:
        from Code.env.arrival_process import generate_poisson_arrivals
        from Code.evaluation.eval_rl_agent import run_heuristic

        rho = args.arrival_rate / 12.0
        config = generate_poisson_arrivals(
            seed=args.online_seed, arrival_rate=args.arrival_rate,
            horizon=args.online_horizon, max_jobs=args.online_max_jobs,
            job_size_distribution=args.job_size_distribution,
        )
        config["max_jobs"] = args.online_max_jobs
        n_realized = int((config["job_arrival_times"] <= args.online_horizon).sum())
        print(f"Retrospective CP-SAT oracle (online case): seed={args.online_seed}, "
              f"arrival_rate={args.arrival_rate} (rho~{rho:.2f}), horizon={args.online_horizon}, "
              f"{n_realized} realized arrivals (of {args.online_max_jobs} slots), "
              f"time_limit={args.time_limit}s\n")

        result = solve_retrospective(config, time_limit_seconds=args.time_limit,
                                      num_search_workers=args.num_search_workers)
        print(f"({result['unschedulable_by_horizon']} of {n_realized} realized arrivals are "
              f"unschedulable by ANY scheduler -- arrived too close to the horizon to ever "
              f"complete -- excluded from the oracle's must-schedule requirement)\n")
        if result["schedule"]:
            sub_config, _ = make_retrospective_config(config, result["included_jobs"])
            replayed = replay_schedule(sub_config, result["schedule"])
            oracle_str = (f"reward={replayed['total_reward']:.2f} tardiness={replayed['tardiness'].sum():.2f} "
                          f"late={replayed['late_jobs']} scheduled={replayed['jobs_scheduled']}/{len(result['included_jobs'])} "
                          f"(objective={result['objective']}, best_bound={result['best_bound']}, "
                          f"solve_time={result['wall_clock_seconds']:.2f}s, status={result['status']})")
        else:
            oracle_str = (f"NO SOLUTION FOUND (status={result['status']}, best_bound={result['best_bound']}, "
                          f"time_limit={args.time_limit}s)")

        edf = run_heuristic("EDF", config=config)
        atc = run_heuristic("ATC", config=config)
        print(f"Retrospective CP-SAT oracle: {oracle_str}")
        print(f"EDF (live online, no foreknowledge):  reward={edf['total_reward']:.2f} "
              f"tardiness={edf['tardiness'].sum():.2f} late={edf['late_jobs']} scheduled={edf['jobs_scheduled']}")
        print(f"ATC (live online, no foreknowledge):  reward={atc['total_reward']:.2f} "
              f"tardiness={atc['tardiness'].sum():.2f} late={atc['late_jobs']} scheduled={atc['jobs_scheduled']}")
        return

    if args.fixed_instance:
        print(f"CP-SAT baseline: real deployed fixed instance (seed=0, num_jobs=100, "
              f"num_machines=10, horizon=100), time_limit={args.time_limit}s\n")
        instances = [(0, generate_env_config(seed=0, num_jobs=100, num_machines=10, horizon=100))]
        instances[0][1]["max_jobs"] = 100
    else:
        print(f"CP-SAT baseline: {args.num_instances} instances, num_jobs={args.num_jobs}, "
              f"num_machines={args.num_machines}, horizon={args.horizon}, time_limit={args.time_limit}s\n")
        instances = []
        for i in range(args.num_instances):
            seed = RANDOM_INSTANCE_SEED_CEILING + i
            config = generate_env_config(seed=seed, num_jobs=args.num_jobs, num_machines=args.num_machines,
                                          horizon=args.horizon)
            config["max_jobs"] = args.num_jobs
            instances.append((seed, config))

    for seed, config in instances:
        result = solve(config, time_limit_seconds=args.time_limit, num_search_workers=args.num_search_workers)
        if result["schedule"] is not None:
            replayed = replay_schedule(config, result["schedule"])
            cpsat_str = (f"reward={replayed['total_reward']:.2f} tardiness={replayed['tardiness'].sum():.2f} "
                         f"late={replayed['late_jobs']} (objective={result['objective']}, "
                         f"best_bound={result['best_bound']}, "
                         f"solve_time={result['wall_clock_seconds']:.2f}s, status={result['status']})")
        else:
            cpsat_str = (f"NO SOLUTION FOUND (status={result['status']}, best_bound={result['best_bound']}, "
                         f"time_limit={args.time_limit}s)")

        edf = run_heuristic("EDF", config=config)
        lst = run_heuristic("LST", config=config)
        print(f"[seed {seed}] CP-SAT: {cpsat_str}")
        print(f"           EDF:    reward={edf['total_reward']:.2f} tardiness={edf['tardiness'].sum():.2f} late={edf['late_jobs']}")
        print(f"           LST:    reward={lst['total_reward']:.2f} tardiness={lst['tardiness'].sum():.2f} late={lst['late_jobs']}\n")


if __name__ == "__main__":
    _main()
