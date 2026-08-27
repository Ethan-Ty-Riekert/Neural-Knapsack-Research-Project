# Exact-solver baseline via OR-Tools CP-SAT, and a structural finding about the environment

**Date:** 2026-08-28 (S2W6)

## 1. Motivation

Stage A (classical heuristics) and Stage B (PSO metaheuristic) both compare
methods against each other, but none of them can say "this is how far from
optimal we are." This project's own literature review
(`NotesForAI/ResearchProjectAsOf_23-07-2026.pdf`, p.5, p.40) names MILP/exact
solvers as the classical-optimal reference point for VM/resource-allocation
scheduling -- this is the third and final stage of the baseline suite the
user asked for (heuristics -> metaheuristic -> exact solver).

## 2. Why CP-SAT, not a raw MILP/PuLP formulation

Google OR-Tools' CP-SAT solver provides `NewOptionalIntervalVar` and
`AddCumulative` primitives that map directly onto this environment's
per-machine, per-resource-dimension capacity constraints -- a job occupies
a machine's resource capacity for an interval `[start, start+duration)`,
and `AddCumulative` natively enforces that the sum of concurrent demands on
one resource never exceeds that machine's capacity, exactly matching
`SchedulingEnv.is_feasible()`'s own check. A raw MILP time-indexed
formulation would need one binary variable per (job, machine, timestep)
triple and a big-M-style linearization for the same constraint; CP-SAT's
interval variables avoid both.

## 3. A structural finding about the environment, made while building this

**This environment does not model truly parallel machines with independent
clocks**, despite the "vector bin packing across `num_machines` machines"
framing used everywhere else in this project's documentation.
`SchedulingEnv.step()`/`step_idle()` (`Code/env/scheduling_env.py`)
increment `self.time` by exactly 1 after *every* accepted action -- a
placement or an idle step -- unconditionally. There is a **single global
decision clock**: at most one job can be *started*, system-wide, per tick,
regardless of how many machines happen to be idle at that moment. A
machine's resource capacity is still consumed for a job's full duration
starting at that tick (so `num_machines` still matters for how many jobs
can be *in flight* concurrently), but two jobs can never begin in the same
tick on two different machines.

This directly implies **this environment can schedule at most `horizon`
jobs in a single episode**, independent of `num_machines`. The deployed
fixed instance (100 jobs, horizon=100) sits exactly at that ceiling, with
zero slack for any idle tick if every job is to be scheduled -- a fact
that had never been made explicit anywhere in this project's prior 10
phases, despite every heuristic and RL policy compared so far already
being bound by it (they are all driven through this same env).

**Verification:** built a synthetic instance with `num_jobs=15 >
horizon=10` and confirmed the CP-SAT model reports `INFEASIBLE` (not a
crash or a silently-wrong answer) -- direct empirical confirmation of the
derived cap.

This meant the CP-SAT model had to bake this constraint in explicitly
(`model.AddAllDifferent(start)` over every job's start-time variable) to
be *replayable* through the real env at all -- a "true parallel machines"
relaxation without this constraint would find solutions the environment
physically cannot realise (confirmed the hard way: an early version without
`AddAllDifferent`-driven replay-gap-filling logic threw a desync error the
first time two jobs' optimal starts happened to be adjacent but not
consecutive -- fixed by having `replay_schedule()` insert explicit idle
steps to advance through any gap CP-SAT leaves between start times, since
gaps cost nothing in CP-SAT's tardiness-only objective but the real env has
no "skip ahead" action).

## 4. Formulation

For instance with jobs `J`, machines `M`, resources `R`, horizon `H`:

```
variables:
  start[j] in [0, H - duration[j]]                for each job j
  assign[j,m] in {0,1}                             for each job j, machine m
  interval[j,m] = OptionalInterval(start[j], duration[j], present=assign[j,m])
  tardiness[j] in [0, H]                           for each job j

constraints:
  sum_m assign[j,m] == 1                           for each job j  (every job scheduled once)
  AddAllDifferent(start)                            (single global decision clock, Section 3)
  AddCumulative(interval[.,m], demand=resource[.,r], capacity=machine_capacity[r])
                                                     for each machine m, resource r
  tardiness[j] >= start[j] + duration[j] - deadline[j]
  tardiness[j] >= 0

objective:
  minimize sum_j weight[j] * tardiness[j]
```

## 5. Scope and limitations (stated explicitly, per CLAUDE.md)

- **Small instances only.** Resource-constrained scheduling with tardiness
  objectives is strongly NP-hard (Lenstra, Rinnooy Kan, & Brucker, 1977) --
  this is the reason exact solving doesn't scale to the deployed 100-job
  instance, not a convenience choice. Ran on freshly-generated instances at
  `num_jobs=10, num_machines=3, horizon=15` (held-out seeds, same
  `RANDOM_INSTANCE_SEED_CEILING` convention as every other baseline).
- **Every job must be scheduled** (`sum_m assign[j,m] == 1`) -- unlike the
  RL/heuristic protocol, which permits leaving a job unscheduled. This
  makes the CP-SAT result an *oracle upper bound under full completion*,
  not a directly comparable "best of all possible behaviours" -- it answers
  "what's the best schedule if every job must finish," not "what's the
  best possible reward including the option to skip jobs."
- **Integer weights assumed.** `job_weights` are rounded to the nearest
  integer for CP-SAT's objective (every instance this project generates
  currently uses uniform weight 1.0 -- see the same honesty note in
  `2026-08-28-classical-heuristic-baselines.md` Section 3.1 -- so this has
  no effect today, but would need revisiting, e.g. integer-scaling by
  1000, if non-uniform non-integer weights are ever used).

## 6. Results

Ran on 5 small held-out instances (`num_jobs=10, num_machines=3,
horizon=15`, seeds 500000-500004), comparing CP-SAT against `EDF` and
`LST` (Stage A's two strongest heuristics) on the exact same instances:

| seed | CP-SAT reward | CP-SAT tardiness | EDF reward | EDF tardiness | LST reward | LST tardiness |
|---|---|---|---|---|---|---|
| 500000 | 75.13 | 0.00 | 20.50 | 0.00 | 20.50 | 0.00 |
| 500001 | 74.03 | 0.00 | 20.57 | 0.00 | 20.74 | 0.00 |
| 500002 | 74.77 | 0.00 | 77.00 | 0.00 | 77.22 | 0.00 |
| 500003 | 74.00 | 0.00 | 77.00 | 0.00 | 77.00 | 0.00 |
| 500004 | 73.63 | 0.00 | 21.86 | 0.00 | 21.86 | 0.00 |

Every solve completed in ~0.05s (well within the 60s time limit) and
reported `status=OPTIMAL`. CP-SAT's objective matched an independently
computed replay through the real env on every instance (unit-verified
before this run, see Section 3).

### Honest reading

**All three methods achieve zero tardiness on every instance here, yet
CP-SAT beats both heuristics on reward by a wide margin on 3 of 5 seeds
(500000, 500001, 500004) -- and the gap is not about lateness at all.**
Checked directly on seed 500000: EDF schedules only **9 of 10 jobs**, not
10 -- it gets greedily stuck (an earlier placement consumes resource
capacity a later, unscheduled job would have needed) even though a
globally-optimal ordering exists that completes all 10 jobs on time (which
CP-SAT finds, because `sum_m assign[j,m] == 1` forces it to). This is a
**different failure mode from the RCPO abandonment issue** documented
elsewhere in `training-log.md`'s 2026-08-28 entries: RCPO's policy
*chose* not to schedule jobs it could have scheduled, to game an
under-specified constraint; EDF/LST here *fail* to schedule every job
purely from greedy, no-lookahead resource packing, with no constraint
gaming involved. Both point at the same broader lesson though: **a
"beats EDF on tardiness" claim says nothing about completion rate** --
this project's evaluation should track jobs-scheduled alongside
reward/tardiness/late-jobs by default going forward, not just when a
result looks suspicious enough to investigate by hand (as it did twice
this session).

## 7. Conclusion / next step

The exact-solver baseline confirms both classical heuristics here already
recover the tardiness-optimal solution on these small instances (matching
CP-SAT on tardiness exactly, every time), but leaves real reward on the
table through incomplete scheduling. A useful, cheap follow-up (not done
tonight): run the current best RL checkpoints on these same small instances
(padding `max_jobs` to match their trained observation/action space size,
`GymSchedulingEnv`'s `max_jobs` decoupling already supports this) to see
whether the pointer network's learned policy also leaves jobs unscheduled
here, or whether it's closer to CP-SAT's full-completion behaviour than
either classical heuristic is.

## References

1. Perron, L., & Furnon, V. *OR-Tools*. Google.
   https://developers.google.com/optimization/cp/cp_solver -- CP-SAT solver
   and its interval-variable/cumulative-constraint modelling primitives.
2. Lenstra, J. K., Rinnooy Kan, A. H. G., & Brucker, P. (1977). "Complexity
   of machine scheduling problems." *Annals of Discrete Mathematics*, 1,
   343-362. -- NP-hardness of resource-constrained scheduling with
   tardiness objectives, the reason this baseline is restricted to small
   instances.
3. `NotesForAI/ResearchProjectAsOf_23-07-2026.pdf`, pp. 5, 40 -- this
   project's own literature review naming MILP/exact solvers as the
   classical-optimal baseline for VM/resource-allocation scheduling.
