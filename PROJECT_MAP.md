# Project Map (read this first)

*One page: what the project is, what has been found, and what is changing. Last updated 2026-09-28 (S2W11).
Detail lives in `Future/research/`, but you shouldn't need it to explain the project.*

## The 30-second version

Cloud data centres must decide **which machine runs each job, and when**. Machines have limited capacity,
and jobs have deadlines and importance. Simple rules like "earliest deadline first" are what's used in
practice. **Can a reinforcement-learning (RL) policy learn to schedule better, and under what conditions?**
To find out, RL is compared against heuristic rules, a search method (PSO) and an exact solver that proves
the true optimum on small cases.

## The problem

- **Jobs:** each needs 4 resources (CPU, memory, storage, network; resource 0 is *assumed* to be CPU), runs
  for 1–10 ticks, and has a deadline and a weight (importance).
- **Machines:** 10, each with capacity 30 per resource. A job fits only if every resource fits.
- **Offline setting:** 100 jobs, all known at the start, horizon H = 100 ticks, one job started per tick
  (a tick is one decision point). Simple, and it has a provable optimum to compare against.
- **Online setting:** jobs arrive at random, about 9 per tick (~75% load), with realistic heavy-tailed sizes
  and about 900 jobs per episode. Decisions are made without knowing the future, which is where learning
  *should* help.
- **Goal:** minimise lateness (tardiness) and dropped jobs; energy is to be added (the number of machines
  switched on over time).

## Methods compared

| Family | Methods | Role |
|---|---|---|
| Heuristic rules | EDF (earliest deadline), LST (least slack), ATC (lateness-cost rule), SPT, and others, each combined with a machine rule (first/best/worst fit) | The practical baseline |
| Metaheuristic | PSO (searches for a good priority order for each instance) | Heavy search, no learning |
| Exact | CP-SAT (constraint solver) | Proves the true optimum on small instances |
| RL | PPO and A2C (actor-critic policy-gradient methods), in 5 action designs: (0) choose the job × machine pair directly (1000+ choices); (1) **choose which heuristic rule to apply** each step (a *hyper-heuristic*); (2) output a priority score per job; (3) priority score starting from ATC; (4) job and machine as two separate choices | What's being studied |

## What has been found (all numbers verified; lower tardiness is better)

| Setting | Best result | Best RL | Verdict |
|---|---|---|---|
| Offline, one fixed instance | optimum **8** (CP-SAT, proven); LST also 8; EDF 16 | 16 (Option 1, ties EDF) | RL doesn't beat the best rule |
| Offline, 50 unseen instances | LST **23.9**, EDF 37.3 | none better without dropping more jobs | Simple rules are near-optimal here |
| Offline, weighted, 50 unseen | LST **68.6** weighted tardiness, EDF 109.4 | 96.3 (Option 3) | RL beats EDF, but not LST |
| Online, weighted, 50 unseen | ATC **648** weighted tardiness | 798 (Option 1, which learned to copy SPT) | RL doesn't beat the best rule |

**The biggest finding is a methodological one.** The reward used so far paid **+3 per job and +50 for
finishing everything**, and only charged lateness weakly. So methods could "win" on reward while scheduling
badly. For example, PSO beat EDF on reward by finishing every job very late. EDF looked perfect on
tardiness partly by *dropping* the jobs that would have been late, and the metric didn't count dropped
jobs. So several RL results were produced by a reward that pointed at the wrong target.

## What is changing now, and why

1. **The reward will equal the objective** (variant v2). It will be weighted lateness, plus a penalty for
   each dropped job that is provably worse than finishing it late, plus energy optionally, and nothing else.
   See `Future/research/2026-09-28-v2-objective-formal-definition.md`.
2. **Every method reports the same metrics:** QoS (on-time %, dropped, tardiness), latency (waiting time),
   completion time, and energy (machine-ticks switched on).
3. **One launcher (`run.py`)** runs any method on any preset, so every comparison is reproducible.
4. Old results are **kept**, as `v1_legacy_reward`. They are valid results *under that reward*.

## Where things are

| You want… | Look at |
|---|---|
| Run something | `python run.py` (menu) or `python run.py --list` |
| The problem code | `Code/core/`; methods in `Code/methods/` |
| All past numbers | `Results/v1_legacy_reward/ALL_RESULTS_*/` |
| Why a decision was made | `Future/research/2026-09-28-objective-redesign-discussion.md` |
| The full history | `PROGRESS.md` (story), `Future/research/training-log.md` (every run) |

## Research directions

The anchor question: *On a multi-resource cloud job-scheduling problem with deadlines, under which
objectives and workload conditions does a learned (RL) policy outperform classical heuristics, a
metaheuristic and an exact solver, and at what computational cost?*

1. **When does learning beat dispatching rules? (recommended headline).** Sweep the difficulty (load, how
   tight deadlines are, how varied the weights are) and the objective (lateness only, or lateness plus
   energy). Run every method on every combination and produce a "regime map" showing where RL wins, ties
   or loses. Current evidence (rules near-optimal offline, small gaps online) is a valid first data point.
   It needs v2 and the difficulty settings implemented.
2. **Reward misspecification in RL scheduling (strong secondary, mostly done).** Three different methods
   gamed the old reward: PSO finished everything late, a constrained RL variant abandoned half its jobs,
   and Option 1 collapsed to a single rule. Pair this with the formal fix (reward = objective, a provable
   drop penalty, shaping that doesn't change the optimum). The evidence already exists; it needs writing up.
3. **Lateness versus energy (extension).** A multi-objective trade-off. Energy-aware heuristic rules against
   RL, compared as Pareto fronts. It needs v2 with energy implemented.
4. **RL as a heuristic selector online (narrow, feasible).** Option 1 learning *which rule to use when*,
   as load changes. Early sign: a mix of SPT and LST beat EDF online (20 instances), but not ATC.
5. **Action-space design for RL schedulers.** The data already covers five action designs. Seven other PPO
   fixes left tardiness unchanged. Reducing the choice set was the first change that moved it.
6. **Jobs with dependencies (DAGs).** A later extension; the design already allows for it.
