# Design + Findings: Retrospective CP-SAT Oracle for the Online Case, and rho>1 Testing

**Date:** 2026-09-17 (S2W10)
**Environment:** `Code/baselines/exact_solver.py`
**Status:** Retrospective oracle implemented and validated (small-scale OPTIMAL,
large-scale correct-but-intractable-to-prove, matching the offline precedent). Two
real findings surfaced while validating rho>1 testing -- see Sections 3 and 4.

---

## 1. Motivation

The published online-case results artifact showed no CP-SAT reference line (unlike
every offline comparison) and every heuristic at exactly 0.00 tardiness even at
rho~0.95. The user asked why, and approved three follow-ups: build a retrospective
CP-SAT oracle for the online case, enable genuine rho>1 testing, and continue the
action-space-reduction work already in progress -- all three together, not
sequenced.

## 2. Retrospective CP-SAT oracle

CP-SAT cannot run *live* online (it needs the whole problem upfront). Once an
episode's arrival sequence is fully realized, though, it's exactly an offline
instance except each job also needs a new causality constraint: `start[j] >=
arrival_time[j]` -- a job cannot be scheduled before it has actually arrived, even
with hindsight.

`Code/baselines/exact_solver.py::solve()` gained an optional `earliest_start`
parameter (per-job start-time lower bounds; `None` default is unchanged behaviour
for every existing offline caller). `solve_retrospective()` (new) filters a
`generate_poisson_arrivals()`-style config down to realized arrivals, builds the
earliest-start array, and calls `solve()`. `make_retrospective_config()` (new)
factors out the filtering so the identical filtered instance can be replayed via
the existing `replay_schedule()` (which already auto-detects online configs via
`make_env()`, so no separate online replay function was needed).

Two exclusions happen before the model is built, both filtering out jobs no
scheduler -- oracle, heuristic, or RL -- could ever help:
- **Padding** (`arrival_time > horizon`): never actually arrived, matches
  `arrival_process.py`'s existing convention.
- **Unschedulable by horizon** (new): `arrival_time + duration > horizon`. The
  online arrival process's `deadline_slack_range` is relative to arrival, not
  bounded by the horizon, so a job arriving late enough simply cannot ever complete
  -- if left in the model, `solve()`'s "every included job must be scheduled"
  requirement (an existing, documented offline-case scope limitation) would make
  the *whole* model infeasible over one uncompletable job, rather than reporting a
  real bound. Excluding these doesn't advantage the oracle: nothing else could
  complete them either.

Validated at small scale (12 realized arrivals, rho~0.04): oracle reaches
**OPTIMAL, tardiness=0.00**, matching EDF/ATC exactly on the same instance. At
large/deployed scale (200-1400+ realized arrivals): matches the offline case's own
documented NP-hardness limit (Lenstra/Rinnooy Kan/Brucker 1977) -- `UNKNOWN` status
within short time limits, no proven bound yet. This mirrors `solve()`'s existing
`--fixed-instance` behaviour exactly and is not a new limitation; a longer
time-limit run (matching the offline case's own eventual "final" runs) is future
work, not attempted here.

### 2.1 A real bug found and fixed before it produced a wrong number

First implementation reused `solve()`'s existing `AddAllDifferent(start)`
constraint unchanged (one global clock, no two jobs start on the same tick) --
correct for the *offline* case, where `SchedulingEnv.step()` unconditionally
advances `self.time` after every action. It is **wrong** for the online case:
`OnlineSchedulingEnv` deliberately does NOT do this (Option 2's tick-advance
relaxation -- `step()` never advances time, only `step_idle()` does, specifically
so multiple placements can share a tick).

This wasn't caught by inspection -- it was caught by running the oracle at
realistic scale and getting a suspicious result: **`status=INFEASIBLE`** (not
`UNKNOWN`) at 200-300 realized jobs over `horizon=100`, while EDF/ATC, run through
the real env in the same script, scheduled every one of them with zero tardiness.
`INFEASIBLE` is a *proof*, not a timeout -- worth investigating rather than
shrugging off. `AddAllDifferent` forcing 200+ jobs into 100 distinct integer start
values is a textbook pigeonhole contradiction; it had nothing to do with the
earliest-start constraint and everything to do with reusing a constraint that
doesn't hold for this case. Fixed with a new `enforce_single_start_per_tick`
parameter on `solve()` (default `True`, unchanged for every offline caller;
`solve_retrospective()` passes `False`). After the fix, the same instance reports
`UNKNOWN` (not `INFEASIBLE`) -- the correct, expected status at this scale.

## 3. Finding: `max_jobs` must be sized for the WHOLE horizon, or rho>1 tests nothing real

`generate_poisson_arrivals()` stops drawing new arrivals once `max_jobs` total
slots are filled (its documented "phantom padding" convention). At high
`arrival_rate`, this cap is reached almost immediately: at `arrival_rate=14,
horizon=100, max_jobs=300`, all 300 slots filled by **tick 20**, leaving **80
further ticks with zero arrivals** -- an enormous, fully uncontested tail in which
any heuristic drains its entire backlog to zero tardiness. This produced a
misleading first result (0.00 tardiness at nominal rho~1.17) that had nothing to
do with genuine overload resilience and everything to do with an under-sized
`max_jobs` silently truncating the arrival *process* itself, not just the array.

Fix: no code change needed, just size `max_jobs` well above `arrival_rate *
horizon` (with margin for Poisson variance) so arrivals continue for the entire
horizon. At `max_jobs=2000` for the same `arrival_rate=14, horizon=100`, arrivals
are sustained through the full horizon (1411 realized, last arrival at tick 97 not
tick 20) -- a genuine sustained-overload instance.

**This changes the plan's own earlier framing** (`encapsulated-questing-cray.md`'s
"rho>1 testing... likely needs little to no new code" is only half right): the
mechanism needs no *env* code changes, but naive re-use of the `max_jobs` values
from rho<1 sweeps silently produces a fake overload test. Documented here so this
doesn't get rediscovered the hard way in the action-space-comparison sweep.

## 4. Finding: at rho>1, overload shows up as job ABANDONMENT, not scheduled-job tardiness

Re-running EDF/ATC on the corrected (`max_jobs=2000`) rho~1.17 instance:
**tardiness is still exactly 0.00** -- but only **1136/1145 of 1352 completable
jobs were ever scheduled at all** (~216/207 permanently abandoned by horizon end,
charged via `_finalize_unscheduled_job_cost()`, not via `tardiness`).

Mechanism (observed, not yet independently re-derived beyond this back-of-envelope
argument -- flagged per CLAUDE.md as an empirical finding, not a proof): EDF/ATC
always schedule the currently-most-urgent available job; deadline slack (10-60
ticks) is generous enough that whenever a job *does* get machine capacity, it is
essentially always still on time. Under sustained rho>1, the backlog of
*unserved* jobs grows roughly linearly (arrivals exceed the ~12-jobs/tick
throughput implied by this project's own `rho = arrival_rate/12` derivation) --
so the deficit shows up as an ever-growing queue that never gets its turn before
`horizon`, not as jobs being served late. **`tardiness=0.00` at rho>1 does not
mean "no problem" -- it means the wrong metric was being read.**
`jobs_scheduled`/completion-rate, already tracked by every eval path in this
project (`eval_rl_agent.py`'s `jobs_scheduled` field, added 2026-08-28), is the
metric that actually moves under genuine overload, not tardiness.

**Implication for the action-space-reduction comparison**: any future rho>1 online
comparison between Options 1/2/3 (or PPO/A2C generally) must report
scheduled/completion-rate alongside tardiness, or a policy that "solves" overload
by quietly abandoning the hardest jobs will look identical to one that doesn't, on
tardiness alone.

## 5. Next steps (not done here -- time-boxed alongside the action-space work)

- A longer-time-limit retrospective-oracle run at deployed scale, to get a real
  (non-zero, proven or `best_bound`) reference number for the rho>1 case, matching
  how the offline case's own "final" CP-SAT numbers were eventually obtained.
- Re-run the online 3-load sweep (`sweep_ppo_*`/`sweep_a2c_*` in
  `eval_results.csv`) with `max_jobs` sized correctly for sustained load, and with
  `jobs_scheduled` reported alongside tardiness, per Section 4.
- Consider exposing `deadline_slack_range` as a CLI-tunable lever (currently a
  fixed default in `generate_poisson_arrivals`) as a complementary way to force
  infeasibility even below rho=1 -- named in the original plan, not implemented
  here.

## 6. References

1. Lenstra, J. K., Rinnooy Kan, A. H. G., & Brucker, P. (1977). "Complexity of
   machine scheduling problems." *Annals of Discrete Mathematics*, 1, 343-362.
