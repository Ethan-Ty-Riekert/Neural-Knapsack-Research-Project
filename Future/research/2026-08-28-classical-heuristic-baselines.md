# Classical heuristic baselines: filling the machine-selection gap

**Date:** 2026-08-28 (S2W6)

## 1. Motivation

Ten phases into this project, the comparison baseline had stayed thin: only
three job-priority rules (EDF, SPT, LST), each just taking the first
feasible machine in ascending action-id order -- not a real machine-selection
rule. There was no First-Fit/Best-Fit/Worst-Fit bin-packing heuristic, no
multi-resource joint scorer, and no FCFS/LPT/WSPT despite these being named
directly in the project's own literature review
(`NotesForAI/ResearchProjectAsOf_23-07-2026.pdf`) as the baselines this
project's RL/NCO approach should be judged against ("Option B", p.7:
*"Baselines will include threshold autoscaling, heuristic placement
(FF/BF), and evolutionary RA methods"*; p.5's RA taxonomy lists Round
Robin/FCFS/SJF as classical scheduling techniques).

This doc covers the first of three stages the user asked for (heuristics,
then a metaheuristic, then an exact solver) to build the RL model's
benchmark out properly.

## 2. What existed before this change

`Code/evaluation/eval_rl_agent.py`'s `run_heuristic()`/`choose_action()`
implemented three job-priority rules -- EDF (min deadline), SPT (min
duration), LST (min slack) -- each picking from the mask-feasible
`(job, machine)` action ids in ascending `job * num_machines + machine`
order. Because action ids are built in that order, "pick the min action id
among a job's feasible machines" is *already* First-Fit by construction --
it was just never named or generalised as one, and no Best-Fit/Worst-Fit/
Tetris alternative existed to compare it against.

## 3. What was added

A new `Code/baselines/` package, decomposing every heuristic into a
job-priority rule (which job to schedule next) and a machine-placement rule
(which machine to put it on), composed automatically into named combos, plus
one joint scorer that doesn't decompose this way:

### 3.1 Job-priority rules (`Code/baselines/priority_rules.py`)

| Rule | Key | Citation |
|---|---|---|
| EDF | `min(deadline)` | Classical scheduling theory (e.g. Pinedo, *Scheduling: Theory, Algorithms, and Systems*) |
| SPT | `min(duration)` | Classical scheduling theory |
| LST | `min(deadline - duration - now)` | Classical scheduling theory |
| FCFS | `min(job index)` | NotesForAI RA taxonomy, p.5 |
| LPT | `max(duration)` | Graham (1969), "Bounds on Multiprocessing Timing Anomalies" -- proves the (4/3 - 1/(3m)) makespan approximation bound on identical parallel machines that justifies scheduling long jobs first |
| WSPT | `min(duration / weight)` | Smith's rule (W.E. Smith, 1956, "Various optimizers for single-stage production", *Naval Research Logistics Quarterly*) -- proven optimal for minimizing total weighted completion time on a single machine |

**Honesty note (per CLAUDE.md's math-rigor mandate):** every instance
`Code/env/env_config.py::generate_env_config` produces sets `job_weights`
uniformly to 1, so WSPT is numerically *identical* to SPT in every result
below. This is not a bug in WSPT's implementation -- it is a direct
consequence of no experiment in this project ever having exercised
non-uniform job weights. WSPT will only diverge from SPT once an instance
generator that actually varies `job_weights` is used.

### 3.2 Machine-placement rules (`Code/baselines/placement_rules.py`)

| Rule | Definition | Citation |
|---|---|---|
| First-Fit | first feasible machine by index | Johnson (1973), "Near-optimal bin packing algorithms", *JACM* |
| Best-Fit | feasible machine minimizing total remaining capacity (tightest fit) | Johnson (1973) -- multi-resource generalisation (sum across resource axes) of the classical 1-D rule |
| Worst-Fit | feasible machine maximizing total remaining capacity (load-spreading) | Johnson (1973) |
| Tetris (joint) | `argmax dot(job_resources, machine_available_capacity)` over `(job, machine)` pairs directly, not decomposed into priority+placement | Grandl et al., "Multi-resource packing for cluster schedulers", *SIGCOMM* 2014 -- **not** in the project's own literature review, so flagged here as an externally-sourced citation per CLAUDE.md's rule that literature-review gaps be stated explicitly rather than silently filled |

Composing every priority rule with every placement rule gives 18 named
combos (`Code/baselines/registry.py`'s `HEURISTICS` dict), plus `Tetris` and
a `Random` fallback -- 23 registered baselines' `PRIORITY_RULES` x
`PLACEMENT_RULES` in total. `EDF`/`SPT`/`LST` (bare names, no `+`) are kept
as aliases of their `+FirstFit` combo so every existing `eval_results.csv`
row and training-log reference to these three names keeps meaning exactly
what it always meant -- confirmed by re-running them post-refactor and
checking bit-for-bit identical reward/tardiness/late-jobs against the
pre-refactor numbers (see Verification below).

### 3.3 Wiring (`Code/evaluation/eval_rl_agent.py`)

- `choose_action()`'s inline if/elif dispatch was replaced with a lookup
  into `Code.baselines.registry.HEURISTICS` -- no other logic changed.
- `main()`'s hardcoded `"EDF"` comparison target became a `--heuristics`
  list argument (default: a curated 8-name subset covering every priority
  and placement rule at least once; `all` runs the full registered set).
- `evaluate_multiple()` gained a `model_runs=` passthrough so comparing one
  checkpoint against N heuristics in one invocation evaluates the model
  once and reuses those episodes for every heuristic comparison, instead of
  re-running N x the model rollouts for no reason.

## 4. Verification

Ran every registered heuristic against the fixed `ENV_CONFIG_PATH` instance
directly (bypassing any RL checkpoint) as a smoke test:
- All 23 heuristics complete an episode without error.
- `EDF`, `SPT`, `LST` (bare names) reproduce their pre-refactor numbers
  exactly and deterministically across repeated runs (no capacity-state
  leakage, no accidental randomness).
- `WSPT+*` results are numerically identical to the corresponding `SPT+*`
  results, confirming the uniform-weights honesty note above rather than
  masking it.
- `LST` (min slack, First-Fit) is the best-performing heuristic on the
  fixed instance by both tardiness and reward, ahead of `EDF` -- worth
  keeping in mind when reading "beats EDF" framing elsewhere in this
  project's docs: EDF is the traditionally-cited baseline, but it is not
  actually the strongest classical heuristic available here.

## 5. Results

Ran `python -m Code.evaluation.eval_rl_agent --heuristics all` (fixed
instance, n=50) and `--randomized-eval --heuristics all` (50 held-out
instances) for both the Phase 8 pointer+shaping checkpoint and the Phase 10
pointer+RCPO checkpoint, against all 23 registered heuristics. Full rows in
`rl_training/results/eval_results.csv` (tags `pointer_shaped_S2W6_baselines`,
`pointer_shaped_S2W6_baselines_heldout_randomized_eval`,
`pointer_rcpo_S2W6_baselines`, `pointer_rcpo_S2W6_baselines_heldout_randomized_eval`).
`MPLBACKEND=Agg` was required to run this in reasonable time -- the default
interactive backend blocks on `plt.show()` per plot with no display
attached, turning an ~18s heuristic comparison into ~2 minutes each; see
`Code/utils/plotting_utils.py::save_and_show()`.

### Held-out (50 unseen instances) -- headline comparison

| Method | reward | tardiness | late jobs |
|---|---|---|---|
| **LST** (best heuristic) | **288.28** | **23.94** | **8.26** |
| EDF | 284.95 | 37.30 | 12.22 |
| Tetris | 267.10 | 1320.18 | 42.48 |
| SPT / WSPT (identical, see 3.1) | 256.99-257.03 | 1150.86 | 37.74 |
| FCFS+FirstFit | 272.41 | 1299.52 | 42.06 |
| LPT+FirstFit | 332.26 | 1392.46 | 44.92 |
| Random | 264.92 | 1313.72 | 42.48 |
| **A2C pointer + shaping** | 254.75 | 28.66 | 9.56 |
| **A2C pointer + RCPO** | 135.67 | 19.84 | 3.20 |

(Fixed-instance results follow the same ranking and are within a few
percent of these; full table in the CSV.)

### Honest reading of these numbers

1. **LST, not EDF, is the strongest classical baseline here**, on every
   metric, on both the fixed instance and held-out. EDF is the
   traditionally-cited comparison point in this project's earlier docs
   (Phases 1-10), but it was never actually the best available heuristic --
   this had simply never been checked before Stage A added LST as a
   registered, comparable baseline. Any future claim of "beats EDF" should
   be read alongside "does it also beat LST," which is the harder bar.
2. **Neither RL checkpoint beats LST or EDF on reward** under this
   comparison. Pointer+shaping is close on tardiness/late-jobs (28.66/9.56
   vs. LST's 23.94/8.26) but behind on reward (254.75 vs. 288.28).
3. **LPT-based rules and the job-selection-only rules without a deadline
   signal (FCFS, SPT, WSPT, Random, Tetris) are all far worse** on
   tardiness/late-jobs (1100-1400 tardiness, 37-45 late jobs out of ~100) --
   confirming that *which job* is prioritized (deadline-aware vs. not)
   dominates *which machine* it's placed on for this reward/tardiness
   trade-off in this environment. Best-Fit/Worst-Fit/First-Fit make only a
   few points of difference within any fixed priority rule; EDF/LST's choice
   of job matters far more than their choice of machine.
4. **Pointer+RCPO's reward (135.67) is not a fair "RL is worse" reading in
   isolation** -- see `Future/research/training-log.md`'s 2026-08-28 entry
   and the correction appended to
   `Future/research/2026-08-21-rcpo-constrained-tardiness.md`: this
   checkpoint achieves its low tardiness/late-jobs by leaving roughly half
   the jobs unscheduled, not by scheduling better. It is included in this
   table for completeness, not as a fair "RL vs. heuristics" comparison
   point until the RCPO constraint gap identified there is fixed.

**Conclusion:** the classical-heuristic baseline was worth building out --
it changed the reference point (LST, not EDF, is the bar to beat) and gives
the RL results here their first honest multi-method comparison. Neither
current RL checkpoint clears that bar yet; closing the gap (or fixing RCPO
so it can be judged fairly) is the natural next step, tracked in the
overnight autonomous-work plan.

**Addendum (2026-08-28, later same night):** after adding `jobs_scheduled`
tracking to `eval_rl_agent.py` (see `training-log.md` and
`Code/utils/results_log.py`'s field comment -- prompted by the Stage C
exact-solver comparison finding the same gap independently), re-checked
this Stage A result and found even **EDF and LST do not fully complete the
fixed instance**: EDF schedules 98/100 jobs, LST 99/100 -- a fact that was
invisible in every prior report of these exact numbers across this
project's entire history, since completion rate was never tracked before
tonight. Every "EDF: reward=289.38, tardiness=16.00, late_jobs=10" cited
anywhere in this project's docs should be read with "on 98/100 jobs" now
implicit alongside it.

**Second addendum (2026-08-28, same night, prompted by a request to add
this column to the results review artifact):** measured the same field on
the 50-held-out-instance protocol for the full curated roster:

```
EDF            96.84/100      LPT+FirstFit  100.00/100
LST            97.76/100      FCFS+FirstFit  96.86/100
SPT/WSPT       92.08/100      Tetris         96.80/100
Random         97.14/100
```

`LPT+FirstFit` completes every instance perfectly; `SPT`/`WSPT` complete
the fewest. Combined with Section 6's tardiness/late-jobs numbers, this
means `LPT+FirstFit`'s apparently-bad tardiness (1392.46, worst of any
heuristic here) is not from incomplete scheduling -- it schedules
everything, just very late. The reverse is also worth noting: `SPT`/`WSPT`
combine mediocre tardiness *and* the worst completion rate in the roster,
making them the weakest heuristics on both axes at once, not just on
tardiness as Section 5's table alone would suggest.

## References

1. Pinedo, M. L. *Scheduling: Theory, Algorithms, and Systems* (5th ed.),
   Springer, 2016 -- standard reference for EDF/SPT/LST priority-rule
   scheduling theory.
2. Graham, R. L. (1969). "Bounds on Multiprocessing Timing Anomalies."
   *SIAM Journal on Applied Mathematics*, 17(2), 416-429 -- LPT makespan
   approximation bound.
3. Smith, W. E. (1956). "Various optimizers for single-stage production."
   *Naval Research Logistics Quarterly*, 3(1-2), 59-66 -- WSPT / Smith's
   rule, optimal for minimizing total weighted completion time on a single
   machine.
4. Johnson, D. S. (1973). "Near-optimal bin packing algorithms." *Journal
   of the ACM* -- First-Fit/Best-Fit/Worst-Fit classical bin-packing rules.
5. Grandl, R., Ananthanarayanan, G., Kandula, S., Rao, S., & Akella, A.
   (2014). "Multi-resource packing for cluster schedulers." *ACM SIGCOMM
   Computer Communication Review*, 44(4), 455-466 -- Tetris multi-resource
   dot-product alignment score.
6. `NotesForAI/ResearchProjectAsOf_23-07-2026.pdf`, pp. 5, 7 -- this
   project's own literature review naming FF/BF, FCFS, and SJF as baseline
   methods for the RA/scheduling framing used here.
