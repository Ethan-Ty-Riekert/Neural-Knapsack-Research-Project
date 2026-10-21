# Final-week findings (2026-10-21): can RL beat the best online heuristic?

**Date:** 2026-10-21 (S2W14). **Status:** exploratory results for the end-of-year presentation. Every number is a test J
(weighted squared tardiness, lower is better) on the 50 held-out instances of one preset unless stated; per-seed values are
given because 3-5 seeds do not support tight intervals. Detailed, dated evidence: `training-log.md`, 2026-10-20 and 2026-10-21
entries. Citations named here are from memory and still have to be verified before they go into a report.

## Headline (preset `on_rho095`: 10 machines, heavy load rho = 0.95, lognormal job sizes, Poisson arrivals)

| Method | Mean J | per-seed J |
|---|---|---|
| EDF + Consolidate (best of ~40 heuristics on this test set) | 21,986 | single deterministic run |
| EDFL + Consolidate (EDF, ties by longest job; new) | 24,061 | single deterministic run |
| Plain RL from scratch, all job slots, FirstFit placement, 1M steps | 28,863 | 22,927 / 31,652 / 24,965 / 32,507 / 32,264 |
| Plain RL from scratch, all slots, Consolidate placement, 1M steps | 26,274 | 28,947 / 30,323 / 23,164 / 26,884 / 22,054 |
| All-slot clone of EDF+Consolidate, then PPO (A3) | 23,654 | 23,613 / 24,001 / 22,258 / 23,766 / 24,633 |
| All-slot clone only (no PPO), 3 seeds | 23,666 | 22,620 / 23,667 / 24,709 |
| Windowed (20 earliest-deadline jobs) clone of EDFL only | 23,811 | 22,567 / 25,371 / 24,293 / 22,912 / 23,910 |
| Windowed clone + PPO fine-tuning (A3w) | 24,794 | 27,162 / 24,342 / 23,185 / 25,009 / 24,270 |
| Windowed PPO from scratch, 1M steps | 38,986 | 26,245 / 42,187 / 47,085 / 46,243 / 33,171 |

What the data support:

1. **No learned policy beat the best online heuristic.** The paired per-instance interval (seed-averaged) never lies below zero.
2. **A policy cloned from a heuristic is not significantly different from it** (windowed clone vs EDFL+Consolidate: -250, interval
   -3,059 to +2,276). It is not shown *equivalent* at +/-5% either (the interval is wider).
3. **PPO fine-tuning added nothing measurable** to the clone (all-slot: -375, interval -2,684 to +1,873; windowed: +983,
   interval -569 to +2,583), at a learning rate chosen on validation instances; higher rates destroy the clone (validation J of
   37k-52k at 1e-4). The statement is conditional: lr 1e-5, ~0.9M steps, 3-5 seeds.
4. **Training from scratch is far behind and unreliable**: 20-30% above the best heuristic with all slots, much worse still in the
   windowed setting, with diverging seeds.

## Other scenarios (windowed recipe, EDFL expert, 3 seeds each, exploratory)

| Scenario | Best heuristic (test J) | Clone only, per seed | Clone + PPO (A3w), per seed |
|---|---|---|---|
| `on_rho075` | WLST+Consolidate 5,990 | 6,197 / 6,192 / 6,101 (+3%, small but significant) | 6,139 / 7,225 / 6,298 (+9%) |
| `on_rho110` | MDC+Consolidate 106,463 | 108,806 / 116,352 / 124,408 (+9%, n.s.) | 127,894 / 107,692 / 120,271 (+11%, n.s.) |
| `off_tf05` | WLST+FirstFit 26,670 | 40,845 / 40,806 / 40,772 (+53%) | 31,917 / 32,261 / 31,507 (+20%) |

The expert was EDFL everywhere (not matched to the scenario). Offline, PPO fine-tuning consistently improved a weak clone by
~8.9k but stayed significantly behind the best heuristic; around a good online heuristic it added nothing.

### Scenario-matched experts (the best heuristic of each scenario as the expert; 3 seeds each)

| Scenario | Best heuristic | Clone only, per seed | Clone + PPO, per seed |
|---|---|---|---|
| `on_rho075` | WLST+Consolidate 5,990 | 6,025 / 6,330 / 6,168 (+3%, sig.) | 6,084 / 6,015 / 6,116 (+1.4%, sig., tiny) |
| `on_rho110` | MDC+Consolidate 106,463 | 116,293 / 115,045 / 111,610 (+7%, n.s.) | 119,062 / 116,578 / 109,300 (+8%, n.s.) |
| `off_tf05` | WLST+FirstFit 26,670 | 27,420 / 27,418 / 27,432 (+2.8%, sig.) | 28,480 / 28,551 / 27,959 (+6.2%, sig.) |

The matched expert is much better than EDFL offline (27.4k vs 40.8k): the expert decides the result; fine-tuning does not help.

### Late-jobs specialist (Option 3 windowed, ATC + Consolidate as the expert, `on_rho095`)

| | J | on-time rate | weighted late jobs |
|---|---|---|---|
| ATC+Consolidate | 40,676 | 0.960 | 97.2 |
| EDF+Consolidate | 21,986 | 0.954 | 124.6 |
| clone only (3 seeds) | 39,341 | 0.960 | 97.3 |
| clone + PPO on J (3 seeds) | 37,466 | 0.960 | 98.5 |

A learned model for the "fewest late jobs" metric (22% fewer weighted late jobs than EDF+Consolidate, at a much higher J), obtained by
imitation; no significant difference from ATC+Consolidate.

### Multi-objective anchors (windowed EDFL clone + PPO on a composite reward, `on_rho095`, 3 seeds)

Training on J + late jobs, J + energy or J + linear tardiness left the policy at the heuristic's trade-off (weighted late jobs 127.3 vs
129.3 for the J-only fine-tuned policy and 126.1 for the clone; machine-ticks 1,251 vs 1,253): no measurable change at lr 1e-5. For the
explorer the learned models for different metrics are imitation clones of different heuristics (EDF, ATC, WLST, MDC).

## Why the gap and the "ties" matter

* **Tie-breaking in EDF moves J by about +/-15%.** Deadlines are integers; several pending jobs share the earliest one. On 20
  validation instances EDF with different tie-breaks scores 28.3k (longest job first) to 36.5k (heaviest weight first), job-index
  30.9k; on the 50 test instances the order flips (EDFL 24,061 vs job-index EDF 21,986). So differences of ~10% inside the EDF
  family are instance-set noise, and "best heuristic out of ~40, picked on the test set" is a winner's-curse reference (critic
  pass 2).
* **Foresight is worth a lot, but an online policy cannot have it.** A clairvoyant one-step rollout over EDF+Consolidate (it
  simulates the base heuristic on a copy of the environment that knows the future arrivals) reduced mean J on three validation
  instances from 24,682 to 7,339 (-70%; -12%, -73%, -80%). That is an upper bound for an online policy, not a result; the
  squared-tardiness objective is dominated by a few heavy instances where knowing arrivals matters.

## What was learned about the method

* **Compute:** 88% of the time of an update step went into running the network over 1,235 job slots, of which ~10 are schedulable
  and ~30 pending (one cProfile run, indicative). Exact compaction of the active slots (`--compact-slots`, identical outputs and
  gradients on selectable actions, `tests/test_compact_policy.py`) gave ~6x; the 20-job window ~40x; a 1M-step windowed run takes
  ~30 minutes. The earlier GPU/CPU contention explanation is not established.
* **Behaviour cloning needs a cloneable expert.** A per-slot network cannot see the job index, so EDF with index tie-breaks
  cannot be cloned exactly (agreement stalls at ~97%); EDFL (ties by duration) is cloned at 99.5%. DAgger-style labelling,
  critic warm-up with the actor frozen, cosine-decay cloning are in `imitation.py`.
* **Placement:** Consolidate placement helps every priority rule online by 30-40% in J, but did not change plain RL at 1M steps
  (26.3k vs 28.9k, overlapping seed ranges); at 300k steps (3 seeds) 28.2k vs 39.9k, suggestive only.
* **Rejected / retracted along the way:** J + linear tardiness online (a one-seed tie in 5 seeds); "RL beats every FirstFit
  heuristic by 25-30%" (it is 13-18% against the best FirstFit rule); "matches the heuristic" and "the clone explains the result"
  (rephrased as non-significant differences, conditional nulls).

## Caveats

* One preset carries the headline; other scenarios are queued (windowed clone and A3w at `on_rho075`, `on_rho110`, `off_tf05`).
* Seeds: 3-5 per arm; arms use different seed numbers; seed bootstrap intervals are not reported for n <= 5.
* The 50 test instances informed many go/no-go decisions; treat the numbers as exploratory, not as confirmatory.
* Tier 2 (default or lightly chosen hyperparameters) throughout; the full tuned protocol was only run earlier (Tier 1) for the
  all-slot network.

## For the presentation

Say: learning from scratch does not beat strong scheduling heuristics under heavy online load in this setup; a policy that
starts from the best heuristic reproduces it reliably but further learning did not improve it; most of the apparent differences
inside the EDF family are tie-breaking noise; with perfect foresight there is large headroom (upper bound), which points to
arrival forecasting as the place to look. Do not say: "RL beats / matches the heuristics".
