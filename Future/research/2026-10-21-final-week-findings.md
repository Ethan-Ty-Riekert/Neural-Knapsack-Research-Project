# Final-week findings (2026-10-21): can RL beat the best online heuristic?

**Date:** 2026-10-21 (S2W14). **Status:** exploratory results for the end-of-year presentation. J = weighted squared tardiness (lower is
better). Two evaluation sets are used and every table says which: **OLD** = the 50 test instances (seeds >= 500000) that informed many
decisions, with reference heuristics picked on them (flattered; winner's curse); **FRESH** = 50 new instances (seeds 700000+), scored for
about 11 arms and up to 7 heuristics with no multiplicity correction (also exploratory). Intervals are bootstrap intervals over
instances of the seed-averaged difference, conditional on the trained seeds: training-seed variance is NOT in any interval, and verdicts
from 3-seed arms are indicative only. Per-seed values are given instead. Detailed, dated evidence: `training-log.md`, 2026-10-20 and
2026-10-21 entries. Citations named here are from memory and still have to be verified before they go into a report.

## Confirmation on 50 FRESH instances (not used by any decision, but scored for ~11 arms and 7 heuristics) -- read this first

The 50 test instances informed many go / no-go decisions, and EDF+Consolidate was the winner of ~40 heuristics on them (a lucky pick).
On 50 new instances (seeds 700000-700049), mean J:

| Heuristic | MDC+C | EDFL+C | EDF+C | WLST+C | ATC+C |
|---|---|---|---|---|---|
| fresh J | 26,809 | 29,583 | 31,090 | 31,854 | 45,972 |

| Arm (members) | per-member mean J | mean J | vs EDF+C (95% CI over instances) | vs MDC+C |
|---|---|---|---|---|
| windowed clone only (5) | 29,839 / 31,317 / 33,658 / 30,030 / 30,011 | 30,971 | -119 (-3,814, +3,858) | +4,162 (-312, +8,906) |
| windowed clone + PPO (5) | 28,572 / 29,751 / 32,960 / 28,381 / 29,287 | 29,790 | -1,300 (-4,147, +1,384) | +2,981 (-540, +6,704) |
| windowed from scratch (5) | 34,475 / 54,213 / 59,493 / 52,540 / 35,933 | 47,331 | +16,241 (+12,013, +20,986) | +20,522 (+14,900, +26,525) |
| all-slot clone only (3) | 28,574 / 27,152 / 28,856 | 28,194 | -2,896 (-6,720, +858) | +1,385 (-2,149, +4,676) |
| all-slot clone + PPO (5) | 32,303 / 27,885 / 26,403 / 31,783 / 31,643 | 30,003 | -1,087 (-4,297, +1,979) | +3,194 (-336, +6,903) |
| plain RL, all slots, FirstFit, 1M (5) | 26,205 / 35,632 / 28,656 / 40,497 / 37,829 | 33,764 | +2,674 (-166, +5,675) | +6,955 (+2,586, +11,438) |
| plain RL, all slots, Consolidate, 1M (5) | 34,684 / 34,341 / 27,059 / 29,361 / 25,550 | 30,199 | -891 (-4,064, +2,407) | +3,390 (-595, +7,236) |

What this shows (FRESH set, exploratory): (1) no learned arm is significantly below MDC+Consolidate, the best of the 7 heuristics scored on
this set (MDC+C is itself selected on this set), and none is significantly below EDF+Consolidate; some arms are significantly below
other, weaker heuristics (COVERT+C, WMDD+C, ATC+C; see the CSV summary), which says nothing about the best rule. (2) clones (with or without
PPO) and all-slot plain RL with Consolidate placement are not significantly different from the EDF family (differences -3k to +1k,
intervals spanning zero); (3) the old-set statement that plain RL from scratch is "20-30% behind" used a flattered reference: on fresh
instances all-slot RL with Consolidate placement is not shown to be behind EDF+Consolidate (-891, interval -4,064 to +2,407), but its
distance to MDC+C (+3,390, interval -595 to +7,236, i.e. up to about +27%) is not excluded; plain RL with FirstFit placement is
significantly behind MDC+C and windowed from scratch is far behind everything; (4) PPO fine-tuning shows no improvement on the fresh set
either. Per-seed spreads are large (e.g. 26k-40k for plain RL).

### FRESH instances at the other loads and for the explorer's expert clones (50 fresh instances each, 5 seeds unless noted; 3-seed rows indicative)

| Scenario | Best heuristic on the fresh set (J) | Clone only | Clone + PPO |
|---|---|---|---|
| `on_rho075` | MDC+C 6,263 (WLST+C 6,407, EDF+C 6,494) | 6,504 (vs WLST+C +97, interval -323..+466; vs MDC+C +241, n.s.) | 6,650 (vs MDC+C +387, +52..+813: slightly worse) |
| `on_rho110` | EDF+C 91,502 (MDC+C 93,590) | 98,618 (vs MDC+C +5,028, n.s.) | 113,223 (vs MDC+C +19,633, significantly worse; one seed 155k) |

Expert clones at `on_rho095` (clone only, 3 seeds, fresh set): MDC expert 30,902 vs MDC+C 26,809 (+4,093, n.s.); WLST expert 36,021 vs
WLST+C 31,854 (+4,168, significantly above); WMDD expert 39,831 vs WMDD+C 40,170 (n.s.); COVERT expert 32,232 vs COVERT+C 35,121 (n.s.).
The picture matches the old test set: imitation reproduces its heuristic within noise, fine-tuning adds nothing (and hurts at overload),
nothing beats the best heuristic, which differs between scenarios.

## Headline (OLD test set; preset `on_rho095`: 10 machines, heavy load rho = 0.95, lognormal job sizes, Poisson arrivals; reference heuristics picked on this set)

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
4. **Training from scratch is unreliable**: on the old test set it is 20-30% above the (lucky) EDF+Consolidate reference with all slots,
   and much worse in the windowed setting, with diverging seeds; on fresh instances all-slot RL with Consolidate placement is not
   significantly different from the EDF family (see the fresh-instance section), while windowed scratch stays far behind.

## Other scenarios (OLD test set, reference picked on it; windowed recipe, EDFL expert, 3 seeds each, indicative)

| Scenario | Best heuristic (test J) | Clone only, per seed | Clone + PPO (A3w), per seed |
|---|---|---|---|
| `on_rho075` | WLST+Consolidate 5,990 | 6,197 / 6,192 / 6,101 (+3%, small but significant) | 6,139 / 7,225 / 6,298 (+9%) |
| `on_rho110` | MDC+Consolidate 106,463 | 108,806 / 116,352 / 124,408 (+9%, n.s.) | 127,894 / 107,692 / 120,271 (+11%, n.s.) |
| `off_tf05` | WLST+FirstFit 26,670 | 40,845 / 40,806 / 40,772 (+53%) | 31,917 / 32,261 / 31,507 (+20%) |

The expert was EDFL everywhere (not matched to the scenario). Offline, PPO fine-tuning consistently improved a weak clone by
~8.9k but stayed significantly behind the best heuristic; around a good online heuristic it added nothing.

### Scenario-matched experts (OLD test set; the best heuristic of each scenario as the expert; 3 seeds each, indicative)

| Scenario | Best heuristic | Clone only, per seed | Clone + PPO, per seed |
|---|---|---|---|
| `on_rho075` | WLST+Consolidate 5,990 | 6,025 / 6,330 / 6,168 (+3%, sig.) | 6,084 / 6,015 / 6,116 (+1.4%, sig., tiny) |
| `on_rho110` | MDC+Consolidate 106,463 | 116,293 / 115,045 / 111,610 (+7%, n.s.) | 119,062 / 116,578 / 109,300 (+8%, n.s.) |
| `off_tf05` | WLST+FirstFit 26,670 | 27,420 / 27,418 / 27,432 (+2.8%, sig.) | 28,480 / 28,551 / 27,959 (+6.2%, sig.) |

The matched expert is much better than EDFL offline (27.4k vs 40.8k): the expert decides the result; fine-tuning does not help.

### Five-seed arms at `on_rho075` (OLD test set; WLST expert, reference WLST+Consolidate 5,990 picked on it, margin +/-5% = +/-300; on the FRESH set MDC+C is best, 6,263)

| Arm | per-seed J | paired diff (CI) | verdict under the pre-registered rules |
|---|---|---|---|
| clone only | 6,246 / 6,103 / 6,083 / 6,103 / 6,207 | +159 (+52, +288) | equivalent within 5% (but significantly +2.7% above) |
| clone + PPO | 6,195 / 5,995 / 6,080 / 6,194 / 6,353 | +174 (+50, +331) | equivalence not shown; no gain from PPO |

### Five-seed arms at `on_rho110` (OLD test set; MDC expert, reference MDC+Consolidate 106,463 picked on it, margin +/-5% = +/-5,300; on the FRESH set EDF+C is best)

| Arm | per-seed J | paired diff (CI) | verdict |
|---|---|---|---|
| clone only | 121,708 / 110,476 / 119,515 / 125,106 / 112,443 | +11,386 (-562, +22,976) | no significant difference; equivalence not shown |
| clone + PPO | 131,077 / 124,023 / 133,774 / 182,172 / 133,408 | +34,427 (+23,347, +44,628) | significantly worse (+32%): fine-tuning degrades it at overload |

### Five-seed arms at `on_rho075_tight` (WLST expert, reference WLST+Consolidate 29,502, margin +/-5% = +/-1,475; old test set)

| Arm | per-seed J | paired diff (CI) | verdict |
|---|---|---|---|
| clone only | 29,644 / 30,043 / 30,042 / 30,370 / 30,219 | +562 (+158, +1,022) | equivalent within 5%; significantly +1.9% above |
| clone + PPO | 29,717 / 30,707 / 29,978 / 30,392 / 29,565 | +570 (+166, +1,034) | equivalent within 5%; no gain from PPO |

### Late-jobs specialist (OLD test set; Option 3 windowed, ATC + Consolidate as the expert, `on_rho095`; 3 seeds, indicative)

| | J | on-time rate | weighted late jobs |
|---|---|---|---|
| ATC+Consolidate | 40,676 | 0.960 | 97.2 |
| EDF+Consolidate | 21,986 | 0.954 | 124.6 |
| clone only (3 seeds) | 39,341 | 0.960 | 97.3 |
| clone + PPO on J (3 seeds) | 37,466 | 0.960 | 98.5 |

A learned model for the "fewest late jobs" metric (22% fewer weighted late jobs than EDF+Consolidate, at a much higher J), obtained by
imitation; no significant difference from ATC+Consolidate.

### Multi-objective anchors (OLD test set; windowed EDFL clone + PPO on a composite reward, `on_rho095`, 3 seeds, indicative)

Training on J + late jobs, J + energy or J + linear tardiness left the policy at the heuristic's trade-off (weighted late jobs 127.3 vs
129.3 for the J-only fine-tuned policy and 126.1 for the clone; machine-ticks 1,251 vs 1,253): no measurable change at lr 1e-5. For the
explorer the learned models for different metrics are imitation clones of different heuristics (EDF, ATC, WLST, MDC).

### Matched clones at more scenarios (OLD test set, reference picked on it; clone only, 3 seeds, indicative)

| Scenario | Best heuristic (test J) | Clone, per seed | vs heuristic |
|---|---|---|---|
| `on_rho050` | WLST+Consolidate 4,085 | 4,090 / 4,087 / 4,093 | +0.1% |
| `on_rho075_tight` | WLST+Consolidate 29,502 | 29,734 / 29,777 / 30,622 | +1.8% |
| `off_tf08` | WLST+FirstFit 257,155 | 266,353 / 266,280 / 266,349 | +3.6% |
| `off_tf08_w1` (unit weights) | WLST+FirstFit 129,805 | 129,805 x3 | identical |
| `off_tf05_w1` (unit weights) | LST+FirstFit 13,098 | 13,098 x3 | identical |

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

* One preset carries the headline; the other scenarios have 3-5 seeds per arm. Arms use different seed numbers, so comparisons are by arm
  means, not by seed pairs; 3-seed verdicts are indicative (the pre-registered "beats" rule needs 5 seeds per arm).
* Protocol mismatches: fine-tuned arms use lr 1e-5 (chosen on validation to preserve the clone) while the scratch arms use the PPO default
  3e-4, untuned, so "scratch is unreliable" is confounded with "untuned"; the ~180k+ expert / DAgger environment steps used for cloning are
  not counted in the 1M budget.
* Multiplicity and selection: no correction over ~11 arms x 7 heuristics on the fresh set (the doc shows a subset of the comparisons);
  the OLD test set informed many go/no-go decisions and its reference heuristic was picked on it; the fresh set is now also used to name
  the best heuristic. Training-seed variance is not part of any interval.
* Open: windowed PPO from scratch is much worse than the all-slot network although the architectures look equivalent; window 40 and 60
  do not help (test J 34k-76k; runs still finishing), so the cause is unexplained.
* Tier 2 (default or lightly chosen hyperparameters) throughout; the full tuned protocol was only run earlier (Tier 1) for the
  all-slot network.

## For the presentation (wording suggested by critic pass 3, adopted)

Say: "On 50 fresh instances no learned policy was significantly better than the best of the five-to-seven heuristics we scored (MDC+Consolidate
at 0.95 load). A policy cloned from a heuristic was, in most cases, not significantly different from it, but a few clones were
significantly worse. Further PPO fine-tuning did not help (lr 1e-5, 3-5 seeds). PPO from scratch with default hyperparameters varied
widely across seeds. A three-instance experiment shows large headroom for an algorithm that knows future arrivals; whether a forecast could
capture any of it is untested. All results are exploratory."
Do not say: "RL beats / matches the heuristics", "RL from scratch is 20-30% behind", "reproduces it reliably", "tie-breaking explains the
differences inside the EDF family" (only suggested by 20 validation instances and one order flip), or "points to arrival forecasting".
