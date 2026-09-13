# PSO metaheuristic baseline

**Date:** 2026-08-28 (S2W6)

## 1. Motivation

Second of the three baseline stages requested (heuristics -> metaheuristic
-> exact solver). Classical heuristics (Stage A) apply a fixed rule
instantly; a metaheuristic instead *searches* for a good solution per
instance, trading wall-clock cost for solution quality -- a different point
on the same spectrum the RL model and CP-SAT's exact solver (Stage C) also
occupy, and worth having as its own comparison point.

## 2. Why PSO, and why this encoding

**PSO over a genetic algorithm:** this project's own literature review
(`NotesForAI/ResearchProjectAsOf_23-07-2026.pdf`, p.7-8) cites Rodriguez &
Buyya's application of PSO to deadline-constrained workflow scheduling with
a concrete algorithm description, whereas GA is only named generically
alongside several other metaheuristics (p.5-7) with no specific scheduling
application detailed. PSO is the metaheuristic this project's own
literature actually grounds a scheduling use-case for.

**Smallest-Position-Value (SPV) encoding** (Tasgetiren, Liang, Sevkli, &
Gencyilmaz, 2004): PSO's native representation is a continuous vector, but
this problem's decisions (which job next, which machine) are discrete. SPV
resolves this the standard way used in PSO-for-sequencing literature:
`argsort` a continuous position vector to obtain a discrete priority rank.
Extended here to two SPV blocks per particle -- one over `num_jobs`
dimensions (job priority order) and one over `num_machines` dimensions
(machine preference order) -- since this problem needs both a job choice
and a machine choice, unlike the single-permutation flowshop problem
Tasgetiren et al. targeted. Decoding follows exactly the same "min-priority
feasible job, then min-priority feasible machine" structure already used by
`Code/baselines/registry.py`'s fixed priority+placement rules (Stage A) --
see `Code/baselines/pso.py::_simulate()` -- so PSO is directly comparable to
every Stage A baseline: same decision structure, only the priority *values*
differ (learned vs. hand-picked).

**Base algorithm + inertia weight:** Kennedy & Eberhart (1995)'s velocity
update `v <- w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)`, with Shi & Eberhart
(1998)'s inertia weight `w` -- the standard, near-universally-adopted
convergence-stabilising extension of the base 1995 rule, not a separate
algorithm.

**Fitness = total episode reward**, computed by replaying the decoded
priority order through the *same* `SchedulingEnv`/`GymSchedulingEnv` every
other baseline uses (`Code/baselines/pso.py::_simulate()`) -- not a custom
tardiness-only score -- so PSO is optimized against, and compared on,
exactly the same objective as every heuristic, the RL model, and (for
tardiness specifically) CP-SAT.

## 3. Search budget, and why it's honestly smaller than the standard protocol

Unlike every other baseline in this comparison, PSO's fitness evaluation
*is* a full episode replay -- one full env rollout per particle per
iteration, not an O(1) lookup or single forward pass. A swarm of
`swarm_size=15` particles run for `iterations=30` costs 450 episode
replays *per instance* (`Code/baselines/pso.py::optimize_and_run()`
docstring). This project's standard evaluation protocol (50 held-out
instances) would cost 50x that -- disproportionately expensive for a
baseline that exists for comparison, not as the main result. This doc
therefore reports PSO on the fixed instance plus a **smaller held-out
sample (`--num-heldout`, default 15, not 50)**, stated here plainly rather
than silently matching the "50" figure used everywhere else in this
project's tables.

`swarm_size=15, iterations=30` were chosen as a modest budget expected to
show a visible convergence trend (the `fitness_curve` returned by
`optimize_and_run()` is monotonically non-decreasing by construction, since
it tracks the running global best) without dominating tonight's overall
time budget across three baseline stages plus the RCPO refix -- not
independently re-tuned via a formal search of their own, which would be a
reasonable follow-up if PSO turns out to be a serious contender rather than
a reference point.

## 4. Results

Ran on the fixed instance plus 15 held-out instances (seeds 500000-500014),
`swarm_size=15, iterations=30`, compared against `EDF` on the same
instances:

| | reward | tardiness | late jobs | wall-clock/instance |
|---|---|---|---|---|
| PSO (fixed instance) | 337.37 | 963.00 | 39 | 208.5s |
| PSO (15 held-out, mean) | 329.68 | 1012.40 | 40.93 | 182.9s |
| EDF (same 15 held-out, mean) | 286.00 | 50.33 | 14.20 | ~0.4s |

### Honest reading

**PSO beats EDF on reward by ~15% on every instance tested, but with
20x the tardiness and 3x the late-jobs rate.** This is not PSO failing to
optimize -- `fitness_curve` (tracked per run) is monotonically improving by
construction, and 450 episode evaluations comfortably finds a
high-reward solution. It is PSO succeeding at exactly what it was told to
optimize: total episode reward under this environment's actual reward
function, unconstrained by any hand-picked priority heuristic's implicit
biases.

**This independently reproduces Phase 6/7's reward-hacking finding
(`Future/research/2026-08-17-literature-review-improving-rl-agent.md`,
grounded in Skalse et al. 2022's formal definition) via a completely
different, gradient-free search method.** Phase 6 found Optuna-tuned RL
policies could achieve high reward with poor tardiness; that was always
possible to attribute (even if unlikely) to some RL-training-specific
artifact -- a bad value-function estimate, an exploration quirk, a
curriculum-transfer issue. PSO has none of those: it directly searches the
space of (job-priority, machine-priority) encodings and evaluates each one
by literally replaying it through the unmodified reward function. Finding
the same high-reward/high-tardiness exploit here is much stronger evidence
that **the misalignment is a property of the reward function's own
weighting** (the `+3` per-placement and `+50` completion bonuses
outweighing `lambda_2=1.0`'s tardiness penalty at this scale), not an
artifact specific to how RL happens to be trained.

**Search cost is real and non-trivial** -- ~185s/instance here, versus
EDF's sub-second, sub-millisecond-per-episode cost. This is the expected,
stated trade-off (Section 3): PSO trades wall-clock for solution quality
against whatever fitness it's given, and given *this* fitness (raw
reward), what it finds is a demonstration of a reward-design problem, not
a genuinely useful scheduling policy by this project's actual goals
(low tardiness).

## 5. Conclusion / next step

PSO is confirmed as a working metaheuristic baseline (correct SPV
decoding, correct env-replay fitness, sensible convergence), and in the
process independently corroborates that this project's reward function
rewards throughput over promptness more than intended -- the third
distinct method (Optuna-tuned RL, this PSO run) and the third distinct
angle (Phase 6, Phase 7's literature grounding, now a non-gradient search)
converging on the same conclusion. A natural, not-yet-tried follow-up:
rerun PSO with fitness = *negative total tardiness* (or a reward variant
with a much larger `lambda_2`) instead of raw reward, to see what the best
achievable tardiness actually is at this instance scale when that's the
explicit target -- a useful complement to CP-SAT's small-instance-only
optimum (Stage C), since PSO doesn't have CP-SAT's scaling limit.

## References

1. Kennedy, J., & Eberhart, R. (1995). "Particle Swarm Optimization."
   *Proceedings of ICNN'95 -- International Conference on Neural
   Networks*, 4, 1942-1948.
2. Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer."
   *1998 IEEE International Conference on Evolutionary Computation
   Proceedings*, 69-73. -- inertia weight term.
3. Tasgetiren, M. F., Liang, Y.-C., Sevkli, M., & Gencyilmaz, G. (2004).
   "Particle swarm optimization algorithm for makespan and total flowtime
   minimization in the permutation flowshop sequencing problem." -- Smallest-
   Position-Value (SPV) encoding for discrete sequencing via continuous PSO.
4. Rodriguez, M. A., & Buyya, R. (2014), cited via
   `NotesForAI/ResearchProjectAsOf_23-07-2026.pdf` (p.7-8) -- PSO applied
   to deadline-constrained workflow scheduling; this project's own
   literature grounding for choosing PSO as its metaheuristic baseline.
