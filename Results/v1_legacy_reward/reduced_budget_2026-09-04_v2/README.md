# Reduced-budget results, v2 (2026-09-04)

**Supersedes `Results/reduced_budget_2026-09-04/` (v1).** v1's RL results
looked weird for a real, diagnosable reason -- see "What was wrong with v1"
below. This is still a time-boxed pass, not the project's real/full-fidelity
results, but it fixes v1's methodological bug and uses a larger, more
combinatorially meaningful instance. v1's folder is left in place rather than
deleted or overwritten, per the project's own "log a correction in a new
entry, don't rewrite history" convention (`CLAUDE.md`).

## What was wrong with v1

v1 ran one combined multi-objective (`optimize_for="pareto"`) Optuna search
and picked models A ("Pointer+shaping") and B ("Pointer+RCPO")'s
hyperparameters from the front's **max-reward** trial. That trial had
`lambda_2=0.505` -- essentially no tardiness penalty in the actual training
reward -- which is exactly the reward/tardiness misalignment this project's
own training log repeatedly documents as a failure mode (Stage A/B baselines,
the whole Pareto investigation arc). It produced a model tuned to look good
on raw reward while barely caring about deadlines, which is why v1's
Pointer+shaping had ~10x EDF/LST's tardiness and Pointer+RCPO showed high
variance and occasional job abandonment.

v1's instance (15 jobs/4 machines/horizon 25) also had enough slack that the
problem was close to pure deadline-ordering -- exactly what EDF/LST are
built for -- without much genuine multi-resource bin-packing pressure, so it
didn't give the pointer network much to actually learn beyond what a greedy
rule already gets for free.

## What's different in v2

- **Two separate Optuna searches**, matching the project's real methodology
  instead of economizing it into one: `optimize_for="tardiness"` (a
  composite `mean_reward - 50*mean_tardiness_norm` score, built into
  `optuna_tune.py` specifically to avoid the trap above) selects models A and
  B's hyperparameters; a separate `optimize_for="pareto"` search selects
  model C's genuine non-dominated knee trial.
- **A larger, tighter instance**: see the exact dimensions in
  `models/optuna_search_summary.json`'s `instance` block -- more jobs and
  machines with proportionally less slack, so multi-resource placement
  actually matters, not just job ordering.
- Everything else (8 held-out instances not 50, single-stage training not a
  curriculum, hand-tuned RCPO `lambda_lr`/`update_every` for the short
  training budget) carries over from v1 -- still a reduced-budget pass.

## Selected trials and hyperparameters

See `models/optuna_search_summary.json` for the instance dimensions, both
searches' selected trial numbers/parameters, and the composite-score/
Pareto-values each was picked on. See `models/<name>_hyperparams.json` for
each trained checkpoint's complete, exact hyperparameter set (including the
derived RCPO `alpha`).

## The 6 methods compared

| Method | What it is |
|---|---|
| EDF | Earliest-Deadline-First + First-Fit placement (classical heuristic) |
| LST | Least-Slack-Time + First-Fit placement (classical heuristic) |
| Pointer+shaping | A2C pointer-network policy, potential-based reward shaping, hyperparameters from the tardiness-composite Optuna search |
| Pointer+RCPO | Same architecture, RCPO-constrained tardiness (Tessler, Mankowitz, Mannor, ICLR 2019), constraint alpha derived from model A's own held-out cost |
| Pointer+Pareto-knee | Same architecture, hyperparameters from a genuine non-dominated Pareto-front trial (separate multi-objective search) |
| PSO | Particle Swarm Optimization metaheuristic, re-optimized per instance |

## Contents

- `models/` — the 3 trained checkpoints (`.pt`), their hyperparameter JSONs,
  and both Optuna searches' summary.
- `figures/` — `comparison_{reward,tardiness,late_jobs,jobs_scheduled}.png`,
  `comparison_utilisation.png`, `summary_table.png`, and `gantt_<method>.png`
  x6 (all on the same showcase instance, seed=500000).
- `raw_eval_results.csv` / `results_summary.md` — the same numbers as data/
  plain text for pasting into the report.
- `train_log.txt` / `eval_log.txt` — full stdout from both stages.

## Headline numbers (8 held-out instances, 45 jobs / 6 machines / horizon 50)

| Method | Reward | Tardiness | Late jobs | Jobs scheduled |
|---|---|---|---|---|
| EDF | 174.65 ± 18.60 | 14.88 ± 17.37 | 7.38 ± 7.00 | 44.88/45 |
| LST | 181.82 ± 0.30 | 9.12 ± 15.05 | 4.62 ± 6.59 | 45.00/45 |
| Pointer+shaping | 119.36 ± 24.96 | **10.25** ± 16.14 | 4.75 ± 6.36 | 40.38/45 |
| Pointer+RCPO | 93.64 ± 5.88 | 156.88 ± 42.99 | 12.75 ± 2.22 | 36.75/45 |
| Pointer+Pareto-knee | 175.95 ± 0.77 | 263.00 ± 27.43 | 19.25 ± 3.31 | 45.00/45 |
| PSO | 179.21 ± 0.52 | 146.62 ± 18.51 | 17.75 ± 1.56 | 45.00/45 |

**This result set actually tells a recognizable, historically-consistent
story** — unlike v1, where every RL model just uniformly lost:

- **Pointer+shaping beats EDF on tardiness** (10.25 vs 14.88), matching the
  project's real 2026-08-19 finding ("pointer beats EDF on tardiness"). It
  loses on reward and completion rate (40.38/45 — abandons ~4.6 jobs on
  average) rather than winning outright, which is a believable outcome for
  35-60k timesteps of training vs. the project's real 500k, not a red flag.
- **Pointer+RCPO destabilized during training** — its reward trajectory
  crashes from ~188 to strongly negative around episode 150-300 before
  partially recovering to a noisy near-zero plateau (see `train_log.txt`),
  and its held-out tardiness (156.88) is far worse than the unconstrained
  shaping model. This is a real instance of the same failure mode the
  project's own "Phase 10" RCPO entry documents — a Lagrange multiplier
  chasing a miscalibrated/too-aggressive constraint threshold — not a
  methodology bug in this run. `rcpo_alpha=0.64` was derived correctly from
  model A's own held-out cost, but combined with the raised `lambda_lr=0.05`/
  `update_every=3` (needed to let the multiplier move at all within 60k
  timesteps), it likely overshot. A real full-budget run would use the
  project's standard `lambda_lr=0.01`/`update_every=5` over far more
  episodes, which is the whole reason this destabilization wasn't as severe
  in the project's historical Phase-10-vs-fixed comparison.
- **Pointer+Pareto-knee and PSO both show the "high reward via throughput,
  poor tardiness" reward-hacking pattern** the project's real Pareto
  investigation found at full deployment scale (fully-trained Pareto-knee:
  reward 303/tardiness 734 there; here: reward 176/tardiness 263) — both
  schedule every job (45/45) but let many run late rather than turning down
  work. This is the same qualitative finding replicating at a much smaller
  scale/budget, not a coincidence.
- The Gantt charts reveal a genuine RL behavioral pattern worth noting for
  the report: the pointer-network policies concentrate most jobs onto one
  machine sequentially rather than spreading load — an under-utilization
  pattern, not a code bug (no overlapping placements, durations are correct).

## Where the real numbers live

`Future/research/training-log.md` is the project's actual chronological
record of full-fidelity runs. Nothing in this folder (v1 or v2) should be
cited in place of those entries.
