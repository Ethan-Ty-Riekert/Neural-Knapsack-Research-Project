# Reduced-budget results (2026-09-04)

**This is a time-boxed (~1 hour), reduced-budget pass, not the project's
real/full-fidelity results.** It was produced because no trained checkpoints
survived from the `autonomous-overnight-2026-08-28` branch's run
(`rl_training/` is gitignored and was never committed) and the user needed
something to work from immediately for their mathematical report, with the
proper full-scale run explicitly deferred to a future session. Every
deviation from the project's standard methodology (documented throughout
`Future/research/training-log.md`) is listed below — read this before citing
any number in this folder as a headline project result.

## What's different from a proper run

| Aspect | This run | Project standard |
|---|---|---|
| Instance scale | 15 jobs / 4 machines / horizon 25 | 100 jobs / 10 machines / horizon 100 |
| Optuna search | 1 combined multi-objective (Pareto) search, 15 trials x 6,000 timesteps each | separate reward-tuned (50 trials) and, where used, Pareto (15-40 trials) searches, each at 30,000 timesteps/trial |
| Final training budget | 35,000 timesteps, single-stage (no curriculum) | 500,000 timesteps, 4-stage curriculum |
| Held-out evaluation set | 8 instances (seeds 500000-500007) | 50 instances (seeds 500000-500049) |
| PSO | swarm=10, iterations=20 | swarm=15, iterations=30 |
| RCPO multiplier | `rcpo_lambda_lr` and `rcpo_update_every_episodes` raised above project defaults so the multiplier can move within the much shorter training run | `rcpo_lambda_lr=0.01`, `update_every=5` |
| `deadline_range` | explicitly set to `(4, 25)`, proportional to horizon | project's `generate_env_config` default `(10, 110)` is NOT proportional to horizon at small scale — a documented footgun (`training-log.md`'s 2026-08-28 "self-inflicted test confound" entry) that this run works around explicitly rather than inheriting |

## Why one combined Optuna search, not two

`Code/training/optuna_tune.py`'s `optimize_for="pareto"` mode already returns
a full `(mean_reward, mean_tardiness_norm)` non-dominated front. Rather than
running a separate reward-only search for models A/B and a second
multi-objective search for model C (which the project's own full-fidelity
runs do separately), this pass runs the Pareto search once and reads two
different points off the same front:

- **Models A (shaping) and B (RCPO)** use the front's **max-reward trial**'s
  full hyperparameter set — standing in for what a reward-only search would
  likely find, since both are optimizing the same underlying objective.
- **Model C (Pareto-knee)** uses a genuine non-dominated **knee trial** from
  the same front (selected to trade the most tardiness away from the
  max-reward point, per unit reward given up).

This is an economization to fit the time budget, not a shortcut around
having a real search — every model's hyperparameters trace back to an
actual Optuna trial, not a hand-picked value.

## Selected trials and hyperparameters

The search found **3 non-dominated trials** out of 15 total:

| Trial | Reward | Tardiness (norm) | Used for |
|---|---|---|---|
| 9 | 96.28 | 0.88 | Models A (shaping) and B (RCPO) -- max-reward point |
| 10 | 95.36 | 0.48 | Model C (Pareto-knee) |
| 2 | -18.81 | 0.00 | *not used* -- zero-tardiness extreme, but reward collapsed near the idle-penalty floor (this is the "abandon everything to trivially satisfy tardiness" failure mode the project's RCPO investigation already documented, not a useful trade-off point) |

Trial 10 was chosen over trial 2 for the knee specifically because it trades
a small reward cost (95.36 vs 96.28, ~1%) for a real tardiness improvement
(0.48 vs 0.88, ~45%) without collapsing -- trial 2 gives up almost all reward
for its zero tardiness, which is a degenerate point, not a genuine
"knee." The selection rule (`train_reduced_budget_models.py::select_trials`)
picks whichever non-max-reward trial buys the most tardiness reduction per
unit of reward given up, automatically avoiding exactly this kind of
collapsed extreme.

Model B's RCPO `alpha` was derived from model A's own held-out mean
`episode_cost`: **0.98** (over the 8 held-out instances, per-instance costs
ranged 0.36-1.36).

Full parameter dicts: `models/optuna_search_summary.json` (the front) and
`models/<name>_hyperparams.json` (each trained checkpoint's complete,
exact hyperparameter set).

## The 6 methods compared

| Method | What it is |
|---|---|
| EDF | Earliest-Deadline-First + First-Fit placement (classical heuristic) |
| LST | Least-Slack-Time + First-Fit placement (classical heuristic) |
| Pointer+shaping | A2C pointer-network policy, potential-based reward shaping (Ng, Harada, Russell 1999), no constraint |
| Pointer+RCPO | Same architecture, RCPO-constrained tardiness (Tessler, Mankowitz, Mannor, ICLR 2019) with the project's fixed constraint definition (unscheduled jobs charged worst-case cost) |
| Pointer+Pareto-knee | Same architecture, hyperparameters from a genuine non-dominated Pareto-front trial rather than the reward-tuned point |
| PSO | Particle Swarm Optimization metaheuristic, re-optimized per instance (Kennedy & Eberhart 1995; SPV decoding per Tasgetiren et al. 2004) |

## Contents

- `models/` — the 3 trained checkpoints (`.pt`), their hyperparameter JSONs,
  and the Optuna search summary.
- `figures/` — `comparison_{reward,tardiness,late_jobs,jobs_scheduled}.png`
  (6-way grouped bar charts, mean±std over the 8 held-out instances),
  `comparison_utilisation.png`, `summary_table.png`, and `gantt_<method>.png`
  x6 (one per method, all on the same showcase instance, seed=500000).
- `raw_eval_results.csv` / `results_summary.md` — the same numbers as the
  figures, as data/plain text for pasting into the report.
- `train_log.txt` — full stdout from the training run.

## Headline numbers (8 held-out instances, 15 jobs / 4 machines / horizon 25)

| Method | Reward | Tardiness | Late jobs | Jobs scheduled |
|---|---|---|---|---|
| EDF | 93.06 ± 0.21 | 2.50 ± 4.47 | 1.75 ± 2.86 | 15.00/15 |
| LST | 93.03 ± 0.39 | 1.75 ± 3.90 | 1.12 ± 2.26 | 15.00/15 |
| Pointer+shaping | 90.10 ± 0.56 | 24.50 ± 8.35 | 4.50 ± 1.32 | 15.00/15 |
| Pointer+RCPO | 61.74 ± 29.67 | 10.62 ± 8.44 | 3.00 ± 1.41 | 14.38/15 |
| Pointer+Pareto-knee | 92.61 ± 0.61 | 2.62 ± 4.77 | 1.25 ± 1.64 | 15.00/15 |
| PSO | 93.24 ± 0.29 | 4.62 ± 5.76 | 2.00 ± 2.12 | 15.00/15 |

At this reduced scale/budget, the qualitative pattern actually echoes the
project's own full-scale history (`training-log.md`): the shaped-only model
trades tardiness for throughput, and RCPO (here undertrained relative to its
usual budget, see the `lambda_lr`/`update_every` deviation above) shows high
variance and occasionally abandons a job rather than a clean improvement —
read this as a training-budget artifact of the 1-hour constraint, not a
finding about RCPO itself. The Pareto-knee checkpoint is the standout here,
essentially matching the heuristics on every metric — but with only 8
held-out instances and 35,000 training timesteps, treat this as a single
directional data point, not a validated result.

## Where the real numbers live

`Future/research/training-log.md` is the project's actual chronological
record of full-fidelity runs. Nothing in this folder should be cited in place
of those entries — this folder exists to give the report something concrete
to work from today, ahead of a proper rerun.

*(Full stdout, wall-clock time, and reward-trend sanity checks: see
`train_log.txt` / `eval_log.txt`. Total pipeline wall-clock: ~13 minutes
training [Optuna search + 3 models] + evaluation.)*
