# v2 results archive (objective = weighted squared lateness, extended horizon)

Every v2 result under the current default objective, rebuilt from `../runs/` by one command:

```
python Results/v2_objectives/ALL_RESULTS/scripts/build_folder.py
```

**Start with [`HIGHLIGHTS.md`](HIGHLIGHTS.md)** (generated): best method per family and per metric,
key figures, and links to every leaderboard.

| Folder | Contents |
|---|---|
| `HIGHLIGHTS.md` | entry page: headline table, best method on each metric per preset, key figures, leaderboard index |
| `leaderboards/<preset>.md` | top-10 table and top-5 / top-10 figures for each metric (J, on-time rate, weighted / max tardiness, mean wait, active machine-ticks) |
| `data/all_results.csv` | one row per preset x method: mean and std of J and the schedule metrics, seeds, instances, git commits, source run folders |
| `tables/<preset>.md` | every method on that preset, ranked by J |
| `tables/summary.md` | best heuristic vs best RL vs PSO vs CP-SAT per preset |
| `figures/` | `J_by_method_<preset>`, `regime_map`, `heuristic_regime_heatmap`, `training_curves_<preset>` (PNG + PDF); `leaderboards/<preset>/top{5,10}_<metric>.png` |

**Setup.** Objective J = sum_j w_j T_j^2 (difficulty presets: w_j ~ Uniform{1,...,5}, i.e. `weights=(1, 6)` with numpy's exclusive upper bound; `off_c_*` presets: w_j = 1), with an extended horizon: jobs never expire,
and unfinished jobs keep accruing lateness past H. The RL reward is exactly -J / (number of jobs)
(`Code/core/objectives.py`). Each preset is evaluated on the same 15 held-out instances for every
method (seeds 500000-500014). Offline presets: 100 jobs, 10 machines, H = 100, deadline tightness
TF 0.2 / 0.5 / 0.8. Online presets: Poisson arrivals at load rho 0.50 / 0.75 / 0.95 / 1.10,
lognormal sizes. Full definitions: `Code/core/difficulty.py`, `Code/variants/v2_objectives/`.

**Methods.** 33 dispatching heuristics (7 priority rules x 4 placement rules, plus Random and
Tetris); two **random rule-selection** baselines (a uniformly random rule at each decision, from the
same menus RL Option 1 uses); PSO (swarm 10 x 20 iterations); CP-SAT (offline `off_c_15` only); and
PPO in four action designs:
Option 1 picks a rule each decision (FirstFit menu, or FirstFit+Consolidate menu, "+Consolidate"),
Option 2 scores job priorities, Option 3 scores priorities starting from ATC, Option 4 picks job and
machine as two choices. PPO budgets: Option 1 750k steps online / 500k offline; Options 2-4 200k
steps. Default PPO hyperparameters (gamma 0.99, GAE lambda 0.95), 4 parallel envs. Option 1 at
rho 0.95 / 1.10 has 3 seeds; everything else 1 seed (first pass).

## Headline results (J, lower is better; from `tables/summary.md` and the per-preset tables)

Hand-written snapshot with interpretation; it can lag new runs. The current numbers are always in
[`HIGHLIGHTS.md`](HIGHLIGHTS.md) and `tables/`.

| Preset | Best heuristic | Best RL | RL vs best heuristic | Random rule selection (best menu) | PSO |
|---|---|---|---|---|---|
| off_c_15 | LST+FirstFit 192 (CP-SAT also 192) | - | - | - | - |
| off_tf05 | LST+FirstFit 37,457 | Opt1 38,885 (1 seed) | +3.8% | 107,422 | 92,729 |
| off_tf08 | LST+FirstFit 383,691 | Opt3 381,318 (1 seed) | -0.6% | 511,737 | - |
| on_rho075 | LST+Consolidate 7,983 | Opt1+Consolidate 12,608 (1 seed) | +58% | 8,823 | - |
| on_rho095 | EDF+Consolidate 21,923 | Opt1 144,248 (3 seeds) | +558% | 51,931 | 108,604 |
| on_rho110 | EDF+Consolidate 99,781 | Opt1+Consolidate 373,310 (3 seeds) | +274% | 247,810 | 422,596 |

## What the results say (proven vs observed)

- **Offline, the best dispatching rule is (near-)optimal, and RL matches it.** On `off_c_15`, CP-SAT
  proves LST optimal on 12/15 instances and equal on all 15 (2026-09-30). On the TF presets, RL
  lands within -0.6% to +3.8% of LST: within instance noise, with 1 seed. Report as "matches the best
  rule", not "beats".
- **Online, the gap grows with load, and RL loses even to random rule selection at rho >= 0.75.**
  At rho 0.95 and 1.10, 6 of the 12 PPO Option 1 runs (2 menus x 3 seeds x 2 loads) converged onto a
  single short-job rule: their J equals **SPT+FirstFit** (5 runs) or **WSPT+FirstFit** (1 run)
  exactly. The other 6 are rule mixtures that beat SPT (by 2-48%), but every one of the 12 is worse
  than random rule selection on the same preset. This is the same
  collapse seen under the v1 reward, so it is not caused by reward misspecification: the v2 reward
  equals the objective. SPT has the lowest mean wait of all methods, but it starves long jobs, and
  squared lateness punishes starvation heavily (max tardiness ~90 vs ~32 for EDF+Consolidate).
  Changing gamma or GAE lambda did not prevent the drift (training-log 2026-10-05).
- **Options 2/3/4 at 200k steps fail online in a different way:** the priority-score policies never
  schedule some jobs (1-72 per instance), which are then charged the forced-drop cost at the
  episode safety cap, so J is 10^6-10^9. Treat these as under-trained at this budget, not as tuned results.
- **Placement matters online.** The best online rule from rho 0.75 up always uses Consolidate:
  EDF+Consolidate at 0.95 / 1.10, LST+Consolidate at 0.75 (`figures/heuristic_regime_heatmap`).
- **PSO** (offline-style priority search per instance) is worse than the best rule everywhere
  here, at this budget.

## Caveats

- 1 seed for every row except Option 1 at rho 0.95 / 1.10 (3 seeds). Multi-seed RL rows report the std
  across seeds; other rows report the std across instances. The figures use the standard error.
- PPO hyperparameters are SB3 defaults, not tuned for v2. Option 1 makes one decision per job
  placement. A per-tick decision epoch (the usual hyper-heuristic setup) is untested.
- Older v2 runs under the earlier fixed-window / drop-surcharge objective (2026-09-29) are excluded
  (`data/build_info.json` counts them).
