# All results, 2026-07-24 to 2026-09-22 (S2W1 - S2W9)

One folder with every evaluation result this project has recorded, split by
**offline / online** and **constant job weights (all w_j = 1) / random job weights
(w_j ~ U{1..5})**, across tardiness, weighted tardiness, reward, late jobs and jobs
scheduled (assigned).

Open `scheduler_results_atlas.html` in a browser for the interactive version (same
data, also published as a claude.ai artifact).

## Why the original figures are missing

Every training/eval script writes to `rl_training/` (see `Code/utils/paths.py`), and
`rl_training/` is gitignored -- so **no commit on any branch ever contained the
figures, checkpoints or `eval_results.csv`**. Git history, both stashes and every
branch were checked: the only image files ever committed are the 5 portfolio figures.
The model paths recorded in the eval CSV point to
`D:\University\Year3\ResearchProject\Neural-Knapsack-Research-Project\rl_training\`,
a drive that does not exist on this laptop. On this machine `rl_training/` only holds
6 Optuna result files. **If you want the original Gantt charts, TensorBoard curves,
per-machine utilisation plots and the full `eval_results.csv`, they should be in
`rl_training/results_by_setting/`, `rl_training/plots/` and `rl_training/results/` on
the D: machine.**

## Layout

| Path | What it is |
|---|---|
| `scheduler_results_atlas.html` | Interactive comparison page (all 217 results) |
| `data/all_results.csv` / `.json` | Master dataset: one row per recorded result, with protocol, config, status, note and source |
| `data/raw_sources/` | The surviving raw data: 99-row `eval_results.csv` excerpt (18-22 Sep), both reduced-budget raw eval CSVs, Optuna trial/Pareto files |
| `figures/regenerated/offline_constant_weights/` | 27 bar charts, one per protocol x metric |
| `figures/regenerated/offline_random_weights/` | 9 bar charts |
| `figures/regenerated/online_constant_weights/` | 27 bar charts |
| `figures/regenerated/online_random_weights/` | 12 bar charts |
| `figures/original_surviving/` | The 29 original figures still on this machine (reduced-budget v1/v2 comparisons + Gantts, portfolio figures) |
| `scripts/results_data.py` | The dataset itself, transcribed with sources |
| `scripts/build_folder.py` | Rebuilds `data/` and `figures/` (`python Results/ALL_RESULTS_2026-07-24_to_2026-09-22/scripts/build_folder.py` from repo root) |

Figure filenames are `<protocol>__<metric>.png`; protocol keys are defined in
`scripts/results_data.py::PROTOCOLS` (e.g. `off_c_50` = offline, constant weights,
50 held-out instances; `on_r_50` = online, random weights, rho~0.75, 50 held-out).

## Rules for reading the numbers

- Only compare rows **within one protocol** (same instances, same metric).
- **Reward is protocol-local**: the reward function changed on 08-09 (T_j/H rescale),
  09-17 (dense per-tick tardiness) and 09-20 (compute_theta fix).
- Before 2026-09-18, weighted runs reported raw tardiness; weighted tardiness is the
  real objective from then on.
- `status=flag`: real number but achieved via job abandonment, reward hacking,
  memorisation or collapse (or, for the online CP-SAT bound, hindsight).
  `status=superseded`: replaced by a later, more careful measurement.

## Sources

`Future/research/training-log.md` (every entry), the dated docs in `Future/research/`,
`EthanTravelDocs/portfolio-artefacts/data/key_results_excerpt.csv`,
`Results/reduced_budget_2026-09-04*/results_summary.md`, and the six earlier results
artifacts (cross-checked; the Tardiness Ledger's small-instance LST mean reward of
54.76 was wrong -- the per-seed values average to 43.46, used here).
