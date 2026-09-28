# Results

One folder per **problem variant**: a variant fixes the reward/objective and the instance generator.
Numbers are only comparable **within one variant and one preset** (same instances, same reward).

| Variant | Status | Definition | Contents |
|---|---|---|---|
| [`v1_legacy_reward/`](v1_legacy_reward/) | frozen | [`Code/variants/v1_legacy_reward/`](../Code/variants/v1_legacy_reward/README.md) | Every result from 2026-07-24 to 2026-09-28, plus new `runs/` of the legacy reward |
| `v2_objectives/` | planned | [`Code/variants/v2_objectives/`](../Code/variants/v2_objectives/README.md) | Selectable objectives + difficulty; created by the first `run.py` run under v2 |

## Layout inside a variant

```
<variant>/
  runs/<timestamp>_<preset>_<method>/   # one folder per run.py run
      run.json                          #   git commit, machine, preset, method args, mean metrics
      per_instance.csv                  #   one row per instance
  <older archives>                      # v1 only: pre-launcher archives, kept as-is
```

## v1_legacy_reward archives (pre-launcher)

- `ALL_RESULTS_2026-07-24_to_2026-09-22/`: every recorded number (217 rows, 24 protocols), 75
  regenerated figures, and an interactive atlas. The protocol keys in `scripts/results_data.py` are the
  same names as the `run.py` presets (`off_c_15`, `off_c_50`, `on_r_50`, ...). Rebuild with
  `python Results/v1_legacy_reward/ALL_RESULTS_2026-07-24_to_2026-09-22/scripts/build_folder.py`.
- `reduced_budget_2026-09-04/`, `reduced_budget_2026-09-04_v2/`: the time-boxed reduced-scale runs
  (`off_c_rb1` / `off_c_rb2`), with their models.

Raw training artefacts (checkpoints, TensorBoard logs, full `eval_results.csv`) are not in git. See
`README.md`, "Two machines: artefact storage".

## Reading a result honestly

- Always read **jobs scheduled / dropped** alongside tardiness. On `off_c_15`, EDF reaches low tardiness
  partly by dropping about 3 jobs per instance (see `Future/research/training-log.md`, 2026-09-28).
- Legacy-reward values are **not** a measure of scheduling quality. They reward finishing every job
  regardless of lateness (see `Future/research/2026-09-28-objective-redesign-discussion.md`).
