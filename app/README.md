# Results explorer (presentation demo)

Streamlit app for the end-of-year presentation: pick a scenario (preset), choose how much each metric matters, and
see the best models ranked, split by family (heuristic / RL / PSO / CP-SAT).

```
pip install streamlit pandas          # in addition to the repository requirements
streamlit run app/app.py
```

- `app.py` - the UI (tab 1: slides from `app/slides/*.png` in filename order; tab 2: explorer).
- `scoring.py` - the scoring logic, independent of Streamlit. A weighted score is the weighted sum of per-metric
  means over the 50 held-out instances; this equals the mean of the weighted objective, so any weighting is scored
  exactly with no retraining. By default each metric is divided by the best heuristic's value on that metric.
- Data: `Results/v2_objectives/ALL_RESULTS/data/all_results.csv`, rebuilt automatically as results arrive
  (`tools/campaign/auto_push.py` / `ALL_RESULTS/scripts/build_folder.py`). Metric names and directions come from
  `build_folder.METRIC_INFO` (the single definition).
- Test: `python -m tests.test_demo_scoring`.

## On the presentation laptop

The explorer reads only `Results/v2_objectives/ALL_RESULTS/data/all_results.csv` (committed and synced by
`tools/campaign/auto_push.py`), so no trained models or training environment are needed:

```
git pull                                  # latest results from the desktop
pip install streamlit==1.66.0 pandas matplotlib numpy
streamlit run app/app.py                  # opens in the browser; works offline
```

The Slides tab shows `app/slides/*.png` in filename order; `python tools/analysis/make_slides.py` rebuilds four slides (headline, fresh-instance
bar chart, scenario overview, caveats) from `Results/v2_objectives/fresh/*.csv`; replace or add PNGs as needed.

Status (2026-10-20): working skeleton with short method labels (`scoring.short_label`) and one-click objectives (`scoring.QUICK_OBJECTIVES`). Trade-off scatter (any two metrics) added. Planned:
frozen data bundle for the laptop.

The app opens on **Race the models**: choose the environment and the metric weights (or a default example), press Start, and the top 2 rules and top 2 learned models race on a 0-100% bar (100% = best score of the scenario). The **Slides** tab is second. Race logic: `race_entries` in `scoring.py`.

Editable slide deck: `python tools/analysis/make_pptx.py` writes `app/slides/deck.pptx` (needs python-pptx; PDF export: open in PowerPoint). Chart PNGs come from `tools/analysis/make_slides.py`.
