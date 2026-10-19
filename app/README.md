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

Status (2026-10-20): skeleton. Planned: short method labels, Pareto plot, one-click "best model for this objective",
frozen data bundle for the laptop.
