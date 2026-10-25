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

The app opens on **Which scheduler wins?**: three plain-language steps (your situation: arrival, how busy, deadlines, job importance; what matters: late jobs, long delays, waiting, energy, or an example; Start). The top 2 classic rules and 3 AI schedulers (1 copied a rule, 2 different ones learned by trial and error, no copying) race on a 0-100% bar (100% = best score of the scenario). The **Slides** tab is second. Question-to-metric mapping, plain names and race logic live in `scoring.py` (`FACTORS`, `EXAMPLES`, `friendly_name`, `race_entries`). Tested situation factors: busyness (4 levels), tight deadlines (at normal busyness), online/offline, offline deadline tightness (3 levels), equal vs unequal job importance (offline); not tested: number of machines, job sizes, hard limits on lateness or waiting.

Editable slide deck: `python tools/analysis/make_pptx.py` writes `app/slides/deck.pptx` (needs python-pptx; PDF export: open in PowerPoint). Chart PNGs come from `tools/analysis/make_slides.py`.

**Browser demo variants (one-screen, large text).** `app/web/` holds two layouts of the same race as standalone HTML: `variant_a.html` (controls left, race right) and `variant_b.html` (three step cards on top, race track below). Shared files: `base.css` (tokens, controls), `race_core.js` (scoring and race selection, mirrors `scoring.py`), `race_ui.js` (controls, animation). `python tools/analysis/make_demo_artifacts.py` exports the numbers from the archive (names, kinds and the question-to-metric mapping come from `scoring.py`) and writes `app/web/dist/variant_*.html`. Checked to fit without scrolling at 1280x720, 1366x768 and 1920x1080. Not yet used by the Streamlit app; pick one layout, then either keep the HTML (open the file in a browser, no server needed) or port the choice to `app.py`.

**Browser demo, race behaviour (layout A is the chosen demo).** Nothing is shown until Start is pressed, and any change to step 1 (situation) or step 2 (what matters) clears the race. Racers start in a random order; each has its own speed curve (`race_ui.js`: `build`, `progress`, `update`), lanes are ranked live and slide past each other when one overtakes, and all finish at their final percentage; the winner is outlined at the end. Layout B (`variant_b.html`) predates this lane mechanism and is not maintained.

**Click-to-explain (browser demo).** Every label, legend item, bar and the winner line opens a slide-over card (`app/web/explain.js`): in plain words, the formulation (MathML, no library), the exact numbers of the clicked racer (every measured value with its spread, the best rule's value, the ratio, your weight and the score, then the bar percentage computed step by step), the model's equations (policy, copying loss, PPO objective, shaped reward) and a deeper section with sources, plus links to related topics and a Back button. All text lives in one table in `explain.js`; formulas are copied from `priority_rules.py`, `difficulty.py`, `metrics.py`, `objectives.py` and the policy files. Exported per row by `make_demo_artifacts.py` (via `scoring.model_meta`): option, algorithm, window, copied rule, placement, runs, instances and spread.

**Wording (write your own).** Every sentence the visitor reads is in one file, `app/web/wording.md` (page text, button labels, rule and model descriptions, notes, and all explanation cards). Its header explains the format: change the text after each `key:`, keep the keys, and use `{{tard}}`-style slots to place the formulas the program builds. After editing run `python tools/analysis/make_demo_artifacts.py`, then `python tests/test_demo_scoring.py` (it checks that every card, link and slot still exists) and republish `app/web/dist/variant_a.html`. Formulas stay in `app/web/explain.js`; names and labels used by the Streamlit app also come from this file through `app/wording.py` and `app/scoring.py`. The Streamlit app's own captions (`app/app.py`) are not in the wording file.
