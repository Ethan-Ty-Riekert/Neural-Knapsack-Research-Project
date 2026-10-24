"""test_demo_scoring.py - checks for app/scoring.py (the demo's metric-weighting logic).

Run from the repo root: python -m tests.test_demo_scoring
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from scoring import race_entries, best_by_family, score, short_label, QUICK_OBJECTIVES, METRIC_INFO, disambiguate  # noqa: E402

df = pd.DataFrame([
    dict(method="H1", family="Heuristic", objective_J=100.0, on_time_rate=0.9, makespan=50.0),
    dict(method="H2", family="Heuristic", objective_J=200.0, on_time_rate=0.8, makespan=40.0),
    dict(method="R1", family="RL", objective_J=150.0, on_time_rate=0.95, makespan=45.0),
])

s = score(df, {"objective_J": 1.0}, normalise=False)
assert list(s["method"]) == ["H1", "R1", "H2"], "J-only ranking must follow J"

s = score(df, {"objective_J": 1.0, "makespan": 1.0}, normalise=True)  # refs: J 100, makespan 40
assert abs(s.set_index("method").loc["R1", "score"] - (150 / 100 + 45 / 40)) < 1e-9

s = score(df, {"on_time_rate": 1.0}, normalise=False)  # higher is better -> negative sign
assert s.iloc[0]["method"] == "R1"
assert best_by_family(s)["Heuristic"]["method"] == "H1"
assert short_label("EDF+Consolidate") == "EDF+Consolidate"
assert short_label("PPO Opt2 priority score look-ahead fixed scaling 1M-step budget [tuned]") == "PPO Opt2 | 1M steps, tuned"
assert short_label("PPO Opt2 priority score +Consolidate +EDF imitation warm start") == "PPO Opt2 | clone only, Consolidate"
for name, weights in QUICK_OBJECTIVES.items():
    assert weights is None or set(weights) <= set(METRIC_INFO), f"unknown metric in quick objective {name!r}"
assert short_label("PPO Opt2 priority score windowed +Consolidate +EDF imitation warm start 1M-step budget") ==     "PPO Opt2 | window 20, clone + PPO, Consolidate, 1M steps"
assert short_label("PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start") ==     "PPO Opt2 | window 20, clone only, Consolidate"
assert short_label("PPO Opt2 priority score +Consolidate windowed window 40 1M-step budget") == "PPO Opt2 | window 40, Consolidate, 1M steps"
assert disambiguate(["a", "b", "a"]) == ["a #1", "b", "a #2"]
print("test_demo_scoring: all checks passed")

# race field: top-2 per family, percent on an absolute scale (best of all = 100)
race_df = pd.DataFrame([dict(method=f"H{i}", family="Heuristic", objective_J=100.0 + 10 * i) for i in range(4)]
                       + [dict(method=f"PPO Opt2 windowed +EDF imitation warm start {x}", family="RL", objective_J=j)
                          for x, j in (("a", 95.0), ("b", 120.0))])
race_df.loc[4, "method"] = "PPO Opt2 windowed +EDF imitation warm start"
race_df.loc[5, "method"] = "PPO Opt2 windowed +EDF imitation warm start"  # same short label -> raced once
r = race_entries(score(race_df, {"objective_J": 1.0}, normalise=False))
assert list(r["family"]).count("RL") == 1 and list(r["family"]).count("Heuristic") == 2
assert abs(r["percent"].max() - 100.0) < 1e-9 and abs(r.set_index("method").loc["H0", "percent"] - 100 * 2.718281828 ** (-4 * 0.05 / 0.95)) < 1e-3
print("race checks passed")
