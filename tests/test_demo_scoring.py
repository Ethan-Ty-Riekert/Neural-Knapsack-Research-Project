"""test_demo_scoring.py - checks for app/scoring.py (the demo's metric-weighting logic).

Run from the repo root: python -m tests.test_demo_scoring
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from scoring import EXAMPLES, FACTORS, factor_weights, friendly_name, race_entries, best_by_family, score, short_label, QUICK_OBJECTIVES, METRIC_INFO, disambiguate  # noqa: E402

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

# plain-language layer
assert friendly_name("MDC+Consolidate", "Heuristic") == ("Marginal delay cost (MDC)", "classic rule; packs jobs onto few machines")
assert friendly_name("PPO Opt2 priority score windowed +Consolidate +EDF imitation warm start WLST expert", "RL")[0] == "AI scheduler"
assert "trial and error" not in " ".join(friendly_name(m, "RL")[1] for m in ("PPO Opt2 priority score 1M-step budget", "A2C Opt1 rule selection")), "no 'trial and error' in racer text"
assert "copies weighted least slack" in friendly_name("PPO Opt2 priority score windowed +Consolidate +EDF imitation warm start WLST expert", "RL")[1].lower()
assert all(set(m) <= set(METRIC_INFO) for _, m in FACTORS.values()) and all(set(e) <= set(FACTORS) for e in EXAMPLES.values())
assert factor_weights({"late": 2, "delay": 0}) == {"late_jobs": 2.0, "objective_J": 0.0}
print("plain-language checks passed")

# race mix: 1 copied + 2 distinct non-copying learned models, never 'copied + trained'
mix = pd.DataFrame([dict(method=m, family="RL", objective_J=j) for m, j in (
    ("PPO Opt2 windowed +EDF imitation warm start WLST expert", 100.0), ("PPO Opt2 windowed +EDF imitation warm start MDC expert", 101.0),
    ("PPO Opt2 windowed +EDF imitation warm start 1M-step budget", 99.0), ("PPO Opt2 priority score 1M-step budget", 120.0),
    ("PPO Opt3 priority score ATC-prior 1M-step budget", 130.0), ("PPO Opt1 rule selection 1M-step budget", 140.0))]
    + [dict(method="EDF+Consolidate", family="Heuristic", objective_J=110.0), dict(method="LST+Consolidate", family="Heuristic", objective_J=115.0)])
kinds = list(race_entries(score(mix, {"objective_J": 1.0}, normalise=False))["kind"])
assert kinds.count("copied") == 1 and kinds.count("trial and error") == 2 and "copied + trained" not in kinds and kinds.count("rule") == 2, kinds
print("race mix checks passed")

# model metadata for the explanation cards
from scoring import model_meta  # noqa: E402
assert model_meta("MDC+Consolidate", "Heuristic") == {"rule": "MDC", "plc": "Consolidate"}
mm = model_meta("PPO Opt3 priority score +Consolidate windowed +EDF imitation warm start ATC expert", "RL")
assert mm["opt"] == 3 and mm["win"] == 20 and mm["exp"] == "ATC" and mm["kind"] == "copied" and mm["plc"] == "Consolidate", mm
assert model_meta("A2C Opt1 rule selection +Consolidate", "RL")["algo"] == "A2C"
print("model meta checks passed")

# wording file: every card id, link and formula placeholder the page uses must exist
import re as _re  # noqa: E402
from scoring import WORDING  # noqa: E402
_SLOTS = {"tard", "J", "late", "wait", "flow", "energy", "rho", "dueOff", "dueOn", "score", "pct", "copy", "ppo", "reward", "policy",
          "policy_atc", "opt1", "opt0", "racer_table", "racer_sum", "racer_what", "racer_math", "racer_deeper", "weights_table"}
_topics = WORDING["topics"]
for _id, _t in _topics.items():
    for _r in _t["related"]:
        assert _r in _topics, f"{_id}: related card {_r!r} does not exist"
    for _s in _t["sections"]:
        for _slot in _re.findall(r"\{\{(\w+)\}\}", _s["text"]):
            assert _slot in _SLOTS, f"{_id}: unknown formula slot {{{{{_slot}}}}}"
_html = (Path(__file__).resolve().parents[1] / "app" / "web" / "variant_a.html").read_text(encoding="utf-8")
for _id in _re.findall(r'data-explain="(\w+)"', _html) + ["f_" + k for k in FACTORS] + ["arrival", "machines", "deadlines", "importance_jobs", "racer"]:
    assert _id in _topics, f"page links to card {_id!r} that is not in wording.md"
assert all(f"chip.{i}" in WORDING["labels"] for i in _topics if i != "racer"), "every card needs a chip.<id> label in ## LABELS"
print("wording checks passed")
