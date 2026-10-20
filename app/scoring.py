"""scoring.py - the demo's scoring logic (no Streamlit): a user-chosen weighting over the recorded metrics.

The weighted objective of a method is the weighted sum of its per-metric means. Every metric is a mean over the
same 50 held-out instances, so the mean of a linear combination equals the combination of the means: any weighting
can be scored exactly from the archive (Results/v2_objectives/ALL_RESULTS/data/all_results.csv) without
retraining. Metrics where higher is better (on-time rate, utilisation) enter with a negative sign.

Normalisation: raw metrics differ by orders of magnitude (J ~ 1e4, makespan ~ 1e2), so by default each metric is
divided by its value for the preset's best heuristic on that metric; weight 1 on two metrics then means equal
importance relative to what a good heuristic achieves.
"""
import sys
from pathlib import Path

import pandas as pd

ARCHIVE = Path(__file__).resolve().parents[1] / "Results" / "v2_objectives" / "ALL_RESULTS"
sys.path.insert(0, str(ARCHIVE / "scripts"))
from build_folder import DISPLAY_METRICS, METRIC_INFO, PRESET_ORDER  # noqa: E402  (single metric definition)

DATA_CSV = ARCHIVE / "data" / "all_results.csv"
HEURISTIC = "Heuristic"


# Full archive labels list every design modifier ("PPO Opt2 priority score look-ahead ... [tuned]"); the demo shows
# the parts an audience cares about. Order = order shown.
_SHORT_TAGS = (("+EDF imitation warm start", "imitation"), ("+Consolidate", "Consolidate"),
               ("+linear-tardiness objective", "+tardiness obj"), ("+late-count objective", "+late-job obj"),
               ("+energy objective", "+energy obj"), ("2M-step budget", "2M steps"), ("1M-step budget", "1M steps"),
               ("[tuned]", "tuned"))


# One-click objectives for the demo: name -> {metric: weight}. Keys must be metrics in METRIC_INFO (tested).
QUICK_OBJECTIVES = {
    "Custom (use the sliders)": None,
    "Squared tardiness (the training objective J)": {"objective_J": 1.0},
    "Fewest late jobs": {"weighted_late_jobs": 1.0},
    "Energy saving (fewest active machine-ticks)": {"active_machine_ticks": 1.0},
    "Balanced: J + late jobs + energy": {"objective_J": 1.0, "weighted_late_jobs": 1.0, "active_machine_ticks": 1.0},
    "Responsiveness (wait and flow time)": {"mean_wait": 1.0, "mean_flow_time": 1.0},
}


def short_label(method):
    """'PPO Opt2 priority score ... 1M-step budget [tuned]' -> 'PPO Opt2 | 1M steps, tuned'. Heuristic and other
    labels are already short and pass through unchanged."""
    if not method.startswith(("PPO", "A2C")):
        return method
    head = " ".join(method.split()[:2])
    tags = [short for key, short in _SHORT_TAGS if key in method]
    return head + (" | " + ", ".join(tags) if tags else "")


def load_results(path=DATA_CSV):
    return pd.read_csv(path)


def presets(df):
    return [p for p in PRESET_ORDER if p in set(df["preset"])]


def metric_label(metric):
    return METRIC_INFO[metric][0]


def reference_values(df_preset):
    """metric -> value of the best heuristic on that metric (the normalisation scale)."""
    refs = {}
    heur = df_preset[df_preset["family"] == HEURISTIC]
    for metric in DISPLAY_METRICS:
        col = heur[metric].dropna() if metric in heur else []
        if len(col):
            higher_better = METRIC_INFO[metric][2]
            ref = col.max() if higher_better else col.min()
            refs[metric] = ref if ref else None  # a zero reference cannot normalise
    return refs


def score(df_preset, weights, normalise=True):
    """-> df_preset with a 'score' column (lower is better); methods missing a weighted metric are dropped.
    weights: {metric: non-negative weight}; zero weights are ignored."""
    active = {m: w for m, w in weights.items() if w}
    refs = reference_values(df_preset) if normalise else {}
    out = df_preset.copy()
    out["score"] = 0.0
    for metric, w in active.items():
        sign = -1.0 if METRIC_INFO[metric][2] else 1.0
        scale = refs.get(metric) if normalise else 1.0
        if not scale:
            out["score"] = float("nan")
            continue
        out["score"] += w * sign * out[metric] / scale
    return out.dropna(subset=["score"]).sort_values("score").reset_index(drop=True)


def best_by_family(scored):
    """-> {family: best row} for the scored frame (already sorted)."""
    return {fam: grp.iloc[0] for fam, grp in scored.groupby("family")}
