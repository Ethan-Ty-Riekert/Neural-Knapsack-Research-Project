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

import numpy as np
import pandas as pd

ARCHIVE = Path(__file__).resolve().parents[1] / "Results" / "v2_objectives" / "ALL_RESULTS"
sys.path.insert(0, str(ARCHIVE / "scripts"))
from build_folder import DISPLAY_METRICS, METRIC_INFO, PRESET_ORDER  # noqa: E402  (single metric definition)

DATA_CSV = ARCHIVE / "data" / "all_results.csv"
HEURISTIC = "Heuristic"


# Full archive labels list every design modifier ("PPO Opt2 priority score look-ahead ... [tuned]"); the demo shows
# the parts an audience cares about. Order = order shown.
_SHORT_TAGS = (("window 40", "window 40"), ("window 60", "window 60"), ("windowed", "window 20"), ("ATC-prior", "ATC prior"), ("+EDF imitation warm start", "imitation"), ("MDC expert", "MDC expert"), ("WLST expert", "WLST expert"),
               ("WMDD expert", "WMDD expert"), ("COVERT expert", "COVERT expert"),
               ("EDF expert", "EDF expert"), ("EDFL expert", "EDFL expert"), ("ATC expert", "ATC expert"), ("+Consolidate", "Consolidate"),
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
    if ("window 40" in method or "window 60" in method) and "window 20" in tags:
        tags.remove("window 20")  # the 'w' (windowed) letter is always present next to the size letter
    if "imitation" in tags:  # clone only = no training budget letter; with one, PPO fine-tuned the clone
        fine_tuned = any(t in tags for t in ("1M steps", "2M steps"))
        tags[tags.index("imitation")] = "clone + PPO" if fine_tuned else "clone only"
    return head + (" | " + ", ".join(tags) if tags else "")


def disambiguate(labels):
    """Labels that occur more than once get a ' #k' suffix (k = 1, 2, ... in input order) so every row of a table is distinct."""
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    seen, out = {}, []
    for lab in labels:
        if counts[lab] > 1:
            seen[lab] = seen.get(lab, 0) + 1
            out.append(f"{lab} #{seen[lab]}")
        else:
            out.append(lab)
    return out


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
            # a zero reference (e.g. no rule has a late job at light load) cannot normalise: use the mean over rules instead,
            # and 1.0 if every rule scores zero (the metric then adds the same 0 to every method)
            refs[metric] = ref if ref else (float(col.abs().mean()) or 1.0)
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


RACE_STEEPNESS = 4.0  # percent = 100 * exp(-RACE_STEEPNESS * relative gap to the best): 5% worse -> 82%, 25% -> 37%, 2x -> 2%
AI_KINDS = ("copied", "copied + trained", "trial and error")  # how a learned scheduler got its skill


def ai_kind(method):
    """Which kind of learned scheduler an archive label is: it copied a rule, copied then trained more, or learned by trial and error."""
    label = short_label(method)
    if "clone only" in label:
        return AI_KINDS[0]
    return AI_KINDS[1] if "clone + PPO" in label else AI_KINDS[2]


AI_MIX = (("copied", 1), ("trial and error", 2))  # the race field: this many learned schedulers of each kind (best first)


def race_entries(scored, per_family=2, ai_mix=AI_MIX):
    """-> frame of the race field: the best `per_family` Heuristic rows plus, per (kind, count) in `ai_mix`, the best learned rows
    of that kind with distinct labels (default: 1 that copied a rule + 2 different ones that did NOT copy; 'copied + trained'
    models are never raced because they started as a copy). A kind with too few models yields fewer racers. Each row gets
    'percent' on a non-linear scale where 100 = the best score among ALL heuristic and RL rows of the scenario (not only the
    racers): percent = 100 * exp(-RACE_STEEPNESS * gap), gap = (score - best) / |best| (or (score - best) / (worst - best)
    when the best score is not positive). The exponential spreads out near-ties so a race is visible; it is a display scale,
    not a statistical statement."""
    pool = scored[scored["family"].isin([HEURISTIC, "RL"])].copy()
    if pool.empty:
        return pool.assign(percent=[], model=[], kind=[])
    pool["model"] = [short_label(m) for m in pool["method"]]
    pool["kind"] = [ai_kind(m) if f == "RL" else "rule" for m, f in zip(pool["method"], pool["family"])]
    best, worst = pool["score"].min(), pool["score"].max()
    scale = abs(best) if best > 0 else ((worst - best) or 1.0)
    pool["percent"] = 100.0 * np.exp(-RACE_STEEPNESS * (pool["score"] - best) / scale)
    picks = [pool[pool["family"] == HEURISTIC].drop_duplicates("model").head(per_family)]
    for kind, count in ai_mix:
        picks.append(pool[(pool["family"] == "RL") & (pool["kind"] == kind)].drop_duplicates("model").head(count))
    return pd.concat(picks).sort_values("score").reset_index(drop=True)


# ---- plain-language layer for the race demo (visitors do not know the rule names or metrics) -------------------------
RULE_NAMES = {"EDFL": "Earliest deadline first (longest first on ties)", "EDF": "Earliest deadline first",
              "WLST": "Weighted least slack", "LST": "Least slack first", "WSPT": "Weighted shortest job first",
              "SPT": "Shortest job first", "LPT": "Longest job first", "WMDD": "Weighted modified due date",
              "MDC": "Modified due date (MDC)", "COVERT": "COVERT (urgency versus cost)",
              "ATC": "Apparent tardiness cost (ATC)", "RandomRule": "Random order"}
OPTION_TEXT = {"Opt0": "picks job and machine directly", "Opt1": "picks which rule to apply", "Opt2": "scores each waiting job",
               "Opt3": "scores each waiting job", "Opt4": "picks a job-machine pair"}
PLACEMENT_NAMES = {"FirstFitConsolidate": "packs jobs onto few machines", "Consolidate": "packs jobs onto few machines",
                   "BestFit": "tightest-fitting machine", "FirstFit": "first machine that fits"}

# What a visitor can care about -> the recorded metrics that measure it (weights are relative; see score()).
FACTORS = {  # key: (question shown, {metric: weight})
    "late": ("Avoid late jobs (how many miss their deadline)", {"late_jobs": 1.0}),
    "delay": ("Avoid long delays (a very late job hurts extra)", {"objective_J": 1.0}),
    "wait": ("Short waiting before a job starts", {"mean_wait": 1.0}),
    "flow": ("Finish each job quickly after it arrives", {"mean_flow_time": 1.0}),
    "energy": ("Save energy (fewer machines switched on)", {"active_machine_ticks": 1.0}),
}
EXAMPLES = {  # one-click examples: label -> {factor key: importance 0-5}
    "Never be late": {"late": 5},
    "Customers hate waiting": {"wait": 5, "flow": 3},
    "Energy Usage": {"energy": 5},
    "A bit of everything": {"late": 2, "delay": 2, "wait": 2, "energy": 2},
}


def factor_weights(importance):
    """{factor key: importance} -> {metric: weight} for score()."""
    out = {}
    for key, level in importance.items():
        for metric, w in FACTORS[key][1].items():
            out[metric] = out.get(metric, 0.0) + level * w
    return out


def friendly_name(method, family):
    """-> (title, detail) in plain language for a method label from the archive."""
    if family == HEURISTIC:
        rule, _, placement = method.partition("+")
        rule = RULE_NAMES.get(rule, rule)
        detail = PLACEMENT_NAMES.get(placement)
        return rule, (f"classic rule; {detail}" if detail else "classic rule")
    label = short_label(method)
    expert = next((r for r in RULE_NAMES if f"{r} expert" in label), None)  # dict order: WLST before LST, EDFL before EDF
    algo, opt = label.split(" | ")[0].split()[:2]
    if "clone only" in label or "clone + PPO" in label:
        what = f"copies {RULE_NAMES[expert].lower()}" if expert else "copies a classic rule"
    else:
        what = OPTION_TEXT.get(opt, "learned model") + (" using an urgency hint" if "ATC prior" in label else "")
    budget = " (2M steps)" if "2M steps" in label else " (1M steps)" if "1M steps" in label else ""
    return "AI scheduler", f"{what[0].upper()}{what[1:]} | {algo}{budget}"  # "[what it does] | [model]"
