"""build_folder.py - export the master dataset and regenerate every comparison figure.

Run from the repo root:  python Results/ALL_RESULTS_2026-07-24_to_2026-09-22/scripts/build_folder.py
"""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(HERE))
from results_data import R, PROTOCOLS, FAMILIES, NO_METRIC_NOTES  # noqa: E402

COLORS = {"rl_options": "#2a78d6", "heuristic": "#eb6834", "rl_legacy": "#1baf7a", "exact": "#eda100", "meta": "#e87ba4"}
METRICS = [("tard", "Total tardiness", "lower is better"), ("wtard", "Weighted tardiness", "lower is better"),
           ("reward", "Episode reward", "higher is better"), ("late", "Late jobs", "lower is better"),
           ("sched", "Jobs scheduled", "higher is better")]
FIELDS = ["id", "date", "case", "weights", "protocol", "protocol_label", "method", "family", "config", "reward",
          "tard", "wtard", "late", "sched", "total", "n", "status", "note", "source"]


def export_data():
    d = ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "all_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(R)
    with open(d / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(protocols=PROTOCOLS, families=FAMILIES, rows=R,
                       no_metric_notes=[dict(date=a, what=b, detail=c) for a, b, c in NO_METRIC_NOTES]), f, indent=1)


def plot(proto, key, label, direction, outdir):
    rows = [r for r in R if r["protocol"] == proto and r[key] is not None]
    if key == "wtard" and PROTOCOLS[proto]["weights"] == "constant":
        return None  # identical to raw tardiness when every weight is 1
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[key], reverse=(direction.startswith("higher")))
    vals = [r[key] for r in rows]
    pos = [v for v in vals if v > 0]
    use_log = key in ("tard", "wtard") and pos and min(vals) >= 0 and max(pos) / max(min(pos), 1) > 30
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(rows) + 1.5))
    names = [r["method"] + (" [flagged]" if r["status"] == "flag" else " [superseded]" if r["status"] == "superseded" else "") for r in rows]
    y = range(len(rows))
    plotv = [math.log10(1 + v) if use_log else v for v in vals]
    ax.barh(list(y), plotv, color=[COLORS[r["family"]] for r in rows],
            hatch=None, edgecolor="white", linewidth=0.8)
    for i, r in enumerate(rows):
        if r["status"] != "ok":
            ax.barh(i, plotv[i], color="none", edgecolor="#444", hatch="///", linewidth=0)
        txt = f"{r[key]:,.2f}" + (f" / {r['total']}" if key == "sched" and r["total"] else "")
        ax.text(plotv[i], i, "  " + txt, va="center", fontsize=8,
                ha="left" if plotv[i] >= 0 else "right")
    ax.set_yticks(list(y), names, fontsize=8)
    ax.invert_yaxis()
    if use_log:
        ticks = [0, 1, 10, 100, 1000, 10000]
        ticks = [t for t in ticks if t <= max(vals) * 1.5]
        ax.set_xticks([math.log10(1 + t) for t in ticks], [str(t) for t in ticks])
    if min(plotv) < 0:
        ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlim(min(0, min(plotv)) * 1.15, max(plotv) * 1.25 if max(plotv) > 0 else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    p = PROTOCOLS[proto]
    ax.set_title(f"{p['case'].title()} / {p['weights']} weights / {p['label']}\n{label} ({direction}"
                 + (", log scale" if use_log else "") + ")", fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f]) for f in FAMILIES if any(r["family"] == f for r in rows)]
    ax.legend(handles, [FAMILIES[f] for f in FAMILIES if any(r["family"] == f for r in rows)], fontsize=7,
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 1.1 / (0.34 * len(rows) + 1.5))
    out = outdir / f"{proto}__{key}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def build_figures():
    base = ROOT / "figures" / "regenerated"
    made = []
    for proto, p in PROTOCOLS.items():
        sub = base / f"{p['case']}_{p['weights']}_weights"
        sub.mkdir(parents=True, exist_ok=True)
        for key, label, direction in METRICS:
            out = plot(proto, key, label, direction, sub)
            if out:
                made.append(out)
    return made


def copy_surviving():
    """Copy every figure/data file that still exists anywhere in the working tree."""
    dst = ROOT / "figures" / "original_surviving"
    copied = []
    for src_dir, tag in [(REPO / "Results" / "reduced_budget_2026-09-04" / "figures", "reduced_budget_v1"),
                         (REPO / "Results" / "reduced_budget_2026-09-04_v2" / "figures", "reduced_budget_v2"),
                         (REPO / "EthanTravelDocs" / "portfolio-artefacts" / "figures", "portfolio")]:
        for f in sorted(src_dir.glob("*.png")):
            (dst / tag).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / tag / f.name)
            copied.append(dst / tag / f.name)
    raw = ROOT / "data" / "raw_sources"
    raw.mkdir(parents=True, exist_ok=True)
    for f, name in [(REPO / "EthanTravelDocs/portfolio-artefacts/data/key_results_excerpt.csv", "eval_results_excerpt_2026-09-18_to_22.csv"),
                    (REPO / "Results/reduced_budget_2026-09-04/raw_eval_results.csv", "reduced_budget_v1_raw_eval_results.csv"),
                    (REPO / "Results/reduced_budget_2026-09-04_v2/raw_eval_results.csv", "reduced_budget_v2_raw_eval_results.csv")]:
        shutil.copy2(f, raw / name)
    for f in (REPO / "rl_training" / "optuna_results").glob("*"):
        (raw / "optuna_results").mkdir(exist_ok=True)
        shutil.copy2(f, raw / "optuna_results" / f.name)
    return copied


if __name__ == "__main__":
    export_data()
    figs = build_figures()
    copied = copy_surviving()
    print(f"rows={len(R)} protocols={len(PROTOCOLS)} regenerated_figures={len(figs)} copied_figures={len(copied)}")
