"""Build the presentation slides (PNG) from the fresh-instance CSVs: app/slides/01..04_*.png (filename order = slide order).

    python tools/analysis/make_slides.py

Slide text states the honest headline of Future/research/2026-10-21-final-week-findings.md ("For the presentation"). Numbers come
from Results/v2_objectives/fresh/*.csv (50 fresh instances per scenario, never used by any decision, scored for many arms).
"""
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FRESH = ROOT / "Results" / "v2_objectives" / "fresh"
OUT = ROOT / "app" / "slides"
INK, MUTED, HEUR, LEARN, SCRATCH = "#1b1b1b", "#5b5b57", "#9a9a94", "#2a78d6", "#d6693a"
SIZE = (13.333, 7.5)


def load(name):
    """arm -> member -> mean J over the 50 fresh instances; heuristics have arm == member."""
    per = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(FRESH / f"{name}.csv", encoding="utf-8")):
        per[r["arm"]][r["member"]].append(float(r["J"]))
    arms = {a: {m: float(np.mean(v)) for m, v in mem.items()} for a, mem in per.items()}
    heur = {a: next(iter(m.values())) for a, m in arms.items() if len(m) == 1 and a in m}
    models = {a: list(m.values()) for a, m in arms.items() if a not in heur}
    return heur, models


def text_slide(path, title, bullets, foot):
    fig = plt.figure(figsize=SIZE, dpi=120)
    fig.text(0.05, 0.92, textwrap.fill(title, 52), fontsize=27, fontweight="bold", color=INK, va="top")
    y = 0.72
    for b in bullets:
        wrapped = textwrap.fill(b, 88)
        fig.text(0.07, y, "• " + wrapped.replace("\n", "\n   "), fontsize=17, color=INK, va="top")
        y -= 0.075 * (wrapped.count("\n") + 1) + 0.02
    fig.text(0.05, 0.05, foot, fontsize=11, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def slide_fresh(path):
    heur, models = load("on_rho095")
    rows = [(h, v, None, HEUR) for h, v in heur.items() if h != "ATC+Consolidate"]
    for a, vals in models.items():
        if a.startswith("windowed scratch w"):
            continue
        rows.append((a, float(np.mean(vals)), vals, SCRATCH if "scratch" in a else LEARN))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=SIZE, dpi=120)
    for i, (label, mean, vals, colour) in enumerate(rows):
        ax.barh(i, mean, color=colour, alpha=0.55 if vals else 0.9, height=0.62)
        if vals:
            ax.scatter(vals, [i] * len(vals), s=22, color=INK, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]}  ({r[1]:,.0f})" for r in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("mean J on 50 fresh instances (lower is better); dots = individual training seeds", fontsize=12)
    ax.set_title("No learned policy was significantly better than the best heuristic (MDC+Consolidate)\n"
                 "grey = heuristics, blue = learned (clone / fine-tuned / plain RL), orange = learned from scratch",
                 fontsize=15, loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.01, "on_rho095 (10 machines, load 0.95), 50 fresh instances never used by any decision; exploratory, "
             "no multiplicity correction", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def slide_scenarios(path):
    scen = [("on_rho050", "load 0.50"), ("on_rho075", "load 0.75"), ("on_rho075_tight", "load 0.75, tight"),
            ("on_rho110", "overload 1.10"), ("off_tf05", "offline")]
    fig, ax = plt.subplots(figsize=SIZE, dpi=120)
    x = 0
    ticks, labels = [], []
    for name, label in scen:
        heur, models = load(name)
        best = min(heur.values())
        groups = [("best heuristic", [best], HEUR)]
        for a, vals in models.items():
            if "clone only" in a:
                groups.append(("clone", vals, LEARN))
            elif "A3w" in a or "clone+PPO" in a:
                groups.append(("clone + PPO", vals, SCRATCH))
        for gi, (gl, vals, colour) in enumerate(groups):
            ratios = np.array(vals) / best
            ax.bar(x + gi, ratios.mean(), color=colour, alpha=0.6, width=0.8)
            ax.scatter([x + gi] * len(ratios), ratios, s=18, color=INK, zorder=3)
        ticks.append(x + (len(groups) - 1) / 2)
        labels.append(label)
        x += len(groups) + 1
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.set_ylim(0.9, 1.25)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("J relative to the best heuristic of that scenario (fresh instances)", fontsize=12)
    ax.set_title("Clones land close to their heuristic (0-8%, wider at overload); PPO fine-tuning does not improve them",
                 fontsize=15, loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in (HEUR, LEARN, SCRATCH)]
    ax.legend(handles, ["best heuristic", "clone", "clone + PPO"], frameon=False, fontsize=12, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.01, "Dots = training seeds (3-5 per arm; indicative). 50 fresh instances per scenario; exploratory.",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def slide_simple(path):
    """Plain-language result chart for the deck (no rule names): best rule vs learned copies vs learned from scratch."""
    heur, models = load("on_rho095")
    best_name = min(heur, key=heur.get)
    groups = [("Best classic rule", [heur[best_name]], HEUR),
              ("Learned: copies a rule", models["windowed clone only (EDFL)"], LEARN),
              ("Learned: copy + extra training", models["windowed clone+PPO (A3w)"], LEARN),
              ("Learned: from scratch", models["windowed scratch"], SCRATCH)]
    fig, ax = plt.subplots(figsize=(8.0, 4.7), dpi=200)
    for i, (label, vals, colour) in enumerate(groups):
        ax.barh(i, float(np.mean(vals)), color=colour, alpha=0.75 if len(vals) > 1 else 0.95, height=0.6)
        if len(vals) > 1:
            ax.scatter(vals, [i] * len(vals), s=26, color=INK, zorder=3)
        ax.text(float(np.mean(vals)) + 600 if len(vals) == 1 else max(vals) + 800, i, f"{np.mean(vals):,.0f}", va="center", fontsize=12, color=INK)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 66000)
    ax.set_xlabel("Lateness cost on 50 fresh test cases (lower is better); dots = training runs", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


DECK_BLUE, DECK_SOFT, DECK_GREY, DECK_ORANGE, DECK_INK = "#2a6fd1", "#8fb3e8", "#8793a1", "#e0702a", "#0e2238"


def _deck_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd9ee")
    ax.tick_params(colors=DECK_INK, labelsize=14)
    return fig, ax


def deck_chart_headline(path):
    """Final deck, results graph 1: cost on 50 fresh test cases at busy load (lower is better), best rule vs three learned kinds."""
    heur, models = load("on_rho095")
    best = min(heur.values())
    groups = [("Best classic rule", [best], DECK_GREY), ("AI that copies a rule", models["windowed clone only (EDFL)"], DECK_BLUE),
              ("AI: copy + extra training", models["windowed clone+PPO (A3w)"], DECK_SOFT), ("AI learning on its own", models["windowed scratch"], DECK_ORANGE)]
    fig, ax = _deck_axes((7.2, 4.6))
    for i, (label, vals, colour) in enumerate(groups):
        mean = float(np.mean(vals))
        ax.barh(i, mean, color=colour, height=0.62)
        if len(vals) > 1:
            ax.scatter(vals, [i] * len(vals), s=22, color=DECK_INK, zorder=3, alpha=0.75)
        ax.text(max(vals) + 2200, i, f"{mean:,.0f}", va="center", fontsize=15, color=DECK_INK, fontweight="bold")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim(0, 70000)
    ax.set_xlabel("Total lateness cost (lower is better)", fontsize=14, color=DECK_INK)
    fig.tight_layout()
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def deck_chart_scenarios(path):
    """Final deck, results graph 2: the same two learned kinds in five scenarios, cost relative to the best classic rule (1.0)."""
    scen = [("on_rho050", "Quiet"), ("on_rho075", "Normal"), ("on_rho075_tight", "Normal,\ntight"), ("on_rho110", "Overloaded"), ("off_tf05", "Offline")]
    fig, ax = _deck_axes((7.2, 4.6))
    ax.axhline(1.0, color=DECK_GREY, lw=2)
    for k, (name, label) in enumerate(scen):
        heur, models = load(name)
        best = min(heur.values())
        for off, (key, colour) in enumerate((("clone only", DECK_BLUE), ("A3w", DECK_SOFT))):
            vals = next((v for a, v in models.items() if key in a or (key == "A3w" and "clone+PPO" in a)), None)
            if not vals:
                continue
            ratios = np.array(vals) / best
            x = k + (off - 0.5) * 0.36
            ax.bar(x, ratios.mean(), width=0.32, color=colour)
            ax.scatter([x] * len(ratios), ratios, s=16, color=DECK_INK, alpha=0.75, zorder=3)
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([s[1] for s in scen], fontsize=14)
    ax.set_ylim(0.9, 1.3)
    ax.set_ylabel("Cost relative to best rule", fontsize=14, color=DECK_INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (DECK_BLUE, DECK_SOFT)] + [plt.Line2D([0], [0], color=DECK_GREY, lw=3)]
    ax.legend(handles, ["AI copies a rule", "AI: copy + extra training", "Best classic rule (= 1.0)"], frameon=False, fontsize=13, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deck_assets").mkdir(exist_ok=True)
    slide_simple(OUT / "deck_assets" / "result_simple.png")
    deck_chart_headline(OUT / "deck_assets" / "final_chart1.png")
    deck_chart_scenarios(OUT / "deck_assets" / "final_chart2.png")
    text_slide(OUT / "01_headline.png", "Can reinforcement learning beat the best online scheduling heuristic?",
               ["On 50 fresh instances, no learned policy was significantly better than the best heuristic we scored "
                "(MDC+Consolidate at load 0.95).",
                "A policy cloned from a heuristic was, in most cases, not significantly different from it; a few clones "
                "were significantly worse.",
                "Further PPO fine-tuning did not help (lr 1e-5, 3-5 seeds). PPO from scratch with default settings "
                "varied widely across seeds.",
                "With perfect knowledge of future arrivals there is large headroom (3-instance experiment); whether a forecast "
                "could capture it is untested."],
               "Exploratory results; per-seed values and caveats in Future/research/2026-10-21-final-week-findings.md")
    slide_fresh(OUT / "02_fresh_on_rho095.png")
    slide_scenarios(OUT / "03_scenarios.png")
    text_slide(OUT / "04_caveats.png", "What these results do and do not show",
               ["3-5 training seeds per arm; no interval includes training-seed variance; 3-seed verdicts are indicative.",
                "The original test instances informed many decisions and the best heuristic was picked on them; the fresh "
                "instances were scored for ~11 arms with no multiplicity correction.",
                "Fine-tuned arms used lr 1e-5; from-scratch arms used the untuned default. Cloning steps are not "
                "counted in the 1M-step budget.",
                "The windowed policy trains much worse from scratch than the all-slot network: unexplained."],
               "Pick the explorer (app/app.py) to try any metric weighting; its data is the old test set.")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
