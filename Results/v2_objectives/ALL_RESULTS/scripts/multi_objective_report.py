"""multi_objective_report.py - RL vs heuristics when more than one objective counts (2026-10-09).

Uses the archive's own loading and aggregation (build_folder.load_runs / aggregate): every method evaluated on the
50 test instances under the default evaluation, whose per-instance metrics include squared tardiness J, the
weighted late-job count sum_j w_j U_j and active machine-ticks (the linear energy measure). Because the composite
objective is linear in these, its mean over instances is the same combination of the means:

    J_comp = J + lambda_T * weighted tardiness + lambda_U * sum_j w_j U_j + lambda_E * active machine-ticks

with lambda = reference * m, the per-preset reference weights from Code.variants.v2_objectives.REFERENCE_LAMBDAS (each
equalises its term with J on the preset's best J heuristic's schedule; m = the lambda multiplier, 1 when the tag has
no _lam). Writes, per preset with data:

    multi_objective/<preset>_composite.md     best heuristic vs best RL on each composite objective
    multi_objective/<preset>_pareto_*.png      J vs weighted late jobs, J vs machine-ticks: heuristics (and their
                                               non-dominated front) and every RL model, multi-objective runs marked

    python Results/v2_objectives/ALL_RESULTS/scripts/multi_objective_report.py
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_folder as bf  # noqa: E402
from Code.variants.v2_objectives import REFERENCE_LAMBDAS  # noqa: E402

OUT = bf.ROOT / "multi_objective"
M = (0.25, 0.5, 1, 2, 4)
WEIGHTINGS = ([("J only", 0, 0, 0)] + [(f"J + linear tardiness (x{m:g})", m, 0, 0) for m in M]
              + [(f"J + late jobs (x{m:g})", 0, m, 0) for m in M] + [(f"J + energy (x{m:g})", 0, 0, m) for m in M]
              + [("J + late jobs + energy (x1)", 0, 1, 1), ("all four terms (x1)", 1, 1, 1)])


def composite(rec, m_t, m_u, m_e, refs):
    ref_t, ref_u, ref_e = refs
    terms = (("weighted_tardiness", ref_t * m_t), ("weighted_late_jobs", ref_u * m_u),
             ("active_machine_ticks", ref_e * m_e))
    if any(w and rec.get(k) is None for k, w in terms):
        return None
    return rec["objective_J"] + sum(w * (rec.get(k) or 0) for k, w in terms)


def pareto_front(points):
    """Non-dominated subset (both coordinates lower is better), sorted by x."""
    front, best_y = [], math.inf
    for x, y in sorted(points):
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def is_multi(label):
    return "objective" in label  # "+late-count objective" / "+energy objective" from the tag letters q / g


def write_composite(preset, recs):
    refs = REFERENCE_LAMBDAS[preset]
    lines = [f"# {preset}: best heuristic vs best RL on composite objectives (50 test instances, lower is better)", "",
             f"J_comp = J + {refs[0]:g} m_T * weighted tardiness + {refs[1]:g} m_U * weighted late jobs + "
             f"{refs[2]:g} m_E * active machine-ticks.", "",
             "| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |", "|---|---|---|---|---|---|"]
    for name, m_t, m_u, m_e in WEIGHTINGS:
        scored = [(composite(r, m_t, m_u, m_e, refs), r) for r in recs]
        scored = [(v, r) for v, r in scored if v is not None]
        heur = min((x for x in scored if x[1]["family"] == "Heuristic"), default=None, key=lambda x: x[0])
        rl = min((x for x in scored if x[1]["family"] == "RL"), default=None, key=lambda x: x[0])
        if not heur or not rl:
            continue
        diff = (rl[0] - heur[0]) / heur[0] * 100
        lines.append(f"| {name} | {heur[1]['method']} | {heur[0]:,.0f} | {rl[1]['method'][:70]} "
                     f"({rl[1]['n_seeds']} seeds) | {rl[0]:,.0f} | {diff:+.1f}% |")
    (OUT / f"{preset}_composite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def fig_pareto(preset, recs, metric, xlabel, fname):
    import matplotlib.pyplot as plt
    pts = [(r["objective_J"], r[metric], r) for r in recs if r.get(metric) is not None]
    if not any(r["family"] == "Heuristic" for _, _, r in pts):
        return  # nothing to compare against (heuristics evaluated before the metric was recorded)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    heur = [(x, y) for x, y, r in pts if r["family"] == "Heuristic"]
    rl_single = [(x, y) for x, y, r in pts if r["family"] == "RL" and not is_multi(r["method"])]
    rl_multi = [(x, y) for x, y, r in pts if r["family"] == "RL" and is_multi(r["method"])]
    ax.scatter(*zip(*heur), s=14, c="#9aa0a6", label="heuristics")
    front = pareto_front(heur)
    ax.plot(*zip(*front), c="#5f6368", lw=1.2, ls="--", label="heuristics' trade-off front")
    if rl_single:
        ax.scatter(*zip(*rl_single), s=26, c="#2a78d6", marker="o", label="RL (J only)")
    if rl_multi:
        ax.scatter(*zip(*rl_multi), s=34, c="#eb6834", marker="D", label="RL (multi-objective)")
    ax.set_xscale("log")
    ax.set_xlabel("J = weighted squared tardiness (log scale, lower is better)")
    ax.set_ylabel(xlabel + " (lower is better)")
    ax.set_title(f"{preset}: J vs {xlabel}")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    newest, _, _ = bf.load_runs()
    results = bf.aggregate(newest)
    for preset in REFERENCE_LAMBDAS:
        recs = [r for r in results.get(preset, []) if r.get("objective_J") is not None]
        if not recs:
            continue
        lines = write_composite(preset, recs)
        print("\n".join(lines))
        fig_pareto(preset, recs, "weighted_late_jobs", "weighted late jobs", f"{preset}_pareto_late.png")
        fig_pareto(preset, recs, "active_machine_ticks", "active machine-ticks", f"{preset}_pareto_energy.png")


if __name__ == "__main__":
    main()
