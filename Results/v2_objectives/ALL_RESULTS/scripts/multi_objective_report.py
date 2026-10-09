"""multi_objective_report.py - RL vs heuristics when more than one objective counts (2026-10-09).

Uses the archive's own loading and aggregation (build_folder.load_runs / aggregate): every method evaluated on the
50 test instances under the default evaluation, whose per-instance metrics include squared tardiness J, the
weighted late-job count sum_j w_j U_j and active machine-ticks (the linear energy measure). Because the composite
objective is linear in these, its mean over instances is the same combination of the means:

    J_comp(lambda_U, lambda_E) = J + lambda_U * sum_j w_j U_j + lambda_E * active machine-ticks

with lambda_U = 146 * m_U and lambda_E = 153 * m_E (the reference weights equalise each term with J on WLST's
schedule; m = the lambda multiplier, 1 when the tag has no _lam). Writes, per preset with data:

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

OUT = bf.ROOT / "multi_objective"
REF_U, REF_E = 146.0, 153.0
WEIGHTINGS = [("J only", 0, 0)] + [(f"J + late jobs (x{m:g})", m, 0) for m in (0.25, 0.5, 1, 2, 4)] \
    + [(f"J + energy (x{m:g})", 0, m) for m in (0.25, 0.5, 1, 2, 4)] + [("J + late jobs + energy (x1)", 1, 1)]


def composite(rec, m_u, m_e):
    if m_u and rec.get("weighted_late_jobs") is None or m_e and rec.get("active_machine_ticks") is None:
        return None
    return rec["objective_J"] + REF_U * m_u * (rec.get("weighted_late_jobs") or 0) + REF_E * m_e * (rec.get("active_machine_ticks") or 0)


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
    lines = [f"# {preset}: best heuristic vs best RL on composite objectives (50 test instances, lower is better)", "",
             f"J_comp = J + {REF_U:g} m_U * weighted late jobs + {REF_E:g} m_E * active machine-ticks.", "",
             "| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |", "|---|---|---|---|---|---|"]
    for name, m_u, m_e in WEIGHTINGS:
        scored = [(composite(r, m_u, m_e), r) for r in recs]
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
    if not pts:
        return
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
    for preset in ("off_tf05", "on_rho095"):
        recs = [r for r in results.get(preset, []) if r.get("objective_J") is not None]
        if not recs:
            continue
        lines = write_composite(preset, recs)
        print("\n".join(lines))
        fig_pareto(preset, recs, "weighted_late_jobs", "weighted late jobs", f"{preset}_pareto_late.png")
        fig_pareto(preset, recs, "active_machine_ticks", "active machine-ticks", f"{preset}_pareto_energy.png")


if __name__ == "__main__":
    main()
