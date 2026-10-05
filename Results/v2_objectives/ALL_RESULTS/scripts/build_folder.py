"""build_folder.py - rebuild the v2 results archive (data, per-preset tables, figures) from runs/.

v2 counterpart of Results/v1_legacy_reward/ALL_RESULTS_*/scripts/build_folder.py. Unlike v1 (a
hand-curated list), every number here is read from Results/v2_objectives/runs/*/{run.json,
per_instance.csv}, written by run.py. Rerun after any new run.py result:

    python Results/v2_objectives/ALL_RESULTS/scripts/build_folder.py

Only runs under the current v2 default objective are included (J = sum w_j T_j^2, extended horizon);
older v2 runs (e.g. the 2026-09-29 fixed-window / drop-surcharge runs) are skipped and counted.
For each (preset, method) the newest run is used. RL runs are grouped across seeds by tag
(v2_<preset>_o<option>[c]_s<seed>): mean of per-seed means, spread = std across seeds (n_seeds > 1)
or across instances (n_seeds == 1, flagged).
"""
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT.parent / "runs"
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from Code.variants import get_variant  # noqa: E402

# Current evaluation protocol per preset (number of held-out instances). Runs on an older protocol
# (e.g. the 15-instance difficulty presets used until 2026-10-05) are skipped, so every row of one
# preset is on the identical instance set.
N_INSTANCES = {name: len(p["seeds"]) for name, p in get_variant("v2_objectives").PRESETS.items()}
TB_DIRS = [ROOT.parents[2] / "rl_training" / "models" / "tb_action_space"]  # optional, gitignored

PRESET_ORDER = ["off_c_50", "off_tf02_w1", "off_tf05_w1", "off_tf08_w1", "off_tf02", "off_tf05", "off_tf08", "on_rho050", "on_rho075", "on_rho075_tight",
                "on_rho095", "on_rho110"]
# Validated with the dataviz palette validator (light, all pairs): worst CVD dE 9.2, normal 16.3.
FAMILY_COLORS = {"RL": "#2a78d6", "Heuristic": "#eb6834", "PSO": "#1baf7a", "CP-SAT": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OPTION_NAMES = {"0": "full action space", "1": "rule selection", "2": "priority score", "3": "ATC-prior score",
                "4": "job x machine branching"}
# Tag modifiers (train_action_space_variant.py flags): c = FirstFit+Consolidate menu, a = ATC feature,
# p = pointer network (Option 0; default flat MLP), w = windowed action space, t = per-tick decisions.
MOD_NAMES = {"c": "+Consolidate", "a": "+ATC feature", "p": "pointer", "w": "windowed", "t": "per-tick",
             "n": "non-delay", "f": "placement repair", "m": "Markov obs"}
METRICS = ["objective_J", "on_time_rate", "weighted_tardiness", "max_tardiness", "mean_wait",
           "active_machine_ticks", "dropped"]
RL_TAG = re.compile(r"^v2_(?P<preset>.+?)_o(?P<opt>\d)(?P<mods>[a-z]*)(?P<algo>_a2c)?(?P<hp>_hp\d+|_tuned)?_s(?P<seed>\d+)$")
BASELINES = ["EDF+FirstFit", "LST+FirstFit", "ATC+FirstFit", "RandomRule+FirstFit", "RandomRule+FirstFitConsolidate"]
# Registry back-compat aliases of "<rule>+FirstFit" (Code/methods/heuristics/registry.py): same
# function, so they are folded into the canonical name (newest run wins) instead of listed twice.
ALIASES = {"EDF", "SPT", "LST", "ATC"}


def is_current_objective(env_kwargs):
    obj = (env_kwargs or {}).get("objective") or {}
    return (env_kwargs.get("extend_horizon") is True and obj.get("tardiness_sq") == 1.0
            and not any(obj.get(k) for k in ("tardiness", "drops", "late_count", "energy")))


def rl_label(m, opt):
    """'PPO Opt1 rule selection +Consolidate [tuned]' from a parsed tag (seed excluded, so seeds merge)."""
    if not m:
        return f"PPO Opt{opt} {OPTION_NAMES.get(opt, '')}"
    algo = "A2C" if m.group("algo") else "PPO"
    mods = " ".join(MOD_NAMES.get(c, c) for c in m.group("mods") if c != "m")  # every paper row is Markov
    hp = f" [{m.group('hp').lstrip('_')}]" if m.group("hp") else ""
    return f"{algo} Opt{opt} {OPTION_NAMES.get(opt, '')}" + (f" {mods}" if mods else "") + hp


def family_and_label(method):
    if method.startswith("rl-eval:"):
        _, opt, *tag = method.split(":")
        m = RL_TAG.match(tag[0]) if tag else None
        return "RL", rl_label(m, opt), (m.group("seed") if m else "?")
    if method == "pso":
        return "PSO", "PSO", None
    if method == "cpsat":
        return "CP-SAT", "CP-SAT", None
    return "Heuristic", method, None


def mean_std(xs):
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0)


def load_runs(include_partial_obs=False):
    newest, skipped, stale = {}, 0, 0
    for run_json in sorted(RUNS.glob("*/run.json")):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        if run.get("variant") != "v2_objectives" or not is_current_objective(run.get("env_kwargs")):
            skipped += 1
            continue
        csv_path = run_json.with_name("per_instance.csv")
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != N_INSTANCES.get(run["preset"], -1):
            stale += 1
            continue
        method = run["method"] + "+FirstFit" if run["method"] in ALIASES else run["method"]
        if method.startswith("rl-eval:") and not include_partial_obs:
            tag_m = RL_TAG.match(method.split(":", 2)[2]) if method.count(":") == 2 else None
            if not (tag_m and "m" in tag_m.group("mods")):
                # paper tables: full-MDP (Markov observation) RL runs only, tag modifier "m"
                skipped += 1
                continue
        key = (run["preset"], method)
        if key not in newest or run["timestamp"] > newest[key]["timestamp"]:
            newest[key] = dict(timestamp=run["timestamp"], git=run.get("git_commit"), rows=rows,
                               folder=run_json.parent.name)
    return newest, skipped, stale


def aggregate(newest):
    """-> {preset: [record]} with one record per method (RL seeds merged)."""
    groups = defaultdict(list)
    for (preset, method), run in newest.items():
        family, label, seed = family_and_label(method)
        groups[(preset, label, family)].append((seed, run))
    out = defaultdict(list)
    for (preset, label, family), seeds in groups.items():
        rec = dict(preset=preset, method=label, family=family, n_seeds=len(seeds),
                   n_instances=len(seeds[0][1]["rows"]), git=",".join(sorted({r["git"] or "?" for _, r in seeds})),
                   runs=";".join(r["folder"] for _, r in seeds))
        for metric in METRICS:
            per_run = [[float(r[metric]) for r in run["rows"] if r.get(metric) not in (None, "")] for _, run in seeds]
            if not all(per_run):
                continue
            run_means = [sum(v) / len(v) for v in per_run]
            if len(seeds) > 1:
                rec[metric], rec[f"{metric}_std"] = mean_std(run_means)
                rec[f"{metric}_err"] = rec[f"{metric}_std"]  # training (seed) uncertainty
            else:
                rec[metric], rec[f"{metric}_std"] = mean_std(per_run[0])
                # standard error of the mean over the shared instance set (the std across instances
                # mostly measures how different the instances are, not uncertainty in the mean)
                rec[f"{metric}_err"] = rec[f"{metric}_std"] / math.sqrt(len(per_run[0]))
        rec["spread"] = "std across seeds" if len(seeds) > 1 else "std across instances"
        out[preset].append(rec)
    for preset in out:
        out[preset].sort(key=lambda r: r.get("objective_J", math.inf))
    return out


def write_data(results, skipped, stale=0):
    d = ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    fields = ["preset", "method", "family", "n_seeds", "n_instances", "spread"] + \
             [x for m in METRICS for x in (m, f"{m}_std")] + ["git", "runs"]
    rows = [r for p in PRESET_ORDER for r in results.get(p, [])]
    with open(d / "all_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (d / "build_info.json").write_text(json.dumps(dict(
        n_rows=len(rows), skipped_runs_not_current_objective=skipped,
        skipped_runs_old_instance_protocol=stale, instances_per_preset=N_INSTANCES,
        presets=[p for p in PRESET_ORDER if p in results]), indent=1), encoding="utf-8")
    return rows


def fmt(r, metric, digits=0):
    if metric not in r:
        return "-"
    return f"{r[metric]:.{digits}f} +/- {r[f'{metric}_std']:.{digits}f}"


def write_tables(results):
    t = ROOT / "tables"
    t.mkdir(parents=True, exist_ok=True)
    summary = ["| preset | best heuristic (J) | best RL (J) | RL vs best heuristic | PSO (J) | CP-SAT (J) |",
               "|---|---|---|---|---|---|"]
    for preset in PRESET_ORDER:
        recs = results.get(preset)
        if not recs:
            continue
        lines = [f"# {preset}: all methods (J = sum w_j T_j^2, lower is better)", "",
                 "| rank | method | family | J | on-time rate | weighted tardiness | max tardiness | mean wait | "
                 "active machine-ticks | seeds | instances |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(recs, 1):
            lines.append(f"| {i} | {r['method']} | {r['family']} | {fmt(r, 'objective_J')} | "
                         f"{fmt(r, 'on_time_rate', 3)} | {fmt(r, 'weighted_tardiness')} | {fmt(r, 'max_tardiness', 1)} | "
                         f"{fmt(r, 'mean_wait', 2)} | {fmt(r, 'active_machine_ticks')} | {r['n_seeds']} | {r['n_instances']} |")
        lines += ["", "Spread (+/-): std across seeds for multi-seed RL rows, otherwise std across instances "
                  "(instance heterogeneity; the figures use the standard error instead)."]
        (t / f"{preset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        def best(fam):
            fr = [r for r in recs if r["family"] == fam]
            return fr[0] if fr else None
        h, rl, pso, cp = best("Heuristic"), best("RL"), best("PSO"), best("CP-SAT")
        gap = f"{100 * (rl['objective_J'] / h['objective_J'] - 1):+.1f}%" if (h and rl) else "-"
        rl_cell = (f"{rl['method']} ({rl['objective_J']:.0f}, {rl['n_seeds']} seed{'s' if rl['n_seeds'] > 1 else ''})"
                   if rl else "-")
        pso_cell = f"{pso['objective_J']:.0f}" if pso else "-"
        cp_cell = f"{cp['objective_J']:.0f}" if cp else "-"
        summary.append(f"| {preset} | {h['method']} ({h['objective_J']:.0f}) | {rl_cell} | {gap} | {pso_cell} | {cp_cell} |")
    (t / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return summary


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def family_legend(ax, families):
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_COLORS if f in families]
    ax.legend(handles, [f for f in FAMILY_COLORS if f in families], frameon=False, fontsize=8,
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, labelcolor=MUTED)


def fig_preset_bars(results, figdir):
    for preset in PRESET_ORDER:
        recs = results.get(preset)
        if not recs:
            continue
        heur = [r for r in recs if r["family"] == "Heuristic"]
        keep = [r for r in recs if r["family"] != "Heuristic"] + heur[:6] + \
               [r for r in heur[6:] if r["method"] in BASELINES]
        keep.sort(key=lambda r: r["objective_J"], reverse=True)  # best at the top
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(keep) + 1.2))
        y = range(len(keep))
        bars = ax.barh(list(y), [r["objective_J"] for r in keep], height=0.62,
                color=[FAMILY_COLORS[r["family"]] for r in keep],
                xerr=[r["objective_J_err"] for r in keep], error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        for bar, r in zip(bars, keep):  # texture as secondary encoding: A2C hatched, PPO solid
            if r["method"].startswith("A2C"):
                bar.set_hatch("///")
                bar.set_edgecolor("white")
        ax.set_yticks(list(y), [r["method"] + ("" if r["family"] != "RL" else f"  (n={r['n_seeds']})")
                                for r in keep], color=INK)
        ax.set_xlabel("J = sum of w_j T_j^2, mean over instances (log scale, lower is better)\n"
                      "error bars: std across seeds if n>1, otherwise standard error over instances",
                      fontsize=7)
        ax.set_title(preset, loc="left", fontsize=10, color=INK)
        style(ax)
        positive = [r["objective_J"] for r in keep if r["objective_J"] > 0]
        if positive:  # J spans orders of magnitude across methods (1e4 to 1e9 online)
            ax.set_xscale("log")
            ax.set_xlim(left=min(positive) / 2)
            ax.grid(axis="x", color=GRID, linewidth=0.6, which="both")
        family_legend(ax, {r["family"] for r in keep})
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"J_by_method_{preset}.{ext}", dpi=200)
        plt.close(fig)


def fig_regime_map(results, figdir):
    """Relative J of the best method in each family vs the best heuristic, per preset."""
    presets = [p for p in PRESET_ORDER if p in results]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k, fam in enumerate(["RL", "PSO", "CP-SAT"]):
        xs, ys = [], []
        for i, p in enumerate(presets):
            h = next((r for r in results[p] if r["family"] == "Heuristic"), None)
            f = next((r for r in results[p] if r["family"] == fam), None)
            if h and f and h["objective_J"] > 0:
                xs.append(i + (k - 1) * 0.12)
                ys.append(100 * (f["objective_J"] / h["objective_J"] - 1))
        if xs:
            ax.scatter(xs, ys, s=42, color=FAMILY_COLORS[fam], label=f"best {fam}", zorder=3,
                       edgecolors="white", linewidths=1.5)
    ax.axhline(0, color=FAMILY_COLORS["Heuristic"], lw=1.2, label="best heuristic (0%)")
    ax.set_xticks(range(len(presets)), presets, rotation=30, ha="right")
    ax.set_ylabel("J vs best heuristic (%)", color=MUTED, fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"regime_map.{ext}", dpi=200)
    plt.close(fig)


def fig_heuristic_heatmap(results, figdir):
    """Each heuristic's J relative to the best heuristic on each preset (one hue, light->dark = worse)."""
    presets = [p for p in PRESET_ORDER if p in results]
    names = sorted({r["method"] for p in presets for r in results[p] if r["family"] == "Heuristic"
                    and "+" in r["method"]})
    grid = []
    for name in names:
        row = []
        for p in presets:
            heur = [r for r in results[p] if r["family"] == "Heuristic"]
            r = next((r for r in heur if r["method"] == name), None)
            best = heur[0]["objective_J"] if heur else None
            if not r or best is None:
                row.append(math.nan)
            elif best == 0:  # e.g. off_tf02: every deadline met by the best rules
                row.append(1.0 if r["objective_J"] == 0 else 3.0)
            else:
                row.append(min(r["objective_J"] / best, 3.0))
        grid.append(row)
    if not names:  # no heuristic runs on the current protocol yet
        return
    order = sorted(range(len(names)), key=lambda i: sum(v for v in grid[i] if not math.isnan(v)))
    names, grid = [names[i] for i in order], [grid[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(names) + 1.6))
    im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=1.0, vmax=3.0)
    ax.set_xticks(range(len(presets)), presets, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 2.0 else INK)
    ax.tick_params(colors=MUTED, labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("J / best heuristic on that preset (capped at 3)", color=MUTED, fontsize=7)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"heuristic_regime_heatmap.{ext}", dpi=200)
    plt.close(fig)


def fig_training_curves(figdir):
    """Episode reward vs timesteps from TensorBoard logs, if present on this machine (gitignored)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return 0
    curves = defaultdict(list)
    for tb in TB_DIRS:
        for run_dir in sorted(tb.glob("option*_v2_*")):
            m = re.match(r"option(\d)_(v2_.+)_(\d+)$", run_dir.name)
            tag_m = RL_TAG.match(m.group(2)) if m else None
            if not tag_m:
                continue
            acc = EventAccumulator(str(run_dir))
            acc.Reload()
            if "rollout/ep_rew_mean" not in acc.Tags().get("scalars", []):
                continue
            ev = acc.Scalars("rollout/ep_rew_mean")
            curves[tag_m.group("preset")].append((m.group(2), [e.step for e in ev], [e.value for e in ev]))
    for preset, runs in curves.items():
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        labels = sorted({rl_label(RL_TAG.match(t), RL_TAG.match(t).group("opt")) for t, _, _ in runs})
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
        for tag, xs, ys in runs:
            key = rl_label(RL_TAG.match(tag), RL_TAG.match(tag).group("opt"))
            ax.plot(xs, ys, lw=2, color=palette[labels.index(key) % len(palette)], alpha=0.9,
                    label=key)
        handles, lab = ax.get_legend_handles_labels()
        uniq = dict(zip(lab, handles))
        ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, labelcolor=MUTED)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("episode reward (= -J / scale)", color=MUTED, fontsize=8)
        ax.set_title(f"{preset}: training curves (each line one seed)", loc="left", fontsize=10, color=INK)
        style(ax)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"training_curves_{preset}.{ext}", dpi=200)
        plt.close(fig)
    return len(curves)


def main():
    # --include-partial-obs: also list RL runs trained on the earlier partial observation (no
    # running-job finish times), e.g. for the future with/without-Markov comparison.
    newest, skipped, stale = load_runs(include_partial_obs="--include-partial-obs" in sys.argv)
    results = aggregate(newest)
    rows = write_data(results, skipped, stale)
    summary = write_tables(results)
    figdir = ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_preset_bars(results, figdir)
    fig_regime_map(results, figdir)
    fig_heuristic_heatmap(results, figdir)
    n_curves = fig_training_curves(figdir)
    print(f"{len(rows)} method rows over {len(results)} presets; skipped {skipped} runs (not current objective), "
          f"{stale} runs (old instance protocol); "
          f"training-curve figures for {n_curves} presets")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
