"""run.py - one launcher for every problem variant, preset and method.

    python run.py                      # interactive menu
    python run.py --list               # show variants, presets, methods
    python run.py --variant v1_legacy_reward --preset off_c_15 --method EDF
    python run.py --variant v1_legacy_reward --preset off_c_15 --method pso --limit 2
    python run.py --variant v1_legacy_reward --preset off_c_small --method cpsat
    python run.py --variant v1_legacy_reward --preset off_c_50 --method rl-eval:1
    python run.py --variant v1_legacy_reward --preset on_r_50 --method rl-train:3 --timesteps 300000
    python run.py --experiment experiments/pso_vs_edf_off_c_15.yaml

Heuristic / PSO / CP-SAT runs are evaluated here on every instance of the preset
and saved to Results/<variant>/runs/<timestamp>_<preset>_<method>/ as run.json
(git commit, machine, full config, per-instance and mean metrics) plus
per_instance.csv. RL methods delegate to the existing training/evaluation
scripts with the preset's exact flags (those scripts log to eval_results.csv).

Metrics always include jobs_scheduled and dropped next to tardiness: tardiness
alone hides dropped jobs (see Future/research/2026-09-28-objective-redesign-
discussion.md section 1).
"""
import argparse
import dataclasses
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from Code.variants import VARIANTS, get_variant  # noqa: E402
from Code.utils.paths import MACHINE_NAME  # noqa: E402

RL_OPTIONS = ["1", "2", "3", "4"]


# --------------------------------------------------------------------------- methods
def list_methods():
    from Code.methods.heuristics.registry import HEURISTICS
    return (sorted(HEURISTICS) + ["pso", "cpsat"]
            + [f"rl-eval:{o}" for o in RL_OPTIONS] + [f"rl-train:{o}" for o in RL_OPTIONS])


def _metrics(stats, config):
    """Reward (variant-specific) + the reward-independent schedule metrics of
    Code/core/metrics.py (QoS, latency, completion time, energy)."""
    return {"reward": float(stats["total_reward"]), **stats["metrics"]}


_RL_MODELS = {}


def _base_env(env):
    """Walk wrapper .env links down to the raw SchedulingEnv."""
    while not hasattr(env, "start_times"):
        env = env.env
    return env


def run_rl_instance(option, config, args, env_kwargs):
    """v2 RL evaluation: a trained action-space-variant checkpoint (Option 1-4) run deterministically
    on one preset instance under the variant's env (same J/metrics as every other method)."""
    from Code.methods.rl.evaluation.eval_rl_agent import make_env
    from Code.methods.rl.evaluation.eval_action_space_variant import build_eval_env, load_model, run_episode
    gym_env = make_env(config, env_kwargs).env  # strip eval_rl_agent's full-action-space masker
    env = build_eval_env(option, gym_env)
    key = (option, args.checkpoint_tag)
    if key not in _RL_MODELS:
        _RL_MODELS[key] = load_model(option, env, checkpoint_tag=args.checkpoint_tag)
    result = run_episode(_RL_MODELS[key], env)
    base = _base_env(env)
    from Code.core.metrics import schedule_metrics
    return {"total_reward": result["total_reward"], "metrics": schedule_metrics(base),
            "truncated": result["truncated"]}


def run_one_instance(method, config, args, env_kwargs=None):
    from Code.methods.rl.evaluation.eval_rl_agent import run_heuristic
    t0 = time.time()
    extra = {}
    if method.startswith("rl-eval:"):
        stats = run_rl_instance(method.split(":")[1], config, args, env_kwargs)
        extra = {"truncated": int(stats["truncated"])}
    elif method == "pso":
        from Code.methods.metaheuristic.pso import optimize_and_run
        n_jobs = len(config["job_durations"])
        stats = optimize_and_run(config, n_jobs, int(config["num_machines"]), swarm_size=args.pso_swarm,
                                 iterations=args.pso_iterations, seed=args.seed, fitness=args.pso_fitness,
                                 env_kwargs=env_kwargs)
    elif method == "cpsat":
        from Code.methods.exact.exact_solver import solve, replay_schedule
        objective = (env_kwargs or {}).get("objective")
        extended = bool((env_kwargs or {}).get("extend_horizon"))
        online = "job_arrival_times" in config
        drop_b, horizon_override = None, None
        if extended:  # every job mandatory, window extended exactly as the env does
            if online:
                raise SystemExit("cpsat with the extended horizon is offline-only for now "
                                 "(online padding jobs would need excluding from the model).")
            from Code.core.scheduling_env import extension_needed
            horizon_override = int(config["horizon"]) + extension_needed(config["job_durations"])
        elif objective is not None:  # v2 fixed window: optional jobs, same J as the env
            drop_b = objective.drop_surcharge if objective.drop_surcharge is not None else int(config["horizon"])
        lam = {}
        if objective is not None:  # same lateness objective as the env (linear and/or squared)
            if objective.late_count or objective.energy:
                raise SystemExit("cpsat supports the tardiness / tardiness_sq objectives only")
            lam = dict(lambda_linear=objective.tardiness, lambda_squared=objective.tardiness_sq)
        res = solve(config, time_limit_seconds=args.time_limit, num_search_workers=args.cpsat_workers,
                    earliest_start=config["job_arrival_times"] if online else None,
                    enforce_single_start_per_tick=not online, drop_surcharge=drop_b,
                    horizon_override=horizon_override, **lam)
        extra = {"cpsat_status": res["status"], "cpsat_objective": res["objective"]}
        if res["schedule"] is None:
            return dict(extra, seconds=time.time() - t0)
        stats = replay_schedule(config, res["schedule"], env_kwargs)
    else:
        stats = run_heuristic(method, config=config, env_kwargs=env_kwargs)
    return dict(_metrics(stats, config), **extra, seconds=round(time.time() - t0, 3))


_START_COMMIT = None


def _start_commit():
    """The commit (and dirty flag) when this process started -- the code actually loaded, even if
    files are committed or edited while a long run is in progress."""
    global _START_COMMIT
    if _START_COMMIT is None:
        _START_COMMIT = _git_commit()
    return _START_COMMIT


def _git_commit():
    """Short HEAD hash, suffixed '-dirty' if Code/, run.py or experiments/ have uncommitted
    changes (then the hash alone does not identify the code that produced the run)."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "Code", "run.py", "experiments"],
                               cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def evaluate(variant_name, preset_name, method, args):
    variant = get_variant(variant_name)
    preset = variant.PRESETS[preset_name]
    env_kwargs = variant.env_kwargs(args)
    rows = []
    for k, (seed, config) in enumerate(variant.instances(preset_name)):
        if args.limit and k >= args.limit:
            break
        row = dict(seed=seed, **run_one_instance(method, config, args, env_kwargs))
        rows.append(row)
        shown = {key: (round(v, 2) if isinstance(v, float) else v) for key, v in row.items()}
        print(f"  [{k + 1}] {shown}", flush=True)

    numeric = [key for key in rows[0] if key != "seed" and all(isinstance(r.get(key), (int, float)) for r in rows)]
    means = {key: round(float(np.mean([r[key] for r in rows])), 4) for key in numeric}
    print(f"\nMEAN over {len(rows)} instance(s): {means}")

    if args.no_save:
        return means
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = REPO_ROOT / "Results" / variant_name / "runs" / f"{stamp}_{preset_name}_{method.replace(':', '-')}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant": variant_name, "preset": preset_name, "method": method,
        "preset_config": {k: v for k, v in preset.items() if k != "seeds"},
        "seeds": [r["seed"] for r in rows],
        "method_args": {k: getattr(args, k) for k in ("pso_swarm", "pso_iterations", "pso_fitness",
                                                        "time_limit", "cpsat_workers", "seed")},
        "env_kwargs": {k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v)
                       for k, v in env_kwargs.items()},
        "git_commit": _start_commit(), "machine": MACHINE_NAME, "timestamp": stamp,
        "mean": means,
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "per_instance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out.relative_to(REPO_ROOT)}")
    return means


def rl_command(variant_name, preset_name, method, args):
    """Build the RL script command for a preset: v1 = legacy reward; v2 (rl-train only -- v2
    rl-eval runs in-process) = --reward-mode objective with this run's objective flags, and
    --difficulty for the difficulty presets."""
    kind, option = method.split(":")
    preset = get_variant(variant_name).PRESETS[preset_name]
    module = ("Code.methods.rl.evaluation.eval_action_space_variant" if kind == "rl-eval"
              else "Code.methods.rl.training.train_action_space_variant")
    cmd = [sys.executable, "-m", module, "--option", option]
    if variant_name == "v2_objectives":
        if args.no_extend_horizon:
            cmd += ["--no-extend-horizon"]
        cmd += ["--reward-mode", "objective", "--objectives", args.objectives,
                "--lambda-late", str(args.lambda_late), "--lambda-energy", str(args.lambda_energy),
                "--power-model", args.power_model]
        if args.drop_surcharge is not None:
            cmd += ["--drop-surcharge", str(args.drop_surcharge)]
        from Code.core.difficulty import DIFFICULTIES
        if preset_name in DIFFICULTIES:
            cmd += ["--difficulty", preset_name]
            if args.timesteps:
                cmd += ["--timesteps", str(args.timesteps)]
            if args.checkpoint_tag:
                cmd += ["--save-tag", args.checkpoint_tag]
            if args.resume:
                cmd += ["--resume"]
            return cmd
    if preset["case"] == "online":
        cmd += ["--online", "--arrival-rate", str(preset["arrival_rate"]),
                "--online-horizon", str(preset["horizon"]), "--online-max-jobs", str(preset["max_jobs"]),
                "--job-size-distribution", preset["job_size_distribution"]]
    elif (preset["num_jobs"], preset["num_machines"], preset["horizon"]) != (100, 10, 100):
        raise SystemExit(f"{method} supports only the 100-job/10-machine/H=100 scale; {preset_name} differs.")
    if preset["weights"]:
        cmd += ["--job-weight-min", str(preset["weights"][0]), "--job-weight-max", str(preset["weights"][1])]
    if kind == "rl-eval":
        if len(preset["seeds"]) > 1:
            cmd += ["--randomized-eval", "--eval-runs", str(args.limit or len(preset["seeds"]))]
        if args.checkpoint_tag:
            cmd += ["--checkpoint-tag", args.checkpoint_tag]
    else:
        if len(preset["seeds"]) > 1 or preset["case"] == "online":
            cmd += ["--randomize-instances"]
        if args.timesteps:
            cmd += ["--timesteps", str(args.timesteps)]
        if args.checkpoint_tag:
            cmd += ["--save-tag", args.checkpoint_tag]
        if args.resume:
            cmd += ["--resume"]
    return cmd


COMPARE_COLUMNS = ["objective_J", "reward", "dropped", "completed_past_horizon", "weighted_sq_tardiness",
                   "max_tardiness", "weighted_tardiness",
                   "weighted_tardiness_past_horizon", "late_jobs", "on_time_rate",
                   "mean_wait", "mean_flow_time", "active_machine_ticks", "energy_specpower", "seconds"]


def compare(variant_name, preset_name, methods, args):
    """Run several methods on one preset and write a ranked comparison table (markdown)."""
    results = {}
    for m in methods:
        results[m] = dispatch(variant_name, preset_name, m, args, return_means=True)
    key = "objective_J" if all("objective_J" in r for r in results.values()) else "reward"
    order = sorted(results, key=lambda m: results[m][key], reverse=(key == "reward"))
    cols = [c for c in COMPARE_COLUMNS if any(c in r for r in results.values())]
    lines = [f"# {variant_name} / {preset_name}: comparison ({'lower J is better' if key == 'objective_J' else 'higher reward is better'})",
             "", f"git {_start_commit()}, machine {MACHINE_NAME}, {datetime.now():%Y-%m-%d %H:%M}, "
             f"method args: pso {args.pso_swarm}x{args.pso_iterations}, cpsat {args.time_limit}s/{args.cpsat_workers}w, "
             f"objectives: {getattr(args, 'objectives', '-')}, limit: {args.limit or 'all'}", "",
             "| rank | method | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for i, m in enumerate(order, 1):
        cells = [f"{results[m][c]:.2f}" if isinstance(results[m].get(c), float) else str(results[m].get(c, "")) for c in cols]
        lines.append(f"| {i} | {m} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if not args.no_save:
        out = REPO_ROOT / "Results" / variant_name / "comparisons"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{datetime.now():%Y%m%d-%H%M%S}_{preset_name}.md"
        path.write_text(text, encoding="utf-8")
        print(f"saved -> {path.relative_to(REPO_ROOT)}")
    return 0


def dispatch(variant_name, preset_name, method, args, return_means=False):
    variant = get_variant(variant_name)
    if not variant.PRESETS:
        raise SystemExit(f"{variant_name}: {variant.STATUS} -- no presets yet.")
    if preset_name not in variant.PRESETS:
        raise SystemExit(f"unknown preset {preset_name!r} for {variant_name}; see --list")
    if method == "heuristics" or "," in method:
        from Code.methods.heuristics.registry import HEURISTICS
        methods = sorted(HEURISTICS) if method == "heuristics" else [m.strip() for m in method.split(",")]
        return compare(variant_name, preset_name, methods, args)
    if method not in list_methods():
        raise SystemExit(f"unknown method {method!r}; see --list")
    print(f"== {variant_name} / {preset_name} / {method} ==\n   {variant.PRESETS[preset_name]['desc']}")
    if method.startswith("rl-eval:") and variant_name != "v1_legacy_reward":
        means = evaluate(variant_name, preset_name, method, args)  # in-process, v2 env + metrics
        return means if return_means else 0
    if method.startswith("rl-"):
        cmd = rl_command(variant_name, preset_name, method, args)
        print("   running:", " ".join(cmd[1:]))
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode
    means = evaluate(variant_name, preset_name, method, args)
    return means if return_means else 0


# --------------------------------------------------------------------------- UI
def print_listing():
    for name in VARIANTS:
        v = get_variant(name)
        print(f"\n{name}  [{v.STATUS}]\n  {v.DESCRIPTION}")
        for p, cfg in v.PRESETS.items():
            print(f"    {p:<13} {cfg['desc']}  ({len(cfg['seeds'])} instance(s))")
    print("\nmethods:\n  " + ", ".join(list_methods()))


def _choose(prompt, options):
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        s = input(f"{prompt} [1-{len(options)}]: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(options):
            return options[int(s) - 1]
        if s in options:
            return s


def interactive(args):
    names = list(VARIANTS)
    labels = [f"{n}  [{get_variant(n).STATUS}]" for n in names]
    variant_name = names[labels.index(_choose("variant", labels))]
    variant = get_variant(variant_name)
    if not variant.PRESETS:
        raise SystemExit(f"{variant_name}: {variant.STATUS}")
    presets = list(variant.PRESETS)
    preset_name = _choose("preset", [f"{p}: {variant.PRESETS[p]['desc']}" for p in presets]).split(":")[0]
    method = _choose("method", list_methods())
    lim = input("limit to first N instances (blank = all): ").strip()
    args.limit = int(lim) if lim.isdigit() else None
    return dispatch(variant_name, preset_name, method, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--experiment", type=Path, help="YAML file with any of the keys below")
    ap.add_argument("--variant")
    ap.add_argument("--preset")
    ap.add_argument("--method")
    ap.add_argument("--limit", type=int, default=None, help="only the first N instances of the preset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pso-swarm", type=int, default=15)
    ap.add_argument("--pso-iterations", type=int, default=30)
    ap.add_argument("--pso-fitness", choices=["reward", "tardiness"], default="reward")
    ap.add_argument("--objectives", default="tardiness_sq",
                    help="v2 only: comma list from tardiness_sq (squared lateness, default), tardiness, "
                         "late_count, energy (dropped-job cost is always on)")
    ap.add_argument("--drop-surcharge", type=float, default=None,
                    help="v2 only: B in ticks (default: H with --no-extend-horizon, 0 with the extended horizon)")
    ap.add_argument("--no-extend-horizon", action="store_true",
                    help="v2 only: fixed window H with dropped jobs (the pre-2026-09-29 behaviour) instead of "
                         "the default extended horizon, where unfinished jobs run past H and pay true lateness")
    ap.add_argument("--lambda-late", type=float, default=1.0, help="v2 only: weight of late_count")
    ap.add_argument("--lambda-energy", type=float, default=1.0,
                    help="v2 only: weight of energy (late job-ticks per normalised energy unit)")
    ap.add_argument("--power-model", default="linear", choices=["linear", "specpower_ml110g5"],
                    help="v2 only: server power model for the energy objective")
    ap.add_argument("--time-limit", type=float, default=60.0, help="CP-SAT seconds per instance")
    ap.add_argument("--cpsat-workers", type=int, default=8, help="CP-SAT parallel search workers")
    ap.add_argument("--timesteps", type=int, default=None, help="rl-train only")
    ap.add_argument("--checkpoint-tag", default=None, help="rl-eval: tag to load; rl-train: tag to save")
    ap.add_argument("--resume", action="store_true", help="rl-train: continue from the latest checkpoint of this tag")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    _start_commit()  # record provenance before anything runs

    if args.experiment:
        import yaml
        for k, v in yaml.safe_load(args.experiment.read_text(encoding="utf-8")).items():
            setattr(args, k.replace("-", "_"), v)
    if args.list:
        return print_listing()
    if not (args.variant and args.preset and args.method):
        return interactive(args)
    return dispatch(args.variant, args.preset, args.method, args)


if __name__ == "__main__":
    sys.exit(main() or 0)
