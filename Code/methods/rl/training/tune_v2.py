"""tune_v2.py - Hyperparameter search for v2 RL, selected on VALIDATION instances (2026-10-05).

Random search (Bergstra & Bengio 2012, JMLR 13) over the on-policy hyperparameters found to matter
most by Andrychowicz et al. 2021 ("What Matters in On-Policy Reinforcement Learning?", ICLR):
learning rate, rollout size, minibatch size, epochs, discount, GAE lambda, clip range, entropy bonus.

Selection uses validation instances (difficulty-preset seeds 600000..), disjoint from training
(seeds below 500000) and from the reported test instances (500000..500049), so the test numbers
are not used to choose hyperparameters.

    # 1. write trial jobs (one per line: "<tag> <run.py args>") for the queue runner
    python -m Code.methods.rl.training.tune_v2 generate --preset on_rho095 --option 1 --mods c \
        --trials 16 --timesteps 300000 > trials.txt
    # 2. after training, rank trials on validation instances and write best.json
    python -m Code.methods.rl.training.tune_v2 evaluate --preset on_rho095 --option 1 --mods c
"""
import argparse
import csv
import json
import math
import re
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
VALIDATION_SEEDS = range(600_000, 600_020)
OUT_DIR = REPO / "Results" / "v2_objectives" / "tuning"

# Search space. log = sample log-uniformly between the bounds; lists = sample uniformly.
SPACE = {
    "ppo": {
        "learning-rate": ("log", 3e-5, 1e-3),
        "rollout-size": [2048, 4096, 8192, 16384],
        "batch-size": [64, 256, 1024],
        "n-epochs": [3, 5, 10],
        "gamma": [0.99, 0.995, 0.999],
        "gae-lambda": [0.9, 0.95, 0.98],
        "clip-range": [0.1, 0.2, 0.3],
        "ent-coef": ("log", 1e-4, 3e-2),
    },
    "a2c": {
        "learning-rate": ("log", 1e-4, 3e-3),
        "rollout-size": [20, 80, 320],  # 5 / 20 / 80 steps per env with 4 envs
        "gamma": [0.99, 0.995, 0.999],
        "gae-lambda": [0.9, 0.95, 1.0],
        "ent-coef": ("log", 1e-4, 3e-2),
    },
}
MOD_FLAGS = {"c": "--rule-placements FirstFit,Consolidate"}  # "t" goes into --train-args (see cmd_generate)


def sample(space, rng):
    cfg = {}
    for k, v in space.items():
        if isinstance(v, tuple):
            cfg[k] = float(math.exp(rng.uniform(math.log(v[1]), math.log(v[2]))))
        else:
            cfg[k] = v[int(rng.integers(len(v)))]
    if "batch-size" in cfg:
        cfg["batch-size"] = min(cfg["batch-size"], cfg["rollout-size"])
    return cfg


def study_name(a):
    return f"{a.preset}_o{a.option}{a.mods}{'_a2c' if a.algo == 'a2c' else ''}"


def trial_tag(a, k):
    return f"v2_{a.preset}_o{a.option}{a.mods}{'_a2c' if a.algo == 'a2c' else ''}_hp{k}_s{a.seed}"


def cmd_generate(a):
    rng = np.random.default_rng(a.search_seed)
    out = OUT_DIR / study_name(a)
    out.mkdir(parents=True, exist_ok=True)
    trials = {}
    for k in range(a.trials):
        cfg = sample(SPACE[a.algo], rng)
        trials[trial_tag(a, k)] = cfg
        train_args = " ".join(f"--{n} {v:.6g}" if isinstance(v, float) else f"--{n} {v}" for n, v in cfg.items())
        mods = " ".join(MOD_FLAGS[m] for m in a.mods if m in MOD_FLAGS)
        extra = ("--decision-epoch tick " if "t" in a.mods else "")
        print(f"{trial_tag(a, k)} --variant v2_objectives --n-envs 4 --vec-backend subproc --seed {a.seed} "
              f"--preset {a.preset} --method rl-train:{a.option} --algo {a.algo} {mods} --timesteps {a.timesteps} "
              f"--train-args {extra}{train_args}".replace("  ", " "))
    (out / "trials.json").write_text(json.dumps(dict(space=str(SPACE[a.algo]), search_seed=a.search_seed,
                                                     timesteps=a.timesteps, trials=trials), indent=1))


def cmd_evaluate(a):
    import run  # repo-root launcher: reuse its exact v2 RL evaluation path
    from Code.core.difficulty import DIFFICULTIES, generate as generate_difficulty
    from Code.variants import get_variant
    from Code.methods.rl.training.train_action_space_variant import checkpoint_path
    out = OUT_DIR / study_name(a)
    trials = json.loads((out / "trials.json").read_text())["trials"]
    args = types.SimpleNamespace(objectives=None, drop_surcharge=None, lambda_late=1.0, lambda_energy=1.0,
                                 power_model="linear", no_extend_horizon=False, checkpoint_tag=None)
    env_kwargs = get_variant("v2_objectives").env_kwargs(args)
    configs = [generate_difficulty(DIFFICULTIES[a.preset], s) for s in VALIDATION_SEEDS]
    rows = []
    for tag, cfg in trials.items():
        if not checkpoint_path(str(a.option), tag).exists():
            print(f"  {tag}: no model yet")
            continue
        js = [run.run_rl_instance(str(a.option), tag, c, args, env_kwargs)["metrics"]["objective_J"] for c in configs]
        rows.append(dict(tag=tag, val_J_mean=float(np.mean(js)), val_J_std=float(np.std(js, ddof=1)), **cfg))
        print(f"  {tag}: validation J {np.mean(js):.0f}", flush=True)
    rows.sort(key=lambda r: r["val_J_mean"])
    with open(out / "trials_ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = rows[0]
    hp = {k: v for k, v in best.items() if k not in ("tag", "val_J_mean", "val_J_std")}
    (out / "best.json").write_text(json.dumps(dict(best_tag=best["tag"], validation_J=best["val_J_mean"],
                                                   validation_seeds=[VALIDATION_SEEDS.start, VALIDATION_SEEDS.stop - 1],
                                                   hyperparameters=hp), indent=1))
    lines = [f"# Tuning study {study_name(a)} ({len(rows)} trials, ranked on {len(configs)} validation instances)",
             "", "| rank | tag | validation J | " + " | ".join(hp) + " |", "|---|---|---|" + "---|" * len(hp)]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['tag']} | {r['val_J_mean']:.0f} +/- {r['val_J_std']:.0f} | "
                     + " | ".join(f"{r[k]:.3g}" if isinstance(r[k], float) else str(r[k]) for k in hp) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print("best train-args:", " ".join(f"--{k} {v:.6g}" if isinstance(v, float) else f"--{k} {v}" for k, v in hp.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["generate", "evaluate"])
    ap.add_argument("--preset", required=True)
    ap.add_argument("--option", type=int, required=True)
    ap.add_argument("--mods", default="", help="tag modifiers, e.g. c (Consolidate menu), t (per-tick)")
    ap.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0, help="training seed used for every trial")
    ap.add_argument("--search-seed", type=int, default=0, help="seed of the random search itself")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z]*", a.mods):
        ap.error("--mods must be lowercase letters")
    {"generate": cmd_generate, "evaluate": cmd_evaluate}[a.command](a)


if __name__ == "__main__":
    main()
