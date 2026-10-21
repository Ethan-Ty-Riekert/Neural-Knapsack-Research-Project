"""Confirmatory check on FRESH instances: score finished models and heuristics on instances that no decision used.

The 50 test instances of a preset (seeds >= 500000) informed many go / no-go decisions in this project (critic pass 2), so
they act as a soft validation set. This script generates a new set (default seeds 700000 + i, i < N) with the preset's own
generator, scores every listed arm (heuristics by method name, models by campaign tag), and writes per-instance J to a CSV.
Nothing here is used to choose anything; it reports.

    python tools/analysis/fresh_eval.py --preset on_rho095 --n 50 --out Results/v2_objectives/fresh/on_rho095.csv \
        --heuristic EDF+Consolidate --heuristic EDFL+Consolidate \
        --arm "windowed clone only=v2_on_rho095_o2cwimlubrze_s50,v2_on_rho095_o2cwimlubrze_s51"

Arm syntax LABEL=tag1,tag2 (tags as in the campaign); the option digit is read from the tag. Rows already in the CSV are skipped,
so the script can be re-run to add arms.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=700_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heuristic", action="append", default=[])
    ap.add_argument("--arm", action="append", default=[], help="LABEL=tag,tag,...")
    a = ap.parse_args()

    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import run
    from Code.core.difficulty import DIFFICULTIES, generate
    from Code.utils.run_tags import TAG
    from Code.variants import get_variant

    ns = run.build_parser().parse_args(["--variant", "v2_objectives", "--preset", a.preset, "--method", "heuristics"])
    env_kwargs = get_variant("v2_objectives").env_kwargs(ns)
    configs = [generate(DIFFICULTIES[a.preset], a.seed_base + i) for i in range(a.n)]
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["arm"], r["member"]) for r in csv.DictReader(open(out, encoding="utf-8"))}
    new = not out.exists()
    f = open(out, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["arm", "member", "instance", "J"])

    def record(arm, member, js):
        for i, j in enumerate(js):
            w.writerow([arm, member, i, f"{j:.4f}"])
        f.flush()
        print(f"{arm} | {member}: mean J {np.mean(js):,.0f}", flush=True)

    for m in a.heuristic:
        if (m, m) not in done:
            record(m, m, [run.run_one_instance(m, c, ns, env_kwargs)["objective_J"] for c in configs])
    for spec in a.arm:
        label, tags = spec.split("=", 1)
        for tag in tags.split(","):
            if (label, tag) in done:
                continue
            opt = TAG.match(tag)["opt"]
            js = [run.run_rl_instance(opt, tag, c, ns, env_kwargs)["metrics"]["objective_J"] for c in configs]
            run._RL_MODELS.clear()
            record(label, tag, js)
    f.close()


if __name__ == "__main__":
    main()
