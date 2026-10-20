"""Paired comparison of RL arms against a reference method on one preset's 50 held-out test instances.

An arm is a set of training seeds (run tags). For each arm the report gives
  * the per-seed mean J and the mean over seeds with a percentile-bootstrap 95% interval over SEEDS (uncertainty of
    training; with 3-5 seeds the interval is wide, which is the point);
  * the paired per-instance difference to the reference (mean over the arm's seeds of J_instance minus the
    reference's J_instance): mean difference with a bootstrap 95% interval over INSTANCES, and the number of instances on
    which the arm's seed-mean J is below the reference's.
J is read from Results/v2_objectives/runs/*_<preset>_<method>/per_instance.csv (newest run per method).

    python tools/analysis/paired_report.py --preset on_rho095 --reference EDF+Consolidate \
        --arm "A6 Consolidate 1M=v2_on_rho095_o2cmlubrzex_s0,...s4" ...
Arm syntax: LABEL=tag1,tag2,...   (tags as in the campaign; the rl-eval run folder is looked up from them).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "Results" / "v2_objectives" / "runs"
N_BOOT = 10_000


def per_instance_j(preset, method):
    """J per instance (array) of the newest run of method on preset; method is the run folder suffix."""
    folders = sorted(RUNS.glob(f"*_{preset}_{method}"))
    if not folders:
        raise SystemExit(f"no run folder for {preset} / {method}")
    with open(folders[-1] / "per_instance.csv", encoding="utf-8") as f:
        return np.array([float(r["objective_J"]) for r in csv.DictReader(f)])


def boot_ci(values, rng, stat=np.mean):
    values = np.asarray(values)
    draws = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(N_BOOT)]
    return np.percentile(draws, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", required=True)
    ap.add_argument("--reference", required=True, help="heuristic run-folder suffix, e.g. EDF+Consolidate")
    ap.add_argument("--arm", action="append", required=True, help="LABEL=tag,tag,...")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ref = per_instance_j(a.preset, a.reference)
    print(f"# {a.preset}: arms vs {a.reference} (J {ref.mean():,.0f}, {len(ref)} test instances)\n")
    print("| arm | seeds | per-seed J | mean J | 95% CI over seeds | paired diff vs ref (95% CI over instances) | "
          "instances better |\n|---|---|---|---|---|---|---|")
    for spec in a.arm:
        label, tags = spec.split("=", 1)
        runs = np.array([per_instance_j(a.preset, f"rl-eval-{TAG_OPT(t)}-{t}") for t in tags.split(",")])
        seed_means = runs.mean(axis=1)
        lo, hi = boot_ci(seed_means, rng)
        diff = runs.mean(axis=0) - ref  # per instance, seed-averaged
        dlo, dhi = boot_ci(diff, rng)
        print(f"| {label} | {len(seed_means)} | {' / '.join(f'{x:,.0f}' for x in seed_means)} | {seed_means.mean():,.0f} "
              f"| {lo:,.0f} - {hi:,.0f} | {diff.mean():+,.0f} ({dlo:+,.0f} to {dhi:+,.0f}) | {(diff < 0).sum()} / {len(diff)} |")


def TAG_OPT(tag):
    """The option digit of a campaign tag (v2_<preset>_o<option>...): run folders are named rl-eval-<option>-<tag>."""
    sys.path.insert(0, str(REPO))
    from Code.utils.run_tags import TAG
    return TAG.match(tag)["opt"]


if __name__ == "__main__":
    main()
