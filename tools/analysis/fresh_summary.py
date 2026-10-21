"""Summarise a fresh_eval.py CSV: per-arm means and paired comparisons with every heuristic on the fresh instances.

For each model arm the per-instance J is averaged over the arm's members (seeds); the paired difference to each heuristic is the
mean over instances with a 95% bootstrap interval over instances (the instance set is the only resampled unit; training-seed
uncertainty is shown by the per-member values, not by an interval). Nothing is selected here; all heuristics are reported.

    python tools/analysis/fresh_summary.py Results/v2_objectives/fresh/on_rho095.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10_000


def main(path):
    per = defaultdict(lambda: defaultdict(dict))  # arm -> member -> instance -> J
    for r in csv.DictReader(open(path, encoding="utf-8")):
        per[r["arm"]][r["member"]][int(r["instance"])] = float(r["J"])
    arms = {}
    members = {}
    for arm, mem in per.items():
        mats = [np.array([m[i] for i in sorted(m)]) for m in mem.values()]
        arms[arm] = np.mean(mats, axis=0)
        members[arm] = [x.mean() for x in mats]
    heur = {a: v for a, v in arms.items() if len(per[a]) == 1 and a in per[a]}  # heuristics: arm == member
    models = {a: v for a, v in arms.items() if a not in heur}
    print(f"# fresh instances: {len(next(iter(arms.values())))} per arm\n")
    print("| heuristic | mean J |\n|---|---|")
    for h, v in sorted(heur.items(), key=lambda x: x[1].mean()):
        print(f"| {h} | {v.mean():,.0f} |")
    print("\n| arm | members | per-member mean J | mean J | " + " | ".join(f"vs {h}" for h in heur) + " |")
    print("|---|---|---|---|" + "---|" * len(heur))
    for a, v in models.items():
        cells = []
        for h, hv in heur.items():
            d = v - hv
            rng = np.random.default_rng(0)  # same resampling for every comparison: output does not depend on the CSV size
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cells.append(f"{d.mean():+,.0f} ({lo:+,.0f}, {hi:+,.0f})")
        print(f"| {a} | {len(members[a])} | {' / '.join(f'{x:,.0f}' for x in members[a])} | {v.mean():,.0f} | "
              + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
