"""test_difficulty.py - Checks for Code/core/difficulty.py (v2 build step 4, 2026-09-29, S2W11):

  1. Online load: the realised bottleneck load of generated instances matches the target rho
     (mean over 15 held-out seeds within 0.05), for every online difficulty preset.
  2. Offline TF/RDD: deadlines tighten monotonically with TF, stay within the scheme's range
     (after the >= P_j clip), and the instance otherwise equals generate_env_config's.
  3. Determinism: the same (difficulty, seed) always gives the same instance.

Run from the repo root: python -m tests.test_difficulty
"""
import numpy as np

from Code.core.difficulty import DIFFICULTIES, generate, makespan_lower_bound
from Code.core.env_config import generate_env_config

SEEDS = range(500_000, 500_015)

for name, d in DIFFICULTIES.items():
    if d.case != "online":
        continue
    loads = []
    for s in SEEDS:
        c = generate(d, s)
        real = c["job_arrival_times"] <= d.horizon
        assert real.sum() < c["num_jobs"], f"{name}: arrivals truncated by max_jobs"
        P = c["job_durations"][real][:, None].astype(float)
        a = c["job_resources"][real].astype(float)
        loads.append(((P * a).sum(axis=0) / (d.horizon * d.num_machines * c["machine_capacity"])).max())
    assert abs(np.mean(loads) - d.rho) < 0.05, (name, np.mean(loads), d.rho)
    print(f"  {name}: realised load {np.mean(loads):.3f} (target {d.rho})")

means = []
for name in ("off_tf02", "off_tf05", "off_tf08"):
    d = DIFFICULTIES[name]
    for s in SEEDS:
        c = generate(d, s)
        base = generate_env_config(seed=s, num_jobs=100, num_machines=10, horizon=100, job_weight_range=(1, 6))
        for k in ("job_durations", "job_resources", "job_weights"):
            assert np.array_equal(c[k], base[k]), (name, k)
        C = makespan_lower_bound(c)
        hi = C * (1 - d.tf + d.rdd / 2)
        assert np.all(c["job_deadlines"] >= c["job_durations"]), name
        assert np.all(c["job_deadlines"] <= max(hi + 0.5, c["job_durations"].max())), name
    means.append(np.mean([generate(d, s)["job_deadlines"].mean() for s in SEEDS]))
assert means[0] > means[1] > means[2], means
print(f"  offline mean deadlines tighten with TF: {[round(m, 1) for m in means]}")

for name, d in DIFFICULTIES.items():
    a, b = generate(d, 500_003), generate(d, 500_003)
    assert all(np.array_equal(a[k], b[k]) for k in a if isinstance(a[k], np.ndarray)), name
print("  deterministic per (difficulty, seed)")
print("PASS test_difficulty")
