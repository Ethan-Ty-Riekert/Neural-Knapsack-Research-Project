"""test_heavy_tailed_arrivals.py - Regression/validation checks for
generate_poisson_arrivals()'s job_size_distribution="lognormal" option
(2026-09-17, S2W10 -- see
Future/research/2026-09-17-heavy-tailed-arrivals.md).

Follows this project's existing tests/ convention (runnable script with
assert invariants, not a pytest suite) -- see test_online_env.py.

Run from the repo root: python -m tests.test_heavy_tailed_arrivals
"""
import numpy as np

from Code.env.arrival_process import generate_poisson_arrivals

HORIZON = 200
MAX_JOBS = 2000
ARRIVAL_RATE = 2.0
MACHINE_CAPACITY = 30


# ============================================================
# Check 1: "uniform" (default) is byte-identical to the pre-existing
# behaviour -- this feature must be purely additive/opt-in.
# ============================================================
print("=== Check 1: default distribution unchanged (regression safety) ===")
cfg_default = generate_poisson_arrivals(seed=0, arrival_rate=ARRIVAL_RATE, horizon=HORIZON, max_jobs=MAX_JOBS)
cfg_explicit_uniform = generate_poisson_arrivals(seed=0, arrival_rate=ARRIVAL_RATE, horizon=HORIZON,
                                                  max_jobs=MAX_JOBS, job_size_distribution="uniform")
assert np.array_equal(cfg_default["job_durations"], cfg_explicit_uniform["job_durations"])
assert np.array_equal(cfg_default["job_resources"], cfg_explicit_uniform["job_resources"])
print("  default and explicit job_size_distribution='uniform' produce identical draws")


# ============================================================
# Check 2: lognormal mean approximately matches the uniform baseline's mean
# (the mean-matching derivation: mu = ln(target_mean) - sigma^2/2)
# ============================================================
print("=== Check 2: lognormal mean matches uniform baseline mean (rho calibration) ===")
cfg_lognormal = generate_poisson_arrivals(seed=1, arrival_rate=ARRIVAL_RATE, horizon=HORIZON, max_jobs=MAX_JOBS,
                                           job_size_distribution="lognormal")
realized_uniform = cfg_default["job_arrival_times"] <= HORIZON
realized_lognormal = cfg_lognormal["job_arrival_times"] <= HORIZON

uniform_dur_mean = cfg_default["job_durations"][realized_uniform].mean()
lognormal_dur_mean = cfg_lognormal["job_durations"][realized_lognormal].mean()
target_mean = (1 + 10) / 2.0  # job_duration_range default
assert abs(uniform_dur_mean - target_mean) < 1.0, f"uniform sanity check failed: mean={uniform_dur_mean}"
assert abs(lognormal_dur_mean - target_mean) < 1.5, (
    f"lognormal duration mean ({lognormal_dur_mean:.2f}) should stay close to the uniform "
    f"baseline's mean ({target_mean}) -- the whole point of the mean-matching mu derivation"
)
print(f"  uniform duration mean={uniform_dur_mean:.2f}, lognormal duration mean={lognormal_dur_mean:.2f} "
      f"(target={target_mean}) -- both close, as required")


# ============================================================
# Check 3: lognormal is genuinely heavier-tailed (higher variance, a real
# "elephant" tail beyond the uniform baseline's max) -- otherwise this
# wouldn't actually change anything.
# ============================================================
print("=== Check 3: lognormal produces genuine heavy-tail variance ===")
uniform_dur_std = cfg_default["job_durations"][realized_uniform].std()
lognormal_dur_std = cfg_lognormal["job_durations"][realized_lognormal].std()
lognormal_dur_max = cfg_lognormal["job_durations"][realized_lognormal].max()
assert lognormal_dur_std > uniform_dur_std * 1.5, (
    f"lognormal std ({lognormal_dur_std:.2f}) should be meaningfully higher than uniform's "
    f"({uniform_dur_std:.2f}) -- that's the entire point of switching distributions"
)
assert lognormal_dur_max > 10, f"expected at least one 'elephant' job exceeding the old range's max (10), got max={lognormal_dur_max}"
print(f"  uniform std={uniform_dur_std:.2f}, lognormal std={lognormal_dur_std:.2f}, "
      f"lognormal max duration={lognormal_dur_max} (old range max was 10)")


# ============================================================
# Check 4: resource demand is always strictly schedulable on an empty
# machine (never exceeds machine_capacity_value) -- no job should ever be
# impossible to schedule by size alone, only by timing/congestion.
# ============================================================
print("=== Check 4: resource demand never exceeds machine capacity ===")
res = cfg_lognormal["job_resources"][realized_lognormal]
assert res.max() < MACHINE_CAPACITY, (
    f"a job's resource demand ({res.max()}) reached/exceeded machine_capacity_value "
    f"({MACHINE_CAPACITY}) -- this should be impossible by construction (see "
    f"_make_job_size_sampler's res_high clip)"
)
print(f"  max realized per-resource demand={res.max()} (machine_capacity_value={MACHINE_CAPACITY}) -- always fits")


# ============================================================
# Check 5: reproducibility (same seed -> identical draws) survives the new
# sampling path, matching test_online_env.py's existing convention for the
# uniform case.
# ============================================================
print("=== Check 5: lognormal reproducibility ===")
cfg_a = generate_poisson_arrivals(seed=7, arrival_rate=ARRIVAL_RATE, horizon=HORIZON, max_jobs=MAX_JOBS,
                                   job_size_distribution="lognormal")
cfg_b = generate_poisson_arrivals(seed=7, arrival_rate=ARRIVAL_RATE, horizon=HORIZON, max_jobs=MAX_JOBS,
                                   job_size_distribution="lognormal")
assert np.array_equal(cfg_a["job_durations"], cfg_b["job_durations"])
assert np.array_equal(cfg_a["job_resources"], cfg_b["job_resources"])
print("  same seed reproduces identical lognormal draws")


# ============================================================
# Check 6: unknown distribution name raises, rather than silently falling
# back to something unexpected.
# ============================================================
print("=== Check 6: unknown job_size_distribution raises ===")
try:
    generate_poisson_arrivals(seed=0, arrival_rate=ARRIVAL_RATE, horizon=HORIZON, max_jobs=MAX_JOBS,
                               job_size_distribution="bogus")
    raise AssertionError("expected ValueError for an unknown job_size_distribution")
except ValueError:
    print("  unknown distribution name correctly raised ValueError")

print("\nALL HEAVY-TAILED-ARRIVAL CHECKS PASSED")
