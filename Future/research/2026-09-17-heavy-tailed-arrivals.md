# Design: Heavy-Tailed (Log-Normal) Job Sizes for the Online Case

**Date:** 2026-09-17 (S2W10)
**Environment:** `Code/env/arrival_process.py`, `Code/training/train_optimized.py`
**Status:** Implemented, opt-in (`job_size_distribution="lognormal"`), regression/validation
tested (`tests/test_heavy_tailed_arrivals.py`). Not yet trained against -- next step.

---

## 1. Motivation

Session finding (see `2026-09-17-retrospective-cpsat-oracle-and-rho-testing.md`): pushing
`rho` toward/above 1 to force lateness doesn't test the thing the online case actually
exists to test. Raising `rho` tests raw throughput under volume -- a purely greedy,
non-anticipatory policy survives it just fine as long as it's fast. It does **not** test
whether a policy can make good decisions *under uncertainty about the future*, which was
the entire point of moving from the offline to the online case in the first place (the
same reasoning that motivated choosing Option 2's tick-advance relaxation and, earlier in
this project, the interest in "hold back capacity for a more important job that might
arrive soon"). The user's own diagnosis: raising `rho` "essentially just makes it
offline" -- a scheduler doesn't need foresight to survive sustained high load, it just
needs speed.

The user asked which realistic-workload lever would create genuine difficulty --
"perfect assignments almost impossible, always something going to be late" -- **without**
relying on raw volume.

## 2. What real cloud workload traces actually show

Three independent, large-scale published cluster-trace studies were checked for how real
job-size/resource-demand distributions actually look, since CLAUDE.md requires grounding
design decisions in a citation rather than an assumed "realistic-sounding" shape:

1. Reiss, C., Tumanov, A., Ganger, G. R., Katz, R. H., & Kozuch, M. A. (2012).
   "Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis." *SoCC 2012*.
2. Cortez, E., Bonde, A., Muzio, A., Russinovich, M., Fontoura, M., & Bianchini, R.
   (2017). "Resource Central: Understanding and Predicting Workloads for Improved
   Resource Management in Large Cloud Platforms." *SOSP 2017* (Microsoft Azure traces).

Both independently report the same qualitative finding: job duration and resource-demand
distributions in real production clusters are **heavily right-skewed** ("mostly mice,
occasionally elephants"), not uniform or even close to symmetric -- a small number of
jobs/VMs consume a disproportionate share of resources and/or run far longer than the
typical job. This is one of the most consistently replicated facts across independent
cloud providers' published trace analyses, which is why it was chosen over the two other
options discussed (bursty/non-homogeneous arrival rate; tighter deadline slack) -- both
plausible, but weaker on independent literature grounding for a specific parametric
shape.

**Honesty note (per CLAUDE.md):** these papers give qualitative skew (log-scale
histograms, percentile tables), not a directly reusable shape parameter for this
project's own (1,10)-range job-size convention. The log-normal family itself is the
standard parametric choice for this kind of skew (used throughout the queueing-theory and
cloud-systems literature for exactly this purpose); the specific `sigma` used here is an
illustrative, untested constant, flagged as such in the code and here.

## 3. Design

### 3.1 Why log-normal, and why mean-matched to the existing uniform range

`generate_poisson_arrivals()` gained `job_size_distribution: "uniform" | "lognormal"`
(default `"uniform"`, fully backward compatible -- see Section 5). Under `"lognormal"`,
both job duration and each resource dimension are drawn from a log-normal distribution
instead of `Uniform(1, 10)`.

The mean is deliberately matched to the uniform baseline's mean (5.5), not left free:
for `X ~ LogNormal(mu, sigma)`, `E[X] = exp(mu + sigma^2/2)` is an exact, standard
property of the distribution (not an approximation) -- solving for `mu` given a target
mean `m` gives `mu = ln(m) - sigma^2/2`, which `_make_job_size_sampler()` uses directly.
This means **switching distributions changes only the variance/skew, not the average
system load** -- this project's own `rho = arrival_rate/12` calibration (derived
elsewhere for the uniform case) stays approximately valid, so a heavy-tailed sweep can be
run at the exact same nominal `rho` values already used for the uniform case, isolating
distribution shape as the only changed variable (this project's established "change one
thing at a time" convention).

### 3.2 Bounds

- **Duration**: clipped to `[1, 4 * old_range_high]` (default up to 40). No tighter
  ceiling is imposed -- a job that arrives too late in the horizon to complete such a
  long duration is a *real, intended* difficulty (a genuine "you couldn't have known this
  was coming and now there's no time left" scenario), not a bug. `exact_solver.py`'s
  `solve_retrospective()` (built the same day) already has to handle exactly this case
  for the retrospective oracle (`unschedulable_by_horizon`), so this isn't a new class of
  problem for the surrounding tooling.
- **Resource demand**: clipped to `[1, min(4 * old_range_high, machine_capacity_value -
  1)]` -- the *additional* cap below `machine_capacity_value` is deliberate and
  different from duration's treatment: a job whose resource demand exceeds every
  machine's total capacity can **never** be scheduled, by any policy, regardless of
  timing or congestion. That would be a degenerate failure mode (untestable scheduling
  skill, just an impossible instance), not the "sometimes there's genuinely no room, and
  you had to have planned for it" difficulty this change is meant to introduce.

### 3.3 Scope: online case only

This only touches `generate_poisson_arrivals()` (the online case). The offline case's
`generate_env_config()` was deliberately left untouched -- changing its distribution
would invalidate every existing offline comparison result this session (CP-SAT floor,
every PPO/A2C variant, all three action-space-reduction options) without being asked for.

## 4. Implementation

- `Code/env/arrival_process.py`: new `_make_job_size_sampler()` helper (builds a
  `sample(rng) -> (duration, resources)` closure once per call, not once per arrival);
  `generate_poisson_arrivals()` gained `job_size_distribution`, `heavy_tail_sigma`
  (default 1.0), `heavy_tail_max_multiplier` (default 4.0) parameters, all backward
  compatible.
- `Code/training/train_optimized.py`: `job_size_distribution` threaded through
  `make_online_resampler()`, `make_env()`, `train_with_optimized_params()` (all 4
  `make_env()` call sites), and a new `--job-size-distribution {uniform,lognormal}` CLI
  flag -- matching this session's established per-feature opt-in-flag convention.

## 5. Validation

`tests/test_heavy_tailed_arrivals.py` (6 checks): default distribution unchanged
(regression safety -- byte-identical draws to before this change); lognormal mean stays
within ~1.5 of the uniform baseline's mean (the mean-matching derivation actually holds
empirically, not just algebraically); lognormal std is meaningfully higher than uniform's
(>1.5x) and produces genuine "elephant" jobs beyond the old range's max; resource demand
never reaches `machine_capacity_value` (no job is ever impossible to schedule by size
alone); same-seed reproducibility; unknown distribution name raises rather than silently
falling back. All 6 pass. `tests/test_online_env.py` re-run clean (no regression).

## 6. Follow-up finding (same day): heavy-tailed sizes alone need moderate-to-high rho to actually differentiate heuristics

Evaluated all 9 `DEFAULT_HEURISTICS` (`Code/baselines/registry.py`) against
`job_size_distribution="lognormal"` at four `rho` levels (`horizon=100`, `max_jobs` sized
generously per Section 3 of `2026-09-17-retrospective-cpsat-oracle-and-rho-testing.md` to
avoid the truncation-into-early-burst artifact found there):

```
rho~0.25 (309 realized):  1 distinct (tardiness,late,scheduled) outcome among 9 heuristics
rho~0.50 (577 realized):  3 distinct outcomes (LPT+WorstFit, Tetris start diverging)
rho~0.75 (864 realized):  9 distinct outcomes -- every heuristic genuinely different
   best: ATC (tardiness=120.00, 799/864 scheduled)  worst: LPT+WorstFit (634.00, 759/864)
rho~1.00 (1199 realized): 9 distinct outcomes, much larger spread
   best: WSPT+BestFit (333.00, 1071/1199)  worst: LPT+WorstFit (1066.00, 941/1199)
```

At `rho~0.25` (the level first tried), every heuristic produced *identical*
tardiness/late/scheduled numbers -- traced to the 11 never-scheduled jobs being exactly
the jobs whose (heavy-tailed) duration made them arrive-too-late-to-ever-finish
regardless of policy (Section 3.2's intended "unschedulable_by_horizon" difficulty). That
is a real difficulty, but it doesn't test *scheduling skill* -- every rule fails on the
same jobs for the same reason, so it can't separate a good policy from a bad one.
Genuine, policy-sensitive differentiation (different heuristics genuinely disagreeing,
producing a real spread of outcomes) only emerged once load was pushed toward
`rho~0.75-1.0` -- i.e., heavy-tailed sizes are the right *shape* lever, but still need a
non-trivial baseline load to create actual resource *contention* between jobs, not just
isolated arrival-timing bad luck. `rho~0.75` is the recommended level for training
comparisons going forward: full differentiation without `rho~1.0`'s much larger,
noisier collapse.

## 7. Next steps (not done here)

- Train Options 1/2/3 (or a plain online PPO/A2C baseline) against
  `job_size_distribution="lognormal"` at `rho~0.75-1.0` (see Section 6 -- not `rho~0.25`,
  which doesn't differentiate) and compare tardiness/jobs_scheduled against the uniform
  baseline, to see whether genuine "elephant" jobs produce unavoidable lateness that a
  myopic/greedy policy can't anticipate but a capacity-reserving one could.
- If this alone doesn't create enough difficulty, bursty (non-homogeneous Poisson rate)
  arrivals was the next-best-grounded lever discussed, not implemented here.
- `job_weights` are still uniformly 1 everywhere in this project (a standing, separately
  flagged fact -- see `Code/baselines/priority_rules.py::wspt_key`'s docstring) --
  pairing heavy-tailed *size* with heavy-tailed/bimodal *weight* (e.g. occasional
  high-priority "VIP" jobs) would be a natural, similarly-grounded follow-up, not
  attempted here.

## 7. References

1. Reiss, C., Tumanov, A., Ganger, G. R., Katz, R. H., & Kozuch, M. A. (2012).
   "Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis." *SoCC 2012*.
2. Cortez, E., Bonde, A., Muzio, A., Russinovich, M., Fontoura, M., & Bianchini, R.
   (2017). "Resource Central: Understanding and Predicting Workloads for Improved
   Resource Management in Large Cloud Platforms." *SOSP 2017*.
