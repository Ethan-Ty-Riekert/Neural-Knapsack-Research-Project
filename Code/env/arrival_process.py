"""arrival_process.py - Job arrival generation for the online (dynamic-arrival) case.

Poisson-process arrivals: N(t) ~ Poisson(rate * t), with independent
Poisson(rate) new arrivals per discrete tick -- the standard queueing-theory
construction (Kleinrock, 1975, "Queueing Systems Vol. 1"), chosen because the
exponential inter-arrival distribution's memoryless property means the
arrival process introduces no correlation structure beyond the single rate
parameter. See mathformulation.tex's "Job Arrivals and Assignments per Tick"
subsection for the full derivation and the arrival-rate selection math.

Deliberately the ONLY place Poisson-specific logic lives -- a future
non-Poisson arrival model (e.g. non-homogeneous/bursty) is a second function
here, not a change to OnlineSchedulingEnv or its Gym wrapper.
"""
import numpy as np


def _make_job_size_sampler(job_duration_range, job_resource_range, num_resources,
                            distribution, sigma, max_multiplier, machine_capacity_value):
    """Return sample(rng) -> (duration:int, resources:np.ndarray[num_resources])
    for one job, per job_size_distribution -- factored out of
    generate_poisson_arrivals() so the sampler is built once per call (not
    once per arrival) and the two distributions' logic stays clearly
    separated. See generate_poisson_arrivals()'s docstring for citations and
    the mean-matching derivation.
    """
    if distribution == "uniform":
        def sample(rng):
            dur = int(rng.integers(*job_duration_range))
            res = rng.integers(*job_resource_range, size=num_resources)
            return dur, res
        return sample

    if distribution == "lognormal":
        dur_mean = (job_duration_range[0] + job_duration_range[1]) / 2.0
        res_mean = (job_resource_range[0] + job_resource_range[1]) / 2.0
        dur_mu = np.log(dur_mean) - 0.5 * sigma ** 2
        res_mu = np.log(res_mean) - 0.5 * sigma ** 2

        dur_low, dur_high_base = job_duration_range
        res_low, res_high_base = job_resource_range
        dur_high = dur_high_base * max_multiplier
        # Strictly below machine_capacity_value (not just clipped to the
        # multiplier) so no job is ever impossible to schedule by size alone
        # -- see docstring.
        res_high = min(res_high_base * max_multiplier, machine_capacity_value - 1)

        def sample(rng):
            dur = int(np.clip(np.round(rng.lognormal(dur_mu, sigma)), dur_low, dur_high))
            res = np.clip(np.round(rng.lognormal(res_mu, sigma, size=num_resources)),
                          res_low, res_high).astype(int)
            return dur, res
        return sample

    raise ValueError(f"Unknown job_size_distribution {distribution!r} (expected 'uniform' or 'lognormal')")


def generate_poisson_arrivals(
    seed: int,
    arrival_rate: float,
    horizon: int,
    max_jobs: int,
    job_duration_range: tuple = (1, 10),
    job_resource_range: tuple = (1, 10),
    deadline_slack_range: tuple = (10, 60),
    num_resources: int = 4,
    num_machines: int = 10,
    machine_capacity_value: int = 30,
    job_size_distribution: str = "uniform",
    heavy_tail_sigma: float = 1.0,
    heavy_tail_max_multiplier: float = 4.0,
    job_weight_range: tuple = None,
) -> dict:
    """Generate one online-case problem instance with Poisson arrivals.

    Same key set as Code.env.env_config.generate_env_config(), plus
    `job_arrival_times` -- so OnlineSchedulingEnv can consume this dict the
    exact same way SchedulingEnv consumes generate_env_config()'s output.

    Fixed-array ("phantom padding") convention: always returns exactly
    `max_jobs` job records, matching GymSchedulingEnv's fixed-num_jobs
    invariant. Only `arrival_rate * horizon` jobs are expected to actually
    arrive on average; any slots beyond the realized arrival count within
    [0, horizon) get `arrival_time = horizon + 1` -- guaranteed unreachable,
    since SchedulingEnv/GymSchedulingEnv episodes always terminate at
    `time > horizon` -- so they are never revealed, never scheduled, and
    never penalised at the terminal step. This lets `num_jobs` stay constant
    across episodes (required by SchedulingEnv.set_jobs()) while the
    *realized* arrival count varies per episode, exactly as an unknown-in-
    advance online arrival stream should.

    Deadlines are relative to arrival (d_j = tau_j + slack), not an absolute
    range independent of tau_j -- an absolute deadline would let a
    late-arriving job be born already overdue, which is not a meaningful
    notion of deadline once arrival itself is random. `deadline_slack_range`
    is (Delta_min, Delta_max) in the paper's notation.

    Job content (duration, resource demand) is drawn from the *same*
    distributions the offline case uses by default (job_size_distribution=
    "uniform"), so online results stay comparable to offline ones on a
    like-for-like basis -- only the *timing* of what's known changes.

    job_size_distribution (added 2026-09-17, S2W10 -- see
    Future/research/2026-09-17-heavy-tailed-arrivals.md for the full
    grounding and findings): "uniform" (default, unchanged) draws duration
    and per-resource demand i.i.d. Uniform(*range) as above. "lognormal"
    instead draws both from a log-normal distribution -- the standard
    parametric family for job-size/resource-demand skew independently
    documented in every major published real-world cluster-trace study this
    project could find (Google: Reiss, Tumanov, Ganger, Katz, Kozuch, 2012,
    "Heterogeneity and Dynamicity of Clouds at Scale: Google Trace
    Analysis," SoCC; Microsoft Azure: Cortez, Bonde, Muzio, Russinovich,
    Fontoura, Bianchini, 2017, "Resource Central: Understanding and
    Predicting Workloads for Improved Resource Management in Large Cloud
    Platforms," SOSP) -- real cloud workloads are consistently "mostly mice,
    occasionally elephants," not uniformly spread. This targets the same
    MEAN as the uniform default (see heavy_tail_sigma below for the
    derivation), so overall system load / this project's own rho =
    arrival_rate/12 calibration stays valid when switching distributions --
    only the *variance/skew* changes, which is the point: an occasional
    much-larger-than-average job is what makes "no idea what's coming next"
    actually matter, without inflating rho into disguised-offline overload
    (the user's own diagnosis this session of why raising rho alone doesn't
    test genuine online decision-making).

    heavy_tail_sigma: log-normal shape parameter (scale of variability in
    log-space). Standard property of the log-normal distribution: if
    X ~ LogNormal(mu, sigma), E[X] = exp(mu + sigma^2/2); solving for mu
    given a target mean m (the uniform range's midpoint here) gives
    mu = ln(m) - sigma^2/2 -- exact algebra, not an approximation, used
    below so E[X] matches the uniform baseline's mean regardless of sigma.
    sigma=1.0 (default) is an illustrative, moderately-heavy-tailed choice
    -- NOT re-derived from these papers' raw trace data (their published
    figures give qualitative skew, not a directly reusable sigma for this
    project's own value ranges) -- flagged per CLAUDE.md's rule that
    untested constants must say so.

    heavy_tail_max_multiplier: clips the log-normal draw's upper tail at
    max_multiplier * range_high (default 4x), so an "elephant" job is
    genuinely large relative to the baseline range without an unbounded
    tail risking pathological single-job outliers. Duration's clip has no
    other ceiling (a job that arrives too late to finish such a long
    duration within the horizon is a real, intended difficulty -- see
    exact_solver.py's solve_retrospective() "unschedulable_by_horizon"
    handling, added the same day for exactly this scenario). Resource
    demand's clip is ADDITIONALLY capped strictly below
    machine_capacity_value, so unlike duration, no job is ever impossible
    to schedule purely by size (regardless of timing/congestion) -- that
    would be a degenerate, uninteresting failure mode (no machine could
    EVER fit it), not a genuine scheduling-skill test.

    job_weight_range (added 2026-09-18, S2W10): see
    Code.env.env_config.generate_env_config()'s matching parameter for the
    full derivation/honesty note -- None (default) keeps every weight at
    1.0 (unchanged); a (low, high) tuple draws each REALIZED job's weight
    i.i.d. Uniform{low,...,high-1} (numpy.integers convention, exclusive
    high). Only realized arrivals get a real draw -- padding slots
    (never revealed) keep the array's np.ones() initialization, which is
    irrelevant since they're never read.
    """
    rng = np.random.default_rng(seed)

    arrival_times = np.full(max_jobs, horizon + 1, dtype=int)
    job_durations = np.zeros(max_jobs, dtype=int)
    job_resources = np.zeros((max_jobs, num_resources), dtype=int)
    job_deadlines = np.zeros(max_jobs, dtype=int)
    job_weights = np.ones(max_jobs)

    sample_job_size = _make_job_size_sampler(
        job_duration_range, job_resource_range, num_resources,
        job_size_distribution, heavy_tail_sigma, heavy_tail_max_multiplier,
        machine_capacity_value,
    )

    n_arrived = 0
    for t in range(horizon):
        if n_arrived >= max_jobs:
            break
        n_new = min(rng.poisson(arrival_rate), max_jobs - n_arrived)
        for _ in range(n_new):
            arrival_times[n_arrived] = t
            job_durations[n_arrived], job_resources[n_arrived] = sample_job_size(rng)
            job_deadlines[n_arrived] = t + rng.integers(*deadline_slack_range)
            if job_weight_range is not None:
                job_weights[n_arrived] = rng.integers(*job_weight_range)
            n_arrived += 1

    machine_capacity = np.array([machine_capacity_value] * num_resources)

    return {
        "job_durations": job_durations,
        "job_resources": job_resources,
        "job_deadlines": job_deadlines,
        "job_weights": job_weights,
        "job_arrival_times": arrival_times,
        "machine_capacity": machine_capacity,
        "num_jobs": max_jobs,
        "num_machines": num_machines,
        "num_resources": num_resources,
        "horizon": horizon,
    }
