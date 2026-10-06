"""difficulty.py - named problem-difficulty settings (v2 build step 4, 2026-09-29).

Two families of knob, each grounded in a standard definition:

1. Load factor rho (online). Offered work per tick divided by capacity per tick, per resource:
       rho_r = lambda * E[P_j * a_jr] / (M * C_r),   rho = max_r rho_r
   (standard queueing utilisation; Kleinrock 1975, bib key kleinrock1975queueing). rho < 1 means
   the cluster can keep up on average, rho >= 1 means work necessarily accumulates. E[P a] is
   estimated from a large fixed-seed sample of the project's own job-size generator, so the same
   code works for uniform and heavy-tailed (log-normal) sizes. The arrival rate for a target rho
   is then lambda = rho / rho(lambda=1).

2. Deadline tightness (offline). The tardiness-factor / due-date-range scheme of Potts & Van
   Wassenhove (1985, bib key potts1985wtardiness): d_j ~ U[C(1 - TF - RDD/2), C(1 - TF + RDD/2)],
   where TF in [0, 1] is how tight deadlines are on average and RDD how spread out they are. Their
   C is the single-machine makespan sum_j P_j; here the natural analogue is a lower bound on this
   environment's makespan: C = max(n_jobs (one start per tick), max_r sum_j P_j a_jr / (M C_r)
   (capacity)). Deadlines are clipped below at P_j (a job must at least be able to finish on time
   if started at t = 0). ADAPTATION NOTE: using this makespan bound in place of sum_j P_j is this
   project's own adaptation of the scheme to a parallel, capacity-constrained machine setting --
   not taken from the paper.

   Online deadline tightness is a slack range added to each job's arrival (deadline = arrival +
   U[slack range]), as already used by arrival_process.py.

The generated instances use the same generators as everything else (generate_env_config,
generate_poisson_arrivals), so every method runs on them unchanged.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

from .env_config import generate_env_config
from .arrival_process import generate_poisson_arrivals


@dataclass(frozen=True)
class Difficulty:
    case: str                                   # "offline" or "online"
    num_jobs: int = 100                         # offline
    num_machines: int = 10
    horizon: int = 100
    weights: Optional[Tuple[int, int]] = (1, 6)  # integer weights 1..5 (rng.integers is exclusive)
    tf: Optional[float] = None                  # offline: tardiness factor (None = original deadlines)
    rdd: float = 0.6                            # offline: range of due dates
    rho: float = 0.75                           # online: load factor
    size_distribution: str = "lognormal"        # online: "uniform" or "lognormal"
    slack: Tuple[int, int] = (10, 60)           # online: deadline slack after arrival
    desc: str = ""

    def as_dict(self):
        return asdict(self)


def makespan_lower_bound(config) -> float:
    """max(one start per tick, capacity bound) -- see module docstring."""
    P = np.asarray(config["job_durations"], dtype=float)
    a = np.asarray(config["job_resources"], dtype=float)
    cap = np.asarray(config["machine_capacity"], dtype=float)
    work = (P[:, None] * a).sum(axis=0) / (int(config["num_machines"]) * cap)
    return float(max(len(P), work.max()))


def apply_tardiness_factor(config, tf: float, rdd: float, seed: int):
    """Replace a config's deadlines with Potts-Van Wassenhove TF/RDD deadlines (in place)."""
    C = makespan_lower_bound(config)
    rng = np.random.default_rng(10_000_000 + seed)  # separate stream from the instance generator
    lo, hi = C * (1 - tf - rdd / 2), C * (1 - tf + rdd / 2)
    d = np.round(rng.uniform(lo, hi, size=len(config["job_durations"])))
    config["job_deadlines"] = np.maximum(d, config["job_durations"]).astype(int)
    return config


_RHO_AT_RATE_1 = {}


def load_at_unit_rate(size_distribution: str, num_machines=10, horizon=100, samples=40) -> float:
    """rho produced by arrival rate 1, estimated from `samples` fixed-seed generated sequences."""
    key = (size_distribution, num_machines, horizon)
    if key not in _RHO_AT_RATE_1:
        vals = []
        for s in range(samples):
            cfg = generate_poisson_arrivals(seed=90_000 + s, arrival_rate=4.0, horizon=horizon, max_jobs=2000,
                                            num_machines=num_machines, job_size_distribution=size_distribution)
            real = cfg["job_arrival_times"] <= horizon
            P = cfg["job_durations"][real][:, None].astype(float)
            a = cfg["job_resources"][real].astype(float)
            vals.append(((P * a).sum(axis=0) / (horizon * num_machines * cfg["machine_capacity"])).max() / 4.0)
        _RHO_AT_RATE_1[key] = float(np.mean(vals))
    return _RHO_AT_RATE_1[key]


def arrival_rate_for(rho: float, size_distribution: str, num_machines=10, horizon=100) -> float:
    return rho / load_at_unit_rate(size_distribution, num_machines, horizon)


def generate(difficulty: Difficulty, seed: int) -> dict:
    """One instance at the given difficulty."""
    d = difficulty
    if d.case == "offline":
        cfg = generate_env_config(seed=seed, num_jobs=d.num_jobs, num_machines=d.num_machines,
                                  horizon=d.horizon, job_weight_range=d.weights)
        if d.tf is not None:
            apply_tardiness_factor(cfg, d.tf, d.rdd, seed)
        return cfg
    rate = arrival_rate_for(d.rho, d.size_distribution, d.num_machines, d.horizon)
    max_jobs = int(np.ceil(rate * d.horizon * 1.3 + 50))  # generous: arrivals must never truncate
    return generate_poisson_arrivals(seed=seed, arrival_rate=rate, horizon=d.horizon, max_jobs=max_jobs,
                                     num_machines=d.num_machines, deadline_slack_range=tuple(d.slack),
                                     job_size_distribution=d.size_distribution, job_weight_range=d.weights)


# Named difficulty presets (evaluated on 50 held-out instances: seeds 500000..500049, set in Code/variants/v2_objectives)
DIFFICULTIES = {
    # offline: deadline tightness sweep at fixed size (100 jobs, 10 machines, H = 100, weights 1..5)
    "off_tf02": Difficulty("offline", tf=0.2, desc="offline, loose deadlines (TF 0.2, RDD 0.6)"),
    "off_tf05": Difficulty("offline", tf=0.5, desc="offline, medium deadlines (TF 0.5, RDD 0.6)"),
    "off_tf08": Difficulty("offline", tf=0.8, desc="offline, tight deadlines (TF 0.8, RDD 0.6)"),
    # Constant-weight counterparts (2026-10-05): identical jobs, durations, resources and deadlines
    # (weights are drawn last from the generator stream), w_j = 1 -- isolates the effect of weights.
    "off_tf02_w1": Difficulty("offline", tf=0.2, weights=None, desc="offline, loose deadlines, unweighted"),
    "off_tf05_w1": Difficulty("offline", tf=0.5, weights=None, desc="offline, medium deadlines, unweighted"),
    "off_tf08_w1": Difficulty("offline", tf=0.8, weights=None, desc="offline, tight deadlines, unweighted"),
    # online: load sweep (heavy-tailed sizes, weights 1..5, slack 10-60)
    "on_rho050": Difficulty("online", rho=0.50, desc="online, light load (rho 0.50)"),
    "on_rho075": Difficulty("online", rho=0.75, desc="online, moderate load (rho 0.75)"),
    "on_rho095": Difficulty("online", rho=0.95, desc="online, heavy load (rho 0.95)"),
    "on_rho110": Difficulty("online", rho=1.10, desc="online, overload (rho 1.10)"),
    # online: deadline tightness at moderate load
    "on_rho075_tight": Difficulty("online", rho=0.75, slack=(5, 25), desc="online, rho 0.75, tight deadlines (slack 5-25)"),
}

# Largest job weight any preset can draw (weights Uniform{1..5}; unweighted presets use w = 1): the fixed
# weight scale of the observation (GymSchedulingEnv fixed scaling).
JOB_WEIGHT_MAX = max(d.weights[1] - 1 for d in DIFFICULTIES.values() if d.weights)


def max_job_duration(difficulty: Difficulty) -> int:
    """Longest job duration a preset can generate, read from the generators' own defaults (generate()
    does not override them): offline Uniform{1..9} (generate_env_config, exclusive upper bound); online
    lognormal clipped at job_duration_range[1] x heavy_tail_max_multiplier = 40 (uniform: 9). Used as the
    capacity look-ahead length K: every processing job finishes within K ticks, so the window shows
    every future capacity release (GymSchedulingEnv.set_markov_obs)."""
    from inspect import signature
    if difficulty.case == "offline":
        return signature(generate_env_config).parameters["job_duration_range"].default[1] - 1
    p = signature(generate_poisson_arrivals).parameters
    high = p["job_duration_range"].default[1]
    if difficulty.size_distribution == "lognormal":
        return int(high * p["heavy_tail_max_multiplier"].default)
    return high - 1
