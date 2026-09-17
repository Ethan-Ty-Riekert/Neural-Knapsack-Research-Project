"""Standardised environment configuration"""

import numpy as np

def generate_env_config(
    seed=0,
    num_jobs=100,
    num_machines=10,
    num_resources=4,
    horizon=110,
    job_duration_range=(1, 10),
    job_resource_range=(1, 10),
    deadline_range=(10, 110),
    machine_capacity_value=30,
    job_weight_range=None,
):
    """job_weight_range (added 2026-09-18, S2W10): None (default, unchanged)
    keeps every job_weights entry at 1.0 -- the historic behaviour every
    "final" result in this project (including CP-SAT's proven 8.0 floor and
    every EDF/LST/ATC reference number) was generated and compared under, so
    the seed=0 fixed instance stays byte-identical unless this is passed
    explicitly. Passing a (low, high) tuple (numpy.integers convention:
    inclusive low, EXCLUSIVE high, matching job_duration_range/
    job_resource_range's existing semantics) instead draws each job's weight
    i.i.d. Uniform{low, ..., high-1}, so the reward's lambda_2*w_j*T_j term
    and WSPT/ATC's w_j/p_j ratio actually vary per job -- previously dead
    code, since every instance this project ever generated set w_j=1
    uniformly (see Code/baselines/priority_rules.py::wspt_key's own
    docstring, which already flagged this). No literature-grounded
    distribution shape is claimed here (job "weight" is a business-priority
    abstraction, not a physically-measurable trace quantity like duration or
    resource demand, so the heavy-tailed-arrivals doc's Google/Azure
    cloud-trace grounding does not directly transfer) -- flagged per
    CLAUDE.md's rule that untested constants/choices must say so; a
    literature-grounded bimodal "VIP job" alternative (mostly weight=1,
    occasionally much higher) was proposed but not implemented here.
    """
    rng = np.random.default_rng(seed)

    job_durations = rng.integers(*job_duration_range, size=num_jobs)
    job_resources = rng.integers(*job_resource_range, size=(num_jobs, num_resources))
    job_deadlines = rng.integers(*deadline_range, size=num_jobs)
    job_weights = (
        np.ones(num_jobs) if job_weight_range is None
        else rng.integers(*job_weight_range, size=num_jobs).astype(float)
    )

    machine_capacity = np.array([machine_capacity_value] * num_resources)

    return {
        "job_durations": job_durations,
        "job_resources": job_resources,
        "job_deadlines": job_deadlines,
        "job_weights": job_weights,
        "machine_capacity": machine_capacity,
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "num_resources": num_resources,
        "horizon": horizon,
    }
