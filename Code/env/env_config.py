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
):
    rng = np.random.default_rng(seed)

    job_durations = rng.integers(*job_duration_range, size=num_jobs)
    job_resources = rng.integers(*job_resource_range, size=(num_jobs, num_resources))
    job_deadlines = rng.integers(*deadline_range, size=num_jobs)
    job_weights = np.ones(num_jobs)

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
