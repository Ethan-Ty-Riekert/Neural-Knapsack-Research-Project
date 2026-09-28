"""Variant v1 -- the legacy reward (everything produced 2026-07-24 .. 2026-09-28).

Reward (Code/core/scheduling_env.py, reward_mode="legacy", lambda_1=lambda_2=lambda_3=1):
    per valid placement:  -lambda_1 [first use of a machine]  - lambda_2 * w_j * T_j / H
                          - lambda_3 * delta_theta  + 3.0   (+50 when every job is placed)
    per idle tick:        -0.5
Frozen: do not change behaviour here -- it must keep reproducing the recorded
results (e.g. off_c_15 EDF: reward 286.00, tardiness 50.33, late 14.20).

PRESETS are the evaluation protocols of
Results/v1_legacy_reward/ALL_RESULTS_2026-07-24_to_2026-09-22/scripts/results_data.py,
restricted to those whose instances can be regenerated exactly.
"""
from Code.core.env_config import generate_env_config
from Code.core.arrival_process import generate_poisson_arrivals

DESCRIPTION = "Legacy reward: +3/job, +50 completion, bounded tardiness, activation, hotspot, idle terms."
STATUS = "frozen (reproduces recorded results)"

HELDOUT_SEED_BASE = 500_000  # == train_optimized.RANDOM_INSTANCE_SEED_CEILING

_OFF = dict(case="offline", num_jobs=100, num_machines=10, horizon=100)
_ON = dict(case="online", arrival_rate=9.0, horizon=100, max_jobs=1300, job_size_distribution="lognormal")

PRESETS = {
    "off_c_fixed": dict(_OFF, seeds=[0], weights=None,
                        desc="Fixed deployed instance (seed 0), 100 jobs / 10 machines / H=100"),
    "off_c_15": dict(_OFF, seeds=list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 15)), weights=None,
                     desc="15 held-out instances (PSO study)"),
    "off_c_50": dict(_OFF, seeds=list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 50)), weights=None,
                     desc="50 held-out instances (official offline generalisation protocol)"),
    "off_c_small": dict(case="offline", num_jobs=10, num_machines=3, horizon=15,
                        seeds=list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 5)), weights=None,
                        desc="Small exact-solver instances: 10 jobs / 3 machines / H=15 (CP-SAT proves optimality)"),
    "off_c_rb2": dict(case="offline", num_jobs=45, num_machines=6, horizon=50,
                      seeds=list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 8)), weights=None,
                      desc="Reduced-budget v2 scale: 45 jobs / 6 machines / H=50, 8 held-out"),
    "off_r_fixed": dict(_OFF, seeds=[0], weights=(1, 6),
                        desc="Fixed instance with random weights w_j in {1..5}"),
    "off_r_50": dict(_OFF, seeds=list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 50)), weights=(1, 6),
                     desc="50 held-out weighted instances"),
    "on_r_50": dict(_ON, seeds=list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 50)), weights=(1, 6),
                    desc="Canonical online protocol: Poisson rate 9 (rho~0.75), log-normal sizes, weights {1..5}, 50 held-out"),
    "on_r_20": dict(_ON, seeds=list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + 20)), weights=(1, 6),
                    desc="As on_r_50, first 20 sequences"),
}


def instances(preset_name: str):
    """Yield (seed, config dict) for every instance of a preset, generated
    exactly as the original evaluation scripts did."""
    p = PRESETS[preset_name]
    for seed in p["seeds"]:
        if p["case"] == "offline":
            yield seed, generate_env_config(seed=seed, num_jobs=p["num_jobs"], num_machines=p["num_machines"],
                                            horizon=p["horizon"], job_weight_range=p["weights"])
        else:
            yield seed, generate_poisson_arrivals(seed=seed, arrival_rate=p["arrival_rate"], horizon=p["horizon"],
                                                  max_jobs=p["max_jobs"],
                                                  job_size_distribution=p["job_size_distribution"],
                                                  job_weight_range=p["weights"])
