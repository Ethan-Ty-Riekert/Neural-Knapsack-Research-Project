"""placement_rules.py - Machine-selection (placement) rules for baseline
scheduling heuristics.

Given a job already chosen by a priority rule (priority_rules.py) and the
set of machines it fits on at the current time t, pick which machine to
place it on. See Future/research/2026-08-28-classical-heuristic-baselines.md.
"""
import numpy as np


def _remaining_capacity_after(base_env, job, machine, t):
    return base_env.capacity[machine, :, t] - base_env.job_resources[job]


def first_fit(base_env, job, feasible_machines, t):
    """First-Fit (Johnson, 1973, "Near-optimal bin packing algorithms",
    JACM): take the first machine (by index) that fits. This is exactly
    what eval_rl_agent.py's original EDF/SPT/LST dispatch did implicitly,
    since it iterated action ids in ascending job*num_machines+machine
    order -- kept as the default placement rule for those three names so
    every existing eval_results.csv row keeps its original meaning."""
    return feasible_machines[0]


def best_fit(base_env, job, feasible_machines, t):
    """Best-Fit: choose the machine leaving the least total remaining
    capacity (tightest fit) -- the standard multi-dimensional
    generalisation (summed across resource axes) of 1-D best-fit
    (Johnson, 1973)."""
    return min(feasible_machines,
               key=lambda m: (_remaining_capacity_after(base_env, job, m, t).sum(), m))


def worst_fit(base_env, job, feasible_machines, t):
    """Worst-Fit: choose the machine leaving the MOST remaining capacity,
    spreading load rather than packing tightly (Johnson, 1973)."""
    return max(feasible_machines,
               key=lambda m: (_remaining_capacity_after(base_env, job, m, t).sum(), -m))


PLACEMENT_RULES = {
    "FirstFit": first_fit,
    "BestFit": best_fit,
    "WorstFit": worst_fit,
}


def tetris_score(base_env, job, machine, t):
    """Tetris alignment score (Grandl et al., "Multi-resource packing for
    cluster schedulers", SIGCOMM 2014): dot product between the job's
    resource-demand vector and the machine's currently-available capacity
    vector at time t. Higher score means the job's largest demand aligns
    with the machine's largest slack -- packing tightly along the dimension
    that matters most for that job, across ALL resource dimensions at once
    (unlike every rule above, which ranks jobs and machines separately).
    Deliberately left unnormalised, matching Grandl et al.'s own formulation
    (see the dated doc for why normalising would change the ranking here)."""
    return float(np.dot(base_env.job_resources[job], base_env.capacity[machine, :, t]))
