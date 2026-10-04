"""obs_atc_feature.py - Shared per-job-slot ATC-priority observation feature,
factored out (2026-09-23, S2W9) from priority_only_gym_wrapper.py's Option 3
(use_atc=True) so rule_selection_gym_wrapper.py's new use_atc_feature flag
can reuse the identical slot-layout logic instead of duplicating it across
two wrapper files (per the project's modular-design convention -- one place
to change if the ATC feature computation ever needs to change).

Motivated by Future/research/training-log.md's 2026-09-23 observation-
informativeness-probe entry: a linear/nonlinear probe found the raw online
observation only weakly encodes SPT-vs-ATC job-choice disagreement (AUC
0.62/0.65), suggesting a representation gap rather than a pure optimization
one. This feature gives the network an explicit ATC-priority signal per job
slot, the same one Option 3 already benefits from, rather than requiring it
to be re-derived from raw duration/deadline/weight/time features.
"""
import numpy as np

from Code.methods.heuristics.priority_rules import atc_priority, _atc_mean_p


def append_atc_priority_feature(base_obs, base_env, max_jobs, job_slot_width, machine_block_end):
    """Appends one per-job-slot feature -- the ATC composite priority index
    (Code/methods/heuristics/priority_rules.py::atc_priority, clamped to [0,1]) -- to
    an existing GymSchedulingEnv/OnlineGymSchedulingEnv-layout observation.

    base_obs: the [time, machine_block, job_slots...] observation as produced
    by GymSchedulingEnv._get_obs() (or the online equivalent), UNCHANGED.
    base_env: the raw SchedulingEnv/OnlineSchedulingEnv (.env on the gym
    wrapper), needed to compute atc_priority() and (online case) to know
    which slots are padding vs. not-yet-revealed vs. real.
    job_slot_width: width of one job's feature block (num_resources + 4),
    matching GymSchedulingEnv._get_obs()'s own per-slot layout exactly.
    machine_block_end: offset where the job-slot block begins (1 +
    num_machines * num_resources), matching the same layout.

    Padding/not-yet-revealed slots get feature value 0.0 (no job to prioritize
    yet), exactly matching priority_only_gym_wrapper.py's existing Option 3
    convention -- this function is a straight extraction of that logic, not a
    reimplementation, so Option 3's already-validated behaviour is unchanged
    by this refactor.

    PERF (2026-10-04, S2W11, found while diagnosing an unexpectedly slow
    --use-atc-feature training run on the new machine): this used to call
    atc_priority(base_env, j) without a precomputed mean_p, inside this
    per-job-slot loop -- the exact same O(J*num_jobs) mistake already found
    and fixed once in registry.py's choose() on 2026-09-21 (see
    priority_rules.py::_atc_mean_p's docstring), just reintroduced here when
    this function was factored out on 2026-09-23. mean_p depends only on
    base_env's state (not on which job slot j is), and nothing mutates
    base_env between the up-to-max_jobs calls in one _get_obs() call, so
    it's safe to compute once here (O(num_jobs)) and reuse for every slot --
    collapsing this function from O(max_jobs * num_jobs) to O(max_jobs +
    num_jobs) per call. This had been silently paid on every step of every
    Option 3 (use_atc=True) run since 2026-09-17, and every --use-atc-feature
    Option 1 run since 2026-09-23 -- a correctness-preserving, pure
    throughput fix (verified: identical output values, see
    tests/test_action_space_wrappers.py's existing ATC-feature-range checks,
    re-run clean after this change).
    """
    head = base_obs[:machine_block_end]
    job_block = base_obs[machine_block_end:]
    slots = job_block.reshape(max_jobs, job_slot_width)

    revealed = getattr(base_env, "revealed_jobs", None)
    mean_p = _atc_mean_p(base_env)
    out_slots = np.empty((max_jobs, job_slot_width + 1), dtype=np.float32)
    for j in range(max_jobs):
        out_slots[j, :job_slot_width] = slots[j]
        is_real = (j < base_env.num_jobs) if revealed is None else (j in revealed)
        out_slots[j, -1] = float(np.clip(atc_priority(base_env, j, mean_p=mean_p), 0.0, 1.0)) if is_real else 0.0

    return np.concatenate([head, out_slots.reshape(-1)]).astype(np.float32)
