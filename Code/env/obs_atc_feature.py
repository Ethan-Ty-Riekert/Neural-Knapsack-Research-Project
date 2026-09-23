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

from Code.baselines.priority_rules import atc_priority


def append_atc_priority_feature(base_obs, base_env, max_jobs, job_slot_width, machine_block_end):
    """Appends one per-job-slot feature -- the ATC composite priority index
    (Code/baselines/priority_rules.py::atc_priority, clamped to [0,1]) -- to
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
    """
    head = base_obs[:machine_block_end]
    job_block = base_obs[machine_block_end:]
    slots = job_block.reshape(max_jobs, job_slot_width)

    revealed = getattr(base_env, "revealed_jobs", None)
    out_slots = np.empty((max_jobs, job_slot_width + 1), dtype=np.float32)
    for j in range(max_jobs):
        out_slots[j, :job_slot_width] = slots[j]
        is_real = (j < base_env.num_jobs) if revealed is None else (j in revealed)
        out_slots[j, -1] = float(np.clip(atc_priority(base_env, j), 0.0, 1.0)) if is_real else 0.0

    return np.concatenate([head, out_slots.reshape(-1)]).astype(np.float32)
