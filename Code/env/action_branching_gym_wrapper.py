"""action_branching_gym_wrapper.py - Option 4 (action-branching, learned
placement), added 2026-09-20 (S2W9) to isolate an unresolved confound flagged
by an external critical review of this project (report.md Section 1.4):
Options 1/2/3's win over the historic ~1290-1330-tardiness PPO band could be
from the smaller action space, or simply from removing machine-placement-
learning entirely (both changed at once, never separated). This wrapper gives
placement back to the learner via action-branching (Tavakoli, Pardo &
Kormushev 2018, arXiv:1711.08946, "Action Branching Architectures for Deep
Reinforcement Learning" -- Code/references.bib key tavakoli2018actionbranching)
instead of the fixed FirstFit used by Options 2/3.

RL picks TWO things per step instead of one: which job slot, and which
machine -- MultiDiscrete([max_jobs+1, num_machines]) instead of a single
Discrete(max_jobs+1) (job only, FirstFit-placed) or the original joint
Discrete(max_jobs*num_machines+1) (multiplicative, the action-space size that
seven independent mechanisms failed against before this session's
action-space-reduction work). This is additive (|J|+1+|M|), not
multiplicative (|J|*|M|), matching action-branching's whole point.

DESIGN DECISION (parallel independent branches, not autoregressive) -- see
this session's 2026-09-20 plan / training-log.md entry for the full
derivation. Verified directly against the installed sb3_contrib package
(sb3_contrib/common/maskable/distributions.py):
MaskableMultiCategoricalDistribution.proba_distribution()/apply_masking()
both split ONE shared forward pass's flat logits/mask tensor into independent
per-branch chunks (th.split(..., list(action_dims), dim=1)) -- there is no
supported hook for the machine branch's mask or logits to depend on which job
the job branch actually sampled. Building true autoregressive masking would
require bypassing MaskableActorCriticPolicy's standard flow and threading a
live env reference into the network (breaks under vectorized training and
checkpoint reload). The parallel-branch design used here is also the more
faithful reading of the cited paper -- Tavakoli et al.'s own BDQ architecture
IS parallel independent branches off a shared trunk, not sequential/
autoregressive (the bib entry's "sequential" wording should be corrected).

HONEST LIMITATION (logged, not glossed over): since the machine branch can't
condition on the sampled job, its mask is the UNION of feasible machines
across ALL remaining jobs (a weak filter -- with num_machines=10 it will
often be close to all-ones, only excluding a machine that fits literally no
remaining job), with an all-ones fallback if that union is empty. Real
constraint enforcement therefore falls on a graceful idle-fallback in step()
when a mask-legal (job, machine) pair fails the ground-truth is_feasible()
check -- this is registered as a wasted step via info["mask_mismatch"]=True
(NOT silently auto-repaired via first_fit, which would defeat the point of
testing whether the machine branch can learn anything). See
Code/utils/training_diagnostics.py for the callback that measures how often
this fallback actually fires during real training.

No ATC-feature observation variant in this first pass (unlike Options 2/3) --
deferred as a straightforward follow-up, not attempted here, to keep this
first branching pass minimal and isolate the one new variable (learned
placement) cleanly, per this project's "change one thing at a time"
convention.

Wraps an already-constructed GymSchedulingEnv/OnlineGymSchedulingEnv the same
composition-over-inheritance way priority_only_gym_wrapper.py and
rule_selection_gym_wrapper.py do.
"""
import gymnasium as gym
import numpy as np


class ActionBranchingGymSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, full_gym_env):
        super().__init__()
        self._full = full_gym_env
        self.env = full_gym_env.env
        self.max_jobs = full_gym_env.max_jobs
        self.num_machines = full_gym_env.num_machines
        self.num_resources = full_gym_env.num_resources
        self.horizon = full_gym_env.horizon

        # action[0] in [0, max_jobs] (max_jobs == idle), action[1] in [0, num_machines)
        self.action_space = gym.spaces.MultiDiscrete([self.max_jobs + 1, self.num_machines])
        self.observation_space = full_gym_env.observation_space  # unchanged obs layout

        self._invalid_action_count = 0
        self._max_invalid_actions = 2 * (self.max_jobs + 1)

    def _get_obs(self):
        return self._full._get_obs()

    def _feasible_machines_for(self, job):
        t = self.env.time
        return [m for m in range(self.num_machines) if self.env.is_feasible(job, m, t)]

    def get_action_mask(self):
        """Flat [job/idle mask][machine union-mask], matching the layout
        MaskableMultiCategoricalDistribution.apply_masking() expects for a
        MultiDiscrete([max_jobs+1, num_machines]) action space (one flat
        boolean array, split internally by the distribution using
        self.action_space.nvec, in declared branch order)."""
        job_mask = np.zeros(self.max_jobs + 1, dtype=np.int8)
        t = self.env.time
        for j in self.env.remaining_jobs:
            if any(self.env.is_feasible(j, m, t) for m in range(self.num_machines)):
                job_mask[j] = 1
        job_mask[self.max_jobs] = 1  # idle always legal

        # Machine mask: union of feasible machines across ALL remaining jobs
        # (see module docstring's "HONEST LIMITATION" -- this is an
        # approximation, not per-job conditioning).
        machine_mask = np.zeros(self.num_machines, dtype=np.int8)
        for j in self.env.remaining_jobs:
            for m in range(self.num_machines):
                if self.env.is_feasible(j, m, t):
                    machine_mask[m] = 1
        if not machine_mask.any():
            machine_mask[:] = 1  # nothing fits anywhere right now -- don't degenerate to all-zero

        return np.concatenate([job_mask, machine_mask])

    def reset(self, *, seed=None, options=None):
        self._full.reset(seed=seed, options=options)
        self._invalid_action_count = 0
        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action):
        # Same 0-d ndarray guard as the other action-space-variant wrappers
        # (model.predict() on a single non-batched obs can hand back 0-d
        # ndarrays, which support == and indexing but are not hashable --
        # crashes `job in self.env.remaining_jobs`). Cast BOTH components.
        job = int(action[0])
        machine = int(action[1])

        mask_mismatch = False
        if job == self.max_jobs:
            _, reward, done = self.env.step_idle()
        else:
            t = self.env.time
            if self.env.is_feasible(job, machine, t):
                was_pending = job in self.env.remaining_jobs
                _, reward, done = self.env.step((job, machine))
                if was_pending and job not in self.env.remaining_jobs:
                    self._invalid_action_count = 0
                else:
                    self._invalid_action_count += 1
            else:
                # Mask-legal (union said this machine fits SOME remaining job)
                # but not feasible for THIS specific job -- the parallel-branch
                # approximation's cost, made visible rather than silently
                # auto-repaired. Idle-fallback, not first_fit: auto-repair
                # would defeat the point of testing whether the machine branch
                # can learn anything.
                mask_mismatch = True
                _, reward, done = self.env.step_idle()

        obs = self._get_obs()
        info = {
            "action_mask": self.get_action_mask(),
            "episode_cost": self.env.episode_cost,
            "mask_mismatch": mask_mismatch,
        }

        terminated = done
        truncated = self._invalid_action_count >= self._max_invalid_actions

        return obs, float(reward), terminated, truncated, info
