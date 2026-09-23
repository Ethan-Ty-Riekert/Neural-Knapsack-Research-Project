"""priority_only_gym_wrapper.py - Options 2 & 3 (raw-feature / ATC-primed
priority-only selection) action-space reduction, part of the 2026-09-17
action-space-reduction work (see rule_selection_gym_wrapper.py's module
docstring for the shared motivation -- this file covers the other two of
the three options).

RL picks WHICH job slot to run next (DeepRM's own action-space pattern --
Mao et al. 2016, HotNets: pick a job, let the system place it), not which
(job, machine) pair. Placement, once a job is chosen, is FirstFit
(Code/baselines/placement_rules.py, already implemented/validated), not
learned. Action space shrinks from max_jobs*num_machines+1 to max_jobs+1.

Option 2 (use_atc=False): job-slot observation features are exactly today's
GymSchedulingEnv/OnlineGymSchedulingEnv per-slot block (duration, deadline,
weight, resources..., scheduled) -- network learns the priority ordering
end-to-end from raw features, matching DeepRM/Decima's own design.

Option 3 (use_atc=True): identical, plus one appended per-slot feature: the
ATC composite priority index (Code/baselines/priority_rules.py::
atc_priority -- the same formula the ATC+FirstFit baseline uses).

Wraps an already-constructed GymSchedulingEnv/OnlineGymSchedulingEnv the
same composition-over-inheritance way rule_selection_gym_wrapper.py does,
so this one file is correct for both the offline and online case without
duplicating either's _get_obs().
"""
import gymnasium as gym
import numpy as np

from Code.baselines.placement_rules import first_fit
from Code.env.obs_atc_feature import append_atc_priority_feature


class PriorityOnlyGymSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, full_gym_env, use_atc: bool = False):
        super().__init__()
        self._full = full_gym_env
        self.env = full_gym_env.env
        self.max_jobs = full_gym_env.max_jobs
        self.num_machines = full_gym_env.num_machines
        self.num_resources = full_gym_env.num_resources
        self.horizon = full_gym_env.horizon
        self.use_atc = use_atc

        # Must match GymSchedulingEnv._get_obs()'s per-job-slot layout:
        # [duration, deadline, weight, resource_0..R-1, scheduled].
        self._job_slot_width = self.num_resources + 4
        self._machine_block_end = 1 + self.num_machines * self.num_resources

        self.action_space = gym.spaces.Discrete(self.max_jobs + 1)  # +1 idle
        obs_dim = full_gym_env.observation_space.shape[0] + (self.max_jobs if use_atc else 0)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._invalid_action_count = 0
        self._max_invalid_actions = 2 * (self.max_jobs + 1)

    def _get_obs(self):
        base_obs = self._full._get_obs()
        if not self.use_atc:
            return base_obs
        return append_atc_priority_feature(
            base_obs, self.env, self.max_jobs, self._job_slot_width, self._machine_block_end,
        )

    def _feasible_machines(self, job):
        t = self.env.time
        return [m for m in range(self.num_machines) if self.env.is_feasible(job, m, t)]

    def get_action_mask(self):
        mask = np.zeros(self.max_jobs + 1, dtype=np.int8)
        t = self.env.time
        for j in self.env.remaining_jobs:
            if any(self.env.is_feasible(j, m, t) for m in range(self.num_machines)):
                mask[j] = 1
        mask[self.max_jobs] = 1  # idle always legal
        return mask

    def reset(self, *, seed=None, options=None):
        self._full.reset(seed=seed, options=options)
        self._invalid_action_count = 0
        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action_id):
        # model.predict() (e.g. eval_action_space_variant.py, called directly
        # on a single non-batched obs rather than through a VecEnv) can hand
        # back a 0-d numpy ndarray rather than a python int/numpy scalar --
        # supports == and indexing (so RuleSelectionGymSchedulingEnv's step()
        # never noticed) but is NOT hashable, which crashes the `job in
        # self.env.remaining_jobs` set-membership check below with
        # "TypeError: unhashable type: 'numpy.ndarray'". Cast once, up front.
        action_id = int(action_id)
        if action_id == self.max_jobs:
            _, reward, done = self.env.step_idle()
        else:
            job = action_id
            feasible = self._feasible_machines(job)
            if not feasible:
                # RL picked a job with no currently-feasible machine (mask
                # said otherwise a step ago, or the job was already handled)
                # -- fall through to idle rather than force a real invalid
                # action through the underlying env.
                _, reward, done = self.env.step_idle()
            else:
                machine = first_fit(self.env, job, feasible, self.env.time)
                was_pending = job in self.env.remaining_jobs
                _, reward, done = self.env.step((job, machine))
                if was_pending and job not in self.env.remaining_jobs:
                    self._invalid_action_count = 0
                else:
                    self._invalid_action_count += 1

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask(), "episode_cost": self.env.episode_cost}

        terminated = done
        truncated = self._invalid_action_count >= self._max_invalid_actions

        return obs, float(reward), terminated, truncated, info
