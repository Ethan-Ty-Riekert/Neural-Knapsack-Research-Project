"""windowed_priority_gym_wrapper.py - DeepRM-style bounded action-space
window for Options 2/3 (2026-09-18, S2W10 follow-up -- see
Future/research/2026-09-17-action-space-reduction.md Section 7's "Next
steps," and the "how do papers do it" discussion the same day). PREPARED
FOR REVIEW, NOT YET WIRED INTO ANY TRAINING RUN -- window size and the
window-selection ordering are real design choices the user asked to decide,
not just an engineering fix; see this module's docstring for the choices
made here and why, so they can be reviewed rather than silently baked in.

Motivation: PriorityOnlyGymSchedulingEnv (Options 2/3) already shrunk the
action space from max_jobs*num_machines+1 to max_jobs+1 (~101-400+
depending on scale) -- a real improvement, but still far from DeepRM's own
~11-21 choices (Mao et al. 2016, HotNets). DeepRM's actual mechanism is not
just "pick a job" -- it's "pick a job from a small BOUNDED VISIBLE WINDOW
(M~10-20 slots), plus a scalar 'backlog' count summarizing everything
beyond the window." This module implements that missing piece.

Design choices made here (flag for review before training):
- Window selection: the M jobs among currently-unscheduled, already-revealed
  jobs with the EARLIEST DEADLINE (EDF order) are the ones made visible/
  choosable each tick -- not raw arrival/FIFO order (DeepRM's own queue is
  FIFO since it has no deadlines; this problem's tardiness objective makes
  deadline urgency the more natural windowing key). This still leaves WHICH
  of the M visible jobs to run, and WHEN, entirely up to the learned policy
  -- windowing only bounds the candidate set, it does not pre-select an
  answer the way a full priority-rule choice (Option 1) would.
- Backlog feature: one extra scalar, (jobs waiting beyond the window) /
  max_jobs, appended once at the very end of the observation (not per-slot)
  -- summarizes "how much unaddressed backlog exists" without exposing any
  individual backlogged job's identity, matching DeepRM's own backlog
  design intent.
- Default window_size=15: a round number inside DeepRM's own ~10-20 range,
  not independently re-derived for this project's instance distribution --
  flagged as an untested constant per CLAUDE.md, same treatment as ATC's
  k=2.0 or the heavy-tail sigma.

Wraps an already-constructed GymSchedulingEnv/OnlineGymSchedulingEnv the
same composition-over-inheritance way rule_selection_gym_wrapper.py and
priority_only_gym_wrapper.py do.
"""
import gymnasium as gym
import numpy as np

from Code.baselines.priority_rules import atc_priority
from Code.baselines.placement_rules import first_fit


class WindowedPriorityGymSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, full_gym_env, window_size: int = 15, use_atc: bool = False):
        super().__init__()
        self._full = full_gym_env
        self.env = full_gym_env.env
        self.max_jobs = full_gym_env.max_jobs
        self.num_machines = full_gym_env.num_machines
        self.num_resources = full_gym_env.num_resources
        self.horizon = full_gym_env.horizon
        self.window_size = window_size
        self.use_atc = use_atc

        self._job_slot_width = self.num_resources + 4
        self._machine_block_end = 1 + self.num_machines * self.num_resources
        self._out_slot_width = self._job_slot_width + (1 if use_atc else 0)

        self.action_space = gym.spaces.Discrete(window_size + 1)  # +1 idle
        obs_dim = 1 + self._machine_block_end - 1 + window_size * self._out_slot_width + 1  # +1 backlog scalar
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._window_jobs = [None] * window_size  # slot index -> real job id, or None (padding)
        self._invalid_action_count = 0
        self._max_invalid_actions = 2 * (window_size + 1)

    def _candidate_jobs(self):
        """Unscheduled jobs with >=1 currently-feasible machine, sorted by
        EARLIEST DEADLINE (EDF order) -- see module docstring for why this
        key, not arrival/FIFO order."""
        t = self.env.time
        candidates = [
            j for j in self.env.remaining_jobs
            if any(self.env.is_feasible(j, m, t) for m in range(self.num_machines))
        ]
        candidates.sort(key=lambda j: (self.env.job_deadlines[j], j))
        return candidates

    def _refresh_window(self, candidates):
        self._window_jobs = list(candidates[:self.window_size])
        while len(self._window_jobs) < self.window_size:
            self._window_jobs.append(None)
        return max(0, len(candidates) - self.window_size)

    def _get_obs(self):
        base_obs = self._full._get_obs()
        head = base_obs[:self._machine_block_end]
        job_block = base_obs[self._machine_block_end:]
        all_slots = job_block.reshape(self.max_jobs, self._job_slot_width)

        candidates = self._candidate_jobs()
        backlog = self._refresh_window(candidates)

        out = np.zeros((self.window_size, self._out_slot_width), dtype=np.float32)
        for i, job in enumerate(self._window_jobs):
            if job is None:
                out[i, -1 if self.use_atc else self._job_slot_width - 1] = 1.0  # scheduled/padding flag
                continue
            out[i, :self._job_slot_width] = all_slots[job]
            if self.use_atc:
                out[i, -1] = float(np.clip(atc_priority(self.env, job), 0.0, 1.0))

        backlog_scalar = np.array([min(1.0, backlog / max(1, self.max_jobs))], dtype=np.float32)
        return np.concatenate([head, out.reshape(-1), backlog_scalar]).astype(np.float32)

    def get_action_mask(self):
        mask = np.zeros(self.window_size + 1, dtype=np.int8)
        for i, job in enumerate(self._window_jobs):
            if job is not None:
                mask[i] = 1
        mask[self.window_size] = 1  # idle always legal
        return mask

    def reset(self, *, seed=None, options=None):
        self._full.reset(seed=seed, options=options)
        self._invalid_action_count = 0
        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action_id):
        action_id = int(action_id)  # see priority_only_gym_wrapper.py's identical fix/comment
        job = self._window_jobs[action_id] if action_id < self.window_size else None

        if job is None:
            _, reward, done = self.env.step_idle()
        else:
            t = self.env.time
            feasible = [m for m in range(self.num_machines) if self.env.is_feasible(job, m, t)]
            if not feasible:
                _, reward, done = self.env.step_idle()
            else:
                machine = first_fit(self.env, job, feasible, t)
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
