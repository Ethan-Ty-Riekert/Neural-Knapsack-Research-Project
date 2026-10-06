"""online_gym_wrapper.py - Gym wrapper for OnlineSchedulingEnv.

Overrides exactly two things from GymSchedulingEnv, both documented in
mathformulation.tex / the online-case plan:

1. _get_obs()'s not-yet-arrived-job observation leak (a real bug, not a
   style nit): the parent's per-slot branch is `if j < self.num_jobs`,
   which is always true once num_jobs == max_jobs (the online case's
   fixed-array/"phantom padding" convention -- see arrival_process.py).
   That would leak a not-yet-arrived job's real duration/deadline/weight/
   resource-demand into the observation even though the action mask
   correctly hides it from *selection* -- silently violating the causality
   constraint (x_jmt = 0 for t < tau_j) at the observation level.

2. step()'s invalid-action detection proxy: the parent uses "did
   self.env.time change?" as a cheap stand-in for "was that action valid?"
   -- correct for the offline case, where every valid placement advances
   time by exactly 1. It is NOT correct for the online case: under Option 2
   (see OnlineSchedulingEnv), a *valid* placement also leaves time
   unchanged, so the parent's proxy would misclassify every successful
   same-tick placement as invalid and could eventually truncate an episode
   for the agent doing exactly what it should. Fixed by checking whether
   the targeted job was actually removed from remaining_jobs instead, which
   stays correct under either tick-advance rule.

get_action_mask() needs no override -- it already only iterates
self.env.remaining_jobs, which is causally correct by construction once
self.env is an OnlineSchedulingEnv.
"""
import gymnasium as gym
import numpy as np

from .gym_scheduling_wrapper import GymSchedulingEnv


class OnlineGymSchedulingEnv(GymSchedulingEnv):
    def _get_obs(self):
        """Same layout as GymSchedulingEnv._get_obs, except only REVEALED jobs expose features
        (unrevealed slots are zero with scheduled-flag 1.0, like padding).

        PERF (2026-09-29, S2W11): vectorised, mirroring the offline wrapper's 2026-09-21 fix. The
        original per-element Python loop (kept verbatim in tests/test_online_obs_vectorised.py as
        the reference) was the dominant cost of every online episode -- ~75% of an ATC episode
        at rho 0.75 (2.8M list appends), and the main reason online RL training was slow. The
        equivalence test checks bit-identical float32 observations at every step."""
        R, J, n = self.num_resources, self.max_jobs, self.num_jobs
        H_phys, H_pref = self.env.horizon, self.env.preferred_horizon  # see GymSchedulingEnv._get_obs
        t = min(self.env.time, H_phys)
        capacity_block = self._capacity_block()  # (+ y_m in full-state mode, see GymSchedulingEnv)

        max_dur, max_wgt, max_res = self._job_scales()  # see GymSchedulingEnv._job_scales

        job_feats = np.zeros((J, self.obs_layout.job_slot_width), dtype=np.float32)
        if self.env.revealed_jobs:
            idx = np.fromiter(self.env.revealed_jobs, dtype=int)
            job_feats[idx, 0] = self.env.job_durations[idx] / max_dur
            job_feats[idx, 1] = self.env.job_deadlines[idx] / H_pref
            job_feats[idx, 2] = self.env.job_weights[idx] / max_wgt
            job_feats[idx, 3:3 + R] = self.env.job_resources[idx] / max_res
        scheduled = np.ones(J, dtype=np.float32)
        if self.env.remaining_jobs:  # remaining is always a subset of revealed
            scheduled[list(self.env.remaining_jobs)] = 0.0
        job_feats[:, R + 3] = scheduled
        if self.obs_layout.markov and self.env.revealed_jobs:
            # revealed jobs only: an unrevealed slot's deadline would leak a future arrival
            self._add_markov_job_feats(job_feats, idx)

        return np.concatenate(([t / H_pref], capacity_block.ravel(), job_feats.ravel())).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if self.job_resampler is not None:
            fresh = self.job_resampler()
            self.env.set_jobs_and_arrivals(
                fresh["job_durations"],
                fresh["job_resources"],
                fresh["job_deadlines"],
                fresh["job_weights"],
                fresh["job_arrival_times"],
            )
        else:
            self.env.reset()

        self._invalid_action_count = 0
        self.initial_capacity = self.env.capacity[:, :, 0].copy()

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action_id):
        idle_action = self.max_jobs * self.num_machines

        if action_id == idle_action:
            _, reward, done = self.env.step_idle()
        else:
            job = action_id // self.num_machines
            machine = action_id % self.num_machines
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
