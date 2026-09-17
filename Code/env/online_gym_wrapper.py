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
        obs = []

        t = min(self.env.time, self.horizon)
        obs.append(t / self.horizon)

        t_idx = min(self.env.time, self.horizon - 1)
        for m in range(self.num_machines):
            for r in range(self.num_resources):
                cap = self.env.capacity[m, r, t_idx] / (self.initial_capacity[m, r] + 1e-8)
                obs.append(cap)

        max_dur = max(1.0, float(np.max(self.env.job_durations)))
        max_wgt = max(1.0, float(np.max(self.env.job_weights)))
        max_res = np.maximum(1.0, np.max(self.env.job_resources, axis=0))

        for j in range(self.max_jobs):
            if j in self.env.revealed_jobs:
                obs.append(self.env.job_durations[j] / max_dur)
                obs.append(self.env.job_deadlines[j] / self.horizon)
                obs.append(self.env.job_weights[j] / max_wgt)
                for r in range(self.num_resources):
                    obs.append(self.env.job_resources[j, r] / max_res[r])
                scheduled = 0.0 if j in self.env.remaining_jobs else 1.0
                obs.append(scheduled)
            else:
                obs.extend([0.0] * (3 + self.num_resources))
                obs.append(1.0)

        return np.array(obs, dtype=np.float32)

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
