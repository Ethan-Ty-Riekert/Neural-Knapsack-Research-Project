"""online_scheduling_env.py - Dynamic-arrival specialization of SchedulingEnv.

Subclasses rather than modifies SchedulingEnv (see mathformulation.tex's
online MDP subsection and the online-case plan's Part 3.1): every mechanism
that already operates purely over self.remaining_jobs (is_feasible, reward,
compute_theta, _finalize_unscheduled_job_cost, termination checks) becomes
automatically causally correct once remaining_jobs is populated based on
arrival rather than "all jobs, from t=0" -- only *how* and *when*
remaining_jobs gains members changes here, plus the tick-advance rule.

Tick-advance relaxation (Option 2, chosen over keeping the offline case's
"every action advances time by 1" -- see mathformulation.tex): only
step_idle() advances self.time. step() below reimplements
SchedulingEnv.step()'s body with the time-advance line removed, so multiple
placements may occur at the same tick across successive external calls,
terminated naturally once no remaining job is feasible anywhere (the action
mask then leaves `idle` as the only legal choice) or the agent idles
voluntarily. This is reimplemented rather than done via
`super().step(); self.time -= 1` because SchedulingEnv.step() may call
_finalize_unscheduled_job_cost() internally based on a `done` check that
uses the (about-to-be-rolled-back) post-increment time -- silently and
permanently tripping that method's idempotency guard on a premature call
would then suppress the real one when the episode actually ends. Also see:
done is deliberately never set True on len(remaining_jobs) == 0 alone here,
unlike the offline case -- an empty remaining_jobs set does not mean "no
more work forever" when more jobs may still arrive before the horizon.
"""
from typing import Tuple

import numpy as np

from .scheduling_env import SchedulingEnv


class OnlineSchedulingEnv(SchedulingEnv):
    def __init__(self, *args, job_arrival_times: np.ndarray, **kwargs):
        self.arrival_times = np.asarray(job_arrival_times)
        self.revealed_jobs = set()
        super().__init__(*args, **kwargs)
        # Parent __init__ sets self.remaining_jobs = set(range(num_jobs))
        # (offline "everything known at t=0" assumption) -- reset() below
        # replaces that with the arrival-based population.
        self.reset()

    def reset(self):
        super().reset()
        self.remaining_jobs = set()
        self.revealed_jobs = set()
        self._reveal_arrivals()
        return None  # PERF: see SchedulingEnv.reset()'s matching comment

    def set_jobs_and_arrivals(
        self,
        job_durations: np.ndarray,
        job_resources: np.ndarray,
        job_deadlines: np.ndarray,
        job_weights: np.ndarray,
        job_arrival_times: np.ndarray,
    ):
        """Online counterpart of SchedulingEnv.set_jobs() -- also swaps in a
        fresh arrival-time array, for a resampler that draws a whole new
        online instance (job content + arrival process) each episode. Sets
        self.arrival_times BEFORE calling set_jobs() (which calls
        self.reset(), resolving polymorphically to this class's reset()
        above) so t=0 arrivals are revealed against the NEW arrival times."""
        self.arrival_times = np.asarray(job_arrival_times)
        self.set_jobs(job_durations, job_resources, job_deadlines, job_weights)

    def _reveal_arrivals(self):
        """Add every job whose arrival tick has passed to remaining_jobs,
        exactly once each -- revealed_jobs guards against re-adding a job
        that has since been scheduled and removed from remaining_jobs."""
        for j in range(self.num_jobs):
            if j not in self.revealed_jobs and self.arrival_times[j] <= self.time:
                self.revealed_jobs.add(j)
                self.remaining_jobs.add(j)

    def step(self, action: Tuple[int, int]) -> Tuple[object, float, bool]:
        job, machine = action

        if job not in self.remaining_jobs:
            return (None, -self.invalidPenalty, False)  # PERF: see SchedulingEnv.reset()'s comment

        if not self.is_feasible(job, machine, self.time):
            return (None, -self.invalidPenalty, False)  # PERF: see SchedulingEnv.reset()'s comment

        machine_was_inactive = self.machine_active[machine] == 0
        if machine_was_inactive:
            self.machine_active[machine] = 1

        # PERF (2026-09-21, S2W9): see SchedulingEnv.step()'s matching comment.
        duration = self.job_durations[job]
        self.capacity[machine, :, self.time:self.time + duration] -= self.job_resources[job][:, None]

        self.start_times[job] = self.time
        self.tardiness[job] = max(0, self.time + duration - self.job_deadlines[job])

        self.remaining_jobs.remove(job)

        new_theta = self.compute_theta()
        delta_theta = max(0, new_theta - self.prev_theta)
        self.prev_theta = new_theta

        reward = self.reward(job, machine, machine_was_inactive, delta_theta)
        if self.reward_mode == "legacy":
            reward += 3.0
            if len(self.remaining_jobs) == 0:
                reward += 50
        # dense_tardiness: no flat bonuses here either -- see
        # SchedulingEnv.__init__'s reward_mode docstring. The dense per-tick
        # charge is applied once per actual tick advance, in step_idle()
        # below, not per placement (placements don't advance time here).

        # Deliberately NOT self.time += 1 here -- see module docstring.
        if self.use_potential_shaping:
            new_potential = self._compute_potential()
            reward += self.shaping_gamma * new_potential - self.prev_potential
            self.prev_potential = new_potential

        # Never done from a placement alone -- see module docstring. Episode
        # end is exclusively a step_idle()/horizon event in the online case.
        done = False

        return (None, reward, done)  # PERF: see SchedulingEnv.reset()'s comment

    def step_idle(self):
        reward = -self.idling_penalty
        if self.reward_mode != "legacy":
            # Charge the elapsing tick's dense tardiness accrual using
            # self.time BEFORE it advances -- see
            # SchedulingEnv._dense_tardiness_tick_charge(). This is the only
            # place a tick actually elapses in the online case (Option 2), so
            # it's the only place this charge belongs.
            reward += self._dense_tardiness_tick_charge()

        self.time += 1
        # BUG FOUND BY test_online_env.py's Check 4: horizon+1 (the "unreachable"
        # padding sentinel -- see arrival_process.py) is NOT actually unreachable
        # for self.time itself -- step_idle() increments up to exactly horizon+1
        # before the done check below fires. Without this guard,
        # _reveal_arrivals() would incorrectly reveal every padding job at that
        # terminal tick. Revealing only while self.time <= self.horizon excludes
        # the sentinel while still correctly revealing a legitimate arrival at
        # the last real tick (self.time == self.horizon).
        if self.time <= self.horizon:
            self._reveal_arrivals()

        if self.use_potential_shaping:
            new_potential = self._compute_potential()
            reward += self.shaping_gamma * new_potential - self.prev_potential
            self.prev_potential = new_potential

        done = self.time > self.horizon
        if done:
            self._finalize_unscheduled_job_cost()

        return None, reward, done  # PERF: see SchedulingEnv.reset()'s comment
