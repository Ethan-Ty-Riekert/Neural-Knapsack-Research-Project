"""gym_scheduling_wrapper.py - The gym wrapper for the scheduling environment to allow for RL implementation
https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium
https://gymnasium.farama.org/
"""
import numpy as np 
import gymnasium as gym
from typing import List
from .scheduling_env import SchedulingEnv


class GymSchedulingEnv(gym.Env):
    """Gymnasium wrapper for SchedulingEnv.
    Converts (job, machine) actions into a single integer action.
    Produces a fixed-size observation vector.
    Includes action masking
    """

    metadata = {"render_modes": []} # for gymnasium

    def __init__(self, env: SchedulingEnv, max_jobs: int = None, restrict_idle: bool = False, job_resampler=None):
        """Initialisation of gym warpper for scheduling environment. Expects passed import
        is class SchedulingEnv from scheduling_env.py.

        job_resampler: optional zero-arg callable returning a dict with keys
        job_durations/job_resources/job_deadlines/job_weights (same shapes/
        num_jobs as `env`). If given, reset() draws a fresh job set from it every
        episode via SchedulingEnv.set_jobs(), instead of reusing the same job set
        `env` was constructed with -- Experiment 2 (randomized-instance
        generalization), see Code/training/train_optimized.py's
        make_random_instance_resampler(). None (default) preserves the previous
        fixed-instance behaviour exactly.

        max_jobs: fixed job-slot capacity used to size the observation and action
        spaces, decoupled from env.num_jobs (the actual/logical number of jobs in
        this instance). Defaults to env.num_jobs, i.e. no padding, matching the
        previous behaviour. Pass a larger, constant max_jobs across curriculum
        stages to let num_jobs vary per stage while keeping the obs/action space
        fixed (required for MaskablePPO's model.set_env() and the hand-rolled A2C
        network, both of which fix their layer sizes from the first env they see).
        Job slots beyond env.num_jobs are zero-padded in the observation and
        always masked out as infeasible actions.

        restrict_idle: if True, the idle action is masked out of get_action_mask()
        whenever at least one non-idle action is currently feasible -- idle stays
        legal only when nothing can be scheduled this step. Solution 1a of the
        2026-08-09 idle-collapse experiments (see Future/research/): idle is
        otherwise always legal and requires no multi-step credit assignment to
        discover, which is why policies collapse onto it early regardless of its
        reward. Defaults to False (existing always-legal-idle behaviour), so this
        is opt-in and A/B-able against the baseline.
        """
        super().__init__()

        self.env = env
        self.num_jobs = env.num_jobs
        self.max_jobs = max_jobs if max_jobs is not None else env.num_jobs
        self.num_machines = env.num_machines
        self.num_resources = env.num_resources
        self.horizon = env.horizon
        self.restrict_idle = restrict_idle
        self.job_resampler = job_resampler

        # For normalisation later on
        self.initial_capacity = env.capacity[:, :, 0].copy()

        ### Action Space ###
        self.action_space = gym.spaces.Discrete(self.max_jobs * self.num_machines + 1) # +1 to allow for the idling action

        # Belt-and-suspenders safety net (this session): invalid actions never
        # advance SchedulingEnv.time and the env has no other truncation signal,
        # so a bug that lets a policy repeatedly select an action the mask
        # (wrongly) reports as valid -- as the horizon-boundary mask/step
        # mismatch fixed above did -- could previously hang an episode forever.
        # This is independent of that specific fix: it guards the general class
        # of bug (invalid action -> no state change -> no other way to end the
        # episode), not just the one instance of it found this session.
        self._invalid_action_count = 0
        self._max_invalid_actions = 2 * self.max_jobs * self.num_machines

        ### Observation Space ###
        obs_dim = self._compute_obs_dim() # gymnasium method
        self.observation_space = gym.spaces.Box(
            low = 0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def _compute_obs_dim(self):
        """Observation vector construction: Compute the dimension of the observation space.

        time (1), remaining capacity (num_machines * resource dimension),
        job features: duration (1), deadline (1), weight (1), requirement dimension (R), scheduled mask (1)
        -> total per job slot = 3 + R + 1 = R + 4, for max_jobs slots"""

        return 1 + (self.num_machines * self.num_resources) + self.max_jobs * (self.num_resources + 4)

    def _get_obs(self):
        """Observation vector construction: Build the observation vector.

        PERF (2026-09-21, S2W9, from Future/research/2026-09-20-optimisation-
        and-efficiency-critique.md Section 1.3): vectorized numpy ops instead
        of Python list.append()/.extend() + np.array() conversion -- this
        runs on every single environment step, previously the second
        highest-value hot-path finding after get_state() (already fixed).
        Produces numerically identical values to the original per-element
        loop, just without the Python-level looping cost."""
        R, M, J, n = self.num_resources, self.num_machines, self.max_jobs, self.num_jobs

        # 1. Normalised time
        t = min(self.env.time, self.horizon)

        # 2. Remaining capacity (normalised) -- machine-major, resource-minor,
        # matching the original nested "for m: for r:" append order exactly.
        t_idx = min(self.env.time, self.horizon - 1)
        capacity_block = self.env.capacity[:, :, t_idx] / (self.initial_capacity + 1e-8)

        # Precompute normalisation constants
        max_dur = max(1.0, float(np.max(self.env.job_durations)))
        max_wgt = max(1.0, float(np.max(self.env.job_weights)))
        max_res = np.maximum(1.0, np.max(self.env.job_resources, axis=0))

        # 3. Job features (max_jobs fixed-size slots; slots beyond this
        # instance's actual num_jobs are zero-padded and marked "scheduled" so
        # the policy treats them as already-handled/irrelevant -- their
        # actions are always masked out in get_action_mask()). job_feats
        # starts all-zero, matching the original's padding-slot branch
        # exactly; only the first n rows get filled with real values.
        job_feats = np.zeros((J, R + 4), dtype=np.float32)
        job_feats[:n, 0] = self.env.job_durations / max_dur
        job_feats[:n, 1] = self.env.job_deadlines / self.horizon
        job_feats[:n, 2] = self.env.job_weights / max_wgt
        job_feats[:n, 3:3 + R] = self.env.job_resources / max_res
        # scheduled mask: 1.0 everywhere by default (matches padding slots'
        # hardcoded 1.0), then 0.0 for real jobs still in remaining_jobs --
        # remaining_jobs only ever holds indices < n, so this can never touch
        # a padding slot.
        scheduled = np.ones(J, dtype=np.float32)
        if self.env.remaining_jobs:
            scheduled[list(self.env.remaining_jobs)] = 0.0
        job_feats[:, -1] = scheduled

        return np.concatenate((
            [t / self.horizon],
            capacity_block.ravel(),
            job_feats.ravel(),
        )).astype(np.float32)

    def set_lambda2(self, value: float) -> None:
        """Set the underlying SchedulingEnv's tardiness weight (lambda2) at
        runtime -- the hook a Lagrangian-constrained trainer (RCPO for A2C,
        PPO-Lagrangian for PPO -- see Code/policies/ppo_lagrangian.py) uses to
        push an adapted multiplier value into the live env.

        Reachable through a VecEnv via env_method("set_lambda2", value):
        SB3's VecEnv.env_method calls env.get_wrapper_attr(method_name)
        (confirmed empirically this session), which chain-walks the
        Monitor/ActionMasker wrapper stack via gymnasium's own wrapper-attr
        resolution rather than the generic __getattr__ forwarding gymnasium
        >=1.x removed -- see the matching comments in
        Code/policies/a2c_policy.py's _resolve_sched_env() for why that
        generic forwarding can no longer be relied on here.
        """
        self.env.lambda2 = float(value)

    def get_action_mask(self):
        """Action mask building:
        mask[a] = 1 if (job, machine) is feasible at current time
        The final action (index = max_jobs * num_machines) is the idle action.
        Job slots beyond this instance's actual num_jobs are never in
        self.env.remaining_jobs, so their actions stay masked out (0) automatically.
        """
        total_actions = self.max_jobs * self.num_machines + 1
        mask = np.zeros(total_actions, dtype=np.int8)

        # BUG FIX (this session): this used to clamp t to horizon-1, which
        # disagreed with SchedulingEnv.step()/is_feasible()'s own uncapped
        # self.time. At env.time == horizon (reachable -- episode only ends once
        # time > horizon), a duration-1 job could read as feasible here
        # ((horizon-1)+1 == horizon, not > horizon) but then fail the real check
        # inside step() (horizon+1 > horizon), landing in the invalid-action
        # branch -- which does not advance time, so a policy trusting this mask
        # could get stuck repeating the same invalid action forever. Using the
        # uncapped self.time here instead matches step() exactly, and is safe
        # because is_feasible() short-circuits on t+duration > horizon before any
        # array indexing, and every job has duration >= 1, so no out-of-bounds
        # read is possible for t >= horizon. NOTE: _get_obs()'s separate
        # t_idx = min(self.env.time, self.horizon - 1) (above) must stay clamped
        # -- that one directly indexes self.env.capacity[m, r, t_idx] and a real
        # out-of-bounds read there.
        t = self.env.time

        # Normal feasible scheduling actions.
        # PERF (2026-09-21, S2W9, from Future/research/2026-09-20-
        # optimisation-and-efficiency-critique.md Section 1.4): vectorized
        # instead of a nested Python "for j: for m:" loop calling
        # is_feasible() O(|remaining_jobs| * num_machines) times. Replicates
        # is_feasible(j, m, t)'s exact two conditions (duration fits before
        # the horizon; every resource dimension has enough remaining
        # capacity) as one array comparison. Preserves the original's OOB
        # safety at t >= horizon defensively (clamped t_idx for indexing
        # only) rather than via short-circuit order, since every job has
        # duration >= 1 -- duration_ok is already False for every row
        # whenever t >= horizon, so the (otherwise out-of-range) capacity
        # values at those rows never affect the final mask.
        remaining = list(self.env.remaining_jobs)
        if remaining:
            remaining_idx = np.array(remaining, dtype=np.int64)
            durations = self.env.job_durations[remaining_idx]                  # (Jr,)
            duration_ok = (t + durations) <= self.horizon                      # (Jr,)
            t_idx = min(t, self.horizon - 1)
            cap_t = self.env.capacity[:, :, t_idx]                             # (M, R)
            resources = self.env.job_resources[remaining_idx]                  # (Jr, R)
            resource_ok = (cap_t[None, :, :] - resources[:, None, :] >= 0).all(axis=2)  # (Jr, M)
            feasible = resource_ok & duration_ok[:, None]                      # (Jr, M)
            action_ids = remaining_idx[:, None] * self.num_machines + np.arange(self.num_machines)[None, :]
            mask[action_ids[feasible]] = 1

        # Idle action: allowed by default, unless restrict_idle is set and at
        # least one non-idle action is feasible this step (Solution 1a).
        idle_action = self.max_jobs * self.num_machines
        if self.restrict_idle and mask[:idle_action].any():
            mask[idle_action] = 0
        else:
            mask[idle_action] = 1

        return mask

    
    def reset(self, *, seed=None, options=None):
        """GYM API"""
        super().reset(seed=seed)
        if self.job_resampler is not None:
            fresh = self.job_resampler()
            self.env.set_jobs(
                fresh["job_durations"],
                fresh["job_resources"],
                fresh["job_deadlines"],
                fresh["job_weights"],
            )
        else:
            self.env.reset()

        self._invalid_action_count = 0

        # Recompute initial capacity (in case reset changed it)
        self.initial_capacity = self.env.capacity[:, :, 0].copy()

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}
        return obs, info
    
    def step(self, action_id):
        # Idle action index
        idle_action = self.max_jobs * self.num_machines

        if action_id == idle_action:
            obs, reward, done = self.env.step_idle()
        else:
            job = action_id // self.num_machines
            machine = action_id % self.num_machines
            # Padding job slots (job >= self.num_jobs) are never in
            # self.env.remaining_jobs, so SchedulingEnv.step() already treats them
            # as an ordinary invalid action -- no special-casing needed here.
            time_before = self.env.time
            _, reward, done = self.env.step((job, machine))
            # SchedulingEnv.step()'s invalid-action branches (job already
            # scheduled / infeasible placement) never advance self.env.time, so
            # an unchanged time is a cheap, exact proxy for "that was an invalid
            # action" without needing step() to return an extra flag.
            if self.env.time == time_before:
                self._invalid_action_count += 1
            else:
                self._invalid_action_count = 0

        obs = self._get_obs()
        # episode_cost: running RCPO constraint cost C(tau) for this episode --
        # see SchedulingEnv.episode_cost and
        # Future/research/2026-08-21-rcpo-constrained-tardiness.md. Exposed every
        # step (cheap float read) so a trainer can read the final value once
        # `terminated` fires without reaching past this wrapper by hand.
        info = {"action_mask": self.get_action_mask(), "episode_cost": self.env.episode_cost}

        terminated = done
        truncated = self._invalid_action_count >= self._max_invalid_actions

        return obs, float(reward), terminated, truncated, info




        