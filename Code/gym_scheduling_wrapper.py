"""gym_scheduling_wrapper.py - The gym wrapper for the scheduling environment to allow for RL implementation
https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium
https://gymnasium.farama.org/
"""
import numpy as np 
import gymnasium as gym
from typing import List
from scheduling_env import SchedulingEnv


class GymSchedulingEnv(gym.Env):
    """Gymnasium wrapper for SchedulingEnv.
    Converts (job, machine) actions into a single integer action.
    Produces a fixed-size observation vector.
    Includes action masking
    """

    metadata = {"render_modes": []} # for gymnasium

    def __init__(self, env: SchedulingEnv, max_jobs: int = None, restrict_idle: bool = False):
        """Initialisation of gym warpper for scheduling environment. Expects passed import
        is class SchedulingEnv from scheduling_env.py.

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

        # For normalisation later on
        self.initial_capacity = env.capacity[:, :, 0].copy()

        ### Action Space ###
        self.action_space = gym.spaces.Discrete(self.max_jobs * self.num_machines + 1) # +1 to allow for the idling action

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
        """Observation vector construction: Build the observation vector"""
        obs = []

        # 1. Normalised time
        t = min(self.env.time, self.horizon)
        obs.append(t/self.horizon)

        # 2. Remaining capacity (normalised)
        t_idx = min(self.env.time, self.horizon - 1)
        for m in range(self.num_machines):
            for r in range(self.num_resources):
                cap = self.env.capacity[m, r, t_idx] / (self.initial_capacity[m, r] + 1e-8) # for non zero division
                obs.append(cap)

        # Precompute normalisation constants
        max_dur = max(1.0, float(np.max(self.env.job_durations)))
        max_wgt = max(1.0, float(np.max(self.env.job_weights)))
        max_res = np.maximum(1.0, np.max(self.env.job_resources, axis=0))

        # 3. Job features (max_jobs fixed-size slots; slots beyond this instance's
        # actual num_jobs are zero-padded and marked "scheduled" so the policy
        # treats them as already-handled/irrelevant. Their actions are always
        # masked out in get_action_mask().)
        for j in range(self.max_jobs):
            if j < self.num_jobs:
                # duration, deadline, weights
                obs.append(self.env.job_durations[j] / max_dur)
                obs.append(self.env.job_deadlines[j] / self.horizon)
                obs.append(self.env.job_weights[j] / max_wgt)

                # resource requirements (R dims)
                for r in range(self.num_resources):
                    obs.append(self.env.job_resources[j, r] / max_res[r])

                # scheduled mask
                scheduled = 0.0 if j in self.env.remaining_jobs else 1.0
                obs.append(scheduled)
            else:
                obs.extend([0.0] * (3 + self.num_resources))
                obs.append(1.0)

        return np.array(obs, dtype=np.float32) # use numpy for efficiency

    def get_action_mask(self):
        """Action mask building:
        mask[a] = 1 if (job, machine) is feasible at current time
        The final action (index = max_jobs * num_machines) is the idle action.
        Job slots beyond this instance's actual num_jobs are never in
        self.env.remaining_jobs, so their actions stay masked out (0) automatically.
        """
        total_actions = self.max_jobs * self.num_machines + 1
        mask = np.zeros(total_actions, dtype=np.int8)

        t = min(self.env.time, self.horizon - 1)

        # Normal feasible scheduling actions
        for j in self.env.remaining_jobs:
            for m in range(self.num_machines):
                action_id = j * self.num_machines + m
                if self.env.is_feasible(j, m, t):
                    mask[action_id] = 1

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
        self.env.reset()

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
            _, reward, done = self.env.step((job, machine))

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}

        terminated = done
        truncated = False

        return obs, float(reward), terminated, truncated, info




        