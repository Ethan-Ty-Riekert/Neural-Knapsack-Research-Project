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

    def __init__(self, env:SchedulingEnv):
        """Initialisation of gym warpper for scheduling environment. Expects passed import
        is class SchedulingEnv from scheduling_env.py"""
        super().__init__()

        self.env = env
        self.num_jobs = env.num_jobs
        self.num_machines = env.num_machines
        self.num_resources = env.num_resources
        self.horizon = env.horizon

        # For normalisation later on
        self.initial_capacity = env.capacity[:, :, 0].copy()

        ### Action Space ###
        self.action_space = gym.spaces.Discrete(self.num_jobs * self.num_machines + 1) # +1 to allow for the idling action

        ### Observation Space ###
        obs_dim = self._compute_obs_dim() # gymnasium method
        self.observation_space = gym.spaces.Box(
            low = 0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def _compute_obs_dim(self):
        """Observation vector construction: Compute the dimension of the observation space.
        
        time (1), remaining capacity (num_machines * resource dimension),
        job features: duration (1), deadline (1), weight (1), requirement dimension (R), scheduled mask (1)
        -> total per job = 3 + R + 1 = R + 4"""

        return 1 + (self.num_machines * self.num_resources) + self.num_jobs * (self.num_resources + 4)

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

        # 3. Job features
        for j in range(self.num_jobs):
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

        return np.array(obs, dtype=np.float32) # use numpy for efficiency

    # OLD VERSION, AI OPTIMISED
    # def _get_obs(self):
    #     """Observation vector construction: Build the observation vector"""
    #     obs = []

    #     # 1. Normalised time
    #     t = min(self.env.time, self.horizon)
    #     obs.append(t/self.horizon)

    #     # 2. Remaining capacity (normalised)
    #     t_idx = min(self.env.time, self.horizon - 1)
    #     for m in range(self.num_machines):
    #         for r in range(self.num_resources):
    #             cap = self.env.capacity[m, r, t_idx] / (self.initial_capacity[m, r] + 1e-8) # for non zero division
    #             obs.append(cap)

    #     # Precompute normalisation constants
    #     max_dur = max(1.0, float(np.max(self.env.job_durations)))
    #     max_wgt = max(1.0, float(np.max(self.env.job_weights)))
    #     max_res = np.maximum(1.0, np.max(self.env.job_resources, axis=0))   

    #     # 3. Job features
    #     for j in range(self.num_jobs):
    #         # duration, deadline, weights
    #         obs.append(self.env.job_durations[j] / max_dur)
    #         obs.append(self.env.job_deadlines[j] / self.horizon)
    #         obs.append(self.env.job_weights[j] / max_wgt)

    #         # resource requirements (R dims)
    #         for r in range(self.num_resources):
    #             obs.append(self.env.job_resources[j, r] / max_res[r])

    #         # scheduled mask
    #         scheduled = 0.0 if j in self.env.remaining_jobs else 1.0
    #         obs.append(scheduled)

    #     return np.array(obs, dtype=np.float32) # use numpy for efficiency

    def get_action_mask(self):
        """Action mask building:
        mask[a] = 1 if (job, machine) is feasible at current time
        The final action (index = num_jobs * num_machines) is the idle action.
        """
        total_actions = self.num_jobs * self.num_machines + 1
        mask = np.zeros(total_actions, dtype=np.int8)

        t = min(self.env.time, self.horizon - 1)

        # Normal feasible scheduling actions
        for j in self.env.remaining_jobs:
            for m in range(self.num_machines):
                action_id = j * self.num_machines + m
                if self.env.is_feasible(j, m, t):
                    mask[action_id] = 1

        # Idle action is ALWAYS allowed
        idle_action = self.num_jobs * self.num_machines
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
        idle_action = self.env.num_jobs * self.env.num_machines

        if action_id == idle_action:
            obs, reward, done = self.env.step_idle()
        else:
            job = action_id // self.env.num_machines
            machine = action_id % self.env.num_machines
            # Let SchedulingEnv handle invalid actions (it applies invalidPenalty)
            _, reward, done = self.env.step((job, machine))

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}

        terminated = done
        truncated = False

        return obs, float(reward), terminated, truncated, info




        