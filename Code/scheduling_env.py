"""scheduling.py: Revised environment to fit new mathematical formulation 
as of 10/05/2026. Resource constrained scheduling environment for cloud resource
allocation. Jobs are deterministic and static. It is assumed that the job requirements 
can fit on at least an empty machine"""
import numpy as np
from typing import Tuple, List, Dict, Union

class SchedulingEnv:
    """Time-indexed scheduling environment for cloud resource allocation"""
    def __init__(
        self,
        job_durations: np.ndarray,        # shape (J,)      Processing Durations P_j
        job_resources: np.ndarray,        # shape (J, R)    Resource requirements a_jr
        job_deadlines: np.ndarray,        # shape (J,)      Deadlines d_j
        job_weights: np.ndarray,          # shape (J,)      Weights w_j
        num_machines: int,                #                 Machine set M
        machine_capacity: np.ndarray,     # shape (R,)      Capacity vector C_r
        horizon: int,                     #                 Planning horizon H
        lambda_1: float = 1.0,            # machine activation penalty
        lambda_2: float = 1.0,            # tardiness penalty
        lambda_3: float = 1.0,            # hotspot penalty
        invalid_penalty: float = 5.0,     # invalid placement penalty
        idle_penalty: float = 0.5         # penalty for idling (doing nothing)
    ):
        """Initiate the scheduling environment"""

        # We can use the mathematical symbols as per the latex document

        # Store job data
        self.num_jobs = len(job_durations) # Job set J
        self.num_resources = job_resources.shape[1] # Resources set R
        self.num_machines = num_machines # Machine set M
        self.horizon = horizon # Planning horizon H

        self.job_durations = job_durations # Set of processing durations
        self.job_resources = job_resources # Set of requirements of each job
        self.job_deadlines = job_deadlines # Set of job deadlines 
        self.job_weights = job_weights # Set of job weights w

        # Machine capacity over time: shape (num_machines, num_resources, horizon)
        self.capacity = np.zeros((self.num_machines, self.num_resources, self.horizon))
        for m in range(self.num_machines):
            for t in range(self.horizon):
                self.capacity[m, :, t] = machine_capacity
        # We have no need to store the entire capacity matrix as for everything we subtract
        # from this capacity over time matrix, we will eventually add back (- for job starting,+ for job finishing)

        # Machine activation flags
        self.machine_active = np.zeros(self.num_machines, dtype=int) # y_m vector

        # Job start times and tardiness
        self.start_times = np.full(self.num_jobs, fill_value=-1, dtype=int) # s_t start time
        self.tardiness = np.zeros(self.num_jobs) # Tardiness T

        # Remaining jobs
        self.remaining_jobs = set(range(self.num_jobs)) # Set of unscheduled jobs J_t
        # Implementation becomes a lot cleaner with this set instead of basing things based off of
        # has the start time remained unchanged

        # Time index
        self.time = 0

        # Reward coefficients
        self.lambda1 = lambda_1
        self.lambda2 = lambda_2
        self.lambda3 = lambda_3
        self.invalidPenalty = invalid_penalty
        self.idling_penalty = idle_penalty

        self.prev_theta = 0.0 # For hotspot tracking

        

    def reset(self):
        """Reset the environment to the initial state"""
        for m in range(self.num_machines):
            for t in range(self.horizon):
                self.capacity[m, :, t] = self.capacity[m, :, 0]
        
        self.machine_active[:] = 0
        self.start_times[:] = -1
        self.tardiness[:] = 0
        self.remaining_jobs = set(range(self.num_jobs))
        self.time = 0
        self.prev_theta = 0.0

        return self.get_state()
    
    def is_feasible(self, j:int, m:int, t:int) -> bool:
        """Check if job index j can start on machine m at time t.
        Returns False in two cases:
        - Task will not be completed at the end of the planning horizon
        - Machine can not fit the task"""
        
        # Job will not complete in time
        if t + self.job_durations[j] > self.horizon:
            return False

        # Job does not fit in machine
        diff = self.capacity[m, :, t] - self.job_resources[j]
        # If there is a single negative resource
        if (diff < 0).any():
            return False
        
        return True
    
    def step(self, action: Tuple[int, int]) -> Tuple[object, float, bool]:
        """Perform one environment step with the provide action where,
        action = (job_index, machine_index).
        
        Returns:
            - (state, reward, isDone)"""

        job, machine = action
        reward = 0.0

        # Invalid if job already scheduled
        if job not in self.remaining_jobs:
            return (self.get_state(), -self.invalidPenalty, False)
        
        # Machine activation
        machine_was_inactive = self.machine_active[machine] == 0
        if machine_was_inactive:
            self.machine_active[machine] = 1

        # Feasibility check
        if not self.is_feasible(job, machine, self.time):
            return (self.get_state(), -self.invalidPenalty, False)
        
        ## If made it up to this case the placement is valid ##
        # Apply resource usage
        duration = self.job_durations[job]
        for tau in range(self.time, self.time + duration):
            self.capacity[machine, :, tau] -= self.job_resources[job]

        # Update job timing
        self.start_times[job] = self.time
        self.tardiness[job] = max(0, self.time + duration - self.job_deadlines[job]) # Equation from report

        # Remove job from remaining set
        self.remaining_jobs.remove(job)

        # Check if a hotspot has been created with delta theta (incremental change in hotspot)
        new_theta = self.compute_theta()
        delta_theta = max(0, new_theta - self.prev_theta)
        self.prev_theta = new_theta

        # Compute mathematical reward
        reward = self.reward(job, machine, machine_was_inactive, delta_theta)

        ## Reward shaping to help with convergence of policy methods
        # STRONG positive reward for any valid scheduling action
        # This makes scheduling immediately attractive and helps prevent idle collapse
        reward += 3.0  # Increased from 1.0 to make scheduling more rewarding than idling

        # NOTE: hotspot severity is already penalised by -lambda3 * delta_theta inside
        # self.reward() above; there used to be a second "reward += 0.05 * (0-delta_theta)"
        # term here that double-counted the same penalty on top of lambda3. Removed.

        # Reward for finishing all jobs
        if len(self.remaining_jobs) == 0:
            reward += 50
        ## End reward shaping

        # Advance time
        self.time += 1

        # Check for termination at end of horizon or this was the last job
        done = len(self.remaining_jobs) == 0 or self.time > self.horizon

        return (self.get_state(), reward, done)
    
    def reward(self, j:int, m:int, ym:bool, delta_theta: float, idle: bool = False) -> float:
        """Reward function"""
        reward = 0.0

        # Machine activation penalty
        if ym is True:
            reward -= self.lambda1

        # Tardiness penalty 1.0 is linear
        reward -= self.lambda2 * self.job_weights[j] * self.tardiness[j]

        # Hotspot penalty
        reward -= self.lambda3 * delta_theta

        # Idling penalty, encourage scheduling if possible, but allow to idle
        if idle:
            reward -= self.idling_penalty

        return reward
    
    def compute_theta(self) -> float:
        """Compute theta = max utilisation across all machines and resources and times
        up to the current time index.
        
        This function penalises actions that create 'hotspots':
        if placing a job increases the usage on ANY machine-resource-time
        combination, theta will increase, and the reward will include a larger penalty."""
        initial_capacity = self.capacity[:, :, 0]
        used = initial_capacity[:, :, None] - self.capacity
        utilisation = used / (initial_capacity[:, :, None] + 1e-8)
        return np.max(utilisation)

    def get_state(self) -> Dict:
        """Return a dictionary state representation."""
        return {
            "time": self.time,
            "remaining_jobs": list(self.remaining_jobs),
            "machine_active": self.machine_active.copy(),
            "capacity": self.capacity.copy(),
            "start_times": self.start_times.copy(),
            "tardiness": self.tardiness.copy(),
        }

    def step_idle(self):
        """Idle step: advance time without scheduling a job."""
        self.time += 1

        reward = -self.idling_penalty

        done = len(self.remaining_jobs) == 0 or self.time > self.horizon

        return self.get_state(), reward, done

