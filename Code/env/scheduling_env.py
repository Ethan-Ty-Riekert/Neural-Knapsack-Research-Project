"""scheduling.py: Revised environment to fit new mathematical formulation
as of 10/05/2026. Resource constrained scheduling environment for cloud resource
allocation. Jobs are deterministic and static. It is assumed that the job requirements
can fit on at least an empty machine

Closest directly domain-comparable related work: Zhang et al., "SPANE: A
Symmetry-Preserving Architecture for Multi-NUMA Environments -- A Deep
Reinforcement Learning Approach for Dynamic VM Scheduling" (arXiv:2504.14946,
2025) -- unlike the generic job-shop/routing literature cited in
Code/policies/pointer_policy.py (which justifies the pointer-network
*architecture*), SPANE targets DRL for cloud VM scheduling specifically, i.e.
the same problem *domain* this environment models. See
Future/research/2026-08-09-fixed-instance-bugfix-and-reward-rescale.md."""
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
        idle_penalty: float = 0.5,        # penalty for idling (doing nothing)
        use_potential_shaping: bool = False,  # Solution 3: optional potential-based reward shaping
        shaping_gamma: float = 0.99,      # discount factor used in the shaping term; should match the RL algorithm's gamma
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

        # Pristine per-resource capacity vector, kept separately from self.capacity
        # so reset() has an untouched value to restore from (see reset()).
        self.machine_capacity = np.array(machine_capacity, dtype=float).copy()

        # Machine capacity over time: shape (num_machines, num_resources, horizon)
        self.capacity = np.zeros((self.num_machines, self.num_resources, self.horizon))
        for m in range(self.num_machines):
            for t in range(self.horizon):
                self.capacity[m, :, t] = self.machine_capacity
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

        # Solution 3: optional potential-based reward shaping (Ng, Harada, Russell
        # 1999) -- see _compute_potential() and Future/research/2026-08-09-pointer-
        # network-action-head.md. Off by default so it's A/B-able against the
        # unshaped reward.
        self.use_potential_shaping = use_potential_shaping
        self.shaping_gamma = shaping_gamma
        self.prev_potential = self._compute_potential()



    def set_jobs(
        self,
        job_durations: np.ndarray,
        job_resources: np.ndarray,
        job_deadlines: np.ndarray,
        job_weights: np.ndarray,
    ):
        """Swap in a new job set (same num_jobs/num_machines/horizon/capacity/
        reward config) and reset. Used by GymSchedulingEnv's optional
        job_resampler (Experiment 2, randomized-instance generalization) to draw
        a fresh instance every episode instead of reusing the one fixed at
        construction time -- see Future/research/training-log.md's S2W5 entries
        for why this matters (a flat per-index head can memorize a single fixed
        instance; this is the test that actually rules that out). Does not
        support changing num_jobs (would invalidate GymSchedulingEnv's num_jobs-
        derived observation/action space sizing) -- callers must generate the
        replacement job set with the same num_jobs as this instance.
        """
        if len(job_durations) != self.num_jobs:
            raise ValueError(
                f"set_jobs() requires the same num_jobs ({self.num_jobs}); got {len(job_durations)}. "
                "Changing num_jobs mid-curriculum-stage would invalidate the observation/action space "
                "sizing GymSchedulingEnv already fixed from the first env it saw."
            )
        self.job_durations = job_durations
        self.job_resources = job_resources
        self.job_deadlines = job_deadlines
        self.job_weights = job_weights
        self.reset()

    def reset(self):
        """Reset the environment to the initial state.

        BUG FIX (2026-08-09): this used to copy self.capacity[m, :, 0] into every
        timestep -- but capacity[:, :, 0] is itself mutated by step() whenever a job
        starts at time 0 (which is most episodes), so it was never actually the
        original capacity after the first episode. Every subsequent reset()
        re-broadcast that already-depleted slice across the whole horizon, so
        capacity leaked downward, permanently, across the SchedulingEnv instance's
        entire lifetime (confirmed: repeated reset()+step() calls monotonically
        shrink capacity[:, :, 0] and it never recovers). Given a single instance is
        reused for many hundreds/thousands of episodes per curriculum stage, this
        eventually makes every job infeasible on every machine regardless of policy
        quality, which produces the exact "-idle_penalty * episode_length" signature
        used throughout Future/research/training-log.md to diagnose "policy
        collapse" -- i.e. this bug is a plausible confound (partial or total) for
        that diagnosis, not just a coincidental separate issue. See
        Future/research/2026-08-09-pointer-network-action-head.md.
        """
        for m in range(self.num_machines):
            for t in range(self.horizon):
                self.capacity[m, :, t] = self.machine_capacity

        self.machine_active[:] = 0
        self.start_times[:] = -1
        self.tardiness[:] = 0
        self.remaining_jobs = set(range(self.num_jobs))
        self.time = 0
        self.prev_theta = 0.0
        self.prev_potential = self._compute_potential()

        return self.get_state()

    def _compute_potential(self) -> float:
        """Potential function for optional potential-based reward shaping (Ng,
        Harada, Russell 1999, "Policy Invariance Under Reward Transformations"):
        shaped_reward = reward + shaping_gamma * Phi(s') - Phi(s) is provably
        policy-invariant (does not change which policy is optimal) for any bounded
        Phi. Unlike raw reward-magnitude tuning (e.g. a much larger idle_penalty),
        which DOES change the optimum and, per Future/research/training-log.md,
        still didn't prevent idle collapse anyway, this only reshapes the gradient
        signal, giving credit for progress toward urgent jobs before their
        tardiness penalty would otherwise fire.

        Phi(s) = -sum_{j in remaining_jobs} urgency_j(t), where
        urgency_j(t) = 1 / (max(slack_j(t), 0) + 1), slack_j(t) = deadline_j - t -
        duration_j. urgency_j is bounded in (0, 1]: it saturates at 1 once a job is
        already at risk of being late (slack <= 0) rather than blowing up or
        flipping sign, and decays towards 0 the more comfortably ahead of schedule
        a job is. Phi(s) is therefore bounded in [-len(remaining_jobs), 0] --
        completing a job (removing it from remaining_jobs) always removes its
        (negative) contribution, and letting time pass without scheduling an urgent
        job makes Phi(s) more negative, so the shaping term rewards moving urgent
        jobs to completion before their deadline penalty would fire, without
        altering which final policy is optimal.
        """
        if not self.remaining_jobs:
            return 0.0
        potential = 0.0
        for j in self.remaining_jobs:
            slack = self.job_deadlines[j] - self.time - self.job_durations[j]
            urgency = 1.0 / (max(slack, 0.0) + 1.0)
            potential -= urgency
        return potential
    
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

        # Feasibility check
        if not self.is_feasible(job, machine, self.time):
            return (self.get_state(), -self.invalidPenalty, False)

        ## If made it up to this case the placement is valid ##
        # Machine activation.
        # BUG FIX (this session): this used to flip machine_active[machine] to 1
        # BEFORE the feasibility check above, and never rolled it back if that
        # same action then failed feasibility -- so an infeasible attempt on a
        # never-used machine permanently marked it "active" without the -lambda1
        # activation penalty ever being charged on the real first successful use.
        # Moved here, after feasibility is confirmed, so machine_was_inactive only
        # ever reflects an actual placement.
        machine_was_inactive = self.machine_active[machine] == 0
        if machine_was_inactive:
            self.machine_active[machine] = 1

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

        # Solution 3: optional potential-based shaping term, added on top of
        # (not instead of) the reward computed above -- see _compute_potential().
        if self.use_potential_shaping:
            new_potential = self._compute_potential()
            reward += self.shaping_gamma * new_potential - self.prev_potential
            self.prev_potential = new_potential

        # Check for termination at end of horizon or this was the last job
        done = len(self.remaining_jobs) == 0 or self.time > self.horizon

        return (self.get_state(), reward, done)
    
    def reward(self, j:int, m:int, ym:bool, delta_theta: float, idle: bool = False) -> float:
        """Reward function"""
        reward = 0.0

        # Machine activation penalty.
        # BUG FIX (this session): this was `if ym is True:` -- but every caller
        # passes a numpy bool_ (from `self.machine_active[machine] == 0`), and
        # `np.bool_(True) is True` is False (identity check against a different
        # object than Python's True singleton, not an equality check). So this
        # branch has never actually fired, in the project's entire history,
        # independent of Issue A above: the -lambda1 activation penalty has been
        # silent dead code. Truthiness (`if ym:`) is correct for both a Python
        # bool and a numpy bool_.
        if ym:
            reward -= self.lambda1

        # Tardiness penalty, normalised by horizon.
        # BUG FIX (this session): raw tardiness T_j = max(0, t+P_j-d_j) is
        # unbounded and scales with horizon H (bounded by H-10 given
        # deadline_range=(10,110)), while every other reward term here is a fixed
        # O(1) constant regardless of curriculum stage. Across horizon in
        # {20,40,60,100}, that let this single term's magnitude grow ~5x from the
        # first to the last curriculum stage while the value function/model is
        # reused across all stages with no reset -- badly miscalibrating the value
        # target right at the stage transitions where it matters most (see
        # Future/research/training-log.md). Dividing by self.horizon keeps
        # T_j/H < 1 for any job that is ever actually scheduled (is_feasible
        # guarantees t+P_j <= H, and d_j >= 10), so this term stays on the same
        # O(1) footing as the others at every stage. See
        # Future/research/<dated>-fixed-instance-bugfix-and-reward-rescale.md.
        reward -= self.lambda2 * self.job_weights[j] * (self.tardiness[j] / self.horizon)

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

        # Solution 3: same shaping term as step(). Idling while urgent jobs remain
        # makes their slack shrink without progress, so Phi(s') is more negative
        # than Phi(s) here more often than not -- this is what gives the shaping
        # term its "idling near a deadline crunch is worse than idling with slack
        # to spare" property, on top of the flat idle_penalty above.
        if self.use_potential_shaping:
            new_potential = self._compute_potential()
            reward += self.shaping_gamma * new_potential - self.prev_potential
            self.prev_potential = new_potential

        done = len(self.remaining_jobs) == 0 or self.time > self.horizon

        return self.get_state(), reward, done

