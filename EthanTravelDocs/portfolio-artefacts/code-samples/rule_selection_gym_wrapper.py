"""rule_selection_gym_wrapper.py - Option 1 (hyper-heuristic rule selection)
action-space reduction, part of the 2026-09-17 action-space-reduction work
(see Future/research/2026-09-17-action-space-reduction.md and
C:\\Users\\ethan\\.claude\\plans\\encapsulated-questing-cray.md).

Motivation: every mechanism tried this session for PPO's persistently poor
tardiness (reward weight, four Lagrangian ceilings, a hyperparameter search,
pointer architecture, a comprehensive dense-reward redesign) converged on
the same ~1290-1330 band. The one untested structural difference from
DeepRM/Decima (the literature that reports 20%+ gains over heuristics): this
project's action space is max_jobs*num_machines+1 (1000+ at deployed scale)
vs. DeepRM's ~11 -- a well-documented independent driver of policy-gradient
sample-inefficiency.

RL picks WHICH classical priority rule (Code/baselines/priority_rules.py::
PRIORITY_RULES -- EDF/SPT/LST/FCFS/LPT/WSPT/ATC) to apply this tick, paired
with FirstFit placement; the rule's already-implemented, already-validated
Code/baselines/registry.py::HEURISTICS["{Rule}+FirstFit"] choose() function
decodes that into a real (job, machine) placement, executed on the SAME
underlying SchedulingEnv/OnlineSchedulingEnv unchanged. Action space shrinks
from 1000+ choices to len(PRIORITY_RULES)+1 = 8.

Wraps an already-constructed GymSchedulingEnv or OnlineGymSchedulingEnv
instance (composition, not subclassing) so this one file is correct for
both the offline and online case without duplicating either's _get_obs()
(they differ -- see online_gym_wrapper.py's causality-fix docstring).
Exposes `.env` as the raw SchedulingEnv/OnlineSchedulingEnv (not the wrapped
gym env), matching GymSchedulingEnv's own `.env` meaning, so this class can
be passed through ActionMasker exactly like any other gym env in this
project and existing eval code's `env.env.env`-style base_env access keeps
working unchanged.
"""
import gymnasium as gym
import numpy as np

from Code.baselines.priority_rules import PRIORITY_RULES
from Code.baselines.registry import HEURISTICS

RULE_NAMES = list(PRIORITY_RULES.keys())


class RuleSelectionGymSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, full_gym_env):
        super().__init__()
        self._full = full_gym_env
        self.env = full_gym_env.env
        self.max_jobs = full_gym_env.max_jobs
        self.num_machines = full_gym_env.num_machines
        self.num_resources = full_gym_env.num_resources
        self.horizon = full_gym_env.horizon

        self.rule_names = RULE_NAMES
        self.num_rules = len(self.rule_names)
        self.action_space = gym.spaces.Discrete(self.num_rules + 1)  # +1 idle
        self.observation_space = full_gym_env.observation_space

        self._invalid_action_count = 0
        self._max_invalid_actions = 2 * (self.num_rules + 1)

    def _decode(self, a):
        return a // self.num_machines, a % self.num_machines

    def _job_actions(self):
        mask = self._full.get_action_mask()
        idle_id = self.max_jobs * self.num_machines
        return [a for a in range(idle_id) if mask[a]]

    def get_action_mask(self):
        """All 7 rules are legal whenever >=1 (job, machine) placement is
        currently feasible -- a rule's choose() always returns a feasible
        action given a non-empty job_actions list, so every rule "succeeds"
        by construction. When nothing is feasible, only idle is legal (any
        rule pick would fall through to idle anyway -- masking it out
        instead gives MaskablePPO a cleaner signal)."""
        mask = np.zeros(self.num_rules + 1, dtype=np.int8)
        if self._job_actions():
            mask[:self.num_rules] = 1
        mask[self.num_rules] = 1  # idle always legal
        return mask

    def reset(self, *, seed=None, options=None):
        obs, _ = self._full.reset(seed=seed, options=options)
        self._invalid_action_count = 0
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action_id):
        # Defensive cast -- see the identical fix/comment in
        # priority_only_gym_wrapper.py's step(): model.predict() can hand
        # back a 0-d numpy ndarray, which tolerates == and list-indexing
        # (why this class never actually crashed on it) but is not hashable,
        # which would break `job in self.env.remaining_jobs`-style checks
        # the moment one is added here.
        action_id = int(action_id)
        job_actions = self._job_actions()

        if action_id == self.num_rules or not job_actions:
            _, reward, done = self.env.step_idle()
        else:
            rule_name = self.rule_names[action_id]
            choose = HEURISTICS[f"{rule_name}+FirstFit"]
            real_action = choose(self.env, job_actions, self._decode)
            job, machine = self._decode(real_action)
            was_pending = job in self.env.remaining_jobs
            _, reward, done = self.env.step((job, machine))
            # Generic valid-action check (works for offline SchedulingEnv
            # and OnlineSchedulingEnv alike, unlike the offline-only "did
            # time change?" proxy -- see online_gym_wrapper.py's step() for
            # why that proxy breaks under Option 2's relaxed clock). Should
            # never actually trip here since choose() only ever returns
            # actions drawn from job_actions, which get_action_mask() has
            # already verified feasible -- kept as the same belt-and-
            # suspenders safety net every other gym wrapper in this project
            # carries.
            if was_pending and job not in self.env.remaining_jobs:
                self._invalid_action_count = 0
            else:
                self._invalid_action_count += 1

        obs = self._full._get_obs()
        info = {"action_mask": self.get_action_mask(), "episode_cost": self.env.episode_cost}

        terminated = done
        truncated = self._invalid_action_count >= self._max_invalid_actions

        return obs, float(reward), terminated, truncated, info
