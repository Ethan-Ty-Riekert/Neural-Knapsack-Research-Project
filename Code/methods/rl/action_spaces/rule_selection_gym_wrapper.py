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

RL picks WHICH classical priority rule (Code/methods/heuristics/priority_rules.py::
PRIORITY_RULES -- EDF/SPT/LST/FCFS/LPT/WSPT/ATC) to apply this tick, paired
with FirstFit placement; the rule's already-implemented, already-validated
Code/methods/heuristics/registry.py::HEURISTICS["{Rule}+FirstFit"] choose() function
decodes that into a real (job, machine) placement, executed on the SAME
underlying SchedulingEnv/OnlineSchedulingEnv unchanged. Action space shrinks
from 1000+ choices to len(PRIORITY_RULES)+1 = 8.

use_atc_feature (added 2026-09-23, S2W9, direct follow-up to Future/research/
training-log.md's observation-informativeness-probe entry): optional,
default-off flag that appends the same per-job ATC-priority feature Option 3
already uses (Code/methods/rl/action_spaces/obs_atc_feature.py, extracted from
priority_only_gym_wrapper.py so both options share one implementation) to
this option's observation. Motivation: a linear/nonlinear probe found the
raw observation only weakly encodes SPT-vs-ATC job-choice disagreement (AUC
0.62/0.65) -- this gives the rule-selection policy an explicit signal for
exactly that distinction, testing whether the gap to ATC's tardiness
performance narrows once the feature doesn't have to be re-derived from raw
duration/deadline/weight/time features. Default False preserves every
existing Option 1 checkpoint's observation_space shape unchanged.

placements (added 2026-10-05, S2W12, v2 high-load runs): optional tuple of
placement rule names (Code/methods/heuristics/placement_rules.py::PLACEMENT_RULES).
The action menu becomes every priority rule x every listed placement, decoded
through the matching HEURISTICS["{Rule}+{Placement}"] entry, plus idle. Default
("FirstFit",) keeps the original 8-action menu, order and labels exactly, so every
existing Option 1 checkpoint still loads. Motivation: the 2026-09-30 v2 heuristic
sweep found placement matters as much as priority at high online load
(LST+Consolidate cuts J ~40% vs LST+FirstFit at rho 0.95), and a FirstFit-only
menu makes that unreachable for the RL policy by construction. Grounding:
Option 1 is a selection hyper-heuristic over low-level heuristics (Burke et al.
2013, J. Oper. Res. Soc. 64(12)); widening the low-level set is the standard
lever in that framework.

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

from Code.methods.heuristics.priority_rules import PRIORITY_RULES
from Code.methods.heuristics.placement_rules import PLACEMENT_RULES
from Code.methods.heuristics.registry import HEURISTICS
from Code.methods.rl.action_spaces.obs_atc_feature import append_atc_priority_feature

RULE_NAMES = list(PRIORITY_RULES.keys())
DEFAULT_PLACEMENTS = ("FirstFit",)


DECISION_EPOCHS = ("placement", "tick")


def rule_kwargs_from_args(args):
    """All Option 1 wrapper settings from a script's CLI args, in one dict (2026-10-05) -- the single
    place these flags are read, so training, evaluation and run.py pass one object through."""
    return dict(placements=parse_placements(getattr(args, "rule_placements", None)),
                decision_epoch=getattr(args, "decision_epoch", None) or "placement")


def is_default_rule_kwargs(rule_kwargs):
    rk = rule_kwargs or {}
    return (tuple(rk.get("placements", DEFAULT_PLACEMENTS)) == DEFAULT_PLACEMENTS
            and rk.get("decision_epoch", "placement") == "placement")


def parse_placements(text):
    """--rule-placements CLI value ("FirstFit,Consolidate") -> placements tuple. Shared by the
    training/evaluation scripts and run.py so the format is defined once."""
    if not text:
        return DEFAULT_PLACEMENTS
    return tuple(p.strip() for p in text.split(",") if p.strip())


class RuleSelectionGymSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, full_gym_env, use_atc_feature: bool = False, placements=DEFAULT_PLACEMENTS,
                 decision_epoch: str = "placement"):
        super().__init__()
        self._full = full_gym_env
        self.env = full_gym_env.env
        self.max_jobs = full_gym_env.max_jobs
        self.num_machines = full_gym_env.num_machines
        self.num_resources = full_gym_env.num_resources
        self.horizon = full_gym_env.horizon
        self.use_atc_feature = use_atc_feature

        if decision_epoch not in DECISION_EPOCHS:
            raise ValueError(f"decision_epoch must be one of {DECISION_EPOCHS}, got {decision_epoch!r}")
        self.decision_epoch = decision_epoch
        self.placements = tuple(placements)
        unknown = set(self.placements) - set(PLACEMENT_RULES)
        if unknown or not self.placements:
            raise ValueError(f"placements must be a non-empty subset of {list(PLACEMENT_RULES)}, got {placements!r}")
        self._heuristic_keys = [f"{r}+{p}" for p in self.placements for r in RULE_NAMES]
        # Labels for logs/diagnostics: plain rule names for the default menu (unchanged from
        # before), full "Rule+Placement" labels once more than FirstFit is offered.
        self.rule_names = (RULE_NAMES if self.placements == DEFAULT_PLACEMENTS
                           else list(self._heuristic_keys))
        self.num_rules = len(self.rule_names)
        self.action_space = gym.spaces.Discrete(self.num_rules + 1)  # +1 idle

        if use_atc_feature:
            # Must match GymSchedulingEnv._get_obs()'s per-job-slot layout:
            # [duration, deadline, weight, resource_0..R-1, scheduled].
            self._job_slot_width = full_gym_env.obs_layout.job_slot_width  # Code/core/obs_layout.py
            self._machine_block_end = full_gym_env.obs_layout.machine_block_end
            obs_dim = full_gym_env.observation_space.shape[0] + self.max_jobs
            self.observation_space = gym.spaces.Box(low=float(full_gym_env.observation_space.low.min()),
                                                high=float(full_gym_env.observation_space.high.max()),
                                                shape=(obs_dim,), dtype=np.float32)
        else:
            self.observation_space = full_gym_env.observation_space

        self._invalid_action_count = 0
        self._max_invalid_actions = 2 * (self.num_rules + 1)

    def _get_obs(self):
        base_obs = self._full._get_obs()
        if not self.use_atc_feature:
            return base_obs
        return append_atc_priority_feature(
            base_obs, self.env, self.max_jobs, self._job_slot_width, self._machine_block_end,
        )

    def _decode(self, a):
        return a // self.num_machines, a % self.num_machines

    def _job_actions(self):
        mask = self._full.get_action_mask()
        idle_id = self.max_jobs * self.num_machines
        # PERF (2026-10-05): vectorised. The Python loop over all max_jobs * num_machines ids
        # (~13k online) was ~45% of Option 1 training time (cProfile). Same ascending list of ints.
        return np.flatnonzero(np.asarray(mask[:idle_id])).tolist()

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
        mask[self.num_rules] = 1 if self._full.idle_allowed(mask[:self.num_rules].any()) else 0
        return mask

    def reset(self, *, seed=None, options=None):
        self._full.reset(seed=seed, options=options)
        self._invalid_action_count = 0
        obs = self._get_obs()
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
            _, reward, done = self._full.idle_step()
        elif self.decision_epoch == "tick":
            # One decision per tick: apply the chosen rule to every placement that fits this tick,
            # then advance the clock; the transition's reward is the sum over the tick (exact, since
            # the per-step rewards telescope to -J). Offline, each placement already advances time,
            # so this loop runs once and the mode equals "placement".
            choose = HEURISTICS[self._heuristic_keys[action_id]]
            reward, done, t0 = 0.0, False, self.env.time
            while job_actions and not done and self.env.time == t0:
                job, machine = self._decode(choose(self.env, job_actions, self._decode))
                _, r, done = self.env.step((job, machine))
                reward += r
                job_actions = self._job_actions()
            if not done and self.env.time == t0:
                _, r, done = self._full.idle_step()
                reward += r
            self._invalid_action_count = 0
        else:
            choose = HEURISTICS[self._heuristic_keys[action_id]]
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

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask(), "episode_cost": self.env.episode_cost}

        terminated = done
        truncated = self._invalid_action_count >= self._max_invalid_actions

        return obs, float(reward), terminated, truncated, info
