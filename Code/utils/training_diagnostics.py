"""training_diagnostics.py - Reusable training-time observability tooling,
added 2026-09-20 (S2W9). Answers two things this project previously had NO
standing tool for (both confirmed via direct exploration of
Code/training/train_action_space_variant.py and the whole Code/ tree before
building this -- see this session's plan file / training-log.md entry):

1. Diagnosing the still-open "more online training hurts" mystery (two
   independent designs, Option 1 and Option 3, both got WORSE going from
   300k to 900k online timesteps -- see training-log.md's newest entries).
   The entropy-collapse hypothesis behind this project's earlier attempt at
   this diagnosis was itself only ever done by manually eyeballing SB3's
   periodically-printed stdout table -- never a persisted, reusable signal.
   ActionDistributionCallback below turns that into a standing TensorBoard
   scalar (action_dist/entropy_normalized and friends).
2. An early-warning signal for whether a multi-hour training run is heading
   toward a good or bad outcome, well before it finishes -- decoupled from
   whatever the reward formula says (the entropy-collapse investigation
   itself is a cautionary example of a metric that moved without the
   underlying behaviour actually resolving -- see the 2026-09-19 ent-coef
   entry in training-log.md). TardinessEvalCallback below runs a small,
   FIXED, cheap held-out eval periodically during training and logs the
   actual tardiness trend, not just reward.

Both are plain sb3 BaseCallback subclasses, following the exact pattern
already proven twice in this codebase (Code/utils/plotting_utils.py's
LiveTrainingPlotter, Code/policies/ppo_lagrangian.py's
PPOLagrangianCallback) -- self.locals (populated every step by SB3's
OnPolicyAlgorithm.collect_rollouts() via callback.update_locals(locals()),
verified directly against the installed stable_baselines3 package: actions
and infos are both present in self.locals at _on_step() time, reflecting
the action/info from the step just taken) and self.logger.record(...) (SB3's
cheap custom-TensorBoard-scalar hook).

NOT wired into eval_action_space_variant.py's run_episode()/
_weighted_tardiness() at MODULE level -- that module imports FROM
train_action_space_variant.py, which imports this module, so a top-level
import here would be circular. TardinessEvalCallback imports them lazily
inside its own method instead (harmless: by the time a real eval actually
runs, during model.learn(), both modules have long since finished loading).
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


def _entropy_normalized(counts: np.ndarray) -> float:
    """Shannon entropy of a count histogram, normalised into [0, 1] by
    dividing by log(num_bins) -- 0 = fully collapsed onto one choice, 1 =
    perfectly uniform. Bins with zero count contribute 0 (standard 0*log(0)
    convention), not NaN."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    nonzero = p[p > 0]
    h = -np.sum(nonzero * np.log(nonzero))
    max_h = np.log(len(counts))
    return float(h / max_h) if max_h > 0 else 0.0


class ActionDistributionCallback(BaseCallback):
    """Logs periodic action-distribution statistics as TensorBoard scalars
    -- turns "what is the policy actually doing over the course of
    training" from ad-hoc log-scraping (this project's only prior precedent
    for this, done once for the offline LPT-collapse and once for the
    online SPT-collapse, both one-off/discarded) into a standing dashboard.

    mode is passed explicitly by the caller (build_diagnostics_callbacks()
    derives it from --option), not introspected from the env/action_space,
    to keep this simple and avoid fragile isinstance-based branching:
      - "rule" (Option 1): per-rule frequency + entropy over the 8 choices
        (7 rules + idle).
      - "job" (Options 2/3, windowed or not): idle_frac, entropy over the
        job-index histogram (too many raw indices for per-index TB scalars
        to be useful -- entropy is the summary), top1_frac (share of the
        single most-picked non-idle slot -- a cheap "is it collapsing onto
        one slot" signal).
      - "branching" (Option 4): the "job" stats above, PLUS
        machine_entropy_normalized/machine_top1_frac (same idea for the
        machine branch) and machine_mask_mismatch_frac (from
        ActionBranchingGymSchedulingEnv's info["mask_mismatch"] key -- a
        live measurement of the parallel-branch masking approximation's
        real cost during actual training, not a one-off guess).
    """

    def __init__(self, mode: str, num_job_choices: int, num_machines: int | None = None,
                 rule_names: list[str] | None = None, log_interval: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        if mode not in ("rule", "job", "branching"):
            raise ValueError(f"mode must be 'rule', 'job', or 'branching', got {mode!r}")
        if mode == "branching" and num_machines is None:
            raise ValueError("num_machines is required for mode='branching'")
        self.mode = mode
        self.num_job_choices = num_job_choices  # num_rules+1 (rule mode) or max_jobs+1 (job/branching)
        self.num_machines = num_machines
        self.rule_names = rule_names
        self.log_interval = log_interval

        self._job_counts = np.zeros(num_job_choices, dtype=np.int64)
        self._machine_counts = np.zeros(num_machines, dtype=np.int64) if num_machines else None
        self._mask_mismatches = 0
        self._mismatch_total = 0
        self._steps_since_log = 0

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals["actions"])
        actions = actions.reshape(actions.shape[0], -1)  # (n_envs, 1) or (n_envs, 2)

        for row in actions:
            job_choice = int(row[0])
            if 0 <= job_choice < self.num_job_choices:
                self._job_counts[job_choice] += 1
            if self.mode == "branching":
                machine_choice = int(row[1])
                if 0 <= machine_choice < self.num_machines:
                    self._machine_counts[machine_choice] += 1

        if self.mode == "branching":
            for info in self.locals.get("infos", []):
                if "mask_mismatch" in info:
                    self._mismatch_total += 1
                    if info["mask_mismatch"]:
                        self._mask_mismatches += 1

        self._steps_since_log += 1
        if self._steps_since_log >= self.log_interval:
            self._log_and_reset()
            self._steps_since_log = 0

        return True

    def _log_and_reset(self) -> None:
        job_total = self._job_counts.sum()
        if job_total > 0:
            if self.mode == "rule":
                names = self.rule_names or [f"rule_{i}" for i in range(self.num_job_choices - 1)]
                for i, name in enumerate(names):
                    self.logger.record(f"action_dist/{name}", float(self._job_counts[i]) / job_total)
                self.logger.record("action_dist/idle", float(self._job_counts[-1]) / job_total)
            else:
                self.logger.record("action_dist/idle_frac", float(self._job_counts[-1]) / job_total)
                non_idle = self._job_counts[:-1]
                if non_idle.sum() > 0:
                    self.logger.record("action_dist/top1_frac", float(non_idle.max()) / job_total)
            self.logger.record("action_dist/entropy_normalized", _entropy_normalized(self._job_counts))

        if self.mode == "branching":
            machine_total = self._machine_counts.sum()
            if machine_total > 0:
                self.logger.record("action_dist/machine_entropy_normalized",
                                    _entropy_normalized(self._machine_counts))
                self.logger.record("action_dist/machine_top1_frac",
                                    float(self._machine_counts.max()) / machine_total)
            if self._mismatch_total > 0:
                self.logger.record("action_dist/machine_mask_mismatch_frac",
                                    float(self._mask_mismatches) / self._mismatch_total)

        self._job_counts[:] = 0
        if self._machine_counts is not None:
            self._machine_counts[:] = 0
        self._mask_mismatches = 0
        self._mismatch_total = 0


class TardinessEvalCallback(BaseCallback):
    """Every eval_freq steps, runs the model deterministically over a small,
    FIXED set of held-out envs (built once by the caller, NOT resampled per
    eval -- so the TensorBoard trend line isn't confounded with
    instance-to-instance noise) and logs mean tardiness/weighted-tardiness/
    late-jobs. This is the actual early-warning signal requested: whether a
    run is trending toward success or failure, decoupled from whatever the
    reward formula says -- deliberately NOT derived from
    MaskableEvalCallback's reward-only signal (see build_diagnostics_callbacks()'s
    docstring for why: reward<->tardiness alignment is exactly what several
    of this project's past findings show can silently break).

    Cost: a handful of full episodes with no gradient computation, run once
    per eval_freq -- negligible relative to a multi-hour PPO run in
    practice, but this should be confirmed empirically with a timed run
    before being trusted as "doesn't meaningfully slow training", not
    assumed.
    """

    def __init__(self, held_out_envs: list, eval_freq: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.held_out_envs = held_out_envs
        self.eval_freq = eval_freq
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._run_eval()
            self._last_eval_step = self.num_timesteps
        return True

    def _run_eval(self) -> None:
        # Deferred import -- see module docstring for why this can't be a
        # top-level import (eval_action_space_variant.py imports FROM
        # train_action_space_variant.py, which imports this module).
        from Code.evaluation.eval_action_space_variant import run_episode, _weighted_tardiness

        tardiness_vals, weighted_vals, late_vals = [], [], []
        for env in self.held_out_envs:
            result = run_episode(self.model, env)
            tardiness_vals.append(float(result["tardiness"].sum()))
            weighted_vals.append(_weighted_tardiness(result))
            late_vals.append(result["late_jobs"])

        self.logger.record("eval_tardiness/mean_tardiness", float(np.mean(tardiness_vals)))
        self.logger.record("eval_tardiness/mean_weighted_tardiness", float(np.mean(weighted_vals)))
        self.logger.record("eval_tardiness/mean_late_jobs", float(np.mean(late_vals)))


def build_diagnostics_callbacks(option: str, held_out_envs: list, diagnostics_interval: int,
                                 rule_names: list[str] | None = None, max_jobs: int | None = None,
                                 num_machines: int | None = None) -> list[BaseCallback]:
    """Factory combining ActionDistributionCallback + TardinessEvalCallback +
    MaskableEvalCallback into the single callback list
    train_action_space_variant.py's model.learn(callback=...) needs.
    mode/num_job_choices are derived from option, not left for the caller to
    work out each time.

    MaskableEvalCallback (sb3_contrib, already installed, unused anywhere
    else in this repo before this) is reused as-is rather than reimplemented
    -- it already correctly threads action_masks through masked evaluation
    (plain SB3 EvalCallback would silently get this wrong for this
    project's masked action spaces) and gives periodic eval/mean_reward
    logging for free. It cannot produce a tardiness metric (only sees
    reward/episode-length), which is exactly why TardinessEvalCallback
    exists as a separate, purpose-built addition -- see that class's
    docstring. best_model_save_path is deliberately left unset (no
    unrequested side-effect file writes) -- this project already has its
    own checkpoint-archival convention (Code/utils/results_log.py::
    archive_checkpoint_files).
    """
    if option == "1":
        mode, num_job_choices = "rule", len(rule_names) + 1 if rule_names else 8
    elif option in ("2", "3"):
        mode, num_job_choices = "job", max_jobs + 1
    elif option == "4":
        mode, num_job_choices = "branching", max_jobs + 1
    else:
        raise ValueError(f"Unknown option {option!r}")

    return [
        ActionDistributionCallback(
            mode=mode, num_job_choices=num_job_choices, num_machines=num_machines,
            rule_names=rule_names, log_interval=diagnostics_interval,
        ),
        TardinessEvalCallback(held_out_envs=held_out_envs, eval_freq=diagnostics_interval),
        MaskableEvalCallback(
            held_out_envs[0], eval_freq=diagnostics_interval, n_eval_episodes=1,
            deterministic=True, verbose=0,
        ),
    ]
