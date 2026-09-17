# Design: DeepRM-Style Bounded Action-Space Window (Prepared, Not Yet Trained) + Option 1 Curriculum Integration

**Date:** 2026-09-18 (S2W10)
**Environment:** `Code/env/windowed_priority_gym_wrapper.py`,
`Code/policies/windowed_priority_pointer_policy.py`,
`Code/policies/windowed_priority_pointer_ppo_policy.py`, `Code/training/train_optimized.py`
**Status:** Windowed action space implemented and unit-tested; **deliberately not wired
into any training run** -- window size and the window-selection ordering are real design
choices flagged for the user's review, not just engineering. Option 1 curriculum
integration implemented and smoke-tested (see Section 2).

---

## 1. Windowed action space for Options 2/3

### 1.1 Motivation

`2026-09-17-action-space-reduction.md` Section 7 named this as the highest-leverage
untried lever for Options 2/3: their current `Discrete(max_jobs+1)` action space
(~101-400+ depending on scale) is still far larger than DeepRM's actual ~11-21 choices
(Mao et al. 2016, HotNets -- `mao2016deeprm` in `references.bib`). DeepRM's real
mechanism is not "pick any job" -- it's "pick from a small bounded VISIBLE WINDOW (M
slots), plus a scalar backlog count summarizing everything beyond the window."

### 1.2 Design choices (flagged for review, not unilaterally decided)

- **Window selection: earliest-deadline (EDF) order**, not raw arrival/FIFO order.
  DeepRM's own queue has no deadlines, so FIFO was its only sensible choice; this
  project's tardiness objective makes deadline urgency the natural windowing key
  instead. This still leaves WHICH of the M visible jobs to run, and when, entirely up
  to the learned policy -- windowing only bounds the candidate set, unlike Option 1's
  full priority-rule choice.
- **Backlog feature**: one scalar, `(jobs waiting beyond the window) / max_jobs`,
  appended once at the end of the observation (not per-slot) -- summarizes "how much
  unaddressed backlog exists" without exposing any individual backlogged job's
  identity, matching DeepRM's own backlog design intent.
- **Default window_size=15**: a round number inside DeepRM's own ~10-20 range, NOT
  independently re-derived for this project's instance distribution -- flagged per
  CLAUDE.md as an untested constant, same treatment as ATC's k=2.0 or the heavy-tailed
  arrivals' sigma=1.0.

### 1.3 Implementation

`WindowedPriorityGymSchedulingEnv` (composition over an already-built
`GymSchedulingEnv`/`OnlineGymSchedulingEnv`, same pattern as every other wrapper in this
project): `Discrete(window_size+1)`, observation = time + machine block + `window_size`
job slots (EDF-ordered, zero-padded/marked-scheduled if fewer jobs remain than the
window) + one backlog scalar. `WindowedPriorityPointerActorCritic`
(`windowed_priority_pointer_policy.py`) adapts `PriorityPointerActorCritic` for this
shape: same job/machine encoders and per-job scoring head, plus the backlog scalar
folded into the context vector alongside job/machine context and time. SB3 wiring
(`WindowedPriorityPointerMaskableActorCriticPolicy`) mirrors the existing
`PriorityPointerMaskableActorCriticPolicy` pattern exactly.

7 regression tests (`tests/test_windowed_priority_wrapper.py`): action-space/obs-dim
sizing, EDF-ordered window selection + backlog-scalar correctness (verified against a
hand-constructed deadline sequence), padding when fewer jobs remain than the window,
correct job resolution on placement, idle/infeasible-slot fallback, the 0-d-ndarray
regression guard (same bug class found in the non-windowed wrappers 2026-09-17), and
network forward-pass shapes for both `use_atc` settings. All pass.

### 1.4 Explicitly not done here

CLI wiring into `train_action_space_variant.py` (a new `--window-size` flag analogous to
Options 2/3's `use_atc`) and any training run -- left for after the window-size/ordering
design choices above are reviewed, per the user's own "I decide structure" rule from the
online-arrival MDP work.

## 2. Option 1 curriculum integration

`train_optimized.py::make_env()` gained `action_mode: "placement" | "rule_selection"`
(default `"placement"`, unchanged) and `job_weight_range` (matching
`env_config.py`/`arrival_process.py`'s parameter). `action_mode="rule_selection"` wraps
the built `GymSchedulingEnv`/`OnlineGymSchedulingEnv` with
`RuleSelectionGymSchedulingEnv` (Option 1) before `ActionMasker`, rebuilt fresh at every
curriculum stage transition exactly like every other wrapper here -- lets Option 1 use
this file's real 4-stage curriculum and Optuna-tuned hyperparameters instead of only
`train_action_space_variant.py`'s standalone flat-timestep trainer. Only wired through
the PPO (non-RCPO) path -- Option 1 has only ever used `MaskablePPO` this session. New
`--action-mode`/`--job-weight-min`/`--job-weight-max` CLI flags; a `parser.error` guards
against the nonsensical `--action-mode rule_selection --policy-type pointer` combination
(`PointerActorCritic` is sized for the placement action space). New
`action_mode_suffix`/`weight_suffix` components added to the checkpoint path-suffix
logic so this mode's checkpoints never collide with placement-mode ones at the same
config.

Smoke-tested via `python -m Code.training.train_optimized --algo ppo --action-mode
rule_selection --policy-type flat --no-curriculum ...` -- caught one real, unrelated
pre-existing rough edge in the process: `--no-curriculum` mode ignores
`--stage4-timesteps` entirely and always runs the hardcoded `TOTAL_TIMESTEPS=300_000`
(the flag only affects curriculum mode's 4th stage), so there is currently no CLI-level
way to run a truly tiny (`--no-curriculum`) smoke test through this file -- noted here
rather than fixed, since it isn't blocking anything and fixing it risks touching a
widely-used code path late at night without review.

## 3. References

See `references.bib`. Directly relevant: `mao2016deeprm` (DeepRM's bounded window +
backlog design), `tavakoli2018actionbranching` (the general factored-action-space
principle this and the still-unimplemented learned-placement-head fallback both draw
on).
