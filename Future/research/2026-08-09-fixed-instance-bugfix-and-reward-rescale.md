# Fixed-Instance Bug Fixes, Reward Rescale, Eval Fairness, and Fresh Optuna Search

**Date:** 2026-08-09
**Environment:** `Code/env/scheduling_env.py`, `Code/env/gym_scheduling_wrapper.py`,
`Code/policies/a2c_policy.py`, `Code/training/{optuna_tune.py, train_optimized.py,
train_rl_agent.py}`, `Code/evaluation/eval_rl_agent.py`
**Status:** Complete. Three structural bugs found and fixed, reward function
rescaled, eval-fairness gap closed, per-stage checkpointing added -- all
verified by a new regression suite (`tests/test_bugfixes.py`) and end-to-end
smoke runs. Fresh Optuna hyperparameter search completed for both A2C
architectures (independent studies). Full rescaled-curriculum training runs and
a freshly-measured EDF baseline complete (Section 6): RL reward is now within
7-20% of EDF at the hardest curriculum stage, a complete reversal from every
prior fixed-instance result in this project -- though a tardiness/late-jobs gap
remains and is flagged as a distinct, not-yet-addressed follow-up (Section 6).
Randomized-instance generalization work remains deferred to the next phase
(Section 7), per the user's explicit prioritisation this session.

---

## 1. Motivation

`Future/research/training-log.md`'s most recent entries record RL losing badly
(deep negative reward) to a simple EDF heuristic baseline in the later curriculum
stages -- despite training on the *exact same fixed job instance* (`seed=0`,
identical job set replayed every episode of every stage; see
`Code/training/train_rl_agent.py`'s curriculum and `Code/env/env_config.py`)
that EDF is scored on fresh. Because RL has a strict information advantage here
(thousands of repeated exposures to the one instance EDF sees zero-shot), a
large loss margin points at an optimization/implementation bug rather than a
capability or generalization gap. This document reports on a focused
investigation of exactly that: bugs in the environment/reward path that are
independent of any architecture choice (flat vs. pointer), fixed and verified
before touching hyperparameters or architecture again. Randomized-instance
training (testing the pointer network's actual generalization advantage,
per Section 10 of `2026-08-09-pointer-network-action-head.md`) remains explicitly
deferred to a follow-up phase -- see Section 6 below.

---

## 2. Three bugs found and fixed in `Code/env/scheduling_env.py` / `gym_scheduling_wrapper.py`

### 2.1 `machine_active` set before the feasibility check, never rolled back

`SchedulingEnv.step()` used to flip `machine_active[machine] = 1` *before*
calling `is_feasible()`, and never rolled it back if that same action then
failed feasibility. An infeasible attempt on a never-used machine therefore
permanently marked it "active" without the `-lambda1` activation penalty ever
being charged on the machine's real first successful use -- corrupting the
activation-cost signal and making it order-dependent on whichever invalid
actions the policy happened to try first.

**Fix:** reordered `step()` so the mutation only happens after `is_feasible()`
has already confirmed the placement is valid.

### 2.2 `if ym is True:` -- a numpy-bool identity bug that made the fix above moot

While writing the regression test for 2.1, direct testing surfaced a second,
independent, and considerably more consequential bug: `reward()`'s activation
penalty branch reads

```python
if ym is True:
    reward -= self.lambda1
```

but every caller passes `ym = (self.machine_active[machine] == 0)`, which is a
**numpy `bool_`**, not a Python `bool`. `is` is an identity check, not an
equality check, and `np.bool_(True) is True` evaluates to `False` -- confirmed
directly:

```python
>>> a = np.zeros(1, dtype=int)
>>> ym = a[0] == 0
>>> type(ym), ym, ym is True, bool(ym)
(<class 'numpy.bool'>, True, False, True)
```

**This means the `-lambda1` activation penalty has never actually fired, in
this project's entire history**, independent of bug 2.1 -- verified with a
controlled before/after test (`tests/test_bugfixes.py`, "Issue A" section):
with `lambda_2 = lambda_3 = 0` isolating just this term, a real first
placement's reward read `3.0` (bug present -- activation penalty silently
skipped) vs. the mathematically correct `2.0` (`-lambda1 + 3.0` flat placement
bonus) once fixed.

**Fix:** use plain truthiness (`if ym:`), which is correct for both a Python
`bool` and a numpy `bool_`.

**Implication:** every `lambda_1` value in every prior training run and every
prior Optuna trial logged in this project (including the "best" `lambda_1`
values on disk from before this session) was tuning a parameter that had zero
effect on the actual reward signal. This doesn't explain "RL loses badly to
EDF" on its own -- if anything, a missing penalty term should make RL's
reported reward *higher* than it should be, not lower, and both RL and EDF
share the same `reward()` so the comparison isn't biased by it -- but it is a
material correction for the paper's account of what `lambda_1` was actually
doing in any run before this fix.

### 2.3 Mask/execution time-index mismatch at `t == horizon`

`GymSchedulingEnv.get_action_mask()` used a clamped
`t = min(env.time, horizon - 1)`, but the real feasibility check inside
`SchedulingEnv.step()`/`is_feasible()` used the uncapped `env.time`. At
`env.time == horizon` (a reachable, non-terminal state -- episodes only end
once `time > horizon`), a duration-1 job could read as feasible in the mask
(`(horizon-1)+1 == horizon`, not `> horizon`) but then fail the real check
(`horizon+1 > horizon`), landing in the invalid-action branch. Invalid actions
do not advance `self.time`, and nothing else about state changes on a failed
attempt -- so a policy trusting the (now-stale) mask could, in principle, get
stuck repeatedly selecting the same penalized invalid action with no way for
the episode to terminate on its own, exactly at the point curriculum stages get
long enough for this boundary to matter (stage 3/4).

**Fix:** uncapped the mask's `t` to `env.time`, matching `step()` exactly. Safe
because `is_feasible()` short-circuits on `t + duration > horizon` before any
array indexing, and every job has `duration >= 1`, so no out-of-bounds read is
possible for `t >= horizon`. `_get_obs()`'s *separate* clamp
(`t_idx = min(env.time, horizon - 1)`, which directly indexes
`capacity[m, r, t_idx]`) was deliberately left untouched -- that one is a real
array-bounds guard, not a feasibility-semantics bug.

**Belt-and-suspenders addition:** gave `GymSchedulingEnv` a per-episode
invalid-action counter (incremented whenever `SchedulingEnv.step()` returns its
invalid-penalty branch, detected cheaply by comparing `env.time` before/after
the call) that sets `truncated = True` past a fixed cap
(`2 * max_jobs * num_machines`). This guards the general *class* of bug
(invalid action -> no state change -> no other way to end the episode)
independent of this specific instance of it.

All three fixes verified in `tests/test_bugfixes.py`, run via
`python -m tests.test_bugfixes` -- see Section 4.

---

## 3. Reward-scale correction: tardiness normalised by horizon

Every reward term in `SchedulingEnv.reward()` is a bounded O(1) constant
(`-lambda1 in {0,-1}`, `-lambda3 * delta_theta in [-1,0]` since `delta_theta`
is a utilisation ratio, flat `+3.0` placement bonus, `+50` one-time completion
bonus, `-idle_penalty`) **except** tardiness:

```
T_j = max(0, t + P_j - d_j)
```

which is unbounded and scales with the episode's `horizon` (`T_j <= horizon -
10` given `deadline_range = (10, 110)` in `Code/env/env_config.py`). Across the
curriculum's `horizon in {20, 40, 60, 100}`, this term's achievable magnitude
grows roughly 5x from the first to the last stage while every other term stays
fixed. Since one A2C model/value-head is reused across all four stages
(`reset_num_timesteps=False`, no per-stage learning-rate schedule, no
value-loss clipping in the hand-rolled `MaskableA2C.train()`), the value
function calibrated on stage 1's reward scale is badly miscalibrated the moment
stage 3/4 activate -- right when TD-target stability matters most.

**Fix -- normalise by the episode's own horizon:**

```
T_j^norm = T_j / H

reward(j, m, y_m, delta_theta) = -lambda1 * 1[y_m]
                                  - lambda2 * w_j * T_j^norm
                                  - lambda3 * delta_theta
                                  (- idle_penalty, if idling)
```

i.e. `reward -= self.lambda2 * self.job_weights[j] * (self.tardiness[j] /
self.horizon)` in place of the raw `self.tardiness[j]`.

**Why this formula, over the alternatives considered:**

1. **Provably bounded and comparable across stages.** Since `is_feasible`
   guarantees any job that is ever actually scheduled satisfies
   `t + P_j <= H`, and `d_j >= 10`, `T_j^norm < 1` for *every* curriculum
   stage alike -- not just empirically, but by construction. This puts
   tardiness on the same O(1) footing as every other term.
2. **Rejected: divide by a fixed constant (e.g. stage 1's horizon, 20).** A raw
   tardiness of 20 timesteps means "maximally late" at `H=20` but only mildly
   late at `H=100` -- a fixed divisor doesn't correctly equate *lateness
   meaning* across different horizons the way dividing by the episode's own
   `H` does.
3. **Rejected: globally squash/clip total reward (e.g. `tanh`).** Destroys
   per-term interpretability needed for this paper's analysis (being able to
   say "the tardiness term contributed X, the hotspot term contributed Y"), and
   sits awkwardly alongside the existing `RunningMeanStd` reward normalizer in
   `a2c_policy.py`, which is explicitly documented as scale-only,
   sign-preserving, and training-signal-only -- a global squash at the
   environment level would duplicate/conflict with that design intent rather
   than complement it.
4. **Ties directly into the Optuna search-space change (Section 5).** The
   tuning environment (`Code/training/optuna_tune.py::make_tuning_env`,
   `horizon=30` by default) is deliberately smaller than the curriculum's
   largest stage (`horizon=100`). Under the old, unnormalised formula, whatever
   `lambda_2` Optuna found best at `H=30` (tardiness maxing out around 20) was
   miscalibrated by roughly 3-5x when deployed at `H=100` (tardiness maxing out
   around 90). Under `T_j/H`, tardiness is bounded in `[0, 1)` at *both*
   horizons, so a `lambda_2` tuned cheaply at the small tuning horizon
   transfers approximately correctly to the full curriculum's largest stage.

**Important caveat for anyone reading `training-log.md`:** `run_heuristic()`
(EDF) in `Code/evaluation/eval_rl_agent.py` computes its reward through this
exact same `SchedulingEnv.reward()`, so both sides of any RL-vs-EDF comparison
shift together -- *relative* comparisons made after this fix remain valid. The
*absolute* magnitudes do not match anything measured before this fix (e.g. the
`+275.5` EDF stage-4 reference in `training-log.md`'s 2026-08-09 entry, measured
pre-fix, needs re-measuring under the new formula, not reuse).

---

## 4. Verification

`tests/test_bugfixes.py` (run via `python -m tests.test_bugfixes`), following
this project's existing `tests/` convention (runnable script with `assert`
invariants, not a pytest suite):

```
=== Issue A: machine_active set-before-feasibility-check ===
  Infeasible attempt: reward=-5.0 (invalid_penalty), machine_active=0 (correctly still 0)
  First real placement: reward=2.0 (== -lambda1 + 3.0, activation penalty correctly charged)
  PASS

=== Issue C: mask/execution time-index mismatch at horizon boundary ===
  t=4 (horizon-1): mask[job0,machine0]=1 (correctly feasible)
  t=5 (horizon): mask[job0,machine0]=0 (correctly infeasible), step() reward=-5.0 (invalid, AGREES with mask)
  [pre-fix clamped mask would have said feasible=True -- exactly the divergence that hung episodes]
  PASS

=== Issue D: tardiness term normalised by horizon ===
  H=20: raw_tardiness=18.0, normalised term=0.9000 (pre-fix this term would have been the raw, unbounded 18.0)
  H=100: raw_tardiness=98.0, normalised term=0.9800 (pre-fix this term would have been the raw, unbounded 98.0)
  PASS -- tardiness term stayed O(1) at both a small and large horizon

=== A2C deterministic (greedy) eval mode ===
  deterministic=True across 5 calls: always action 6 (PASS)
  deterministic=False across 50 calls: 7 unique action(s) sampled (stochastic path unchanged)
  PASS
```

End-to-end smoke runs (`python -m Code.training.train_rl_agent --algo a2c
--policy-type {flat,pointer} --smoke-test`, 300 timesteps/stage): both
architectures complete the full 4-stage curriculum without crashing, with 4/4
per-stage checkpoints saved. A follow-up plumbing check
(`Code/evaluation/eval_rl_agent.py`'s `run_ppo`/`run_heuristic` called directly)
confirmed the deterministic A2C eval path and the EDF heuristic both run
cleanly against the rescaled reward end-to-end (numbers not meaningful -- the
model was only smoke-trained, 300 steps/stage).

---

## 5. Optuna search-space changes

`Code/training/optuna_tune.py`:

1. **`objective_a2c` split by `policy_type`.** Previously hardcoded
   `policy_type="pointer"` unconditionally. Now accepts `policy_type` (bound via
   `functools.partial` at study registration): samples `embed_dim`/`hidden` only
   for `"pointer"`; for `"flat"`, no architecture search added in this pass
   (kept at the flat trunk's fixed `256,256` shape) -- keeps scope contained.
2. **Two independent studies**, not one shared search space trading off
   architecture against hyperparameters: `a2c_flat_v2_scheduling_optimization`
   and `a2c_pointer_v2_scheduling_optimization`, with separate output files
   (`a2c_flat_best_params.json` / `a2c_pointer_best_params.json`, etc.) so
   `Code/training/train_optimized.py` can load the right one per architecture.
3. **`lambda_2` search range widened** from `[0.5, 2.0]` to a log-scale
   `[0.5, 20.0]` in both `objective_ppo` and `objective_a2c` -- see Section 3
   item 4 above for why the old range was calibrated against a now-nonexistent
   unbounded tardiness term.
4. **Fresh study names (`_v2` suffix)** so these runs don't resume any
   pre-fix study history under `rl_training/optuna.db` -- both the search
   space and the reward function's numeric meaning changed, so resuming would
   bias the TPE sampler's posterior with results measured against a
   now-nonexistent objective landscape. (Output *file* names deliberately do
   **not** carry the `_v2` tag -- `train_optimized.py` loads
   `a2c_{policy_type}_best_params.json` -- only the internal Optuna study name
   needs the freshness guarantee.)
5. A2C's Optuna eval loop now also calls `agent.act(..., deterministic=True)`,
   consistent with the eval-fairness fix in Section 6 below -- a less noisy
   trial-reward signal for Optuna to compare across trials.
6. `Code/training/train_optimized.py` fixed to actually consume the
   per-architecture files: added `--policy-type {pointer,flat}`; previously it
   always called `MaskableA2C(first_env, device="cpu")` with no `policy_type`/
   `policy_kwargs` at all, silently ignoring any `embed_dim`/`hidden` Optuna
   found and always building the pointer network's hardcoded defaults
   regardless of which best-params file was loaded.

**Sanity checks** (`n_trials=2` each, separate `optuna_sanity.db` storage, kept
isolated from the real study history): both `a2c_flat` and `a2c_pointer`
studies complete without error and report positive mean rewards on the small
tuning environment (`horizon=30`) -- `a2c_flat` best trial 106.06,
`a2c_pointer` best trial 105.84, `ppo` (1-trial sanity) 103.996. These are
*sanity-check* numbers only (2-trial budget, small tuning env, not the full
curriculum) -- not representative results, but a strong signal the search-space
changes and the rescaled reward are wired correctly end-to-end before spending
the full 50-trial budget.

**Full study results (50 trials each, flat then pointer, tuning env
`horizon=30, num_jobs=20, num_machines=5`):**

```
                  best trial   best mean_reward   embed_dim/hidden
a2c_flat          45           108.25             n/a (fixed 256,256 trunk)
a2c_pointer       49           108.01             128 / 32
```

Best hyperparameters (saved to `rl_training/optuna_results/a2c_{flat,pointer}_best_params.json`):

```
                  learning_rate   n_steps  gamma   gae_lambda  ent_coef  value_coef  max_grad_norm  lambda_1  lambda_2  lambda_3  idle_penalty  invalid_penalty
a2c_flat          1.37e-4         5        0.9558  0.9067      0.0079    0.4166      0.6085         0.5908    5.8464    0.5871    1.2598        4.0603
a2c_pointer       1.47e-5         5        0.9545  0.9745      0.0093    0.3757      0.8205         0.6624    3.8529    0.6613    1.7814        7.8358
```

**Observation:** flat and pointer land within noise of each other on the small
tuning environment (108.25 vs 108.01) -- consistent with Section 8 of
`2026-08-09-pointer-network-action-head.md`'s finding that this project's
fixed-instance setup doesn't exercise the pointer network's actual advantage
(it can't lose to memorization here, but it doesn't clearly win either). Both
studies independently converged on a `lambda_2` (tardiness weight) well above
the old `[0.5, 2.0]` range's upper bound (5.85 and 3.85 respectively) --
consistent with Section 3's prediction that the rescaled, now-O(1) tardiness
term needed more relative weight to matter against the fixed `+3.0` placement
bonus once it was no longer inflated by horizon. `n_steps=5` (the minimum
offered) won in both studies, and both landed on a small `ent_coef` (~0.008-0.009,
near the bottom of `[0.001, 0.1]`) -- notably lower than the hand-tuned
default of `0.01` used throughout `training-log.md`'s prior entries, though in
the same order of magnitude. Full-curriculum reruns using these parameters are
in Section 6 (originally Section titled "Full curriculum reruns" below).

---

## 6. Full-curriculum reruns and fresh EDF comparison

Full 500k-timestep curriculum (`Code/training/train_optimized.py`, using the
Optuna best-params from Section 5), both A2C architectures, then evaluated
deterministically (50 episodes each) against a freshly-measured EDF baseline on
the same stage-4 instance (`horizon=100, num_jobs=100`, `seed=0` -- the same
fixed instance both models trained on repeatedly), all under the rescaled
reward:

```
Training, first200->last200 mean episode reward per stage (episode-index segments,
NOT the CSV's raw "timestep" column -- see note below):

                    stage1(15j/h20)     stage2(30j/h40)     stage3(60j/h60)     stage4(100j/h100)
a2c_flat_optimized  57.75  -> 88.16     129.40 -> 135.10    133.21 -> 132.75    187.53 -> 180.85
a2c_pointer_optimized 52.83 -> 39.27    132.46 -> 129.65    137.85 -> 136.67    212.21 -> 227.66

Deterministic eval, 50 episodes, stage-4 instance (mean; std=0.00 for all rows --
both the eval policy and EDF are deterministic on this one fixed, repeated instance):

                    total_reward   total_tardiness   late_jobs (/100)
EDF (fresh)         289.38         16.00              10.00
a2c_flat_optimized  231.84         866.00             27.00
a2c_pointer_optimized 270.23       1227.00            32.00
```

**Note on the training-stats table above:** `MaskableA2C.train()`'s internal
step counter `t` is local to each call and resets to 0 every time
`train_rl_agent.py`/`train_optimized.py` calls `agent.train()` again for the
next curriculum stage -- so the `timestep` column in the saved
`training_rewards.csv` resets near-zero at every stage boundary rather than
accumulating across the whole curriculum (confirmed: exactly 3 downward jumps
in the column, matching the 3 stage transitions in a 4-stage curriculum). This
doesn't affect training correctness (each stage still runs its own
`stage["timesteps"]` budget correctly) -- it's purely a logging/plotting-axis
issue, discovered while building the table above by using episode-index
segments split at the reset points instead of trusting the `timestep` column
directly. **Not fixed this session** (out of the approved scope, which is env/
reward correctness + Optuna, not the plotting pipeline) -- flagged here so the
saved `training_rewards.png` plots for these runs (which visibly wrap 4 times
on the x-axis) aren't mistaken for a training instability, and as a known
follow-up item for whoever next touches `Code/policies/a2c_policy.py`'s
`train()` or `Code/utils/plotting_utils.py`'s `LiveTrainingPlotter`.

**Observation -- this is the headline result of this session's work.** RL reward
is now *competitive* with EDF at stage 4, a complete reversal from every prior
entry in `training-log.md`: the previous best fixed-instance result
(`flat_fixed_full`, pre-tardiness-rescale) ended stage 4 at `-21.8 -> 50.7`
against an (also pre-fix, not directly comparable) EDF reference of `+275.5`;
here, `a2c_pointer_optimized` reaches **270.23** and `a2c_flat_optimized`
**231.84** against a freshly-measured EDF of **289.38** on the identical
instance and reward formula -- RL is within 7% (pointer) / 20% (flat) of EDF's
reward, not off by a factor of ten-plus in the wrong sign. This strongly
supports the session's opening hypothesis: the RL-vs-EDF gap was
overwhelmingly an optimization/implementation-bug problem (Issues A/C/D +
eval-fairness + the stale, wrongly-scaled Optuna search), not evidence that RL
"fundamentally can't" solve this problem, nor primarily an architecture
question.

**But the reward numbers overstate schedule *quality* parity -- read the
tardiness/late-jobs columns too.** EDF's solution has total tardiness 16.0
across 10 late jobs; the RL policies' solutions have total tardiness 866-1227
across 27-32 late jobs, roughly 50-75x and 3x worse respectively on those two
metrics despite a reward gap of only 7-20%. This is explained by the reward
formula itself: the flat `+3.0` placement bonus and `+50` completion bonus are
large relative to the now-normalised (`T_j/H < 1` per job) tardiness penalty,
so a policy that schedules aggressively (getting placement bonuses, finishing
the job set) can score well on reward while still routinely placing jobs late.
This is not a bug -- it is an honest, now-visible property of the *reward
design* (worth stating plainly for the paper): reward-competitive is not the
same claim as tardiness-competitive, and Section 3's rescale, while
correcting the cross-stage scale-imbalance bug, did not and was not intended
to change the *relative* weighting of placement-vs-tardiness objectives. If
minimising tardiness specifically (rather than total reward) is the paper's
actual target metric, `lambda_2` and/or the flat placement bonus likely need
further, deliberate retuning -- distinct from, and downstream of, everything
fixed in this document.

**Pointer vs. flat:** pointer (270.23 reward, stage-4 training ending
212.21->227.66, i.e. still improving) outperforms flat (231.84 reward,
stage-4 training ending 187.53->180.85, essentially flat/plateaued) on this
run. This is a reversal from the previous pointer-vs-flat comparison in
`2026-08-09-pointer-network-action-head.md` Section 8 (where pointer
underperformed flat on the same fixed-instance setup) -- plausibly explained
by the Optuna retune (Section 5) giving each architecture its own properly-
searched hyperparameters this time, rather than reusing the flat head's
incidentally-tuned defaults for the pointer network as that earlier comparison
did. This is *not* evidence of the pointer network's actual hypothesised
advantage (generalising across job identity) -- Section 7 below still applies
unchanged, since this is still the one memorised fixed instance -- but it is
evidence that, once fairly tuned, the pointer network is at least not
structurally disadvantaged on this task.

**Bug found while reviewing these eval plots with the user:**
`Code/evaluation/eval_rl_agent.py::plot_results()` hardcoded the literal string
`"PPO"` for the evaluated model's legend/bar label in all four saved plots
(`mean_utilisation.png`, `total_tardiness.png`, `late_jobs.png`,
`total_reward.png`), regardless of which algorithm/policy_type was actually
evaluated -- so the `a2c_flat_optimized` eval run's plots were indistinguishable
from a real PPO run by anything except the output directory name, and were
initially misread as PPO results. Fixed by threading a real `model_label`
(`"PPO"` or `"A2C ({policy_type})"`) through from `main()`. **Any plot from
before this fix (including this project's very first eval plots) that appears
to show "PPO" should be treated as unverified until its output-directory name
and the checkpoint/args actually used are checked** -- the label alone is not
trustworthy evidence of which algorithm was run.

**Conclusion / next step:** log this as a `training-log.md` entry (done, see
that file's matching 2026-08-10 entry). The next real test of the pointer
network's design claim remains the randomised-instance experiment in Section 7
below. A secondary, independent next step suggested by the tardiness/late-jobs
gap above: if tardiness minimisation (not just total reward) matters for the
paper's claims, treat that as its own follow-up tuning question rather than
assuming the reward-competitive result above already implies it.

---

## 7. Explicitly deferred

Per the user's explicit prioritisation this session: fix bugs and get a
convincing result on the current fixed-instance curriculum *first*; the
harder generalization question is next, not now.

- **Randomized per-episode/per-stage job sets.** Training currently uses a
  single fixed instance (`seed=0`) replayed every episode of every stage --
  `generate_env_config(seed=0)` is called once per `make_env()` invocation, and
  curriculum stages are literal array-slice prefixes of that same draw
  (`Code/training/train_rl_agent.py`). This is a known, deliberate limitation
  of the *current* phase, not an oversight: Bengio et al.'s survey on
  generalization in neural combinatorial optimization [7] documents that
  heuristics trained on a fixed/narrow instance distribution routinely fail to
  transfer to different instances, and that curriculum learning / distributional
  diversification is a standard, necessary intervention for genuine
  generalization claims -- exactly the randomized-instance experiment already
  identified as the next step in `2026-08-09-pointer-network-action-head.md`
  Section 10 (the pointer network's real advantage -- generalizing from job
  *features* to a never-seen slot index -- is not exercised by a curriculum
  where a flat per-index head can simply memorize the one fixed instance).
- GNN inter-entity message passing, PPO+pointer integration, dynamic/
  energy-aware extensions: unchanged from `Future/README.md`'s existing
  queue.

---

## 8. How to tell if it worked

- All four `tests/test_bugfixes.py` checks pass -- **done** (Section 4).
- Both smoke-test curricula complete without crashing -- **done** (Section 4).
- Full Optuna studies (flat, pointer) complete and report sane (non-collapsed,
  `idle_ratio` not pathological) best trials -- **done** (Section 5): both
  architectures converged to mean_reward ~108 on the tuning env, no idle
  collapse in either study's best trial.
- Full-curriculum reruns, evaluated deterministically against a **freshly
  re-measured** EDF baseline on the same rescaled reward: success is RL
  approaching or beating EDF's fresh number in the later stages, not the stale
  pre-fix `+275.5` figure -- **done, and achieved** (Section 6): stage-4 reward
  231.84 (flat) / 270.23 (pointer) vs. a freshly-measured EDF of 289.38 -- RL is
  within 7-20% of EDF on reward, a complete reversal from every prior
  fixed-instance result in `training-log.md`. **Caveat, not a failure of this
  criterion but a distinct one worth tracking separately:** RL's tardiness
  (866-1227) and late-jobs count (27-32) remain far worse than EDF's (16.0 /
  10) despite the reward parity -- see Section 6's discussion of why the reward
  formula doesn't penalise this as strongly as it might.

---

## References

[1] O. Vinyals, M. Fortunato, N. Jaitly, "Pointer Networks," NeurIPS 2015.

[2] W. Kool, H. van Hoof, M. Welling, "Attention, Learn to Solve Routing Problems!,"
ICLR 2019. arXiv:1803.08475.

[3] W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job Shop Scheduling via Dual
Attention Network Based Reinforcement Learning," arXiv:2305.05119, 2023.

[4] R. Hu, J. Yang, A. Ferber, et al., "Attend2Pack: Bin Packing through Deep
Reinforcement Learning with Attention," arXiv:2107.04333, 2021.

[5] "Graph Neural Networks for Job Shop Scheduling Problems: A Survey,"
arXiv:2406.14096, 2024.

[6] A. Y. Ng, D. Harada, S. Russell, "Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping," ICML 1999.

[7] "On the Generalization of Neural Combinatorial Optimization Heuristics,"
arXiv:2206.00787, 2022. Cited in Section 6 above as the methodological
justification for treating fixed-seed/single-instance training as a documented
limitation of this phase rather than an oversight, and for why the next phase
should diversify the training instance distribution rather than only tune
further on one fixed instance.

[8] Zhang et al., "SPANE: A Symmetry-Preserving Architecture for Multi-NUMA
Environments -- A Deep Reinforcement Learning Approach for Dynamic VM
Scheduling," arXiv:2504.14946, 2025. Closest directly domain-comparable prior
art (DRL for cloud VM scheduling specifically, vs. the generic job-shop/routing
framing of references [1]-[5]) -- cited in `Code/env/scheduling_env.py`'s
module docstring.

**Open TODO (not yet formally cited):** a 2024 paper on attention-based RL for
job-shop scheduling ("Lee et al. 2024") surfaced during this session's
literature search, corroborating the pointer/attention architecture choice
with a source independent of Kool et al. [2] and DANIEL [3]. Not added above
pending a confirmed arXiv ID/venue -- do not cite it in the paper until that's
verified.

**Deferred-phase reading (not yet needed, relevant once Section 6's
randomized-instance work begins):** "Instance-Conditioned Adaptation Models for
Large-scale Generalization of Neural Combinatorial Optimization,"
arXiv:2405.01906, 2024 -- directly relevant to this project's existing
`max_jobs=100` padding scheme (`Code/env/gym_scheduling_wrapper.py`) once
per-episode job-set randomization is implemented.
