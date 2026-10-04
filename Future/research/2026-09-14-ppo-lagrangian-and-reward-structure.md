# Why Reward-Tuned PPO Ignores Deadlines, and PPO-Lagrangian as the Structural Fix

**Date:** 2026-09-14 (S2W9)
**Files:** `Code/env/scheduling_env.py`, `Code/env/gym_scheduling_wrapper.py`,
`Code/policies/ppo_lagrangian.py` (new), `Code/training/train_optimized.py`,
`Code/training/optuna_tune.py`
**Status:** Root cause identified and grounded quantitatively (Sections 1-4).
PPO-Lagrangian implemented, tested at 5 different `lambda_max` values across
a 100k-to-500k-step budget range, and ruled out as a fix -- every value sits
on the same failure spectrum (no effect -> slow degradation -> full collapse
at 50), diagnosed as an architectural moving-value-function-target problem,
not a hyperparameter to tune (Sections 8-10). The lower-risk alternative
(`optimize_for="tardiness"` extended to PPO) was implemented, searched, and
validated at full scale -- also did not improve tardiness at all (Sections
11-12). **Net result across every mechanism tried for PPO tonight: no
improvement over the original ~1310-1327 tardiness.** Current leading
hypothesis, not yet tested: the flat MlpPolicy architecture itself, not the
reward weighting, is the bottleneck (Section 12) -- next session's most
promising lead is a pointer/attention-based PPO policy, not further reward
tuning.

---

## 1. Motivation

The final offline-case PPO run (`Future/research/training-log.md`, 2026-09-14
entry above this doc) produced a checkpoint that completes ~97% of jobs but is
not deadline-aware: tardiness ~1310-1320 across all 3 seeds, landing in the
same band as heuristics with *no deadline signal at all* (SPT, FCFS, Tetris:
1150-1320), nowhere near EDF (37) or LST (24) on the identical instance. The
user asked *why*, not just *that* -- this document is that investigation, plus
the resulting structural fix.

## 2. The mechanism, quantified

`SchedulingEnv.reward()` (`scheduling_env.py:373-374`) computes

```
tardiness_cost = job_weights[j] * (tardiness[j] / horizon)
reward -= lambda2 * tardiness_cost
```

`is_feasible()` guarantees `tardiness[j] <= horizon` for any job actually
scheduled, so `tardiness_cost <= 1.0` always -- **the worst possible tardiness
penalty for any single job is exactly `lambda2 * 1.0`.**

`SchedulingEnv.step()` (`scheduling_env.py:314`) then adds a flat completion
bonus **on top of, not instead of,** that penalty:

```python
reward += 3.0  # "Increased from 1.0 to make scheduling more rewarding than idling"
```

So a job can be scheduled *maximally late* and still be net-positive for
reward whenever `lambda2 < 3.0`: worst case contribution is
`+3.0 - lambda2 >= 0`. There is no tardiness level that makes skipping/delaying
a job better than scheduling it immediately, as long as `lambda2` stays below
that threshold.

**PPO's own reward-tuned `lambda_2` is 1.93** (`ppo_best_params.json`) --
below the threshold. Reconstructing the observed reward from this: 97 jobs
scheduled x 3.0 = 291, minus tardiness cost (`sum(tardiness)/horizon * lambda2`
= 13.18 x 1.93 ~= 25.4), minus machine-activation cost (<=10 x lambda1 ~= 9.4)
~= 256 -- close to the observed ~268 (the gap is idle/hotspot terms, both
small here). The arithmetic is consistent with the hypothesis, not just
qualitatively suggestive of it.

**A2C's own reward-tuned `lambda_2` is 3.85** (`a2c_pointer_best_params.json`)
-- *above* the threshold. A2C+shaping's measured tardiness (28.66, from
`eval_results.csv`) is consistent with this: once `lambda2 > 3.0`, the
structural argument above no longer guarantees a late job is net-positive, so
there's room for a policy to learn to avoid it. This isn't proof A2C's
architecture is superior for this task -- it's evidence that the two
algorithms' independent Optuna searches happened to land on opposite sides of
a hard threshold this reward function has always had, and nobody had
previously computed where that threshold was.

## 3. Why this is a proxy-metric problem, not a search-range problem

Optuna's search range for `lambda_2` is `(0.5, 20.0, log-scale)`
(`optuna_tune.py:168`) -- values well above 3.0 were reachable. PPO's search
didn't find one because **it was optimizing reward, not tardiness**, and
reward-maximization genuinely prefers a low `lambda_2` here: raising it makes
the agent more cautious about tight/risky jobs, which costs `+3.0`/`+50`
completion bonuses on jobs it would otherwise have scheduled -- a worse
trade for pure reward than just accepting the (capped) tardiness penalty and
scheduling everything. Optuna did exactly what it was configured to do; the
configured objective (reward) doesn't equal what's actually wanted (low
tardiness). This is a known, named failure mode -- reward/proxy-metric
divergence, sometimes called reward hacking when the divergence is exploited
this cleanly (Amodei et al., "Concrete Problems in AI Safety," arXiv:1606.06565,
2016, Section 2). Searching harder on the same proxy (e.g. widening the
`lambda_2` range further, or running more Optuna trials) would not fix this --
the search would still be free to pick a low value whenever that maximizes
reward, and nothing forces it not to.

## 4. Why "just raise lambda_2" isn't the structural fix

A cheap fix exists and is worth trying (Section 6): force a specific
`lambda_2 > 3` and retrain, or run `optuna_tune.py --optimize-for tardiness`
for PPO (already implemented for A2C only -- `optimize_for` is A2C-only per
`optuna_tune.py:556`). But this only patches the specific numeric threshold
this specific reward function happens to have today. It doesn't decouple
"how many jobs get scheduled" from "how late are they" -- it just hopes a
single scalar weight, chosen once before training starts, holds the right
balance for the whole run. If the completion bonus, `idle_penalty`, or
`invalid_penalty` ever change, the threshold moves and the same failure mode
can reappear silently.

The principled fix is to stop encoding the tardiness *target* as a reward
weight at all, and instead treat it as a **constraint**: maximize reward
subject to `E[tardiness cost] <= alpha`, with a multiplier that *adapts during
training* to whatever weight is actually needed to hold the constraint --
rather than a human (or Optuna) guessing the right fixed weight in advance.

## 5. RCPO already does this for A2C; PPO-Lagrangian extends it to PPO

This project already implements exactly this idea for A2C: RCPO (Tessler,
Mankowitz, Mannor, ICLR 2019, arXiv:1805.11074 --
`Code/policies/a2c_policy.py`, see
`2026-08-21-rcpo-constrained-tardiness.md` for the full CMDP formulation).
RCPO+PPO integration was explicitly flagged as out of scope in that document's
Section 5. This session's PPO result is a direct, concrete demonstration of
why that gap matters -- extending it is the natural next step, not a new
research direction.

**PPO-Lagrangian** (Ray, Achiam, Amodei, "Benchmarking Safe Exploration in
Deep Reinforcement Learning," arXiv:1910.01708, 2019) is the established name
for this exact combination: a Lagrange multiplier adapted on a slower
timescale than the policy, penalizing a cost signal on top of PPO's own
clipped surrogate objective, with no change to PPO's own update rule required.
Implemented as `Code/policies/ppo_lagrangian.py::PPOLagrangianCallback`, using
the *identical* constraint definition and update rule as RCPO
(`lambda += lr * (mean_cost - alpha)`, projected to `[0, lambda_max]`), so
PPO and A2C results stay directly comparable rather than differing by two
independent constrained-RL formulations.

**A known refinement, not yet implemented:** plain Lagrangian ascent (what
both RCPO and this PPO-Lagrangian implementation use) only reacts to the
current constraint violation, which is known to oscillate/overshoot in
practice. Stooke, Achiam, Abbeel ("Responsive Safety in Reinforcement Learning
by PID Lagrangian Methods," ICML 2020, PMLR v119) replace the plain
integral-only update with a PID controller on the multiplier, and report
better stability. Worth considering for *both* RCPO and PPO-Lagrangian if the
plain update proves unstable in practice here -- not implemented now, since
the plain update is what RCPO already uses and this implementation is meant
to be directly comparable to it.

## 6. Implementation notes and open items

- `Code/env/gym_scheduling_wrapper.py::GymSchedulingEnv.set_lambda2()` (new):
  the runtime hook, mirroring RCPO's direct `sched_env.lambda2 = value`
  assignment but reachable through a VecEnv.
- **Reaching a sub-env's raw `SchedulingEnv` through a VecEnv, post-gymnasium
  auto-forwarding removal**: confirmed empirically this session that SB3's
  `VecEnv.env_method(name, ...)` calls `env.get_wrapper_attr(name)(...)`
  internally (checked `DummyVecEnv.env_method`'s source directly), which
  correctly chain-walks the `Monitor -> ActionMasker -> GymSchedulingEnv`
  wrapper stack via gymnasium's own wrapper-attribute resolution -- distinct
  from the generic `__getattr__` forwarding gymnasium >=1.x removed (the
  reason `a2c_policy.py`'s `_resolve_sched_env()` needed manual `.unwrapped`/
  `.env` traversal for A2C's single-env case). So `env_method("set_lambda2",
  value)` works uniformly for both `DummyVecEnv` and `SubprocVecEnv` without
  needing backend-specific code.
- **Episode-boundary detection**: uses `info.get("episode") is not None`
  (the same signal `LiveTrainingPlotter` already relies on, set by SB3's
  `Monitor` only on an episode's terminal step) rather than a
  `self.locals["dones"]` array key name, which isn't part of SB3's stable
  public callback API.
- **Persistence across curriculum stages**: the callback instance (not a
  fresh one per stage) is passed to every `model.learn()` call in the
  curriculum loop, and `_on_training_start()` re-pushes the *current*
  (possibly already-adapted) `lambda_value` at the start of each stage --
  mirroring `MaskableA2C`'s `self._lambda_value` surviving `self.env`
  reassignment across stages.
- **Smoke test** (not a real training run): 4 parallel envs, 4096 timesteps,
  `lambda_init=1.93` (PPO's own tuned value), `alpha=0`. The multiplier
  climbed from 1.93 to ~3.01 within 4096 steps, tracking a per-episode cost
  consistently above 0 -- crossing the Section 2 threshold in the direction
  predicted, and confirmed live in all 4 sub-envs' `SchedulingEnv.lambda2`
  at the end. This validates the mechanism works; it says nothing yet about
  final converged behavior or actual tardiness at full training scale.
- **Not yet done**: an actual full-scale PPO-Lagrangian training run (this
  project's curriculum + multi-seed protocol) to measure whether it closes
  the tardiness gap to EDF/LST the way A2C+RCPO did relative to A2C+shaping.
  Also not yet done: the cheaper `optimize_for="tardiness"` extension to PPO
  (Section 4) as a comparison point -- worth running both and reporting which
  (or whether both) closes the gap.

## 7. The true floor, now proven (not just estimated): CP-SAT solves the real 100-job instance to proven optimality

`Code/baselines/exact_solver.py --fixed-instance` was run with a 900s budget
and `num_search_workers=15` (CPU was otherwise idle at the time -- see
`solve()`'s new `num_search_workers` parameter, added this session
specifically to make this practical). It did **not** need anywhere near the
budget: CP-SAT returned `status=OPTIMAL` in **21.98 seconds**.

```
[seed 0] CP-SAT: reward=339.09 tardiness=8.00 late=5 (objective=8.0, best_bound=8.0, status=OPTIMAL)
           EDF:    reward=289.38 tardiness=16.00 late=10
           LST:    reward=292.92 tardiness=8.00 late=7
```

**The true minimum possible total tardiness for the deployed fixed instance,
scheduling every job, is exactly 8.0** (5 late jobs) -- not an estimate or a
heuristic-relative comparison, a proven bound (`objective == best_bound`).
This is decisive, not just informative: the fixed-instance PPO run's
tardiness (~1310-1320) and A2C+shaping's (28.66) are now both measured
against a real floor, not just each other. LST already gets to tardiness=8
(matching the proven optimum's total, with a different job-by-job split) --
confirming Section 2's earlier account that LST is already close to optimal
for this instance's deadline structure, using only a simple greedy rule.

**This also rules out a possible alternative explanation for PPO's result.**
One might guess PPO's low tardiness competitors (LST/EDF) achieve it by
leaving more jobs unscheduled -- trading completion for punctuality. CP-SAT
disproves that tradeoff exists here: the proven-optimal solution schedules
**all 100 jobs** while holding tardiness to 8. There was never a fundamental
tension between completion rate and deadline-awareness for this instance;
PPO's ~97/100 scheduled + ~1315 tardiness reflects the policy not being
deadline-aware at all, not a forced trade-off it navigated poorly.

With this proven floor in hand, PPO-Lagrangian's success criterion is now
concrete rather than relative: not just "closer to EDF/LST than before," but
how close it gets to 8.0.

## 8. Real result: PPO-Lagrangian collapsed to zero jobs scheduled (all 3 seeds)

The full run (3 seeds, same protocol as the fixed-instance set,
`alpha=0, lambda_init=1.9315, lambda_lr=0.01, lambda_max=50, update_every=5`)
finished 2026-09-15. Evaluated identically to every other checkpoint
(`--randomized-eval`, 50 held-out instances):

```
seed 1: reward=-50.50  tardiness=0.00  late=0/100  scheduled=0/100  (std=0 across all 50 runs)
seed 2: reward=-50.50  tardiness=0.00  late=0/100  scheduled=0/100  (std=0 across all 50 runs)
seed 3: reward=-50.50  tardiness=0.00  late=0/100  scheduled=0/100  (std=0 across all 50 runs)
```

**Zero variance across 50 held-out instances and all 3 seeds -- this is a
fully deterministic collapse to "never schedule anything," not noisy or
partial degradation.** Tardiness=0 is real but worthless: it's trivially
achieved by never placing a job at all (tardiness is only ever computed for
jobs that get scheduled -- see Section 2 -- so a policy that schedules
nothing cannot be late). This is the exact idle-collapse failure mode the
`+3.0` completion bonus and `idle_penalty` were originally added to this
environment to prevent (see the code comments at `scheduling_env.py:312-314`
predating this session).

**Diagnosed mechanism, from `ppo_lambda_history_*.json`:** the multiplier hit
its ceiling (`lambda_max=50`) at timestep ~345,000 -- only 18% into the
1.9M-step run -- and stayed pinned there for the remaining 82%, while the
mean sampled cost *rose* over that time (0.96 -> 15.88 -> 18.15 -> 25.86)
rather than falling toward `alpha=0`. This is a runaway feedback loop, not a
slow convergence that needed more time:
1. Cost exceeds alpha -> lambda climbs (correct, intended behavior).
2. At `lambda2=50`, a single maximally-late job costs up to `-50` against a
   flat `+3.0` completion bonus -- scheduling anything became a large
   net-negative-expected-value bet for a policy that hadn't yet learned
   *which* jobs are safe to schedule without lateness.
3. The policy responds by scheduling fewer jobs -- but `SchedulingEnv`'s
   `_finalize_unscheduled_job_cost()` (line ~207) *also* charges a
   worst-case cost for jobs left unscheduled at episode end, so abandoning
   jobs does not actually reduce the measured cost the way the policy's
   local gradient seems to have predicted it would.
4. Cost stays high (now from abandonment rather than lateness) -> lambda
   stays pinned at its ceiling -> step 2 gets worse, not better.
5. Once collapsed, an all-idle trajectory carries almost no gradient signal
   about scheduling at all (nothing left to differentiate through), so
   there's no mechanism to climb back out. All 3 independent seeds fell into
   this same trap, which rules out random bad luck as the explanation.

**This is exactly the instability Section 5 flagged as a known risk of plain
Lagrangian ascent** (Stooke, Achiam & Abbeel, 2020) *before* this run was
launched -- now with concrete empirical confirmation in this project, not
just a cited risk. Two contributing design choices, both worth revisiting
before trying again:

- **`lambda_max=50` was too permissive.** It was copied directly from A2C's
  RCPO default without re-deriving it for PPO's specific reward scale (the
  `+3.0` completion bonus and Section 2's threshold analysis). A tighter
  cap -- e.g. `lambda_max` on the order of 5-10, keeping the worst-case
  per-job tardiness penalty closer to (not wildly beyond) the completion
  bonus -- would likely prevent the "scheduling is a huge bet" regime this
  run fell into, at the cost of a possibly-looser final constraint
  satisfaction.
- **This implementation (matching RCPO's own mechanism) modifies the
  environment's raw reward in place** (`sched_env.lambda2 = value`) rather
  than the two-critic architecture the original PPO-Lagrangian paper (Ray,
  Achiam & Amodei, 2019) actually uses -- a separate cost value function and
  cost advantage estimate, combined with the reward advantage as
  `A = A_reward - lambda * A_cost`, with the *raw* reward never changing.
  Folding lambda directly into the reward (as done here, faithfully
  replicating RCPO for direct comparability -- see Section 5) means PPO's
  single value function has to keep re-learning a moving target every time
  lambda updates, which is a plausible additional contributor to the
  instability observed. Worth trying the two-critic formulation as a
  follow-up if a tighter `lambda_max` alone doesn't fix it.

**Conclusion:** PPO-Lagrangian as implemented did not close the gap to the
proven floor (8.0) -- it produced a worse outcome than doing nothing useful
at all. This is a real, informative negative result: it confirms the
instability risk in the literature is not merely theoretical for this
problem, and narrows the next attempt to two concrete, testable changes
(lower `lambda_max`; possibly the two-critic architecture) rather than a
vague "try harder."

## 9. Retesting with lower lambda_max: no collapse, but no improvement either -- and it's not simply "lambda too low"

Per the user's explicit request to verify with small excerpts before
committing another multi-hour run: ran fast, faithful smoke tests (real
`train_optimized.py` pipeline, full 4-stage curriculum, but
`--stage4-timesteps 100000` instead of 1,600,000 -- ~27 min each instead of
~3h) at three candidate `lambda_max` values.

**Quantitative motivation for the candidates**: at `lambda_max=8`, the
worst-case per-job penalty (`lambda2 * tardiness_cost`, `tardiness_cost<=1`)
is 8, still less than the actual failure's `lambda_max=50` but also not
obviously enough to change *typical* behavior -- the failed full run's
observed tardiness (1312-1327) implies mean tardiness-per-late-job of
~30 timesteps, i.e. `tardiness_cost ~= 0.31`. For that *typical* case (not
just the worst case) to flip from profitable to unprofitable against the
`+3.0` completion bonus requires `lambda > 3.0/0.31 ~= 9.7`. So `lambda_max=8`
should be expected to barely matter; `12` and `18` should clear this
threshold with margin.

**Results** (50-held-out-instance protocol):
```
lambda_max=8:  reward=264.75  tardiness=1312.62 (P95=57.89)  late=42.40/100  scheduled=96.90/100
lambda_max=12: reward=264.65  tardiness=1291.16 (P95=58.23)  late=42.42/100  scheduled=96.82/100
lambda_max=18: reward=264.76  tardiness=1291.64 (P95=58.23)  late=42.50/100  scheduled=96.84/100
```

**No collapse at any of the three values** (unlike `lambda_max=50`) -- good,
confirms the collapse is specific to very large multipliers, not an
inherent property of this callback. **But also no meaningful improvement at
any of the three** -- 8, 12, and 18 are statistically indistinguishable from
each other and from the ORIGINAL uncontrolled PPO baseline (~1315-1327). The
quantitative "typical tardiness needs lambda>9.7" prediction from above was
*wrong in practice*: even `lambda_max=18` (nearly 2x that threshold) shows no
detectable effect.

**Checked the lambda trajectory directly to rule out "not enough time at the
elevated value"**: in both the 12 and 18 runs, the multiplier reached 90% of
its cap within ~105 timesteps of entering stage 4 -- i.e. it sat at
(essentially) its ceiling for effectively the *entire* 100,000-step stage-4
budget, not just a brief final window. Yet the sampled mean cost plateaued
at ~15-16 in both cases regardless of the cap being 12 or 18. **The
multiplier reaching and holding a theoretically-sufficient value is not
translating into the policy actually changing its scheduling behavior**,
within this training budget.

Two remaining, not-yet-distinguished explanations:
1. **Training-time bottleneck**: 100,000 steps may simply be too few for a
   flat-MLP policy that already converged (from stages 1-3) toward
   "schedule everything ASAP" to unlearn that habit and find a
   meaningfully different one, regardless of how strongly the new reward
   landscape discourages it. The original, uncontrolled full run used
   1,600,000 stage-4 steps (16x more) to reach its own (bad) tardiness
   number -- it's plausible a comparably long budget is needed here too
   before any lambda value shows its effect.
2. **The "moving target" value-function problem flagged in Section 8**:
   mutating the environment's raw reward in place means PPO's single value
   function has to keep re-estimating a shifting regression target,
   independent of how long training runs -- if this is the dominant
   bottleneck, more time at a fixed lambda_max won't help either, and the
   two-critic architecture (separate reward/cost value functions and
   advantages, per Ray/Achiam/Amodei 2019) is the real fix needed.

**Diagnostic result: training-time bottleneck (explanation 1) is ruled out.**
Ran 500,000 stage-4 steps at a fixed `lambda_max=15` (seed 96) and bucketed
the resulting `episode_cost` trace into 10 equal chunks across stage 4:

```
t=300k-349k: avg_cost=14.38   (lambda held at exactly 15.0 throughout)
t=349k-399k: avg_cost=15.20
t=399k-449k: avg_cost=15.63
t=449k-499k: avg_cost=16.07
t=499k-549k: avg_cost=16.37
t=549k-599k: avg_cost=16.54
t=599k-649k: avg_cost=16.77
t=649k-699k: avg_cost=16.89
t=699k-749k: avg_cost=17.12
t=749k-801k: avg_cost=17.27
```

Cost does not fall with more training time -- it **rises monotonically**,
smoothly, across the entire 500k-step window, at a constant `lambda=15`
(never re-adjusting, since it's pinned at its own ceiling the whole time so
there's no confound from the multiplier itself changing). Five times the
training budget of the earlier 100k-step smoke tests produced a *worse*
trend, not a converging one. This is not noise around a plateau; it is
smooth, steady degradation.

**This rules out "needs more time" and points squarely at explanation (2):
mutating the environment's raw reward in place (the mechanism this
implementation shares with A2C's RCPO) is itself destabilizing for PPO**,
independent of which `lambda_max` is chosen -- every value tried (8, 12, 15,
18, 50) sits somewhere on the same underlying failure spectrum (no effect ->
slow degradation -> full collapse), not on a spectrum where some value in
the middle works well. The likely mechanism: PPO's single value function is
permanently chasing a shifting regression target every time the reward
function's own `lambda2` term changes (or, once pinned, is simply
miscalibrated for the current reward scale for an extended period), which
corrupts the advantage estimates the policy gradient relies on -- consistent
with, and now with concrete supporting evidence for, the two-critic-
architecture concern raised in Section 8.

## 10. Decision: pause lambda_max tuning, defer the two-critic rewrite

**Conclusion:** No `lambda_max` value fixes this implementation -- the
problem is architectural (reward-mutation-in-place), not a hyperparameter to
search over. The correct next step is the two-critic PPO-Lagrangian
architecture actually described in Ray, Achiam & Amodei (2019): a *separate*
cost value function and cost advantage estimate, combined with the (never-
mutated) reward advantage as `A = A_reward - lambda*A_cost`, rather than
changing the environment's reward signal at all.

**Deliberately not attempted autonomously tonight.** This is a materially
larger, more novel implementation than the callback-based approach tried so
far (a second value head, a second GAE computation over the cost signal,
and a modified PPO loss combining two advantage streams -- sb3_contrib's
`MaskablePPO` has no extension point for this, so it would likely need a
hand-rolled training loop in the style of `MaskableA2C` rather than a
callback wrapped around the existing model). Given the amount of subtle,
hard-to-verify design surface a first implementation like this has -- and
given this session already produced two increasingly severe failure modes
from the *simpler* approach before diagnosing the real cause -- this is
flagged as a well-scoped, ready-to-implement recommendation for the next
session (with a human in the loop on the design) rather than something to
build and trust unsupervised overnight. Continuing instead with the other,
lower-risk, already-identified follow-ups below, which reuse existing,
already-validated infrastructure rather than a new RL architecture.

## 11. Cheaper alternative pursued instead: `optimize_for="tardiness"` extended to PPO

Rather than attempt the two-critic rewrite unsupervised, implemented the
lower-risk alternative flagged in Section 4: `objective_ppo` in
`Code/training/optuna_tune.py` now supports `optimize_for` ("reward"
(default, unchanged), "tardiness", "pareto") exactly like `objective_a2c`
already did -- same `TARDINESS_PENALTY_WEIGHT`-scalarized composite score
(`mean_reward - 50.0 * mean_tardiness_normalised`), same pareto/NSGA-II
option, same result-file naming convention
(`ppo_tardiness_best_params.json`, already loadable via
`train_optimized.py --params-tag tardiness` -- that flag pre-dates this
session and needed no changes). `run_optimization()`'s suffix/dispatch logic,
previously gated to `algorithm == "a2c"` for all of this, was extended to
apply to PPO too.

Smoke-tested with 2 trials before committing to a real search: worked
end-to-end, and even at 2 trials the found `lambda_2` (17.88) was already far
above both PPO's reward-tuned value (1.93) and every `lambda_max` tried in
the Lagrangian sweep (8-18) -- a promising early signal that a genuine
tardiness-directed *search* (not a hand-picked constant, and not a
runtime-adapting multiplier fighting a moving value-function target) may
reach a working weight where the other approaches didn't.

**Full 50-trial search result -- more interesting than the smoke test
suggested, and needs a validation step before trusting it.** The actual best
trial (42, composite_score=107.66) converged on `lambda_2=1.927` -- almost
identical to the original reward-tuned value (1.9315), NOT a large value.
What differs is everything else: a bigger network (`layer_size=512` vs 256),
`activation=relu` (vs tanh), `learning_rate=0.000196` (vs 1.3e-5, ~15x
larger), `n_steps=1024, batch_size=512, n_epochs=10` (vs 512/64/4),
`ent_coef=0.029` (vs 0.0073, ~4x larger). And the top ~10 trials by score
(42, 45, 48, 40, 49, 51, 25, 44, 41, 50) all report **`mean_tardiness=0.0`
and `mean_late_jobs=0.0` exactly** -- perfect on-time completion at the
tuning scale.

**Critical caveat, already flagged elsewhere in this project's own history
and directly relevant here**: this search tunes at `make_tuning_env`'s
small default scale (20 jobs, 5 machines, horizon 30), not the deployed
100-job/10-machine/horizon-100 instance -- the same small-scale-tuning
setup Eimer, Lindauer & Raileanu (2023, cited in `objective_a2c`'s own
docstring) warn can silently fail to transfer, and which this project's own
S2W5 tardiness-retuning history already experienced directly. Finding
perfect tardiness at 20-job scale is encouraging but proves nothing about
100-job behavior on its own -- it needs to be validated with an actual
full-scale training run using these hyperparameters
(`train_optimized.py --params-tag tardiness`, already wired to load
`ppo_tardiness_best_params.json` with no changes needed) and the project's
normal 50-held-out-instance evaluation, not trusted from the tuning search's
own small-scale numbers.

## 12. Full-scale validation result: the caveat was correct -- it did not transfer

Ran the actual validation (3 seeds, same protocol as every other PPO run
this session, `train_optimized.py --params-tag tardiness`):

```
seed 1: reward=266.19  tardiness=1315.06 (P95=58.98)  late=42.32/100  scheduled=97.06/100
seed 2: reward=264.71  tardiness=1289.04 (P95=57.35)  late=42.46/100  scheduled=96.84/100
seed 3: reward=264.72  tardiness=1304.88 (P95=59.50)  late=41.76/100  scheduled=96.90/100
```

**No improvement at all.** These numbers are statistically indistinguishable
from the original reward-tuned PPO (~1310-1327), the randomized-instance PPO
(~1327), and every PPO-Lagrangian `lambda_max` value tried (8-18:
~1291-1312). Perfect tardiness at the 20-job tuning scale (Section 11)
completely failed to transfer to the deployed 100-job scale -- exactly the
failure mode Eimer et al. (2023) warn about, now demonstrated directly for
PPO (this project had previously only seen it for A2C, S2W5).

**Cumulative picture after tonight's full investigation arc** -- every
mechanism tried for PPO converges to essentially the same ~1290-1330
tardiness band, regardless of how differently each one tries to fix it:

```
Reward-tuned PPO (fixed instance):        ~1310-1320
Reward-tuned PPO (randomized instance):   ~1327
PPO-Lagrangian, lambda_max=8/12/18:       ~1291-1312
PPO-Lagrangian, lambda_max=50:            collapse (0 scheduled) -- worse, not on this band
Tardiness-tuned PPO (full-scale):         ~1289-1315
```

Five different mechanisms -- a fixed weight, a runtime-adapting Lagrangian
multiplier at four different ceilings, and a from-scratch hyperparameter
search directly targeting tardiness -- all land in the same narrow band.
This consistency itself is informative: it's difficult to explain by "the
weight was wrong" (five different weight-setting mechanisms, same result)
and easier to explain by something upstream of the reward weighting
entirely -- most plausibly the **flat MlpPolicy architecture itself**, which
has no explicit mechanism for reasoning about job-to-job relationships or
deadline ordering the way a pointer/attention-based architecture does. A2C's
much better tardiness (28.16-28.66, Section 5) used exactly such an
architecture (`PointerActorCritic`, shared job/machine encoders with
pairwise compatibility scoring -- see `Code/policies/pointer_policy.py`) --
this was previously attributed partly to A2C's `lambda_2` happening to clear
the Section 2 threshold, but tonight's result suggests architecture may be
the dominant factor, not the reward weight. **Not yet tested**: PPO with a
pointer-style architecture (`sb3_contrib.MaskablePPO` supports custom
feature extractors via `policy_kwargs`, so this is plausible without a
hand-rolled training loop, unlike the two-critic Lagrangian rewrite) -- the
natural next experiment, and one that could be pursued independently of
whether the two-critic PPO-Lagrangian architecture (Section 10) is ever
built.

This is a fundamentally different mechanism from the Lagrangian approach in
Sections 1-10: no runtime adaptation during training, no moving reward
target for the value function to chase -- just a fixed (but *searched-for*,
not guessed) `lambda_2`, found by training many independent short trials to
convergence and comparing their *final* tardiness-aware scores. It doesn't
have the instability failure mode diagnosed above, at the cost of not
adapting to a specific instance/distribution at runtime the way a true
constrained-RL method could.

## References

1. Tessler, C., Mankowitz, D. J., & Mannor, S. (2019). *Reward Constrained
   Policy Optimization*. ICLR 2019. arXiv:1805.11074.
2. Ray, A., Achiam, J., & Amodei, D. (2019). *Benchmarking Safe Exploration in
   Deep Reinforcement Learning*. arXiv:1910.01708.
3. Stooke, A., Achiam, J., & Abbeel, P. (2020). *Responsive Safety in
   Reinforcement Learning by PID Lagrangian Methods*. ICML 2020, PMLR v119.
4. Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., &
   Mané, D. (2016). *Concrete Problems in AI Safety*. arXiv:1606.06565.
5. Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems
   Viewpoint*. (Two-timescale stochastic approximation, grounding the
   `update_every_episodes` timescale separation -- carried over from RCPO's
   own citation of this, see `2026-08-21-rcpo-constrained-tardiness.md`.)
