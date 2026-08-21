# RCPO-style constrained optimization for the tardiness penalty

**Date:** 2026-08-21 (S2W5)

## 1. Motivation

Every prior attempt to control tardiness through the fixed linear weight
`lambda_2` in `SchedulingEnv.reward()` has either under- or over-corrected:

- Optuna's reward-tuned search picked `lambda_2 = 3.85` (pointer) --
  `9.00` total tardiness on the fixed instance, beating EDF (Phase 8,
  2026-08-19 training-log entry). Good result, but `lambda_2` was tuned to
  maximize *reward*, with tardiness only an indirect beneficiary.
- Optuna's tardiness-focused search (Phase 6/Experiment 3, 2026-08-17) tuned
  `lambda_2` directly against a composite reward-tardiness score at a small
  (20-job) tuning scale, and picked a *lower* `lambda_2` than the
  reward-tuned run for both architectures -- yet tardiness at full (100-job)
  deployment scale got worse (1699 pointer / 998 flat), not better. See
  `Future/research/2026-08-17-literature-review-improving-rl-agent.md` for the
  HPO tuning/deployment-mismatch explanation (Eimer, Lindauer, Raileanu 2023).

Both results point at the same underlying problem: `lambda_2` is a fixed
*scalarization weight* picked once, before training, by a process (grid/TPE
search over a hand-picked range) that has no principled connection to "how
much tardiness is actually acceptable." Roijers et al. (2013)'s critique of
linear scalarization for multi-objective sequential decision problems
applies directly here -- a single fixed weight can only reach one point on
the reward-tardiness Pareto front, and there is no a priori way to know
which weight reaches the point we actually want (e.g. "tardiness as low as
possible without destroying schedule reward").

This experiment replaces the fixed weight with a **Lagrange multiplier that
adapts automatically during training** to enforce an explicit tardiness
*constraint*, instead of a tardiness *preference*. This is the approach
proposed in the literature review's Section 7 (priority 3, agreed by the
user after the potential-shaping ablation and randomized-instance
generalization experiments were run first).

## 2. Formalization: tardiness as a CMDP constraint

Following Tessler, Mankowitz, and Mannor, *"Reward Constrained Policy
Optimization"* (ICLR 2019, arXiv:1805.11074) [1], we reformulate training as
a **Constrained MDP** (CMDP):

```
maximize_theta   E_tau~pi_theta [ sum_t r_t ]
subject to       E_tau~pi_theta [ C(tau) ] <= alpha
```

where `r_t` is every reward term *except* the tardiness penalty (activation
penalty, hotspot penalty, idle penalty, the placement/completion shaping
bonuses, and the potential-based shaping term when enabled -- all still
fixed, hand-tuned constants exactly as before), and `C(tau)` is the
per-episode tardiness cost:

```
C(tau) = sum_{j scheduled in tau} w_j * (T_j / H)
```

This is *exactly* the quantity `SchedulingEnv.reward()` already computes as
`self.job_weights[j] * (self.tardiness[j] / self.horizon)` -- the only
change is that it is no longer multiplied by a fixed `lambda_2` inside the
per-step reward; instead it is tracked separately as the constraint cost
signal, and `lambda_2` becomes the (adapted) Lagrange multiplier for the
constraint above.

**Constraint threshold alpha = 0.** We set the target to zero weighted
normalised tardiness. This is not a claim that zero tardiness is achievable
-- it is the standard RCPO choice for "drive this cost down as far as
possible" constraints (Tessler et al. use `alpha` this way for their
"avoid unsafe states entirely" guided-navigation experiment, Section 5.2 of
[1]). Under simultaneous gradient dynamics (not exact best-response), the
multiplier does not need to force `C(tau)` to literally reach `alpha` for
the joint (theta, lambda) system to reach a stable point -- it converges to
a saddle point of the Lagrangian where the policy gradient's marginal
reward gain from *more* tardiness exactly balances the multiplier's
marginal cost, at whatever cost level that balance occurs (Tessler et al.,
Theorem 1, via Borkar's two-timescale stochastic approximation theory
[2]). This is precisely the property we want: instead of us guessing a
fixed weight ahead of time, the multiplier finds its own equilibrium
weight during training.

## 3. Lagrangian and the two-timescale update rule

The Lagrangian of the CMDP above (with `alpha = 0`) is:

```
L(theta, lambda) = E_tau [ sum_t r_t ] - lambda * E_tau [ C(tau) ]
                  = E_tau [ sum_t (r_t - lambda * c_t) ]
```

where `c_t = w_j * (T_j / H)` is the per-step cost increment (nonzero only
on the step a job is actually placed, matching the existing per-step
tardiness term's structure -- see `SchedulingEnv.reward()`).

Algorithm 1 of [1] alternates two updates on different timescales:

- **Fast (policy) update, every rollout:** ordinary A2C gradient ascent on
  `theta` using the *shaped* reward `r'_t = r_t - lambda * c_t` as the
  environment reward -- i.e. exactly the reward `SchedulingEnv.reward()`
  already returns, just with `lambda_2` read as the live multiplier value
  instead of a fixed constant.
- **Slow (multiplier) update, every K episodes:** projected gradient
  *ascent* on `lambda`:

  ```
  lambda <- Proj_[0, lambda_max] ( lambda + eta_lambda * (C_hat - alpha) )
  ```

  where `C_hat` is the sample-mean episode cost over the last `K`
  completed episodes. Ascent (not descent) is correct here: `lambda` is
  the dual variable of an inequality constraint, so it must *increase*
  when the constraint is violated (`C_hat > alpha`, i.e. tardiness too
  high, so the tardiness penalty needs to bite harder) and decay back
  toward its lower projection bound of 0 when the constraint is satisfied.

**Why this must be slower than the policy update, not just a different
formula.** The two-timescale requirement is not a tuning nicety -- it is
the condition under which Borkar's stochastic approximation theory [2]
guarantees the coupled `(theta, lambda)` iteration converges to a local
saddle point of `L` at all (Tessler et al.'s Theorem 1 cites this
directly). If `lambda` moved as fast as `theta`, the two would chase each
other's most recent gradient with no stable point to converge to
(non-stationary reward chasing a non-stationary penalty). We enforce the
separation two ways simultaneously, matching the two mechanisms Tessler et
al. describe for satisfying it in practice:

1. **Update frequency:** `lambda` is updated once every `K = 5` completed
   episodes (~500 environment steps at 100 jobs), vs. the policy's
   `n_steps = 5`-step rollout update -- two orders of magnitude less
   frequent.
2. **Step size:** `eta_lambda = 0.01`, a small constant step independent of
   the policy's Adam learning rate (`~1.47e-5` to `7e-4` across this
   project's tuned configs) -- these are different units (a scalar penalty
   coefficient vs. neural network weights) so they are not directly
   comparable, but the small, constant, un-adapted step size (no Adam
   momentum/preconditioning on `lambda`) keeps its movement deliberately
   sluggish relative to the network's per-batch updates.

## 4. Hyperparameter choices and their grounding

| Parameter | Value | Grounding |
|---|---|---|
| `alpha` (constraint threshold) | `0.0` | Matches Tessler et al. [1] Section 5.2's "avoid entirely" constraint pattern; see Section 2 above for why this doesn't require literal zero tardiness to be achieved. |
| `lambda_init` (`lambda(0)`) | `3.8529` (pointer) | Warm start at the value Optuna's *reward*-tuned search already found productive (`rl_training/optuna_results/a2c_pointer_best_params.json`), rather than an arbitrary cold start -- gives the multiplier a sensible starting point instead of spending early training re-discovering it. |
| `lambda_max` (projection bound) | `50.0` | Reuses this project's existing `TARDINESS_PENALTY_WEIGHT = 50.0` calibration anchor (`Code/training/optuna_tune.py`), itself justified as matching the `+50` job-completion bonus magnitude -- so the multiplier cannot grow to dominate the completion bonus by more than a fixed, previously-chosen ratio. Tessler et al. [1] note the theory only requires `lambda >= 0` (no upper bound), but bound it in practice to prevent divergence; we follow that practice with a value already load-bearing elsewhere in this codebase rather than picking a new unjustified constant. |
| `eta_lambda` (multiplier step size) | `0.01` | Small relative to `C(tau)`'s typical O(1)-to-O(10) scale (tardiness terms are individually bounded in `[0, w_j)` per the horizon-normalisation argument in `2026-08-09-fixed-instance-bugfix-and-reward-rescale.md`), so a single update moves `lambda` by a small fraction of its own value -- consistent with the "slow" timescale requirement in Section 3. |
| `K` (episodes between multiplier updates) | `5` | Averaging over 5 episodes reduces the variance of the `C_hat` estimate (tardiness is highly variable episode-to-episode, per this session's held-out-eval std of ~57 on a mean of ~28 -- see the 2026-08-20 training-log entry) before it moves the multiplier, while still being far less frequent than the 5-step policy rollout update. |

All other hyperparameters (`lambda_1`, `lambda_3`, `idle_penalty`,
`invalid_penalty`, `learning_rate`, `gamma`, `gae_lambda`, `ent_coef`,
`value_coef`, `max_grad_norm`, `embed_dim`/`hidden`) are held at Phase 8's
already-tuned reward-optimal values (`a2c_{policy_type}_best_params.json`),
unchanged. This isolates the *one* variable under test -- fixed vs.
adaptive tardiness weighting -- against the same baseline every other
experiment this session was compared to, per this project's ablation
discipline (`CLAUDE.md`).

## 5. Implementation

Structural changes (this branch, `rcpo-constrained-optimization`):

- `SchedulingEnv`: added `self.episode_cost` (reset in `reset()`,
  accumulated inside `reward()` as the *pure* `w_j * (T_j/H)` term,
  independent of `lambda_2`) -- this is the constraint cost signal
  `C(tau)` read back by the trainer. `lambda_2` itself is left as a plain
  mutable attribute (no code path needed to distinguish "fixed" vs.
  "adaptive" mode inside the environment -- from the environment's
  perspective it is always just "the current weight applied to the
  tardiness term," exactly as before).
- `GymSchedulingEnv.step()`/`reset()`: expose `info["episode_cost"]` so the
  trainer can read the running/final cost without reaching past two wrapper
  layers by hand.
- `MaskableA2C` (`Code/policies/a2c_policy.py`): new opt-in constructor
  parameters `use_rcpo`, `rcpo_alpha`, `rcpo_lambda_init`,
  `rcpo_lambda_lr`, `rcpo_lambda_max`, `rcpo_update_every_episodes`. When
  enabled, seeds `env`'s `lambda_2` to `rcpo_lambda_init`, and `train()`
  performs the slow-timescale projected-ascent update on `lambda_2` at
  every `K`-th completed episode, logging `(timestep, lambda, mean_cost)`
  to `self.lambda_history` for post-hoc inspection/plotting.
- `train_optimized.py`: `--use-rcpo` (+ `--rcpo-alpha`,
  `--rcpo-lambda-lr`, `--rcpo-lambda-max`, `--rcpo-update-every`) CLI flags,
  A2C only (PPO+RCPO integration is out of scope here, consistent with
  PPO+pointer already being deferred project-wide). When `--use-rcpo` is
  set, the loaded `lambda_2` from the best-params JSON is used only as
  `rcpo_lambda_init` (the starting point), not as a fixed weight for the
  whole run.

Scope explicitly excluded from this pass (flagged as follow-ups, not
silently dropped): re-tuning the *other* hyperparameters jointly with RCPO
enabled (a fresh Optuna search under RCPO would need its own objective,
since `lambda_2` is no longer a free parameter to search over); PPO
integration; multiple constraints (e.g. a separate hotspot constraint) --
Tessler et al.'s formulation generalizes to a vector of multipliers, but
this project has only ever tuned tardiness as the one clearly proxy-prone
term, so a single constraint is the right scope for a first test.

## 6. Results

Pointer architecture, full curriculum, `alpha=0.0`, `lambda_max=50.0` (see
`Future/research/training-log.md`'s 2026-08-21 entry for the complete
stats table):

|  | reward (held-out) | tardiness (held-out) | late-jobs (held-out) |
|---|---|---|---|
| EDF | 284.95 ± 3.65 | 37.30 ± 60.43 | 12.22 ± 14.58 |
| Phase 8 (fixed lambda_2=3.85) | 254.75 ± 9.14 | 28.66 ± 57.56 | 9.56 ± 13.45 |
| RCPO (adaptive lambda) | 135.67 ± 15.10 | **19.84 ± 17.99** | **3.20 ± 2.12** |

RCPO reaches the best tardiness and late-jobs result in the project on both
axes at once, with much lower variance than every prior result -- direct
evidence that letting the penalty weight adapt can find a better point on
the reward-tardiness trade-off than any fixed weight this project has
searched over. But `lambda` saturated at its `lambda_max=50.0` projection
ceiling (from `lambda(0)=3.85`) rather than converging to an interior
value, and reward roughly halved as a result. This is the expected
behaviour of the CMDP formulation under a target (`alpha=0.0`) the
environment cannot actually satisfy in expectation -- `E[C(tau)] > alpha`
never stops holding, so projected ascent has nothing to stop it short of
the ceiling we imposed. `lambda_max=50.0` was reused from an existing
project anchor (Section 4), not derived for this specific run, so the
ceiling itself -- not the RCPO mechanism -- is the most likely reason
reward was sacrificed more than necessary.

**Flagged follow-up (not yet run):** repeat with a less strict `alpha`
grounded at an achievable target (e.g. Phase 8's own held-out tardiness of
~28.66, or a fraction of EDF's ~37.30) instead of 0.0, so the multiplier
can reach an interior equilibrium rather than saturating at the ceiling.
This should recover reward while keeping most of the tardiness/late-jobs
gain.

**Flat architecture, same RCPO config (`lambda_init=5.85`, `alpha=0.0`,
`lambda_max=50.0`):**

|  | reward (fixed) | tardiness (fixed) | late-jobs (fixed) | tardiness (held-out) | late-jobs (held-out) |
|---|---|---|---|---|---|
| EDF | 289.38 | 16.00 | 10.00 | 37.30 ± 60.43 | 12.22 ± 14.58 |
| Phase 8 flat (fixed lambda_2) | 262.98 | 1252.00 | 26.00 | -- | -- |
| RCPO flat (adaptive lambda) | 198.50 | **0.00** | **0.00** | **570.62 ± 89.14** | **24.02 ± 3.34** |

The flat architecture's fixed-instance result looks perfect (zero
tardiness) but is the worst held-out result recorded in this project --
worse than EDF, worse than every prior flat-architecture number. Unlike
the pointer run, `lambda` here did *not* saturate at `lambda_max` (final
value 20.59, still rising slowly but clearly converging, not stuck at the
ceiling) -- so this is not the same "target unreachable, multiplier maxed
out" failure mode as the pointer run. It is a straightforward severe
overfitting case: `MaskableActorCritic` parameterizes one weight row per
`(job-slot, machine)` action index, giving it a direct route to
memorizing this one instance's optimal schedule rather than learning
transferable job-feature-based placement rules (the pointer architecture's
shared job/machine encoders structurally cannot do this the same way).
RCPO's adaptive, harder-driving penalty appears to have let the flat
network fully exploit that route. This is consistent with -- and sharpens
-- the architecture-level generalization gap already found in the
2026-08-20 randomized-instance generalization experiment.

**Conclusion:** RCPO's benefit is real for the pointer architecture but
does not transfer to flat. Generalization quality in this project appears
to be primarily determined by architecture (shared-encoder pointer vs.
per-index-weight flat), not by the reward/constraint formulation -- no
technique tried so far (shaping, tardiness-focused tuning, RCPO) has made
the flat architecture generalize. Pointer + potential-based shaping (with,
pending the alpha follow-up above, RCPO) remains the only configuration
recommended for further work; flat is retained only as the fixed-instance
A/B baseline.

## References

[1] Tessler, C., Mankowitz, D. J., & Mannor, S. (2019). *Reward Constrained
Policy Optimization*. ICLR 2019. arXiv:1805.11074.

[2] Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems
Viewpoint*. Cambridge University Press / Hindustan Book Agency. (Two-
timescale stochastic approximation convergence theory underlying [1]'s
Theorem 1.)

[3] Roijers, D. M., Vamplew, P., Whiteson, S., & Dazeley, R. (2013). *A
Survey of Multi-Objective Sequential Decision-Making*. Journal of
Artificial Intelligence Research, 48, 67-113. arXiv:1402.0590. (Motivates
why a fixed linear scalarization weight, as used before this experiment,
cannot reliably target a specific point on the reward-tardiness trade-off.)

[4] Eimer, T., Lindauer, M., & Raileanu, R. (2023). *Hyperparameters in
Reinforcement Learning and How To Tune Them*. arXiv:2306.01324. (Explains
why the earlier fixed-weight tardiness-focused Optuna search, tuned at a
small scale, did not transfer to full deployment scale -- the motivating
failure this experiment addresses.)
