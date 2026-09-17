# Design: Pointer-Network Architecture for MaskablePPO

**Date:** 2026-09-16 (S2W9)
**Environment:** `Code/policies/{pointer_policy.py, pointer_ppo_policy.py}`,
`Code/training/{train_optimized.py, optuna_tune.py}`
**Status:** Implemented, smoke-tested, Optuna-tuned, and trained (3-seed
fixed-instance). **Result: the hypothesis is REFUTED** -- see Section 9. Randomized-
instance training not yet run (deprioritized given Section 9's result).

---

## 1. Motivation

The 2026-09-15 overnight session (`2026-09-14-ppo-lagrangian-and-reward-structure.md`,
Sections 9-12; `training-log.md`'s "Overnight session conclusion" entry) found that five
completely different reward/hyperparameter mechanisms for fixing PPO's tardiness --
a fixed weight, four Lagrangian $\lambda_{\max}$ ceilings, and a from-scratch
tardiness-directed hyperparameter search -- all converge on the same ~1290-1330
tardiness band, while A2C (using `PointerActorCritic`, in place of a flat MLP) sits at
~28. That consistency argues against "the reward mechanism was wrong" and points at
something upstream: PPO has never actually been given the pointer/attention
architecture to test whether architecture, not reward weighting, is the real
bottleneck. This had been deliberately not attempted autonomously (flagged as
needing the same kind of design decision as the two-critic Lagrangian rewrite), but
the user has since explicitly prioritized it -- ahead of, and in place of, the
originally-planned full $\lambda_{\max}$ sweep -- once returning to the session.

`PointerActorCritic` (`Code/policies/pointer_policy.py`, added 2026-08-09) already
existed as a `torch.nn.Module` returning raw `(logits, value)` from a flat observation
tensor, used directly by A2C's hand-rolled training loop
(`Code/policies/a2c_policy.py`). It had never been connected to PPO, which uses
`sb3_contrib`'s `MaskablePPO` -- a real third-party library with its own policy
interface, not something `PointerActorCritic` could just be swapped into.

---

## 2. SB3's policy interface: what actually needs overriding

Reading `sb3_contrib/ppo_mask/ppo_mask.py` directly (not assuming) showed
`MaskablePPO` only ever calls four methods on `self.policy` during training/rollout:
`forward()` (rollout: action + value + log-prob), `predict_values()` (value bootstrap
at truncation), `evaluate_actions()` (the PPO loss), and -- for inference, via the
inherited `predict()` -- `get_distribution()`/`_predict()`. Everything else
(`obs_to_tensor`, save/load plumbing, `predict()` itself) is generic and needed no
changes.

This meant `PointerMaskableActorCriticPolicy` (`Code/policies/pointer_ppo_policy.py`)
could subclass `MaskableActorCriticPolicy` and override exactly those four methods,
routing each through a single `self.pointer_net(obs)` call, rather than reimplementing
`BasePolicy.__init__`'s full setup from scratch.

## 3. A hazard found and avoided: optimizer parameter registration

`MaskableActorCriticPolicy.__init__` builds a default `mlp_extractor`/`action_net`/
`value_net` (a 64x64 MLP) and constructs `self.optimizer` over `self.parameters()`
*before* returning. If a subclass simply called `super().__init__()` unchanged and
then bolted on `self.pointer_net = PointerActorCritic(...)` afterward, two things would
go wrong silently: (a) the optimizer would never see `pointer_net`'s parameters at all
-- the whole point of the exercise -- and (b) the unused default MLP's parameters would
still exist, receive gradient updates, and pollute Adam's moment buffers for no reason.

Fix: override `_build()` (called from within `__init__`, after `self.action_dist` is
already constructed but before the parent's own optimizer-construction line) to build
`self.pointer_net` instead of the parent's `mlp_extractor`/`action_net`/`value_net`, and
construct `self.optimizer` there over `self.parameters()` -- at that point in the
subclass's `__init__`, `self.pointer_net` is the only submodule with any parameters
(`self.features_extractor`, built earlier by the parent, is a parameterless
`FlattenExtractor` for this project's flat `Box` observation space), so this is both
correct and leaves no dead parameters behind.

`_get_constructor_parameters()` was also overridden to swap out the now-irrelevant
`net_arch`/`activation_fn`/`ortho_init` keys for the pointer-specific ones
(`max_jobs`, `num_machines`, `num_resources`, `embed_dim`, `hidden`, `clip_c`) --
needed for `MaskablePPO.load()` to reconstruct the policy correctly from a saved
checkpoint.

## 4. Deriving max_jobs/num_machines/num_resources

`PointerActorCritic` needs these three dimensions at construction time.
`a2c_policy.py`'s `MaskableA2C.__init__` already derives them via
`getattr(env.unwrapped, ...)`, reading `GymSchedulingEnv`'s own attributes (not a
`gymnasium.Wrapper` in the inheritance sense -- it's a standalone `gym.Env` that
internally composes a `SchedulingEnv`, which is why `.unwrapped` stops there rather
than at `SchedulingEnv`). For PPO, `first_env` is a `VecEnv`, not a single `gym.Env`, so
the equivalent is `VecEnv.get_attr(name)`. Verified directly from
`stable_baselines3/common/vec_env/{dummy_vec_env,subproc_vec_env}.py` that both
backends' `get_attr` route through `env_i.get_wrapper_attr(attr_name)` -- the same
gymnasium>=1.x-safe wrapper-traversal mechanism already confirmed (2026-09-14 session)
for `env_method()` and used by `PPOLagrangianCallback`. So
`first_env.get_attr("max_jobs")[0]` (etc.), read once after `build_vec_env()` and before
`MaskablePPO(...)` construction, is safe regardless of `--vec-backend {dummy,subproc}`.

## 5. Backward compatibility: `--policy-type`'s CLI default

Before this change, PPO silently ignored `policy_type` entirely (always built
`"MlpPolicy"`), so the CLI's `default="pointer"` for `--policy-type` was harmless for
PPO in practice. Making PPO respect `policy_type` meant that same default would have
silently switched every existing `--algo ppo` invocation (that doesn't pass
`--policy-type` explicitly) onto the new, not-yet-tuned pointer path -- and, before the
Optuna search below existed, straight into a `FileNotFoundError` for a best-params file
that didn't exist yet. Fixed in both `train_optimized.py` and `optuna_tune.py`: the
CLI default is now `None`, resolved after parsing to `"pointer"` for `--algo a2c`
(unchanged) and `"flat"` for `--algo ppo` (unchanged) -- pointer-PPO must be requested
explicitly via `--policy-type pointer`. Verified this resolution doesn't change either
algorithm's default behavior by inspection (both branches produce identical
`policy_type` values to before for every call site that doesn't pass the flag).

The corresponding `file_tag`/`result_tag` construction in
`load_best_params()`/`run_optimization()` also needed the same treatment: rather than
branching on `algorithm == "a2c"` (which silently collapsed every PPO call onto the
unsuffixed `"ppo_..."` tag regardless of `policy_type`), both now branch on
`(algorithm == "ppo" and policy_type == "flat")` to decide whether to omit the
architecture suffix -- preserving every existing `"ppo_..."`/`"ppo_tardiness_..."` file
reference while correctly producing new `"ppo_pointer_..."` files for the new variant.

## 6. Optuna search extension

`objective_ppo` gained a `policy_type` parameter mirroring `objective_a2c`'s existing
branch exactly: `"pointer"` searches `embed_dim`/`hidden` (categorical, same ranges as
A2C's search) instead of `layer_size`/`n_layers`/`activation`, and resolves
`max_jobs`/`num_machines`/`num_resources` from the tuning env the same way (via
`env.env.env`, i.e. `Monitor -> ActionMasker -> GymSchedulingEnv`, one hop less than the
existing `base_env = env.env.env.env` used for reading `.tardiness`). `run_optimization`
threads `policy_type` through to `objective_ppo` via the same `functools.partial`
pattern already used for A2C, and its `--optimize-for reward/tardiness/pareto` modes
all work unchanged for the pointer variant (they only change what the objective
function returns, not how the model is built).

## 7. Validation before any real-scale run

Following this project's own established discipline (small excerpts before a real
launch -- see the PPO-Lagrangian smoke-tests precedent), two checks were run before
committing to the 50-trial Optuna search or any multi-hour training:

1. **Standalone smoke test**: hand-built a tiny `SchedulingEnv`/`GymSchedulingEnv`
   (10 jobs, 3 machines, horizon 15), constructed `MaskablePPO(PointerMaskableActorCriticPolicy,
   ...)` directly, ran `.learn(total_timesteps=128)`, `.predict()`, and a save/load
   round-trip asserting identical deterministic action before and after. All passed;
   `model.policy.pointer_net` reported a non-zero, correctly-scoped parameter count
   (6,818 for the tiny `embed_dim=32, hidden=16` config used).
2. **Integration smoke test**: the actual `build_vec_env()` -> `get_attr()` ->
   `MaskablePPO(...)` path `train_optimized.py` uses, at a small scale (8 jobs, horizon
   15). Confirmed `get_attr` correctly resolved `max_jobs=8, num_machines=10,
   num_resources=4` and training completed without error.

Both are throwaway scripts (scratchpad, not committed to the repo) -- if a permanent
regression test is wanted later, `tests/test_bugfixes.py`'s runnable-assert-script
convention would be the natural place for it, structured like the checks above.

## 9. Result: the hypothesis is refuted

3-seed fixed-instance training (Optuna-tuned: embed_dim=256, hidden=64,
lambda_2=1.89, 1.9M timesteps, identical curriculum/protocol to every other
offline PPO/A2C result):

```
seed1: tardiness=1303.36   seed2: tardiness=1619.24   seed3: tardiness=1403.86
```

Statistically indistinguishable from flat-MLP PPO's ~1290-1330 band (if anything
slightly worse, with higher variance -- seed2's 1619 is the worst PPO result
recorded this campaign outside the lambda_max=50 collapse). Since **A2C achieves
~28 tardiness with this exact same `PointerActorCritic` architecture**, giving PPO
the identical network did not close the gap at all -- architecture cannot be the
explanatory variable for PPO's tardiness, contrary to this document's Section 1
motivation and the overnight session's leading hypothesis.

This leaves five now-tried mechanisms (fixed reward weight, four Lagrangian
lambda_max ceilings, a tardiness-directed hyperparameter search, and now
architecture) all landing PPO in the same band. The remaining plausible
explanation is something intrinsic to **PPO's optimization procedure** on this
environment/reward -- the clipped surrogate objective, multi-epoch minibatch
reuse of rollout data, or GAE/advantage estimation, in some interaction with this
reward's structure -- rather than any single input (reward weight, constraint,
hyperparameters, or network) fixable independently of the algorithm itself. A
direct comparison of PPO's and A2C's *training dynamics* on identical
architecture and reward (not just final performance) is the natural next
investigation, not attempted here.

---

## 10. References

1. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). "Pointer Networks." NeurIPS 2015,
   arXiv:1506.03134. (Cited already in `pointer_policy.py` for the base architecture
   concept -- not re-derived here.)
2. Kool, W., van Hoof, H., & Welling, M. (2019). "Attention, Learn to Solve Routing
   Problems!" ICLR 2019, arXiv:1803.08475. (Scaled dot-product + tanh-clipping
   compatibility scoring, used unchanged by `PointerActorCritic.CompatibilityScorer`.)
3. Raffin, A., et al. (2021). "Stable-Baselines3: Reliable Reinforcement Learning
   Implementations." JMLR 22(268). (`MaskableActorCriticPolicy`'s interface, verified
   directly against the installed `sb3_contrib==2.8.0`/`stable_baselines3==2.8.0`
   source rather than assumed from documentation alone.)
