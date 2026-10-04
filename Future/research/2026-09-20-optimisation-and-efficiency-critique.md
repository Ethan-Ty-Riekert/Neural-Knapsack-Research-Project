# Optimisation & Efficiency Critique

**Date:** 2026-09-20 (S2W9)

**Scope:** This document is deliberately narrower than `report.md` (the existing
methodological/research-quality critical review at the repo root). It does not
re-derive `report.md`'s findings (stale docs, `*_ppo_policy.py` duplication, missing
test runner, citation hygiene) — those stand as already reported. This document asks
a different question: **where is compute being wasted, and what training-methodology
changes (HPO, parallelism, multi-agent decomposition) would get more research signal
per CPU-hour** on a CPU-only, 16-core machine that has already run 1.6M-timestep
training campaigns repeatedly. Findings below were produced by two independent
code-reading passes (training/HPO pipeline; environment/policy hot path) and every
claim quoted as fact was spot-checked directly against the source before being
written here.

---

## 0. Executive Summary

The codebase is not compute-bound in the way a first guess might suggest (huge
networks, expensive attention) — the pointer-network architecture is well-vectorized
and scales fine at deployed size (§3). The actual waste is almost entirely in three
places: (1) **a per-step dead-code hot-path bug** in the environment that copies four
arrays every single step and throws the result away (§1.1) — free to fix, no
behavioural change, and multiplies out over every 1.6M-timestep run this project has
ever done; (2) **an Optuna pruner that is configured but structurally cannot prune**,
because it only reports after a trial's compute is already fully spent (§2.1); (3)
**a fully sequential 50-trial HPO search on a 16-core machine** (§2.2), and A2C's
single-process rollout loop (§2.4) — both leaving most of the machine idle during
exactly the phases (tuning, and the algorithm this project has repeatedly compared
against PPO) where parallelism would help most.

None of this changes any past *result* — these are throughput/efficiency fixes, not
correctness fixes (the environment correctness bugs found today, `compute_theta()`
and the earlier capacity leak, are already logged separately in `training-log.md` and
are out of scope here). But given HPO on the current action-space designs (Options
1-4) is already deferred pending the design space settling
([[project_hpo_deferral_decision]] via memory; see also `training-log.md`'s
2026-09-19 entry) — the pruner and parallelism fixes in §2 are worth doing *before*
that search runs, not after, since they change how many trials/how much search
breadth a fixed compute budget buys.

Section 5 addresses the multi-agent-training question directly: it is not a drop-in
fix, but it is a structurally motivated *next* mechanism for the project's own
still-unresolved root-cause finding (action-space size), complementary to the
action-branching (Option 4) work already implemented today.

---

## 1. Environment / Simulation Hot-Path Efficiency

Every finding in this section executes on **every RL step of every training run** —
including all five completed curriculum-based offline campaigns and the online
6-run campaign already in `training-log.md`. None require retraining to fix; they
are pure throughput improvements with no effect on learned behaviour.

### 1.1 `get_state()` is computed and unconditionally discarded on every step — highest-value finding

`Code/env/scheduling_env.py:556-565` (`get_state()`) deep-copies four arrays
(`machine_active`, `capacity` — the full `(M, R, H)` array — `start_times`,
`tardiness`) and converts `remaining_jobs` (a set) to a list, every call. It is
called from `step()` (`scheduling_env.py:476`) and `step_idle()` (`:594`) — i.e. on
literally every environment step.

**Verified directly**: `Code/env/gym_scheduling_wrapper.py:229` and `:237` bind that
return value to `obs`/`_` and then **unconditionally overwrite it two lines later**
at `:247` with `obs = self._get_obs()` — the actual observation used for training.
`get_state()`'s output is never read anywhere in the RL training path. This is not a
hypothesis; grepping `gym_scheduling_wrapper.py` for any other use of the tuple's
first element returns nothing.

**Fix:** `SchedulingEnv.step()`/`step_idle()` should stop constructing `get_state()`
internally and either return `None` for that slot or take an explicit
`return_state: bool = False` flag, reserved for the (presumably non-hot-path) callers
that do need the dict form (e.g. debugging/inspection scripts, if any exist —
grep for other `.step(` callers before removing entirely). This removes four array
copies and a set→list conversion from the single hottest line of code in the entire
project.

### 1.2 Capacity array rebuilt/updated with Python loops instead of vectorized slicing

- `scheduling_env.py` (`__init__`/`reset()`, ~lines 87-90 and ~197-199): a nested
  `for m: for t:` Python loop broadcasts `machine_capacity` into the `(M, R, H)`
  capacity array on every `reset()` (i.e. every episode, not just once). Fixable with
  `np.broadcast_to(self.machine_capacity[:, None], (M, R, H)).copy()`.
- `scheduling_env.py:~420-421` (`step()`): `for tau in range(time, time+duration):
  capacity[machine, :, tau] -= job_resources[job]` — a Python loop over every
  occupied tick on every valid placement. Fixable with a single slice op:
  `self.capacity[machine, :, time:time+duration] -= job_resources[job][:, None]`.

Both are correct today (no bug), just doing with a Python loop what numpy's
broadcasting already does in one vectorized call — meaningful at 1.6M-timestep scale
purely as constant-factor overhead.

### 1.3 Observation construction reallocates via Python list → `np.array()` every step

`gym_scheduling_wrapper.py:95-137` (`_get_obs()`) builds the observation with
`list.append()`/`.extend()` (≈`1 + M·R + max_jobs·(R+4)` elements at deployed scale)
then converts with `np.array()` — a fresh Python list and a fresh numpy array every
single step. Fix: preallocate `np.empty(obs_dim, dtype=np.float32)` once (per episode
or per env instance) and fill via vectorized slice assignment.

### 1.4 Action-mask computation: unvectorized, and double-computed in one wrapper

- `gym_scheduling_wrapper.py:156-198` (`get_action_mask()`): nested Python loop —
  `for j in remaining_jobs: for m in range(num_machines): is_feasible(...)` — an
  O(|remaining_jobs| · M) Python-level call graph, run on every `step()` and
  `reset()`. This is a classic bin-packing vectorization target: broadcasting
  `capacity[:, :, t]` (shape `(M, R)`) against `job_resources[list(remaining_jobs)]`
  (shape `(J_r, R)`) — e.g. `(capacity[None] >= job_resources[:, None]).all(axis=2)`
  plus a vectorized `t + duration <= horizon` check — replaces the double Python loop
  with one array comparison.
- `Code/env/rule_selection_gym_wrapper.py` (Option 1's wrapper): computes the mask
  **twice per step** — once inside `_job_actions()` (line ~97), and again inside
  `get_action_mask()` (line ~123), which itself calls `_job_actions()` again
  (line ~78). This doubles the cost above for zero benefit. Fix: compute once, reuse
  the cached result within the same step.

### 1.5 `ATC` priority rule: O(job count) recomputation inside an O(job count) sort

`Code/baselines/priority_rules.py:~115` (`atc_priority`): `mean_p =
sum(job_durations[j] for j in known_jobs) / len(known_jobs)` — recomputed from
scratch on every call. `Code/baselines/registry.py:~29`'s `sorted(unique_jobs,
key=lambda j: (priority_key(base_env, j), j))` calls the priority function once per
candidate job, so a single ATC ranking costs O(J · num_jobs) instead of O(num_jobs).
This runs on every RL step whenever ATC is selectable (Option 1's rule set includes
ATC), across all of the project's 1.6M-timestep Option-1 runs. Fix: compute `mean_p`
once outside the per-job key function and pass it in as a precomputed value.

### 1.6 Positive control: the pointer-network architecture itself is not a bottleneck

`Code/policies/pointer_policy.py`'s `CompatibilityScorer` (`torch.einsum("bje,bme->
bjm")`) is fully batched, vectorized, and O(J·M) — exactly matching the action-space
size it scores, with no quadratic blow-up and no redundant forward passes over masked
actions found in either code-reading pass. At deployed scale (100 jobs × 10 machines)
this is not where cycles are going; the Python-loop findings in §1.1-§1.5 are.

---

## 2. Training Pipeline, Parallelism, and HPO Methodology

### 2.1 Optuna's `MedianPruner` is configured but structurally cannot prune

`Code/training/optuna_tune.py:705` configures `MedianPruner(n_startup_trials=5,
n_warmup_steps=0)` for the shared study. But **verified directly**:
`objective_ppo` calls `trial.report(prune_metric, step=0)` exactly once
(`optuna_tune.py:349`), and that call happens *after* `model.learn(total_timesteps=
30_000)` (`:283`) and a full 10-episode evaluation loop (`:298-315`) have both
already completed in full — i.e. after 100% of that trial's compute is already
spent. Pruning at that point saves nothing. **`objective_a2c` never calls
`trial.report()`/`trial.should_prune()` at all** (confirmed: `trial.report` appears
exactly once in the entire file, inside `objective_ppo` only) — the pruner is fully
inert for every A2C search.

**Fix:** report an intermediate metric mid-trial — e.g. after each of a few
sub-checkpoints within the 30k-timestep budget (every 5-10k steps, using a cheap
partial-episode reward proxy rather than the full 10-episode eval), so
`should_prune()` can actually kill a clearly-bad trial (e.g. the idle-collapse
pattern this project already detects post-hoc at `optuna_tune.py:327`) before it
burns its full budget. This directly increases the number of trials a fixed compute
budget can run.

### 2.2 HPO runs fully sequentially on a 16-core machine

`run_optimization`'s `n_jobs` parameter defaults to `1` (`optuna_tune.py:618`, used
at `study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, ...)`, `:756`), and
each trial's env (`make_tuning_env`) is a single non-vectorized environment. The
default 50-trial search therefore runs 50 trials fully sequentially, each using
roughly one core, on a 16-core machine — an ~unused 15/16ths of available
parallelism during every HPO run this project has done. Optuna natively supports
`n_jobs > 1` for in-process parallel trials against a shared study (the project
already writes to `optuna.db`, i.e. an RDB-backed study — a prerequisite for
multi-process parallel `optimize()` calls is already satisfied). Raising `n_jobs`
(or launching several `study.optimize()` processes pointed at the same storage URL)
would let a fixed number of trials finish in a fraction of the wall-clock time, or
let a fixed time budget cover more trials.

### 2.3 Tuning parallelism doesn't match deployment parallelism

HPO trials train on a single (`n_envs=1`) env (`optuna_tune.py`'s
`make_tuning_env`), while real training uses `n_envs=5` (`resolve_ppo_rollout_params`,
`train_optimized.py:367-394` — whose own docstring already documents that Optuna's
`n_steps`/`batch_size` were tuned assuming `n_envs=1` and must be rescaled by a
`n_envs`-dependent conversion at deploy time). This conversion is a real, acknowledged
approximation, and a second source of exactly the small-scale-to-deployed-scale
transfer risk this project has already been burned by twice (Eimer et al. 2023
[[feedback_use_latest_findings]] pattern — see `training-log.md`'s 2026-09-15
tardiness-tuning entry). Tuning directly under `n_envs=5` (the actual deployment
setting) would remove this conversion step entirely, not just make it faster.

### 2.4 A2C has no vectorized rollout — a structural throughput gap independent of the algorithm-effect question `report.md` already raised

`report.md` §1.6 already flags that every "A2C beats PPO" comparison in this
project's history is confounded by implementation/library/vectorization differences,
and recommends training SB3's own `A2C` on PPO's protocol to isolate the algorithm
effect. This section adds the throughput dimension to that same finding:
`Code/policies/a2c_policy.py`'s `MaskableA2C.train()` runs `self.env.step(action)`
in a plain single-process Python loop (`a2c_policy.py:~406`) — no `VecEnv`, no
multiprocessing — so it uses roughly 1 of 16 cores for env-stepping throughput
regardless of machine size, while PPO gets `n_envs`-way (default up to 15)
parallel/subprocess rollout. Two more compounding, lower-priority items found in the
same file: `compute_returns_and_advantages`'s GAE computation
(`a2c_policy.py:182-204`) is a scalar Python `for t in reversed(range(...))` loop
(harmless at the default `n_steps=5`, but Optuna's own A2C search space goes up to
`n_steps=50`, `optuna_tune.py:~492`); and `select_action`
(`a2c_policy.py:76-110`) does per-step `.item()` conversions with no cross-env
batching, consistent with the single-env design.

**Fix candidates, cheapest first:** (a) raise `n_steps` to amortize per-step Python
overhead before each update — free, no architecture change; (b) give `MaskableA2C`
genuine synchronous vectorized rollout (n copies of the env stepped in a batch) —
SB3's own `A2C` already supports `VecEnv` natively, so this is "adopt the existing
library pattern," not new design; either directly closes report.md's §1.6 gap
(the experiment SB3's own A2C on PPO's protocol needs this fix as a prerequisite
anyway) and removes A2C's throughput handicap as a confound in every future
A2C-vs-PPO comparison.

### 2.5 Untested assumptions baked into training defaults

- `train_optimized.py:481` defaults `vec_backend="subproc"`. Its own docstring
  (`:315-332`) explicitly says to "time both backends at the target `n_envs` on this
  machine before committing to one" — **no benchmark script or result exists
  anywhere in the repo.** The parallelism-backend default is an untested assumption,
  not a measured choice, despite already being flagged in-code as needing
  measurement.
- `train_optimized.py:613-614` defaults `n_envs = cpu_count() - 1` with no check
  against concurrent processes. The project's own established protocol runs up to 3
  seeds concurrently (`--n-envs 5 --torch-threads 5` per seed on 16 cores,
  [[project_offline_training_campaign]]) — correct only because the CLI flags are
  set by hand each time. Nothing in code computes or warns on
  `n_envs * torch_threads * num_concurrent_procs > cpu_count()`; a future session
  that forgets to override the defaults when running concurrent seeds would silently
  oversubscribe the machine. A cheap guard (warn, don't block) would remove this as
  a recurring manual-discipline requirement.
- `train_optimized.py:885-908`: every curriculum-stage transition (4 stages) closes
  and rebuilds the full VecEnv — necessary in principle (each stage's
  `num_jobs`/`horizon` changes the observation layout, so the sub-envs must be
  rebuilt), but with the untested `subproc` default this means 4 stage transitions ×
  3 concurrent seeds = 12 OS-process-pool respawns per campaign via Windows' slow
  "spawn" start method — a real, currently unmeasured cost that is specific to
  whichever backend turns out to actually be faster in §2.5's first point.

---

## 3. HPO Search-Space / Budget Observations (methodology, not code)

Not a code defect, but relevant to "more signal per CPU-hour": `optuna_tune.py`'s
default trial budget (30k timesteps, 10 eval episodes, 50 trials) is a *smoke-scale*
search relative to deployed training (1.6M timesteps). This is the same class of
small-scale-tuning-doesn't-transfer risk already twice-confirmed in this project
(A2C then PPO, per [[project_offline_training_campaign]]) — not a new finding, but
worth stating plainly here since §2.1-§2.3's fixes (real pruning, parallel trials,
matched `n_envs`) change what's actually affordable: with pruning working and 16
trials running in parallel instead of 1, the same wall-clock budget could support
either more trials at the current scale, or fewer trials at a larger, more
deployment-representative scale — a real tradeoff worth deciding deliberately
rather than defaulting into via unfixed pruning/parallelism. This search is already
scoped as deferred until the Options 1-4 design space settles
([[project_hpo_deferral_decision]]) — the recommendation here is to land §2.1-§2.3
*before* that search runs, since it changes the search's effective budget, not to
run HPO now.

---

## 4. What Was *Not* Found

For calibration: the pointer-network architecture (§1.6) is well-vectorized and not
a bottleneck; checkpoint/logging I/O (`Code/utils/results_log.py`'s archiving,
`Code/utils/training_diagnostics.py`'s TensorBoard logging) is batched and
interval-gated, not per-step, and is not a hot-path concern. `Code/utils/
training_diagnostics.py`'s `TardinessEvalCallback` docstring itself flags that its
per-`eval_freq` cost claim is unverified ("should be confirmed empirically... not
assumed") — worth a real timing pass before relying on that comment as fact, but not
included as a confirmed finding above since no measurement exists either way.

---

## 5. Multi-Agent Training: A Structurally Motivated Next Step

This section answers the user's explicit question about multi-agent training. It is
a **proposed research direction, not a recommendation to implement immediately** —
consistent with this project's own precedent of flagging bigger design-surface
changes for user input before building them unsupervised (e.g. the two-critic
PPO-Lagrangian architecture and the pointer-network-PPO experiment were both scoped
this way before being either deferred or built). No code changes accompany this
section.

**Why this connects to the project's own strongest open finding, not a generic
suggestion:** `training-log.md` records seven independent mechanisms tried for
fixing flat-MLP PPO's tardiness (reward weight, four Lagrangian variants,
hyperparameter search, pointer architecture, dense-reward redesign) all converging
on the same ~1290-1330 tardiness band, with **action-space size** (~1000+ raw
`(job, machine)` combinations vs. DeepRM's ~11) identified as the most plausible
root cause once architecture was ruled out (2026-09-16/17 entries). Every fix that
worked (Options 1-3, and today's Option 4 action-branching) works by *shrinking or
decoupling the action space* — either delegating placement to a fixed heuristic
(Options 1-3) or splitting one joint decision into parallel independent branches off
one shared network (Option 4, Tavakoli et al. 2018 [`tavakoli2018actionbranching`]).

Multi-agent RL is the next point on that same spectrum, not a different idea:
instead of one policy emitting a factored-but-still-jointly-trained action (Option
4), a genuinely decentralized formulation gives **each machine its own agent**,
choosing only among its own locally-feasible remaining jobs each tick (or idling).
Concretely:

- **Action space per agent** shrinks from the current O(J·M) joint space to O(J) per
  machine — a much larger reduction than action-branching alone, since branching
  still trains through one shared forward pass and one shared optimizer step, while
  per-machine agents decompose the *learning problem* itself, not just the action
  encoding.
- **Architectural fit with existing code:** the project already has, in
  `pointer_policy.py`, a job encoder and a compatibility-scoring mechanism
  (`CompatibilityScorer`, §1.6) that computes per-(job, machine) scores from shared
  embeddings — structurally close to what a per-machine actor would need (score only
  this machine's row) plus a centralized critic that sees the full joint state
  (already exactly what the current flat observation gives the single-agent value
  function). This is the Centralized-Training-Decentralized-Execution (CTDE)
  pattern.
- **Two candidate algorithms, different cost/fit tradeoffs:**
  - **MAPPO** (Yu et al. 2022 [`yu2022mappo`]) — multi-agent PPO with a centralized
    critic and decentralized actors. Lowest engineering lift for this specific
    codebase: it is PPO with a shared critic input, not a new algorithm family, so
    it reuses the project's existing `sb3_contrib.MaskablePPO`-adjacent tooling,
    masking conventions, and Optuna search infrastructure far more directly than a
    from-scratch algorithm would.
  - **QMIX** (Rashid et al. 2018 [`rashid2018qmix`]) — value-based CTDE with
    monotonic value-function factorization (a per-agent Q-value that provably
    composes into a correct joint Q-value under a monotonicity constraint). A
    heavier engineering lift here (this project has zero existing
    value-factorization infrastructure — everything so far is policy-gradient), but
    worth naming as the alternative design point in the literature, since a
    value-based multi-agent approach reasons about credit assignment across
    machines differently than a shared-critic policy-gradient approach does.
  - General grounding for feasibility and design alternatives beyond these two:
    Hady et al. 2025's survey of MARL for resource-allocation problems
    ([`hady2025marlsurvey`]) — cited as a survey, not as evidence either algorithm
    is proven for *this* problem class; author list not independently
    cross-checked (see `references.bib`'s note on the entry).

**Open questions this project would need to resolve before building this (explicitly
not answered here, per the rigor rule against asserting an untested design as
settled):**
1. Credit assignment: with one shared global reward (weighted tardiness across the
   whole job set), does each per-machine agent get a clean enough learning signal
   for *its own* contribution, or does this need a difference-reward / counterfactual
   baseline (a known hard problem in cooperative MARL, not yet investigated here)?
2. Non-stationarity: from any one machine-agent's perspective, the other machines'
   evolving policies make the environment non-stationary during training — a known
   MARL convergence risk that single-agent RL (everything this project has done so
   far) never has to contend with.
3. How does a per-machine-agent formulation interact with job-level features that
   are inherently *shared* across machines (a job's deadline/weight doesn't belong
   to any one machine) — likely resolved via the same job-encoder-then-score pattern
   `pointer_policy.py` already uses, but not yet designed.

**Recommended sequencing, consistent with this project's existing deferred-HPO
decision:** treat this as a third major branch of future work alongside the
already-planned two-critic PPO-Lagrangian rewrite and the online-case work — i.e.
scope it properly with the user once the current action-space design space (Options
1-4) settles, the same gate already applied to HPO
([[project_hpo_deferral_decision]]). Given Option 4 was only implemented earlier
today and hasn't had a real training run yet, this is not the moment to start a
second, larger architectural rewrite in parallel.

---

## 6. Prioritized Recommendations

**Free / zero-risk (no retraining needed, pure throughput fixes, do anytime):**
1. Stop constructing `get_state()` from `step()`/`step_idle()`'s hot path (§1.1) —
   the single highest-value fix in this document.
2. Vectorize the capacity-array reset/update loops (§1.2).
3. Preallocate `_get_obs()`'s output buffer instead of list→`np.array()` per step
   (§1.3).
4. Vectorize `get_action_mask()`; deduplicate the double mask computation in
   `rule_selection_gym_wrapper.py` (§1.4).
5. Fix ATC's O(J·num_jobs) `mean_p` recomputation (§1.5).

**Before the next HPO run (moderate effort, changes what a fixed compute budget
buys — do before, not after, the already-planned Options-1-4 HPO search):**
6. Make the Optuna pruner actually prune: report an intermediate metric mid-trial
   for both PPO and A2C objectives (§2.1).
7. Raise Optuna's `n_jobs` above 1 / run parallel trials against the existing
   RDB-backed study (§2.2).
8. Tune under `n_envs=5` (matching deployment) instead of `n_envs=1` +
   post-hoc rescaling (§2.3).
9. Decide deliberately (not by default) what the freed-up budget from #6-8 buys:
   more trials at smoke scale, or fewer trials at deployment scale (§3).

**Moderate (training-pipeline changes, quick to verify, no new experimental design
needed):**
10. Benchmark `DummyVecEnv` vs `SubprocVecEnv` once at the target `n_envs` and
    commit to a measured default (§2.5) — the code already says to do this.
11. Add an oversubscription guard/warning for concurrent-seed CPU allocation (§2.5).
12. Give `MaskableA2C` real vectorized rollout, or at minimum raise `n_steps` — this
    is also a prerequisite for `report.md` §1.6's own recommended experiment
    (isolate the A2C-vs-PPO algorithm effect) (§2.4).

**Bigger design surface (needs user scoping before building, per this project's own
established process — not to be started unsupervised):**
13. Multi-agent (MAPPO/QMIX-style) decomposition as the next mechanism on the
    action-space-size spectrum, after Options 1-4 (§5).

---

## References

1. Tavakoli, A., Pardo, F., & Kormushev, P. (2018). "Action Branching Architectures
   for Deep Reinforcement Learning." arXiv:1711.08946 — the action-space-decoupling
   principle Option 4 already implements; §5 frames multi-agent decomposition as the
   next point on this same spectrum. [`tavakoli2018actionbranching`]
2. Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022).
   "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games."
   arXiv:2103.01955 — MAPPO, the lowest-engineering-lift multi-agent RL option for
   this codebase given its existing PPO investment. [`yu2022mappo`]
3. Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson,
   S. (2018). "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent
   Reinforcement Learning." *ICML 2018*. — the value-based CTDE alternative to
   MAPPO. [`rashid2018qmix`]
4. Hady, M. A., Hu, S., Pratama, M., Cao, Z., & Kowalczyk, R. (2025). "Multi-Agent
   Reinforcement Learning for Resources Allocation Optimization: A Survey."
   *Artificial Intelligence Review*, 58(11), 354. DOI: 10.1007/s10462-025-11340-5 —
   general MARL-for-resource-allocation feasibility grounding; author list
   unconfirmed against the published article. [`hady2025marlsurvey`]
5. Eimer, T., Lindauer, M., & Raileanu, R. (2023). "Hyperparameters in Reinforcement
   Learning and How To Tune Them." — the small-scale-tuning-doesn't-transfer failure
   mode §2.3/§3 warn against repeating a third time. [`eimer2023hyperparams`]
6. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). "Optuna: A
   Next-Generation Hyperparameter Optimization Framework." *KDD 2019* — the pruning
   (`MedianPruner`) and parallel-trial (`n_jobs`) mechanisms discussed in §2.1-§2.2
   are native Optuna features, not proposed new infrastructure.
   [`akiba2019optuna`]
