# Design: Online (Dynamic-Arrival) Case -- MDP Extension and Implementation

**Date:** 2026-09-16 (S2W9)
**Environment:** `Code/env/{arrival_process.py, online_scheduling_env.py, online_gym_wrapper.py}`,
`Code/training/train_optimized.py`, `Code/evaluation/eval_rl_agent.py`,
`Code/baselines/priority_rules.py`, `tests/test_online_env.py`
**Status:** Core environment implemented, tested (`tests/test_online_env.py`, all
checks pass), and wired into training/eval. Formal MDP writeup:
`mathformulation.tex`'s "Job Arrivals and Assignments per Tick" and "MDP formulation
for the online (dynamic-arrival) case" subsections (both confirmed with the user
before implementation began -- see the approved plan,
`hi-claude-lots-on-cozy-graham.md`).

---

## 1. Design decisions (all made collaboratively, not unilaterally)

Every decision below was presented as options/trade-offs to the user and chosen by
them, not assumed:

- **Arrival process**: Poisson, rate $\nu$ jobs/tick (Kleinrock, 1975, *Queueing
  Systems Vol. 1* -- the standard queueing-theory construction; exponential
  inter-arrival times are memoryless, so no arrival-timing correlation is introduced
  beyond the single rate parameter). Matches the synthetic arrival model in Mao,
  Alizadeh, Menache & Kandula (2016, "Resource Management with Deep Reinforcement
  Learning," HotNets) -- the closest directly-comparable online RL scheduling work.
- **Deadlines relative to arrival**: $d_j = \tau_j + \Delta_j$, not an absolute range
  independent of $\tau_j$ -- an absolute deadline would let a late-arriving job be
  born already overdue, which is not a meaningful notion of deadline once arrival
  itself is random.
- **Decision-clock relaxation (Option 2 of three presented)**: rather than keep the
  offline case's "every action advances time by 1" rule (which would impose an
  artificial $\nu < 1$ decision-throughput ceiling, stricter than actual resource
  capacity), only the `idle` action advances the tick. Multiple placements may occur
  at the same tick across successive steps, limited only by residual capacity
  feasibility -- no explicit cap, since no citation or derivation in this project
  justifies a specific dispatch-rate limit below that.
- **Finite window** $[0, H)$, not the paper's own genuinely infinite-horizon /
  average-reward alternative (Eq. 46-48, $\rho(\pi)$) -- the latter needs a
  fundamentally different training objective with no precedent in this codebase,
  flagged as future work.
- **Arrival rate**: chosen via the standard queueing-theory utilization criterion.
  By Little's Law (Little, 1961), $\rho = \nu \cdot \mathbb{E}[p_j] \cdot
  \mathbb{E}[a_{jr}] / (|M| \cdot C_r) = \nu/12$ under this project's default
  parameters ($|M|=10$, $C_r=30$, $\mathbb{E}[p_j]=\mathbb{E}[a_{jr}]=5$). Target
  $\rho \approx 0.7$ (moderate load), single fixed setting initially.
- **Importance/priority feature** (for the RL policy's observation, not yet
  implemented as of this writing -- see Section 5): an Apparent Tardiness Cost
  (ATC) index (Vepsalainen & Morton, 1987), to be tested via a 4-variant ablation
  (no feature / ATC-only / ATC+resource-fragmentation term / ATC+resource as a
  separate feature) rather than committing to one design by assertion.
- **Hotspot penalty** ($\lambda_3 \Delta\theta$): resolved via ablation (present vs.
  decoupled from training) rather than by argument -- not yet run as of this
  writing.

## 2. Implementation: subclass, don't modify

`OnlineSchedulingEnv(SchedulingEnv)` and `OnlineGymSchedulingEnv(GymSchedulingEnv)`
leave the offline classes completely untouched, satisfying the user's explicit
requirement that offline training/eval keep working unmodified while the online
case is built. Every mechanism that already operates purely over
`self.remaining_jobs` (`is_feasible`, `reward`, `compute_theta`,
`_finalize_unscheduled_job_cost`, termination checks) becomes automatically
causally correct once `remaining_jobs` is populated based on arrival rather than
"all jobs, from t=0" -- only *how* and *when* it gains members changes.

**Phantom padding**: `generate_poisson_arrivals()` (`arrival_process.py`) always
returns exactly `max_jobs` job records; any slot beyond the realized Poisson
arrival count gets `arrival_time = horizon + 1`, keeping `num_jobs` constant across
episodes (required by `SchedulingEnv.set_jobs()`'s invariant) while the realized
count varies.

## 3. Three real bugs found and fixed during implementation (verified by
`tests/test_online_env.py`, not just reasoned about)

**3.1 The "unreachable" sentinel wasn't actually unreachable.** `horizon + 1` was
intended as a guaranteed-never-revealed arrival time for padding jobs. But
`step_idle()` increments `self.time` up to exactly `horizon + 1` *before* checking
`done = self.time > self.horizon` -- so `_reveal_arrivals()`, called at that same
tick, would incorrectly reveal every padding job right at the terminal step. Fixed
by gating revelation on `self.time <= self.horizon`. Caught by Check 4 in the test
suite before this doc was written, not found by inspection.

**3.2 A valid same-tick placement would eventually get misclassified as an invalid
action and truncate the episode.** `GymSchedulingEnv.step()` uses "did
`self.env.time` change?" as a cheap proxy for "was that action valid?" -- correct
for the offline case, where *every* valid placement advances time by exactly 1.
Under the online case's relaxed tick-advance rule, a *valid* placement also leaves
time unchanged (by design -- that's the whole point of allowing multiple placements
per tick), so the base class's proxy would count every successful same-tick
placement toward `_max_invalid_actions` and eventually truncate the episode for the
agent doing exactly what it should. Fixed in `OnlineGymSchedulingEnv.step()` by
checking whether the targeted job was actually removed from `remaining_jobs`
instead -- a proxy that stays correct under either tick-advance rule. This is the
kind of bug that would have silently corrupted every online training run's episode
lengths without ever raising an exception -- worth flagging as the most consequential
find of this implementation pass.

**3.3 The offline `_get_obs()`'s per-slot branch (`if j < self.num_jobs`) always
evaluates true once `num_jobs == max_jobs`** (the phantom-padding convention), so a
not-yet-arrived job's real duration/deadline/weight/resource-demand would leak into
the observation even though the action mask correctly hides it from selection --
silently violating the causality constraint $x_{jmt}=0,\ \forall t<\tau_j$ at the
observation level. This one was anticipated (flagged during planning, before any
code was written) rather than found via testing; fixed in
`OnlineGymSchedulingEnv._get_obs()` by gating on `j in self.env.revealed_jobs`
instead. Still verified by Check 2 in the test suite as a regression guard.

## 4. Baseline adaptation

- `fcfs_key` (`priority_rules.py`): now reads `arrival_times[job]` when the env
  exposes it (`getattr(base_env, "arrival_times", None)`), matching true
  first-come-first-served semantics online; unchanged (job index) for the offline
  case, where every job is available at $t=0$ and "arrival order" was already
  meaningless as anything but index order.
- New `atc_key`: the ATC rule itself (Section 1), online-adapted by computing its
  mean-processing-time normalizer over only `revealed_jobs` (falling back to every
  job offline) -- keeps the baseline causally restricted to information actually
  available at decision time, consistent with the online eval protocol.
  Registered as `"ATC"` in `PRIORITY_RULES`, which automatically generates
  `ATC+FirstFit`/`ATC+BestFit`/`ATC+WorstFit` via `registry.py`'s existing
  priority-x-placement composition.

## 5. Not yet done as of this writing

- Importance-feature ablation (ATC as an observation feature, not just a baseline
  rule) and the hotspot-penalty ablation -- both scoped, neither implemented.
- Pointer-network PPO training for the online case (depends on the offline-case
  pointer-PPO work landing first -- see `2026-09-16-pointer-network-ppo.md`).
- A full training run at deployment scale ($\rho \approx 0.7$) for either
  algorithm -- only small-scale smoke tests have been run so far.
- `optuna_tune.py` has no online-specific search yet; online training currently
  reuses offline-tuned hyperparameters as a starting point.
- Min & Kim (2022)'s state-dependent-tuned ATC variant (a DDPG-tuned lookahead
  parameter) -- a separate small RL-training sub-project, not attempted; `atc_key`
  implements only the original fixed-$k$ rule.

## 6. References

1. Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory.* Wiley.
2. Little, J. D. C. (1961). "A Proof for the Queuing Formula: $L = \lambda W$."
   *Operations Research* 9(3), 383-387.
3. Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). "Resource Management
   with Deep Reinforcement Learning." *HotNets 2016*.
4. Vepsalainen, A. P. J., & Morton, T. E. (1987). "Priority Rules for Job Shops
   with Weighted Tardiness Costs." *Management Science* 33(8), 1035-1047.
5. Min, B., & Kim, C. O. (2022). "State-Dependent Parameter Tuning of the Apparent
   Tardiness Cost Dispatching Rule Using Deep Reinforcement Learning." *IEEE
   Access* 10, 20187-20198.
6. Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy Invariance Under Reward
   Transformations: Theory and Application to Reward Shaping." *ICML 1999*.
   (Cited for the hotspot-penalty ablation's potential-based-reformulation option,
   not yet implemented -- see Section 5.)
