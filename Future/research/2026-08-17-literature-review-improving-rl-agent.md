# Literature Review: Diagnosing and Improving the RL Agent (Reward/Tardiness Misalignment, HPO Generalization)

**Date:** 2026-08-17 (S2W5)
**Status:** Literature review complete. Findings mapped to concrete next actions (Section 7);
none of those actions implemented yet in this session -- this doc is the research basis for
a follow-up implementation session.

## 1. Motivation

Requested directly: "Are there any ways that papers say we can use to examine and improve
our reinforcement learning agent?" This follows two results from earlier today (S2W5,
`Future/research/training-log.md`) that both showed the same shape of failure:

- Doubling stage-4 training time pushed reward up but tardiness up too (flat: 866->1546).
- Retuning Optuna to explicitly optimize a reward-minus-tardiness composite score found
  hyperparameters that hit *zero* tardiness on the small tuning environment, but at full
  scale produced the best-ever reward *and* the worst-ever tardiness simultaneously
  (pointer: 270.23->328.21 reward, 1227->1699 tardiness).

Both results looked like "pushing optimization harder makes the true objective (tardiness)
worse, not better, even while the optimized proxy (reward) improves." That is a named,
studied phenomenon in the RL literature, not something specific to this codebase -- this
review found the relevant papers and checked whether their prescribed fixes apply here.

## 2. Reward hacking: this project has had a named failure mode, not just a bug

**Skalse, Howe, Krasheninnikov, Krueger, "Defining and Characterizing Reward Hacking,"
arXiv:2209.13085 (2022).** Gives the first formal definition of reward hacking: optimizing
an imperfect *proxy* reward function leads to worse performance under the *true* objective.
Their central negative result: for a proxy and a true reward to be jointly "unhackable"
(optimizing one never hurts the other), across all stochastic policies, one of them has to
be constant -- i.e. for any non-trivial pair of objectives, hackability is close to the
default, not the exception. This directly reframes the S2W4/S2W5 pattern: `total_reward`
(the +3 placement / +50 completion bonus mixture) is a *proxy*, and `total_tardiness` is
closer to the actually-wanted objective. Per this paper's result, some degree of
divergence between them under harder optimization should be *expected* given they're
combined via a fixed linear weighting (`lambda_2`), not treated as a rare bug to hunt down.

**Pan, Bhatia, Steinhardt, "The Effects of Reward Misspecification: Mapping and Mitigating
Misaligned Models," ICLR 2022, arXiv:2201.03544.** Empirically maps *when* reward hacking
gets worse: they vary agent capability (model size, action resolution, training duration)
across four RL environments and find hacking increases with capability, often via a sharp
**phase transition** -- a capability threshold past which behavior qualitatively shifts and
true reward drops suddenly, even as proxy reward keeps climbing. This is a close match to
this project's own data: the pointer network's *biggest* single-run capacity/training
increases this session (stage-4 timesteps doubled; then a 4x-larger hidden layer from the
tardiness-focused Optuna search) each independently produced record-high reward *and*
record-high tardiness in the same run, not a monotonic trade-off. Their proposed mitigation
is anomaly detection on the *true* metric during training/selection, not just watching the
proxy -- i.e. checkpoint/trial selection should have a tardiness sanity check, not just a
reward one (see Section 7).

## 3. Why the tardiness-focused Optuna retuning didn't transfer

**Eimer, Lindauer, Raileanu, "Hyperparameters in Reinforcement Learning and How To Tune
Them," arXiv:2306.01324 (2023).** Studies exactly the failure this project's S2W5
tardiness-retuning experiment hit: hyperparameters tuned against one environment/seed can
overfit to it, because "the hyperparameter landscape can strongly depend on the tuning
seed." Their concrete recommendation, imported from AutoML practice: **separate the tuning
environment/seed from the testing (deployment) one**, and validate a tuned configuration on
a held-out setting before trusting it, rather than assuming a result at tuning scale
transfers. This project's Optuna tuning environment (20 jobs, 5 machines, horizon 30) is
much smaller/easier than the deployment target (stage 4: 100 jobs, horizon 100) -- exactly
the kind of tuning/deployment mismatch this paper's recommendation is meant to catch. This
gives a literature-grounded reason (not just a post-hoc guess) for the follow-up already
flagged in the S2W5 training-log entry: evaluate Optuna trials on a harder/larger instance
closer to stage-4 scale, not just the small tuning env.

## 4. An alternative to hand-tuning `lambda_2`: constrained RL

**Tessler, Mankowitz, Mannor, "Reward Constrained Policy Optimization" (RCPO), ICLR 2019,
arXiv:1805.11074.** Names the exact problem this project has been fighting via repeated
manual Optuna re-weighting: "as the goal of the agent is to maximize accumulated reward, it
often learns to exploit loopholes and misspecifications in the reward signal" -- their fix
is to stop folding the secondary objective into the reward as a fixed scalar weight at all.
Instead, RCPO reformulates training as **constrained optimization** (e.g. "keep mean
tardiness under some threshold" rather than "minimize reward minus 50x tardiness") and uses
a Lagrange multiplier, adjusted automatically on a slower timescale than the policy update,
to enforce the constraint -- proven convergent, and the penalty weight is *learned*, not
hand-picked or Optuna-searched. This is structurally different from everything tried so
far in this project (every attempt to date -- the original reward formula, the S2W5
tardiness-focused Optuna search -- has been a fixed linear weighting of reward and
tardiness, just with different weights). Directly relevant as a genuinely different next
architecture to try, not another reweighting pass. See Section 7 for how this would map
onto `SchedulingEnv`/`MaskableA2C`.

## 5. Scalarization vs. multi-objective: what our composite Optuna score gave up

**Roijers, Vamplew, Whiteson, Dazeley, "A Survey of Multi-Objective Sequential
Decision-Making," Journal of Artificial Intelligence Research, 48:67-113, 2013,
arXiv:1402.0590.** Foundational taxonomy for multi-objective sequential decision problems;
establishes that collapsing multiple objectives into one scalar (a "scalarization
function") is only one of several valid strategies, and that in general the object of
interest is a *Pareto front* of non-dominated policies, not a single scalar-optimal one.
Several more recent papers turned up by this same search (SURF, arXiv:2605.20619;
Tchebysheff scalarization, arXiv:2604.13175) state the specific limitation more bluntly:
**linear scalarization cannot recover the full Pareto front** -- a single weighted
combination can only ever find one point on it, and a poorly-chosen weight can miss good
trade-offs entirely. This project's `optuna_tune.py --optimize-for tardiness` (added this
session) is exactly a linear scalarization (`mean_reward - 50 * mean_tardiness_normalised`)
with a hand-picked weight (50, chosen to match the completion bonus's magnitude, explicitly
flagged as a calibration choice not a derived one). Optuna natively supports genuine
multi-objective studies (`directions=["maximize", "minimize"]`, returning a
`(reward, tardiness)` tuple instead of collapsing them) -- worth trying instead of (or
alongside) picking a better scalar weight, per this literature's own critique of the
approach just used.

## 6. Domain-specific: dense/incremental tardiness reward shaping in scheduling RL

Search turned up several 2024-2025 multi-agent DRL scheduling papers (e.g. dynamic
reconfigurable shop scheduling with an "estimated tardiness cost driven reward function";
production-logistics scheduling minimizing total weighted tardiness) that report using a
**dense, incremental tardiness signal** -- computing the change in tardiness at each
decision step and using its negative as an immediate per-step reward, rather than only
scoring tardiness once, sparsely, at job completion. These are ScienceDirect-hosted, not
open-access, and this review could not verify their full method against the primary text
(only search-result summaries) -- flagged explicitly per this project's rule about
not citing unverified sources as if confirmed. The pattern they describe, however, is
already implemented and unused in this codebase: `SchedulingEnv`'s
`use_potential_shaping` flag (Section 6 of
`2026-08-09-pointer-network-action-head.md`) computes exactly this kind of per-step,
urgency-based dense signal via `Φ(s) = -Σ urgency_j(t)`, and is sign-checked but has never
been run through the full curriculum (Experiment 4, still queued). This search result is
independent, if unverified, corroboration that the *type* of fix already built (dense,
per-step deadline-urgency signal vs. a single sparse per-job term) is a direction the wider
scheduling-RL literature has also converged on -- raises the priority of finally running
Experiment 4, rather than deprioritizing it.

## 7. Concrete next actions, mapped from Sections 2-6

In rough order of how directly they follow from what's already built vs. requiring new
implementation:

1. **Run Experiment 4 (potential-based shaping) before further reward-weight tuning.**
   Already implemented, sign-checked, never run full-curriculum. Section 6's (unverified
   but corroborating) literature and the existing Ng/Harada/Russell (1999)
   policy-invariance guarantee both point the same direction.
2. **Redesign the Optuna tardiness search to evaluate trials on a larger/harder instance**
   (closer to stage-4 scale), not the current 20-job tuning env -- directly per Eimer et
   al.'s tuning/testing-environment-separation recommendation (Section 3). This is the
   already-flagged S2W5 follow-up; now literature-grounded, not just a post-hoc guess.
3. **Add a tardiness sanity check to trial/checkpoint selection**, not just reward -- per
   Pan et al.'s anomaly-detection recommendation (Section 2). Concretely: reject/flag an
   Optuna trial or a full-curriculum checkpoint if tardiness moved substantially worse even
   if reward improved, rather than only tracking reward-vs-baseline.
4. **Try a genuine multi-objective Optuna study** (`directions=["maximize","minimize"]` on
   `(reward, tardiness)`) instead of a single hand-weighted scalar, and inspect the Pareto
   front directly -- per Section 5. Bigger implementation lift than 1-3; a real alternative
   to re-guessing `TARDINESS_PENALTY_WEIGHT`.
5. **Prototype constrained RL (RCPO-style)** as a structurally different alternative to
   reward-weight tuning altogether -- per Section 4. Biggest lift (changes the training
   objective, not just the search), but directly targets the root cause this whole
   session's results kept surfacing: a fixed linear reward/tardiness weighting is
   fundamentally hard to get right, no matter how it's searched.

## References

[1] J. Skalse, N. H. R. Howe, D. Krasheninnikov, D. Krueger, "Defining and Characterizing
Reward Hacking," arXiv:2209.13085, 2022.

[2] A. Pan, K. Bhatia, J. Steinhardt, "The Effects of Reward Misspecification: Mapping and
Mitigating Misaligned Models," ICLR 2022, arXiv:2201.03544.

[3] T. Eimer, M. Lindauer, R. Raileanu, "Hyperparameters in Reinforcement Learning and How
To Tune Them," arXiv:2306.01324, 2023.

[4] C. Tessler, D. J. Mankowitz, S. Mannor, "Reward Constrained Policy Optimization," ICLR
2019, arXiv:1805.11074.

[5] D. M. Roijers, P. Vamplew, S. Whiteson, R. Dazeley, "A Survey of Multi-Objective
Sequential Decision-Making," Journal of Artificial Intelligence Research, 48:67-113, 2013,
arXiv:1402.0590.

[6] J. Parker-Holder et al., "Automated Reinforcement Learning (AutoRL): A Survey and Open
Problems," Journal of Artificial Intelligence Research, 2022, arXiv:2201.03916. General
context for systematic RL agent tuning/diagnosis; not directly actioned in Section 7.

**Not formally verified (flagged, not cited as confirmed):** Several 2024-2025 multi-agent
DRL scheduling papers describing dense/incremental tardiness reward shaping (Section 6),
found via search but hosted on ScienceDirect without open-access full text -- summarized
from search-result snippets only, not read directly. If this becomes load-bearing for a
written conclusion, needs primary-source verification first.
