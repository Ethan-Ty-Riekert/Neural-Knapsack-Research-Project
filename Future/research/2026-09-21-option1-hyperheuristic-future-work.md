# Option 1: Where It Stands, and Hyper-/Meta-Heuristic RL as Future Work

**Date:** 2026-09-21 (S2W9)

**Scope:** Reviews Option 1's current, well-established status across offline and
online cases (pulled from `training-log.md`, not re-derived here), grounds its
"selection hyper-heuristic" framing precisely (Burke et al. 2013's taxonomy),
and proposes a concrete, literature-grounded next mechanism -- a genetic-
programming-evolved rule pool combined with RL selection -- as future work. This
is a research-scoping document, not an implementation plan; nothing here has
been built.

---

## 1. Option 1's current status, summarized

**Offline, fixed instance, dense+weighted, full-scale (1.2M timesteps) -- the
project's strongest validated result:**
```
LST (proven-optimal-matching)   8.00
Option 1                       11.00   <- 3 units behind LST, solidly beats EDF
EDF                            16.00
```
Starting point was ~1300 tardiness (seven independent mechanisms tried and
failed before the action-space-size diagnosis); this is a ~118x improvement,
attributable to three compounding fixes: action-space reduction (this design),
dense-tardiness reward, and real job weights (`training-log.md`, 2026-09-18/19
entries).

**Offline, randomized instance, dense+weighted, full-scale (1.2M):**
```
LST     23.94 raw /  68.62 weighted
EDF     37.30 raw / 109.36 weighted
Option 1  38.88 raw / 116.50 weighted   <- ties EDF, behind LST
```
The must-generalize case is harder than the can-memorize fixed-instance case at
the same reward design -- a real, not-yet-closed gap (2026-09-19 entry).

**Online, rho~0.75 (heavy load), dense+weighted:**
```
ATC (best heuristic)     644-648 weighted
Option 1, 300k           730.30 weighted   <- best online RL result, genuinely mixed rule-switching
Option 1, 900k           788-798 weighted  <- WORSE, confirmed 3x independently (2026-09-21 entry)
```
More training reliably makes the online case worse, not better -- the opposite
of every offline result. Root cause still open (entropy regularization ruled
out as both cause and fix, per the same entry); the 300k checkpoint remains the
best available online result and has not been beaten by any longer run tried so
far.

**Net picture:** Option 1 offline is close to optimal and the project's
strongest result overall; Option 1 online is competitive but not yet dominant,
and has a genuine, unresolved training-instability problem distinct from
anything seen offline.

## 2. What kind of method Option 1 actually is

Burke et al. (2013, *Hyper-heuristics: A survey of the state of the art*,
Journal of the Operational Research Society) draws the standard taxonomy this
section uses: hyper-heuristics split into **selection** (choose among a fixed
set of existing low-level heuristics) and **generation** (construct genuinely
new heuristics, typically via genetic programming over a grammar of scheduling
features). Option 1 is a **selection hyper-heuristic**, full stop -- it picks
from `Code/baselines/priority_rules.py::PRIORITY_RULES`'s 7 fixed classical
rules (EDF/SPT/LST/FCFS/LPT/WSPT/ATC) each tick. This was already the honest
framing `report.md` (the external critical review) and this project's own
`2026-09-17-action-space-reduction.md` §6.1 converged on independently: Option
1's ceiling is bounded by the best *achievable combination* of those 7 rules --
it can never discover a genuinely new priority function, only learn when to
defer to an existing one.

This project already has two directly relevant selection-hyper-heuristic
citations in `references.bib`:
- `lassoued2026hyperheuristic` (Lassoued et al. 2026) -- RL hyper-heuristic
  agent switching scheduling rules dynamically based on system state, with
  action-prefiltering and a switching-frequency commitment mechanism --
  directly analogous to Option 1's own design.
- `cie2025dispatchruleselection` -- another RL-based dispatch-rule-selection
  precedent (author list flagged unconfirmed in its own bib note).

Neither of these, nor Option 1 itself, is a generation hyper-heuristic. That
gap is real and is where the literature search below is aimed.

## 3. A concrete next mechanism: GP-generated rules + RL selection

Searched specifically for work combining rule *generation* with RL *selection*
for scheduling, rather than treating "go beyond a fixed rule menu" as a vague
aspiration. Two directly relevant results, verified via multiple independent
listings before citing:

**Chen, Bai, Qu, Dong & Jin (2024)**, `chen2024drlgphh` -- "Deep Reinforcement
Learning Assisted Genetic Programming Ensemble Hyper-Heuristics for Dynamic
Scheduling of Container Port Trucks" (IEEE Transactions on Evolutionary
Computation 29(4)). This is the precise pattern: genetic programming evolves a
**pool** of novel, problem-tailored priority rules from training instances
(the generation half), then a DRL agent **selects** among that evolved pool at
each scheduling decision point (the selection half -- structurally identical
to Option 1's existing mechanism, just operating over a GP-discovered rule set
instead of 7 hand-picked classical ones). Different problem domain (container
trucks, not cloud/job scheduling), but the architecture transfers directly.

**Xu, Mei, Zhang & Zhang (2025)**, `xu2025learntooptimise` -- "Learn to
Optimise for Job Shop Scheduling: A Survey with Comparison Between Genetic
Programming and Reinforcement Learning" (Artificial Intelligence Review 58(6)).
A direct survey of exactly the two paradigms Option 1 sits between -- useful
as the grounding citation for *why* GP and RL are the two dominant
learning-to-schedule paradigms in this literature, not just an arbitrary pairing.

### What this would concretely mean for this project

- **Generation phase (new, not yet attempted anywhere in this project):** run
  genetic programming over a grammar of this project's own per-job/per-machine
  features (duration, deadline, weight, resource demand, slack, remaining
  capacity -- the same features already in `GymSchedulingEnv._get_obs()`) to
  evolve a small pool (e.g. 5-10) of candidate priority functions on held-out
  training instances, analogous to how `Code/baselines/priority_rules.py`'s
  existing 7 rules were hand-derived from the classical scheduling literature,
  except machine-discovered instead.
- **Selection phase (reuse, not rebuild):** swap the evolved rule pool into
  `Code/env/rule_selection_gym_wrapper.py`'s `PRIORITY_RULES` in place of (or
  alongside) the 7 classical ones -- `RuleSelectionGymSchedulingEnv` and
  `Code/policies/action_branching_ppo_policy.py`'s sibling policy machinery
  need no structural change, since the mechanism (RL picks a rule index,
  `HEURISTICS[f"{rule_name}+FirstFit"]` decodes it) is already generic over
  the rule set's size and contents, not hard-coded to the current 7.
- **Honest scope note:** this raises the action space from 8 to roughly
  `pool_size + 1` (still additive, not multiplicative -- the core reason
  action-space-size mattered in the first place is unaffected). A pool of 10
  evolved rules is still a `Discrete(11)` space, well within the regime that
  worked for Option 1's original 8-choice design.

### Alternative/complementary direction: RL-guided metaheuristic search

A different angle from the same "hyper-heuristic vs. generation" question:
this project already has a metaheuristic baseline (`Code/baselines/pso.py`,
PSO) that independently rediscovered the reward-hacking problem earlier this
session (`PROGRESS.md` Phase 12) but was never extended past a single
gradient-free search pass. The broader 2024/2025 literature (general search
results, not yet individually verified/cited -- flagged as unconfirmed rather
than asserted) shows active work on RL *guiding* metaheuristic search
(e.g. RL-tuned neighborhood selection in variable neighborhood search, or
RL-controlled mutation/crossover in genetic algorithms) rather than RL and
metaheuristics competing as separate baselines the way PSO currently does in
this project. This is a real, citable direction but was not chased to a
specific verified paper in this session -- named here as a second candidate
future-work branch, less concretely scoped than the GP+RL generation-hyper-
heuristic direction above, which has two directly-matching precedents.

## 4. Recommended sequencing (not started, needs user scoping)

Consistent with this project's established process for larger design-surface
work (the two-critic PPO-Lagrangian rewrite, the multi-agent MARL proposal,
and Option 4's action-branching were all scoped this way before being built):

1. **Do not start GP rule generation unsupervised.** It is a genuinely new
   piece of infrastructure (this project has no existing GP implementation),
   distinct in kind from every action-space-reduction variant tried so far
   (which all reused or recombined existing machinery).
2. **If pursued, the offline case is the natural first target** -- it is
   already close to optimal (11.00 vs. LST's 8.00) and has none of the online
   case's unresolved instability, so a GP-evolved rule pool would be tested
   against a clean, well-understood baseline rather than confounded with the
   still-open online mystery.
3. **The online case's instability should likely be resolved (or at least
   better understood) before extending Option 1 there** -- adding a
   generation phase on top of a selection mechanism that's already unstable
   at longer training budgets would make it harder, not easier, to attribute
   any new result to the right cause.

## References

1. Burke, E. K., Gendreau, M., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., &
   Qu, R. (2013). "Hyper-heuristics: A survey of the state of the art."
   *Journal of the Operational Research Society*, 64(12), 1695-1724. DOI:
   10.1057/jors.2013.71. [`burke2013hyperheuristicsurvey`] -- selection-vs-
   generation hyper-heuristic taxonomy used throughout this doc.
2. Chen, X., Bai, R., Qu, R., Dong, J., & Jin, Y. (2024). "Deep Reinforcement
   Learning Assisted Genetic Programming Ensemble Hyper-Heuristics for
   Dynamic Scheduling of Container Port Trucks." *IEEE Transactions on
   Evolutionary Computation*, 29(4). DOI: 10.1109/TEVC.2024.3381042.
   [`chen2024drlgphh`]
3. Xu, M., Mei, Y., Zhang, F., & Zhang, M. (2025). "Learn to Optimise for Job
   Shop Scheduling: A Survey with Comparison Between Genetic Programming and
   Reinforcement Learning." *Artificial Intelligence Review*, 58(6). DOI:
   10.1007/s10462-024-11059-9. [`xu2025learntooptimise`]
4. Lassoued, S., Gobachew, A., Lier, S., & Schwung, A. (2026). "Policy-Based
   Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling
   Problems." arXiv:2601.11189. [`lassoued2026hyperheuristic`] -- already in
   `references.bib`, cited in `2026-09-17-action-space-reduction.md`.
