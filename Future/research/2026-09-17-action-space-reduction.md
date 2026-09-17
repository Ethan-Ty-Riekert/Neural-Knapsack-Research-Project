# Design + Results: Action-Space Reduction (Options 1/2/3)

**Date:** 2026-09-17 (S2W10)
**Environment:** `Code/env/rule_selection_gym_wrapper.py`, `Code/env/priority_only_gym_wrapper.py`,
`Code/policies/priority_pointer_policy.py`, `Code/policies/priority_pointer_ppo_policy.py`,
`Code/training/train_action_space_variant.py`, `Code/evaluation/eval_action_space_variant.py`
**Status:** All three options implemented, smoke-tested, and compared at reduced scale.
Option 1 confirmed as the strongest of the three both offline and online; the single
biggest tardiness improvement of the entire session. Full-curriculum (1.9M-timestep)
offline validation and further online tuning are deferred, not done here.

---

## 1. Motivation

Seven separate mechanisms were tried earlier this session to fix PPO's persistently
terrible offline tardiness (~1290-1330, next to CP-SAT/LST's proven floor of 8.0): a
fixed reward weight, four Lagrangian lambda_max ceilings, a tardiness-directed
hyperparameter search, a pointer-network architecture (`2026-09-16-pointer-network-ppo.md`),
and a comprehensive dense-per-tick reward redesign (`2026-09-17-dense-tardiness-reward.md`).
All seven converged on the same band. Comparing against DeepRM (Mao, Alizadeh, Menache,
Kandula, 2016, HotNets) and Decima (Mao et al., 2019, SIGCOMM) -- the literature that
actually reports 20%+ RL gains over heuristics -- surfaced one untested structural
difference: this project's action space is `max_jobs*num_machines+1` (1000+ at deployed
scale) vs. DeepRM's ~11 discrete choices. A large discrete action space is an
independently well-documented driver of policy-gradient sample-inefficiency (high-variance
gradients, poor exploration, softmax dilution over many rarely-taken actions).

User approved researching and implementing all three literature-grounded ways to shrink
it, rather than picking one blind, with Option 1 first and a ~30-minute comparison scale
used as a first-pass filter before committing to a full validation run.

## 2. The three designs

All three keep `SchedulingEnv`/`OnlineSchedulingEnv` and its feasibility/reward/tardiness
machinery completely unchanged (this project's "subclass, don't modify" convention) --
each is a Gym-wrapper-level reinterpretation of what the RL action means, built via
composition (wrapping an already-constructed `GymSchedulingEnv`/`OnlineGymSchedulingEnv`
instance) rather than inheritance, so each file is correct for both the offline and
online case without duplicating either's observation-building logic.

### Option 1 -- hyper-heuristic rule selection (`rule_selection_gym_wrapper.py`)
RL picks WHICH classical priority rule (`Code/baselines/priority_rules.py::PRIORITY_RULES`
-- EDF/SPT/LST/FCFS/LPT/WSPT/ATC) to apply this tick, paired with FirstFit placement.
Action space: `Discrete(8)` (7 rules + idle) -- reuses `Code/baselines/registry.py`'s
already-validated `HEURISTICS["{Rule}+FirstFit"]` `choose()` functions directly to decode
the chosen rule into a real `(job, machine)` placement, executed on the unmodified
underlying env. Falls through to idle when no rule can place anything. Observation
unchanged. Network: stock `MaskablePPO` `"MlpPolicy"` (flat MLP) -- no need for anything
fancier at this action-space size.

### Options 2 & 3 -- priority-only selection (`priority_only_gym_wrapper.py`, shared)
RL picks WHICH job slot to run next (DeepRM's actual pattern -- pick a job, not a
job+machine pair); placement is FirstFit (`Code/baselines/placement_rules.py`), not
learned. Action space: `Discrete(max_jobs+1)`. Network: `PriorityPointerActorCritic`
(`Code/policies/priority_pointer_policy.py`) -- adapted from the existing
`PointerActorCritic` (`Code/policies/pointer_policy.py`), reusing its shared-weight
`JobEncoder`/`MachineEncoder` unchanged but replacing `CompatibilityScorer` (which scores
every job x machine PAIR, sized for the now-bypassed action space) with `JobScoreHead`,
producing one logit per job slot directly, since placement is no longer a policy
decision. Wired into `MaskablePPO` via `PriorityPointerMaskableActorCriticPolicy`
(`priority_pointer_ppo_policy.py`), mirroring the existing `PointerMaskableActorCriticPolicy`
pattern exactly (same four SB3-called methods overridden).
- **Option 2** (`use_atc=False`): job-slot features are the existing raw per-job
  attributes (duration, deadline, weight, resource demand) -- network learns priority
  ordering end-to-end from raw features, matching DeepRM/Decima's own design.
- **Option 3** (`use_atc=True`): identical, plus one appended per-slot feature: the ATC
  composite priority index (`Code/baselines/priority_rules.py::atc_priority` -- factored
  out of `atc_key` the same day specifically so the raw, un-negated score could be reused
  here as an observation feature rather than only as a baseline-ranking key).

Honest scope note: `max_jobs+1` (~101 at the offline deployed scale) is a >10x reduction
from 1000+, not a literal replication of DeepRM's much smaller bounded-window action
space -- windowing to a smaller M is a natural follow-up, not attempted here.

## 3. A real bug found during evaluation, not training

`model.predict()` called directly on a single non-batched observation (as
`eval_action_space_variant.py` does) can return a 0-d numpy ndarray rather than a python
int or numpy scalar. This tolerates `==` comparisons and list-indexing (why
`RuleSelectionGymSchedulingEnv.step()` never noticed it) but is NOT hashable, which
crashed `PriorityOnlyGymSchedulingEnv.step()`'s `job in self.env.remaining_jobs`
set-membership check with `TypeError: unhashable type: 'numpy.ndarray'`. Fixed with an
explicit `action_id = int(action_id)` cast at the top of both wrappers' `step()`, with a
regression test added (`tests/test_action_space_wrappers.py`) verifying a 0-d ndarray
action doesn't crash either wrapper.

## 4. Training/eval protocol

Standalone trainer (`Code/training/train_action_space_variant.py`), deliberately NOT
wired into `train_optimized.py`'s curriculum/Optuna/RCPO machinery -- first-pass
comparison runs only, un-tuned `MaskablePPO` defaults, ~300k timesteps (~30 min,
matching this session's own established "quick check" scale). Supports both the offline
fixed instance and (added later the same day, for the heavy-tailed follow-up) the online
case via `--online`/`--arrival-rate`/`--job-size-distribution`, with a fresh realized
arrival sequence resampled every episode for online training (not memorized).

## 5. Results

### 5.1 Offline fixed instance (100 jobs, 10 machines, horizon=100)

```
                              tardiness  late  scheduled
CP-SAT (proven optimal)           8.00     -          -
LST                                8.00     7     99/100
EDF                               16.00    10     98/100
Option 1 -- 600k steps            19.00    11    100/100
Option 1 -- 300k steps            35.00    10    100/100
ATC                              106.00    14     96/100
Option 3 (ATC-primed priority)   152.00     8    100/100
Option 2 (raw-feature priority)  525.00    16    100/100
Historic PPO (7 prior fixes)   ~1290-1330     -          -
SPT                             1321.00    43     92/100
```

Every one of the three options dramatically beats the historic PPO band -- Option 1 by
~37x at 300k steps, ~68x at 600k. Ranking matches the theory exactly: smaller/more
structured action space wins (Option 1's `Discrete(8)` < Option 3's `Discrete(101)`+ATC
feature < Option 2's plain `Discrete(101)`). Doubling Option 1's training (300k->600k)
roughly halved tardiness again (35->19) with no sign of plateauing, closing in on EDF
(16.00) -- see Section 6 for what a full-scale run might do.

### 5.2 Online, heavy-tailed (log-normal) arrivals, rho~0.25 (309 realized, seed=0)

Option 1 matches every one of 9 `DEFAULT_HEURISTICS` exactly (tardiness=36.00, late=2,
scheduled=298/309) -- this instance turned out to be in the "no differentiation" regime
found in the heuristic-only sweep (`2026-09-17-heavy-tailed-arrivals.md` Section 6): all
failures trace to the same 11 jobs that are unschedulable by ANY policy regardless of
skill (heavy-tailed duration + late arrival = structurally impossible), so there was no
room here to show whether Option 1 could actually out-schedule anyone.

### 5.3 Online, heavy-tailed arrivals, rho~0.75 (864 realized, seed=0) -- the genuinely hard regime

```
                     tardiness  late  scheduled
CP-SAT (hindsight)     105.00     -          -   (best_bound, UNKNOWN status -- NP-hard at scale)
ATC                     120.00    14     799/864  <- best live policy
Option 1                163.00    15     799/864  <- IDENTICAL to SPT
SPT                     163.00    15     799/864
WSPT+BestFit            192.00    19     796/864
EDF                     193.00    25     797/864
EDF+BestFit             235.00    23     792/864
Tetris                  248.00    18     799/864
LST                     275.00    28     789/864
FCFS+FirstFit           294.00    23     793/864
LPT+WorstFit            634.00    34     759/864  <- worst
```

This is the most honest result of the session: Option 1 (300k timesteps, no
curriculum/tuning) beats 6 of 8 other heuristics but **loses to ATC** (163 vs 120) and
sits ~55% above the CP-SAT oracle's hindsight floor vs. ATC's ~14%. Option 1's numbers
are *exactly* identical to SPT's, not just close -- strong evidence the learned policy
converged onto a single classical rule (SPT) rather than discovering a genuinely novel
strategy, unlike the offline case where it clearly outperforms every single-rule
baseline. RL does not automatically dominate in the harder, literature-grounded online
setting the way it did offline, at least not yet at this training budget.

## 6. Conclusion

The action-space-size hypothesis is confirmed as the dominant fix for the offline case --
a genuine root cause, not another incremental tweak, and the first mechanism this session
that actually moved the needle after seven failed attempts. Option 1 (hyper-heuristic
rule selection) is the clear winner among the three designs in every setting tested. The
online picture is more nuanced and not yet settled: at the load level where heuristics
actually differentiate (rho~0.75), Option 1 is competitive but not yet better than the
best classical heuristic, and shows signs of having only learned to imitate one rule
rather than synthesizing a new one.

### 6.1 Honest limitation: Option 1's ceiling is bounded by its own rule menu

User-raised critique (2026-09-17, same day): Option 1's action space is literally "pick
one of 7 pre-written heuristics" -- its ceiling is bounded by whatever the best
*achievable combination* of those 7 rules is; it can never discover a genuinely new
priority function, only learn when to defer to an existing one. This is a real
limitation, not a nitpick, and it's sharpened by Section 5.3's finding: Option 1 didn't
even reach the interesting version of this (adaptive rule-switching conditioned on
system state) -- it converged onto picking SPT statically. That's closer to a
hyperparameter search over 7 discrete options than to learning, and it's grounded in the
same hyper-heuristic/dispatching-rule-selection literature this design borrows from
(`lassoued2026hyperheuristic`, `cie2025dispatchruleselection` -- both found during this
session's earlier research but never previously recorded in this doc; see
`references.bib`). Options 2/3 (learned, continuous per-job priority scoring from raw
features) are the actually-novel-strategy-capable designs; Option 1's offline win, while
real, is a meta-selection result, not evidence RL learned new scheduling behaviour.

## 7. Next steps (not done here, tracked for follow-up)

- **Bounded action-space window for Options 2/3** (DeepRM's actual trick, not yet
  implemented here): cap the number of choosable job slots at a small constant M
  (~10-20, `mao2016deeprm`'s own scale) with a scalar "backlog count" summarizing
  everything beyond the window, instead of `Discrete(max_jobs+1)` (~101-400+ depending on
  scale). Highest-leverage remaining lever for Options 2/3 specifically -- could shrink
  their action space by another 5-10x on top of what's already done here.
- **A learned placement head via a factored/hierarchical action space**
  (`tavakoli2018actionbranching`'s general pattern), as a fallback if neither more
  training nor the windowing above is enough: decouple job selection and machine
  placement into two sequential small discrete choices (additive |J|+|M|, not
  multiplicative |J|x|M|) instead of Options 2/3's current fixed-FirstFit placement --
  lets RL learn placement too, without paying the full joint action space's cost. Named
  explicitly as the thing to reach for "if nothing [else] works" (user, 2026-09-17).
- More training for Option 3 specifically, offline (continue past 300k) and online
  rho~0.75 (untested there entirely) -- queued as the immediate next experiment, on the
  reasoning in Section 6.1 that Options 2/3, not Option 1, are the path to genuinely
  *beating* ATC rather than matching/approaching it via rule selection.
- Full 1.9M-timestep, full-curriculum offline validation of Option 1 (~6hr estimated
  cost at observed throughput -- deferred, not committed to yet).
- More training/tuning on the online rho~0.75 heavy-tailed case for Option 1, mirroring
  the offline 300k->600k improvement, before concluding anything final about RL vs. ATC
  there.

## 8. References

See `references.bib` for full entries (keys noted below); this section stays as a quick
in-context pointer per CLAUDE.md's documentation convention.

1. Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). "Resource Management with
   Deep Reinforcement Learning." *HotNets 2016*. [`mao2016deeprm`]
2. Mao, H., et al. (2019). "Learning Scheduling Algorithms for Data Processing
   Clusters." *SIGCOMM 2019*. [`mao2019decima`]
3. Kool, W., van Hoof, H., & Welling, M. (2019). "Attention, Learn to Solve Routing
   Problems!" -- scaled-dot-product + tanh-clipping compatibility scoring, reused
   unchanged from `pointer_policy.py`'s `CompatibilityScorer`/`JobEncoder` design.
   [`kool2019attention`]
4. Vepsalainen, A. P. J., & Morton, T. E. (1987). "Priority Rules for Job Shops with
   Weighted Tardiness Costs." *Management Science*, 33(8) -- the ATC rule used in Option
   3's feature and Option 1's rule set. [`vepsalainen1987atc`]
5. Lassoued, S., Gobachew, A., Lier, S., & Schwung, A. (2026). "Policy-Based Deep
   Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems." arXiv:2601.11189
   -- directly analogous rule-switching hyper-heuristic design to Option 1; grounds
   Section 6.1's framing. [`lassoued2026hyperheuristic`]
6. "Dynamic Dispatching Rule Selection for the Job Shop Scheduling Problem." *Computers
   and Industrial Engineering* (2025), DOI: 10.1016/j.cie.2025.111471 -- dispatching
   rule-selection systems beat the single best fixed rule by >10%; author list
   unconfirmed (see `references.bib`'s note). [`cie2025dispatchruleselection`]
7. Tavakoli, A., Pardo, F., & Kormushev, P. (2018). "Action Branching Architectures for
   Deep Reinforcement Learning." arXiv:1711.08946 -- the general factored/hierarchical
   action-space pattern behind Section 7's learned-placement-head fallback.
   [`tavakoli2018actionbranching`]
