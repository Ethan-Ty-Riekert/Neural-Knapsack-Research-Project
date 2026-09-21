# Training Log

Chronological record of training runs, what changed since the previous entry, the
resulting stats, and what was concluded from them. This is a running log of results
across the project, separate from the dated deep-dive write-ups in this folder (which
investigate one specific problem in depth and are linked from the relevant entry below).

Newest entries at the top. Each entry should be appended, not edited retroactively --
if a conclusion turns out to be wrong, say so in a later entry rather than rewriting
history.

## Template for new entries

Week label is `S2W<n>` (Monday-Sunday weeks, `S2W1` = 2026-07-20 -- see CLAUDE.md
for the formula). Put it in the header next to the date.

```
## YYYY-MM-DD (S2WN) -- <short description>

**Config:** algo, curriculum/stage, key hyperparameters (only what changed since the
previous entry, or "unchanged" if nothing did)

**Stats:**
\`\`\`
<paste relevant rollout/train stats>
\`\`\`

**Observation:** what the stats show / what changed vs. the previous entry

**Conclusion / next step:** what this means and what to try next
```

---

## 2026-09-22 (S2W9) -- Windowed online FIFO-ordering result: confirms the offline finding -- EDF-ordering wins in both cases, confound question fully closed out

**Context:** Completes the windowed-ordering confound investigation. Same config
as the original EDF-ordered windowed Option 3 online result, `--window-order
fifo` instead of the default.

**Stats:**
```
Option 3 (window=15, FIFO, online)  weighted_tardiness=1190.48+/-496.07  late=31.00  scheduled=826.90/898
Option 3 (window=15, EDF, online, 2026-09-20 entry)  weighted_tardiness= 952.74+/-370.59  (for comparison)
ATC (best heuristic, online)         weighted_tardiness= 648.16+/-338.30
```

**Observation:** FIFO ordering (1190.48) is worse than EDF ordering (952.74)
online too -- the same direction as the offline result (FIFO 2869.10 vs. EDF
105.86), just a smaller relative gap here since the EDF-ordered online result
was already a clear regression itself (second-worst of nine methods, per the
2026-09-20 entry), not a strong baseline to fall further from. Both cases now
agree: EDF-ordering outperforms FIFO-ordering, confirming it is genuinely
informative scaffolding rather than a ceiling-imposing bias, consistent across
offline and online.

**Conclusion: this confound investigation is now fully closed out.** Across
both the offline and online cases, across two independent training runs each,
EDF-ordering beats FIFO-ordering decisively. The original 2026-09-18 design
choice (window candidates ordered by earliest deadline) was the right one, not
an accidental constraint worth revisiting -- an alternative ordering carrying
less scheduling-relevant information makes results worse, not better. The
windowed action space's REMAINING open problem (both EDF-ordered variants
still underperform the unwindowed Options 2/3 at the same training budget) is
therefore NOT explained by window-ordering -- it's most likely genuinely an
undertraining or window-size effect, not a design flaw in the ordering choice
itself. If this thread is revisited, a same-timestep-budget comparison against
unwindowed Options 2/3, or a larger-scale windowed run, are the next things
worth trying -- not a different ordering.

---

## 2026-09-22 (S2W9) -- Precise timing of the entropy_loss collapse -- refines (not reverses) the previous entry

**Context:** The previous entry corrected the 2026-09-21 "argmax locks in before
entropy fully decays" claim as conflating two different metrics. Checked the
exact step-by-step SB3 entropy_loss trajectory from this run's log (not just
start/end snapshots) before letting that correction stand as stated, since it's
itself a precision claim worth verifying rather than asserting from two data
points.

**Stats (evenly-sampled entropy_loss readings across the full 900k-step run):**
```
step   4096: -2.040   step 262144: -0.160   step 605000: -0.0013
step  90000: -1.040   step 348160: -0.100   step 720000: -0.0003
step 175000: -0.196   step 462848: -0.066   step 835000: -0.00008
step 233472: -0.118   step 548864: -0.018   step 900000: -0.00006  (final)
```

**Observation:** SB3's own per-state entropy_loss declines SMOOTHLY and
essentially monotonically across the entire run -- this confirms the ORIGINAL
2026-09-21 entry was right about the shape (a gradual, continuous decline, not
a sudden late event). It crosses into near-total saturation (below -0.02 or so)
around step ~550k-600k, and is numerically indistinguishable from exactly zero
by 900k. What the previous entry's correction actually applies to is narrower
and still stands: comparing THIS metric's trajectory against this project's
own `action_dist/entropy_normalized` (a different quantity -- aggregate
cross-state action diversity, not per-state uncertainty) to conclude something
about "the argmax converging before the full distribution does" was not a valid
comparison between the two metrics in the first place, independent of exactly
when either one bottoms out. The precise, now-verified picture: per-state
entropy declines gradually and smoothly for most of the run (consistent with
the original framing), reaching near-complete saturation only in roughly the
run's final third.

**Conclusion / next step:** No change to the open questions already flagged
(what drives the logits toward saturation in the first place; why the entropy
bonus's diminishing gradient at saturation isn't caught earlier). This entry
exists purely to make sure the corrected record is itself precise, not just
directionally right -- matching this project's own rigor convention of
distinguishing what was actually measured from what was inferred.

---

## 2026-09-22 (S2W9) -- Policy-confidence diagnostic: per-state entropy collapsed to near-EXACTLY zero, not gradually -- corrects the earlier "argmax locks in before entropy decays" framing, and explains why ent_coef failed

**Context:** Direct follow-up to the congestion-adaptivity negative result above,
which flagged "examine the value function's landscape" as the next step. Built
`Code/evaluation/diagnose_policy_confidence.py` to extract the FULL masked
probability distribution (not just the argmax action) and the value estimate
V(s) at every step of 6 deterministic online rollouts (5591 decisions) against
the same SPT-collapsed checkpoint.

**Stats:**
```
SPT probability:  mean=0.8916, std=0.3109, min=0.0000, max=1.0000
Margin (top1 - top2 probability): mean=1.0000, std=0.0000, min=1.0000, max=1.0000
  -- EXACTLY 1.0 on every single one of 5591 decisions, zero exceptions.
Value estimate V(s): mean=-6.81, std=1.44, min=-13.12, max=-1.99
  -- a real, meaningful spread, not collapsed.
corr(SPT probability, congestion)  = +0.073  (essentially no linear relationship)
corr(value estimate, congestion)    = +0.306  (weak positive, not yet investigated further)

Cross-checked against SB3's OWN entropy_loss metric (the standard PPO per-state
policy-entropy term, distinct from this project's custom action_dist/
entropy_normalized diagnostic -- see below) from this exact training run's log:
  step ~2k:   -2.04   (healthy, high per-state entropy)
  step ~900k: -6.0e-05  (essentially EXACTLY zero)
```

**Observation -- a real methodological correction, not just a new data point.**
The margin being EXACTLY 1.0 on every single decision means the policy's
distribution is fully deterministic (one-hot, p~1.0 on one action) at
essentially every individual state it was evaluated on -- not "mostly
confident," completely saturated. SB3's own entropy_loss confirms this
independently: it collapsed to -6.0e-05 (not just small -- numerically
indistinguishable from exactly zero) by 900k steps.

This means the 2026-09-21 entry's framing -- "entropy decays smoothly across the
whole run... the argmax action converges before the full distribution does" --
was based on conflating two DIFFERENT quantities that this project's own
`action_dist/entropy_normalized` diagnostic and SB3's `entropy_loss` measure.
`entropy_normalized` (this project's custom callback) computes Shannon entropy
over a HISTOGRAM of which actions get chosen, AGGREGATED across many different
states within a logging window -- it stays non-trivial (~0.164-0.165 by the
end) as long as DIFFERENT states still deterministically resolve to DIFFERENT
actions (mostly SPT, sometimes idle), even if each individual state's own
decision has zero uncertainty. SB3's entropy_loss is the PER-STATE distribution's
own entropy, averaged across samples -- and that collapsed to essentially
exactly zero. The earlier "argmax locks in before entropy fully decays" claim
had this backwards: per-state entropy collapsed completely and fast, and what
looked like "residual entropy" late in training was actually cross-state action
diversity (the policy still picking DIFFERENT deterministic actions in
DIFFERENT states), not within-state uncertainty. This entry doesn't edit that
2026-09-21 entry (per this file's own rule) -- it supersedes that specific claim.

**A mechanistic explanation this finding provides for why `--ent-coef 0.01`
didn't fix the collapse (2026-09-19 entry):** the standard entropy bonus's
gradient with respect to the policy's logits vanishes as the underlying
probability approaches 0 or 1 (d(entropy)/d(logit) -> 0 at saturation) -- so
entropy regularization becomes progressively LESS effective exactly as a
policy approaches the kind of per-state determinism found here, and once
several states' distributions saturate, the aggregate entropy bonus computed
across a batch containing both saturated and unsaturated states has
diminishing power to pull the saturated ones back. This is consistent with,
not contradicting, the earlier finding that ent_coef=0.01 didn't prevent the
collapse -- it's now clearer WHY a fixed-coefficient entropy bonus alone
wouldn't be expected to, mechanistically, not just empirically.

**Also notable, not yet chased further:** the value function (critic) is NOT
collapsed -- V(s) has a real, substantial spread (std=1.44, range [-13.12,
-1.99]) -- so this is specifically an ACTOR-head saturation, not a general
"the network stopped learning" story. The weak positive value-vs-congestion
correlation (+0.306) is also unexplained and worth a closer look if this thread
is picked up again (a naive expectation would be higher congestion -> worse,
more negative value, not better -- possibly confounded with time-within-episode
rather than a direct congestion effect, not yet separated).

**Conclusion / next step:** The "why does online training collapse" mystery now
has a clearer mechanistic picture (per-state logit saturation, entropy bonus
loses effectiveness exactly when needed) even though the ultimate root cause
(what specifically drives the logits toward saturation in the first place,
starting this early and this completely) remains open. A concrete, literature-
groundable next step if pursued: techniques specifically designed to prevent
policy saturation independent of the standard entropy bonus's vanishing-
gradient problem -- e.g. KL-divergence trust-region penalties, logit
clipping/max-entropy regularization applied directly to pre-softmax logits
rather than the post-softmax entropy term, or simply a much larger ent_coef
applied EARLY (before saturation begins) rather than throughout. Not
implemented -- flagged for the next session or whoever picks this thread up,
consistent with this project's process for design decisions with real
training-budget cost.

---

## 2026-09-22 (S2W9) -- Windowed offline FIFO-ordering result: EDF-ordering is essential scaffolding, not a biasing ceiling -- the confound question answered decisively

**Context:** Direct test of whether the windowed action space's EDF-ordering
design was pre-biasing results toward EDF-like performance (the open question
from the 2026-09-18/20 windowed-offline entries). Same config as the original
EDF-ordered windowed Option 3 offline result (300k, dense+weighted, window_size
=15), just `--window-order fifo` instead of the default `edf`.

**Stats:**
```
Option 3 (window=15, FIFO order)  tardiness=1033.76+/-134.71  weighted_tardiness=2869.10+/-491.58  late=43.16  scheduled=97.10/100
Option 3 (window=15, EDF order, 2026-09-18/20 entries)         weighted_tardiness=105.86+/-137.35   (for comparison)
SPT (worst-case reference)        tardiness=1150.86+/-154.91  weighted_tardiness=3447.56+/-581.72
EDF                                tardiness=  37.30+/- 60.43  weighted_tardiness=  109.36+/-175.73
```

**Observation: the opposite of what the original hypothesis predicted.**
The framing going in was "EDF-ordering might be BIASING the policy toward
mediocre, EDF-like performance -- worth testing whether removing it helps."
Instead, removing it (FIFO/job-index order, which carries zero scheduling-
relevant information by construction) caused the training run to collapse to
near-SPT-level performance (2869.10 weighted vs. SPT's 3447.56) -- close to the
WORST end of every method compared, not an improvement or even a neutral
result. This flips the earlier framing: EDF-ordering was never primarily a
biasing ceiling holding the policy back from doing better -- it was essential
scaffolding. Bounding the visible window to the M EDF-nearest jobs hands the
policy a genuinely informative, pre-filtered candidate set; bounding it to an
arbitrary M-job window (FIFO/index order) instead removes that signal entirely,
and the policy apparently cannot recover the equivalent information on its own
within this training budget -- landing close to what a policy with NO priority
signal at all would achieve.

**Conclusion / next step:** The offline half of this confound question is now
answered, not just narrowed: the windowed design's near-EDF-matching results
found on 2026-09-18/20 were the GOOD outcome, not a ceiling imposed by the
design -- removing the informative ordering makes things dramatically worse.
This means the earlier "maybe an alternative ordering would let the policy do
better" framing was based on an incorrect assumption; the real lesson is that
the window's candidate-selection key needs to carry real scheduling
information (EDF, or potentially a learned/raw-priority-score ordering, not
FIFO) to be useful at all at this training budget. Not yet known whether the
same holds online -- that run is still in progress (2026-09-22 entry above
this one predates it; a follow-up entry will report the online FIFO result
once it lands).

---

## 2026-09-22 (S2W9) -- Direct test of the congestion-adaptivity hypothesis: NOT supported -- collapse looks unconditional, not state-dependent

**Context:** Immediate follow-up to the entry below, which proposed that the
online SPT-collapse might be a congestion-ADAPTIVE local optimum (SPT chosen
specifically under heavy load, per its flow-time-optimality). Built a direct
test (`Code/evaluation/diagnose_rule_choice_congestion.py`) rather than leaving
this as an untested hypothesis -- ran the SPT-collapsed checkpoint
deterministically over 10 fresh online held-out episodes (9332 total decisions),
logging the chosen rule and the congestion level (|remaining_jobs|) at every
step.

**Stats:**
```
Rule frequency: SPT 89.2% (8322/9332), idle 10.8% (1010/9332) -- EVERY OTHER
  RULE (EDF/LST/FCFS/LPT/WSPT/ATC) chosen 0% of the time, zero exceptions
  across all 10 episodes.
Congestion range: min=0, max=97, mean=25.3, median=23.0 -- a genuinely wide
  range of load levels sampled across the 10 episodes.
Mean congestion when SPT chosen: 25.7 -- essentially indistinguishable from
  the OVERALL mean (25.3), which is itself dominated by SPT-choice steps
  since they're 89% of all decisions.
```

**Observation: the congestion-adaptivity hypothesis is NOT supported.** If SPT
were being invoked specifically as a congestion-response strategy (the
Little's-Law-based mechanism proposed below), rule choice should correlate with
load level -- SPT more likely when congestion is high, something else (or idle)
more likely when it's low. Instead, SPT dominates essentially UNCONDITIONALLY
across the full observed congestion range (0 to 97) -- the policy shows no
adaptive variation, just a near-total collapse onto one action regardless of
state. This points back toward a genuine training-dynamics collapse (closer to
the original entropy-decay framing) rather than the policy having discovered a
real, state-conditioned scheduling strategy that happens to coincide with SPT's
classical strength.

**A flaw in this script's own secondary test, caught and corrected rather than
left standing:** the script also checked whether PLACED jobs' weights exceed the
feasible-job median (61.1% did, vs. 50% expected under a weight-blind rule) as a
proposed "is the policy weight-aware" signal. This is not actually meaningful:
Option 1's RL policy only selects a RULE, not an individual job -- once "SPT" is
chosen, WHICH job gets placed is determined entirely by SPT's own fixed,
duration-only formula (`priority_rules.py::spt_key`), which does not read job
weight at all by construction. Any placed-job-weight-vs-median signal reflects
incidental duration-weight correlation in this project's random instance
generation, not anything the RL policy decided. The actually meaningful,
already-visible evidence of weight-blindness is simpler and cleaner: WSPT and
ATC (the two weight-aware rules in the menu) were chosen 0% of the time across
9332 decisions -- not "partially," never.

**Conclusion / next step:** The literature-grounded explanation proposed below
does not survive direct testing in its strong ("SPT is adaptively invoked under
congestion") form -- recording this as a genuine negative result, not quietly
dropping the hypothesis. The online collapse looks more like an unconditional,
state-independent training-dynamics failure than a coherent (if incomplete)
congestion-management strategy. This narrows, not widens, the space of
remaining explanations -- the next diagnostic worth trying is tracking the
VALUE FUNCTION's landscape (does it assign similar values to SPT vs. non-SPT
choices across the state space, or does it also show the same
state-independence?) rather than further action-distribution analysis, which
this entry and the two before it have now used to rule out several specific
mechanisms (entropy regularization, congestion-adaptivity) without yet finding
the actual cause. Not implemented -- flagged as the next step for whoever picks
this up next, autonomous or otherwise.

---

## 2026-09-22 (S2W9) -- A partial, literature-grounded explanation for "why SPT specifically" in the online collapse (analysis of already-collected data, no new training)

**Context:** Continuing the still-open online "more training hurts" mystery while
the user is away -- specifically the unanswered question from the 2026-09-21
entry: WHY does the online policy converge onto SPT specifically, not some other
rule or a genuinely mixed strategy? This entry doesn't resolve the instability
itself, but gives a grounded, testable partial answer to the "why SPT" half.

**Hypothesis:** SPT is not an arbitrary rule to collapse onto -- it is the
provably flow-time-optimal single-machine sequencing rule (Smith 1956,
`smith1956wspt` in `references.bib`; unweighted SPT is the w_j=1 special case of
Smith's more general weighted-completion-time result already cited in this
project for WSPT). By Little's Law (L = lambda*W, mean-number-in-system equals
arrival rate times mean waiting time), minimizing mean flow time also minimizes
mean waiting time under a fixed arrival rate -- a real, if partial, mechanism for
why a policy searching for high reward under SUSTAINED HEAVY LOAD (rho~0.75, this
project's online hard-congestion regime) could find "clear the queue as fast as
possible" (SPT-like behaviour) locally attractive, even though SPT ignores
deadlines and weights entirely and is NOT tardiness-optimal.

**Evidence check against data already collected tonight (2026-09-21 online eval,
50-instance protocol) -- not new training, just re-reading what's already there:**
```
Option 1 (900k, SPT-collapsed)   weighted_tardiness=798.46
SPT                                weighted_tardiness=798.46  (identical, as established)
WSPT+BestFit                       weighted_tardiness=709.42  <- meaningfully better than SPT
ATC                                weighted_tardiness=648.16  <- best (degenerates toward WSPT
                                                                   under low slack, per its own
                                                                   formula's documented design,
                                                                   vepsalainen1987atc)
```
WSPT (weighted SPT) meaningfully beats plain SPT under the identical load
conditions, and ATC -- which explicitly interpolates toward WSPT as slack shrinks
-- beats both. This is a coherent picture, not just three arbitrary heuristic
numbers: the RL policy appears to have found a real, theoretically-grounded local
optimum (minimize queue length / congestion) but stopped short of the natural
refinement (also weight that by job importance), landing on a worse-but-related
strategy than the ones that add weight-awareness on top.

**What this does NOT explain (flagged honestly, not glossed over):** why the
deterministic (argmax) policy locks in and plateaus BEFORE the training-time
entropy metric finishes decaying (the 2026-09-21 entry's other open finding);
why more training pushes TOWARD this local optimum rather than past it to
something closer to WSPT/ATC; and this is a plausibility argument from existing
classical scheduling theory, not a mechanistic proof about THIS specific
network's training dynamics -- it explains why SPT is a coherent thing to
converge to, not why the specific optimization process converges there.

**Conclusion / next step:** A concrete, directly testable follow-up if pursued:
since job weights are the one ingredient distinguishing SPT from the
better-performing WSPT/ATC, and the reward function already includes weighted
tardiness, the gap suggests the policy isn't learning to CONDITION its urgency
response on job weight under heavy load, specifically -- a diagnostic worth
adding to the training-diagnostics tooling (e.g., correlating chosen-rule
frequency against the current backlog/congestion level AND against the
weight-distribution of jobs in the window, not just an aggregate entropy number)
would directly test whether the policy's rule choice is congestion-dependent (as
this hypothesis implies) or genuinely state-independent. Not implemented --
flagged as the next diagnostic to build, not launched unsupervised given it's a
new piece of instrumentation, not just a training-config variation.

---

## 2026-09-21 (S2W9) -- Option 1 review + hyper-/meta-heuristic RL future work (research doc, no code changes)

**Context:** User-requested review of Option 1's current status plus a literature-
grounded look at future work in the hyper-heuristic/meta-heuristic-with-RL space.
Full write-up: `2026-09-21-option1-hyperheuristic-future-work.md`.

**Summary:** Option 1 offline (1.2M, dense+weighted, fixed instance) is the
project's strongest result -- 11.00 tardiness, 3 units behind LST's proven-optimal-
matching 8.00, solidly ahead of EDF's 16.00. Randomized-instance offline ties EDF
(116.50 weighted) but doesn't beat LST. Online is unresolved: 300k (730.30
weighted) is the best result found so far and has not been beaten by any longer
run; 900k reliably regresses (confirmed 3x independently, see the 2026-09-21 entry
above), root cause still open.

Grounded Option 1's framing precisely via Burke et al. (2013)'s hyper-heuristic
taxonomy: it is a SELECTION hyper-heuristic (picks among 7 fixed classical rules),
never a GENERATION one (cannot invent new priority functions) -- confirming
report.md's and this project's own earlier framing, now with the canonical
citation behind it. Found two directly relevant precedents for the natural next
mechanism (not yet built): Chen et al. (2024)'s DRL-GPHH pattern -- genetic
programming evolves a pool of novel priority rules, RL selects among that pool at
each decision point (structurally identical to Option 1's existing selection
mechanism, just over a richer, machine-discovered rule set instead of 7
hand-picked ones) -- and Xu et al. (2025)'s direct GP-vs-RL survey for job shop
scheduling, grounding why these are the two dominant paradigms in this literature.

**Conclusion / next step:** Not started -- explicitly scoped as needing user
input before building (this project's established process for genuinely new
infrastructure, here a GP rule-generation phase this project has never
implemented). Recommended sequencing if pursued: target the offline case first
(already near-optimal, no confounding instability), and likely resolve or better
understand the online instability before extending Option 1 there, so a new
generation phase isn't confounded with the still-open online mystery. Also named
(less concretely scoped, no specific paper verified) RL-guided metaheuristic
search as a second candidate direction, connecting to this project's existing
but never-extended PSO baseline.

---

## 2026-09-21 (S2W9) -- Option 4 detach-fix result: stability confirmed fixed, but still short of the original simplest design -- closing out this arc

**Context:** Third and (for now) final Option 4 pooling variant this session. Same config as both prior attempts (300k, dense+weighted).

**Stats -- all three variants, same protocol, side by side:**
```
                                         weighted_tardiness   std     late    scheduled
Option 4 (original, flat-mean pooling)         355.64        358.80   31.74   96.10/100
Option 4 (weighted, undetached)                795.48        495.38   40.46   92.26/100
Option 4 (weighted, detached)                  558.64        197.53   35.88   94.80/100
EDF                                             109.36        175.73   12.22   96.84/100
ATC                                             298.80        184.13   11.10   94.68/100

Training trajectory shape:
  undetached: wildly oscillating (0 -> 5800 -> 0 -> 2930 ... ended at 1120)
  detached:   smooth, steadily declining (3140 -> ... -> 538), std of the FINAL eval
              (197.53) less than half the undetached version's (495.38)
```

**Observation:** The detach fix did exactly what it was designed to do -- training
stability is unambiguously better (smooth monotonic-looking decline vs. wild
oscillation, final eval std cut by more than half) -- confirming the hypothesis that
the job_score_head dual-role gradient entanglement was a real source of instability,
not a red herring. But stability alone wasn't enough: the detached run's best point
across its entire trajectory (~517) never reached the ORIGINAL simplest design's final
result (355.64), let alone the undetached run's brief mid-training peak of exactly 0.
This suggests a real trade-off rather than a strictly-better fix: the entangled
gradient path the detach removes was unstable, but it may also have been carrying real
signal that helped reach better optima when it happened to work -- removing it
stabilized training onto a WORSE ceiling, not just a safer path to the same one.

**Conclusion / next step: closing out this specific investigation arc, not chasing a
4th variant blind.** Three attempts (flat mean, weighted+undetached, weighted+detached)
across 900k total offline timesteps have not produced a version of Option 4 that beats
the simplest original design at matched budget, let alone the fixed-FirstFit Options
1/3. The honest state of the evidence: action-branching's LEARNED placement has not yet
demonstrated a real advantage over FirstFit in this project, across three genuinely
different pooling designs, though the undetached run's mid-training peak (weighted=0)
is a real existence proof that a much better optimum is reachable by SOME policy in
this action space -- just not yet reliably found by training. Flagging this as a
decision point for the user rather than unilaterally launching a fourth variant or a
larger-scale run: is this thread worth more compute (e.g. a longer/larger-scale run to
see if either weighted variant's ceiling rises with more budget, matching the pattern
that worked for Options 1/2/3), or is "FirstFit-fixed placement beats learned placement
at every budget/design tried so far" now well-evidenced enough to treat as this
session's answer to the report.md Section 1.4 confound?

---

## 2026-09-21 (S2W9) -- "More online training hurts" confirmed a 3rd independent time, now with the full mechanistic trajectory (not just start/end snapshots)

**Context:** Direct use of the new training-diagnostics tooling (`Code/utils/training_diagnostics.py`, implemented 2026-09-20) on the exact protocol that first produced the "more training hurts" mystery -- Option 1 online, 900k timesteps, dense+weighted, `--diagnostics-interval 5000`. Goal: real visibility into WHAT happens over the course of training, not just comparing two endpoint numbers.

**Stats:**
```
Final 50-instance randomized eval:
  Option 1 (900k, diagnostics run)  tardiness=269.20+/-116.80  weighted_tardiness=798.46+/-346.65  late=22.90  scheduled=833.42/898
  SPT                                tardiness=269.20+/-116.80  weighted_tardiness=798.46+/-346.65  late=22.90  scheduled=833.42/898  (IDENTICAL, digit-for-digit)
  ATC                                tardiness=234.36+/-114.67  weighted_tardiness=648.16+/-338.30  late=23.80  scheduled=835.92/898

Comparison across all three 900k online runs so far (all dense+weighted, same protocol):
  900k, ent_coef=0.0 (original, 2026-09-18):   weighted=788.20+/-389.26  (20-instance sample)
  900k, ent_coef=0.01 (2026-09-19 entry):      weighted=798.46+/-346.65  (50-instance) -- BYTE-IDENTICAL to this run
  900k, ent_coef=0.0, this diagnostics run:    weighted=798.46+/-346.65  (50-instance)
  300k (best known online Option 1 result):    weighted=730.30+/-306.91  (20-instance sample, genuinely mixed rule behaviour)

Full-run diagnostics trajectory (TensorBoard, this run):
  action_dist/entropy_normalized: ~1.0 (uniform) at the start, declining CONTINUOUSLY and
    gradually across the entire 900k steps (not a sudden collapse at one point) to a final
    ~0.164-0.165 -- a partial, not total, collapse in entropy terms.
  eval_tardiness/mean_weighted_tardiness (5 fixed held-out instances, sampled every 5000
    steps): genuinely noisy through the run's middle third (spiking as high as 2.02e3 at
    points), before settling into an exactly-stable plateau at 741 for a long final
    stretch -- the deterministic eval policy became fully fixed well before 900k, not
    gradually drifting worse all the way to the end.
```

**Observation:** This is now the THIRD independent 900k online run (two different `ent_coef` settings) landing on the same collapsed-to-SPT outcome, with two of the three producing byte-identical eval numbers -- this rules out noise/seed-luck as the explanation; the collapse is a robust, reproducible property of this training setup at this timestep budget, not an artifact of one run. The new trajectory data adds real mechanism that wasn't visible before: entropy decays smoothly and continuously across the WHOLE run (not a late, sudden event), while the deterministic policy's actual behaviour (the eval_tardiness plateau) locks in earlier and holds fixed for a long final stretch -- meaning most of the run's back half trains a policy whose deterministic behaviour has already stopped changing, even though the underlying entropy metric was still nominally declining. This dissociation (entropy still moving, deterministic behaviour already frozen) is itself informative: it suggests the policy's argmax action converges well before its full probability distribution does, so tracking entropy alone would give a falsely-still-improving signal after the practically-relevant collapse has already happened.

**Conclusion / next step:** The "more training hurts online" finding is now well-evidenced (3 independent confirmations) rather than a single-run curiosity, and entropy regularization is conclusively ruled out as the fix (both ent_coef settings converge to the same place). The genuinely open question -- WHY SPT specifically, and why the deterministic policy locks in well before entropy fully decays -- remains unresolved and is a good candidate for a future session's diagnostic work (e.g. tracking which specific state features correlate with the SPT-favouring decisions, or comparing the value function's landscape around the collapse point). Not chased further tonight; the 300k checkpoint (weighted=730.30) remains the best available online Option 1 result, and this entry's job is documenting the phenomenon precisely, not solving it.

---

## 2026-09-21 (S2W9) -- Option 4 context-fix result: reached a much better optimum mid-training, but instability lost it by the end

**Context:** Direct follow-up to the job-choice-weighted `_pool_job_context()` fix (`Code/policies/action_branching_policy.py`, committed the same evening -- see that commit for the full derivation grounded in Tavakoli et al. 2018). Same config as the original Option 4 result (300k, dense+weighted, `--diagnostics-interval 5000`), save-tag `offline_dense_weighted_ctxfix`, for a direct comparison.

**Stats:**
```
Option 4 (ctxfix, 300k)    tardiness=  427.24+/-221.97  weighted_tardiness=  795.48+/-495.38  late=40.46  scheduled=92.26/100
Option 4 (original, 300k)  tardiness=  172.74+/-159.61  weighted_tardiness=  355.64+/-358.80  late=31.74  scheduled=96.10/100  (2026-09-20 entry above)
ATC                        tardiness=  221.24+/-129.98  weighted_tardiness=  298.80+/-184.13  late=11.10  scheduled=94.68/100

eval_tardiness/mean_weighted_tardiness trajectory across the run (TensorBoard, sampled):
  early (~2900-5800, noisy) -> mid-run: EXACTLY 0.0 for several consecutive readings (perfect
  scheduling on the 5 fixed held-out instances) -> brief noise (10.8-2930) -> late-run climb
  back into the 500-1500 range -> final reading before save: 1120.
```

**Observation:** At face value the final-checkpoint comparison looks like a regression
(795.48 vs. 355.64) -- but the training-time trajectory tells a different story than a
simple "the fix didn't work." This run reached a materially BETTER optimum (weighted
tardiness = 0, i.e. perfect) than the original run ever got close to at any point,
proving the underlying hypothesis (a job-choice-weighted machine-branch context beats a
flat average) is directionally correct -- reachable, not wrong. What actually happened
is instability: the original run's tardiness curve was comparatively smooth and
steadily improving (1250->~65 by the end); this run oscillated by orders of magnitude
throughout, and simply happened to be in a bad phase of that oscillation when training
stopped at exactly 300k timesteps.

**A concrete, testable hypothesis for the instability (not yet diagnosed further,
flagged as the next step, not asserted as proven):** the fix makes `job_score_head`'s
parameters serve two entangled roles that were previously fully decoupled -- picking a
good job directly (via `job_logits`, unchanged) AND shaping what the machine branch
sees (via `job_logits`-weighted pooling into `job_context`, the new part). Before this
fix, `job_score_head`'s gradient signal came only from the job branch's own action
loss; now it also indirectly affects the machine branch and value/idle heads through
the pooling weights, a plausible source of the added optimization noise. A natural,
minimal next test: detach `job_logits` before using them as pooling weights
(`torch.softmax(job_logits.detach(), dim=-1)`), which keeps the core fix (weighted, not
flat, pooling) while removing this specific gradient-coupling path, isolating whether
it's actually the cause.

**Conclusion / next step:** Neither "the fix works" nor "the fix doesn't work" is the
correct takeaway -- the fix demonstrably unlocks a better achievable optimum but also
destabilizes training around it. Given this project's own established discipline
(diagnose mechanism before re-running blind), the `job_logits.detach()` variant is the
next concrete, cheap (one-line change, no new training-scale commitment yet) thing to
try before deciding whether this whole approach is worth a larger-scale validation run.
Not implemented yet -- a genuine next-step decision, not assumed.

---

## 2026-09-20 (S2W9) -- Option 4 (action-branching) first result: learned placement makes things WORSE at matched training budget, not better

**Context:** First real training run for Option 4 (`Code/env/action_branching_gym_wrapper.py`, implemented earlier today -- see the entry below covering that implementation session) -- the direct test of the report.md Section 1.4 confound: is Options 1/2/3's win over the historic PPO band from the smaller action space, or from removing machine-placement-learning entirely (fixed FirstFit)? Option 4 gives placement back to the learner via `MultiDiscrete([max_jobs+1, num_machines])`.

**Config:** 300k timesteps (matching every other option's first-pass-filter scale), `--reward-mode dense_tardiness --job-weight-min 1 --job-weight-max 6`, `--diagnostics-interval 5000` (new tooling's first real use). Evaluated on the standard 50-instance randomized protocol.

**Stats:**
```
                                         raw tardiness   weighted tardiness
Option 4 (action-branching, 300k)          172.74            355.64
Option 3 (FirstFit, 300k, same entry above) 98.82              --
Option 1 (FirstFit, 300k, same entry above)  49.32              --
EDF                                          37.30            109.36
LST                                          23.94             68.62
ATC                                         221.24            298.80

Diagnostics trend across the run (eval_tardiness/mean_tardiness, TensorBoard):
  step ~15k: 1250 -> step ~150k: ~155 -> step ~300k: ~65-150 (noisy but converging)
action_dist/entropy_normalized (job branch): stayed at 1.0 for the ENTIRE run -- never
  specialized away from uniform.
action_dist/machine_mask_mismatch_frac: consistently 0.7-1% throughout -- the parallel-
  branch masking approximation (see the wrapper's design docstring) is cheap in
  practice, not the bottleneck.
```

**Observation:** At the same 300k, dense+weighted, randomized-eval comparison, Option 4
is WORSE than both Option 1 (49.32) and Option 3 (98.82) -- despite having a strictly
more expressive action space that CAN learn placement, where those two are stuck with
fixed FirstFit. This is the opposite of what "placement-learning was the missing piece"
would predict. The job-branch entropy staying at 1.0 (never specializing) for the whole
run is a plausible mechanism: adding a second thing to learn (even as an independent,
additively-sized branch, not a multiplicative joint space) appears to dilute the
optimization budget that would otherwise go into refining job selection, at this
training scale -- consistent with, not contradicting, the original "action-space size
governs sample efficiency" finding. The machine_mask_mismatch_frac staying low all run
is a genuine methodological win regardless of the placement result itself: the parallel-
branch design's approximation cost is empirically small, not the confound's explanation.

**Conclusion / next step:** This is real evidence AGAINST the "confound" hypothesis
report.md raised, not a validation of it -- at matched budget, removing placement-
learning (Options 1/2/3) is better than restoring it (Option 4), suggesting the
FirstFit-fixed designs' win genuinely is action-space-size, not an artifact of skipping
placement-learning. Not fully settled: only one training budget tested, and the job
branch's stuck-at-uniform entropy suggests this may be an undertrained result rather than
a durable one (the same possibility flagged for Options 1/2/3 in earlier entries before
their full-scale validation runs improved substantially). Natural next step if this
thread is worth more compute: a full-scale (1.2M) Option 4 run to see whether the job
branch ever specializes given more budget, mirroring the validation pattern already used
for Options 1/2/3.

---

## 2026-09-20 (S2W9) -- Optimisation & efficiency critique (analysis session, no code behaviour changed)

**Context:** User requested an "optimisation critique" distinct from `report.md`'s
methodological review -- specifically computational/training inefficiencies, HPO
methodology, and multi-agent training as a possible future direction. Two parallel
code-reading passes (training/HPO pipeline; env/policy hot path), each finding
spot-checked directly against source before being written up, per this project's
verification standard.

**Findings (full detail, file:line evidence, and References section):**
`2026-09-20-optimisation-and-efficiency-critique.md`.

**Headline finding:** `SchedulingEnv.get_state()` (`scheduling_env.py:556-565`) is
called from both `step()` and `step_idle()` on every single environment step, deep-
copying four arrays plus a set-to-list conversion -- and its return value is
unconditionally overwritten/discarded by `gym_scheduling_wrapper.py` two lines later
(verified directly, not inferred). This has run on every RL training step of every
campaign this project has ever executed. Free, zero-behaviour-change fix.

**Also found:** Optuna's configured `MedianPruner` cannot actually prune -- both
PPO and (especially) A2C objectives report their pruning metric only after a
trial's full compute is already spent, or (A2C) never report one at all
(`optuna_tune.py`, `trial.report()` appears exactly once in the whole file).
Combined with `n_jobs=1` (fully sequential 50-trial search on a 16-core machine)
and HPO tuning under `n_envs=1` while deployment uses `n_envs=5`, this means the
next HPO run (already deferred per the design-space-not-settled decision,
`2026-09-19` entry below) would benefit from three fixes landing first, since they
change what a fixed compute budget buys, not just how fast it runs.

**Multi-agent training:** framed as a structurally motivated *next* mechanism on the
same action-space-decomposition spectrum as Options 1-4 (per-machine agents shrink
the action space per-agent to O(J) instead of the joint O(J*M)), not a generic
suggestion -- MAPPO (Yu et al. 2022) named as the lowest-engineering-lift option
given the project's existing PPO investment, QMIX (Rashid et al. 2018) as the
heavier value-based alternative. Explicitly scoped as future work needing user
input before building, same gate already applied to the two-critic Lagrangian
rewrite and HPO itself -- not started today.

**Conclusion / next step:** No code changed this session (pure analysis/report).
The free env-hot-path fixes (§1 of the dated doc) can be applied anytime with no
retraining implications. The Optuna pruning/parallelism fixes (§2) should land
before the already-planned Options 1-4 HPO search, not after.

---

## 2026-09-20 (S2W9) -- Implementation session: Option 4 (action-branching) + training-time observability tooling

**Context:** Following the compute_theta() bug discussion below, agreed direction for
"lots of time" available: (1) action-branching to isolate the report.md Section 1.4
confound (is Options 1/2/3's win from the smaller action space, or from removing
machine-placement-learning entirely?), (2) examine the still-open "more online
training hurts" mystery, (3) early-success-indicator tooling ("any ways to see if a
certain model would be successful and worth training for a set amount of hours, or
if there are any indications it will not perform"). (2) and (3) turned out to need
the same underlying infrastructure. HPO remained explicitly deferred. Scoped via
plan mode (2 parallel Explore passes + 1 Plan pass), with the single most
load-bearing technical claim independently re-verified against the installed
sb3_contrib source before committing to the design. Full plan:
`C:\Users\ethan\.claude\plans\encapsulated-questing-cray.md`.

**Option 4 implemented** (`Code/env/action_branching_gym_wrapper.py`,
`Code/policies/action_branching_policy.py`,
`Code/policies/action_branching_ppo_policy.py`): `MultiDiscrete([max_jobs+1,
num_machines])` instead of Options 2/3's fixed-FirstFit placement. Design decision,
verified directly against `sb3_contrib/common/maskable/distributions.py`:
`MaskableMultiCategoricalDistribution` splits one shared forward pass's logits/mask
into independent per-branch chunks, with no hook for the machine branch to condition
on the job branch's sampled value -- built the parallel-independent-branches design
(also the more faithful reading of Tavakoli et al. 2018's actual BDQ architecture;
the `references.bib` note previously said "sequential", corrected). Machine mask is
the union of feasible machines across all remaining jobs (a logged approximation);
real constraint enforcement is a graceful idle-fallback in `step()` when a
mask-legal pair turns out infeasible for the specific job chosen, flagged via a new
`info["mask_mismatch"]` key. 9-check regression suite including a distribution-level
masking sanity check (validates independent per-branch masking directly against the
real library, not just trusted). Smoke-tested end-to-end (train -> save -> load ->
eval).

**Training-time observability implemented** (`Code/utils/training_diagnostics.py`):
`ActionDistributionCallback` (periodic per-mode TensorBoard scalars --
action_dist/entropy_normalized + per-rule frequency for Option 1, idle_frac/entropy/
top1_frac for Options 2/3, plus machine-branch stats and
`machine_mask_mismatch_frac` for Option 4 -- turns the entropy-collapse-style
investigation into "read one TensorBoard scalar" instead of the ad-hoc
stdout-eyeballing this project has done twice before) and `TardinessEvalCallback`
(periodic held-out tardiness eval against 5 FIXED instances, reusing
`eval_action_space_variant.py`'s `run_episode()`/`_weighted_tardiness()` via a
deferred import to avoid a circular import). `build_diagnostics_callbacks()` also
wires in `sb3_contrib.MaskableEvalCallback` (already installed, correctly threads
`action_masks` through masked eval, unused anywhere in this repo before now) for
free reward/episode-length logging. New `--diagnostics-interval` flag on
`train_action_space_variant.py` (default off, existing behaviour unchanged).
Smoke-tested end-to-end for all three modes via real short training runs -- every
TensorBoard scalar group populated correctly, including a real nonzero
`machine_mask_mismatch_frac` reading. 7-check regression suite.

**Conclusion / next step:** Both pieces of infrastructure are implemented, tested,
and committed but NOT yet used for a real finding -- no full-scale Option 4 training
run has been launched yet, and the diagnostics tooling hasn't yet been pointed at a
real multi-hour online run to actually chase the "more training hurts" mystery. Next
session's natural first move: launch Option 4 at the same ~300k first-pass-filter
scale this session's other options used, and launch (or relaunch) an online Option
1/3 run with `--diagnostics-interval` on to get the first real TensorBoard trend
data on the still-open mystery.

---

## 2026-09-20 (S2W9) -- Real bug found and fixed: compute_theta() has been understating utilisation for this project's ENTIRE history

**Context:** Direct follow-up to the previous entry's design discussion -- user asked "well is our machine utilization part of the objective correct?" Checked rather than assumed, per this project's rigor convention, and found a real, confirmed bug, not just a design gap.

**Bug:** `SchedulingEnv.compute_theta()` (the hotspot-penalty term, `-lambda3*delta_theta`) used `self.capacity[:, :, 0]` as its "original capacity" reference. That is the LIVE, mutable capacity array at time-slot 0, not a fixed baseline -- the instant any job occupies time-slot 0 (true for nearly every real episode, since `self.time` starts at 0), that reference silently corrupts for the rest of the episode. The environment already had a correct pristine reference for exactly this purpose (`self.machine_capacity`, kept separate in `__init__` specifically "so reset() has an untouched value to restore from") -- `compute_theta()` never used it.

**Verified empirically (`tests/test_compute_theta_bugfix.py`, all 3 checks pass):**
```
Check 2: 50%-used slot-0 job, then an 80%-used later job -> pre-fix theta=0.6 (wrong), post-fix theta=0.8 (true)
Check 3: a single 90%-used job AT slot 0 -> pre-fix theta=0.0 (hidden entirely!), post-fix theta=0.9 (true)
```

**Observation:** Two distinct failure modes from the same root cause: (1) once anything is scheduled at slot 0, every LATER slot's computed utilisation is biased toward whatever was consumed at slot 0 specifically, generally UNDERSTATING true utilisation for the more heavily-loaded slots -- exactly the ones the hotspot penalty is meant to catch; (2) slot 0's own utilisation always read as exactly 0% by construction (the buggy reference self-cancels there), regardless of how full it actually was -- a fully-loaded slot 0 was invisible to this term for the entire project's history. Net effect: `-lambda3*delta_theta` (hardcoded `lambda_3=1.0` everywhere, never tuned) has been systematically WEAKER than intended, not absent -- it fired, just on understated numbers. This also softens (does not eliminate) the hotspot-vs-energy-consolidation tension flagged in the previous entry: the anti-consolidation force has in practice been weaker than lambda_3's nominal value implies.

**Fix:** `compute_theta()` now uses `self.machine_capacity` (broadcast correctly across machines/time) instead of `self.capacity[:, :, 0]`. One-line semantic change, existing `tests/test_bugfixes.py` suite still passes unchanged (confirms no other mechanism depended on the buggy behaviour).

**Conclusion / next step:** This is a real correctness bug in the environment's core reward computation, present since the hotspot penalty was first added and active in EVERY training run this project has ever done (dense_tardiness mode does not skip this term -- see `reward()`, it's unconditional). Given the magnitude is "systematically weaker penalty," not "term was completely absent" (unlike the job-weights precedent, where the term was fully dead), and `lambda_3=1.0` is one term among several rather than the dominant driver of any result reported so far, this is flagged here rather than unilaterally triggering a full retrain -- **explicitly deferring to the user on retraining scope**, per this project's own established process rule (CLAUDE.md "Follow-through on concrete requests," added after the job-weights incident) rather than assuming either "retrain everything" or "ignore it" on my own.

---

## 2026-09-20 (S2W9) -- Design discussion: this project's reward does not model energy efficiency, and the existing hotspot penalty actively works against the literature's consolidation-based definition of it

**Context:** User asked "how do we incentivise machine utilisation, or do we not explicitly?", then "how should we define it for the purpose of energy efficiency, or what do the papers say?" -- a design-grounding question, not a training run. No code changed; this is a literature check + honest gap statement.

**Finding 1 -- no term currently rewards utilization or penalizes active-machine count in a time-integrated way.** `Code/env/scheduling_env.py::reward()` has exactly two utilization-adjacent terms: (a) `-lambda1` on machine activation (a one-shot, fixed-magnitude penalty at the moment a machine first activates, currently hardcoded `lambda_1=1.0` and never tuned/searched -- see `train_action_space_variant.py`), and (b) `-lambda3 * delta_theta` (`compute_theta()` = max utilization across any machine/resource/time), which penalizes *increases* in peak utilization -- i.e. discourages concentrating load, the opposite of a consolidation incentive. Neither is time-integrated (a machine sitting active-but-idle for 50 ticks costs nothing beyond its one activation charge), and for Options 1/2/3 (this session's entire action-space-reduction line of work) the RL policy doesn't even choose which machine to use -- placement is fixed to `FirstFit` -- so term (a) isn't actionable there regardless of its magnitude.

**Finding 2 -- the standard literature definition (grounded, not assumed).** Fan, Weber & Barroso (2007, ISCA -- `fan2007powerprovisioning`, newly added to `references.bib`) established empirically that server power is close to linear in CPU utilization, but **idle power is ~70% of peak, not 0%** (`k=0.7`): `P = (P_MAX - P_MIN)*u + P_MIN`. Beloglazov, Abawajy & Buyya (2012, `energyaware2012`, already in this bib but previously un-annotated) substitute that constant into the simplified form `P = P_MAX * (0.7 + 0.3*u_cpu)`. Confirmed via a secondary survey (arXiv:2004.12335 Section 2.1.1) quoting Fan et al. directly, after the primary PDF itself failed to text-extract in this session (binary/encoded) -- flagged in the bib entry's note as needing independent re-verification if the exact derivation is needed later, per this project's rigor convention of not asserting an unverified number as settled.

**The key implication for reward design:** because idle-but-active power is high (not free), the energy-optimal policy under this model is NOT "maximize average utilization" -- it's "minimize the number of active machines" (consolidate load onto fewer machines, run them hot, leave the rest fully OFF rather than idle-but-active) and integrate the resulting power draw over time, not charge it once at activation.

**The tension, stated plainly:** this project's existing hotspot penalty (`-lambda3*delta_theta`) actively rewards *spreading* load across more machines to avoid any single machine/resource/time hitting high utilization -- structurally the opposite of the consolidation behaviour the energy literature says saves power. The two objectives (load-balancing to avoid hotspots, and energy-efficient consolidation) are not just unmodelled together, they point in opposite directions as currently implemented. This was not previously flagged anywhere in this project's docs.

**Conclusion / next step:** Energy efficiency is not currently part of this project's objective at all, despite the README's own Project Overview listing "minimize active servers, energy usage" as a goal. If it's to be added, the grounded design would be a per-tick energy term `E(t) = sum_m [active_m(t) * P_max_m * (0.7 + 0.3*u_m(t))]`, normalized and subtracted from reward the same way the tardiness term is normalized by horizon (existing project convention). Not implemented -- this is a scoping/design discussion pending the user's decision on whether to pursue it, and if so, how to resolve the hotspot-vs-consolidation conflict first (e.g. is hotspot avoidance protecting against a real resource-contention/thermal risk that should coexist with consolidation, or was it only ever meant as a rough load-balancing heuristic that an energy term should supersede).

---

## 2026-09-20 (S2W9) -- Option 3 online-longer (900k) result: more training hurt here too, same pattern as Option 1

**Config:** Option 3, online, same protocol as the 300k dense+weighted baseline (`--arrival-rate 9 --online-horizon 100 --online-max-jobs 1300 --job-size-distribution lognormal --reward-mode dense_tardiness --job-weight-min 1 --job-weight-max 6`), 900k timesteps (up from 300k), save-tag `online_lognormal_rho075_dense_weighted_longer`. Evaluated on the 50-instance randomized protocol.

**Stats:**
```
Option 3 (900k)   tardiness=  327.60+/-140.01  weighted_tardiness=  892.84+/-399.23  late=29.96  scheduled=829.20/898
EDF               tardiness=  263.16+/-126.71  weighted_tardiness=  800.90+/-404.57  late=28.78  scheduled=832.62/898
ATC               tardiness=  234.36+/-114.67  weighted_tardiness=  648.16+/-338.30  late=23.80  scheduled=835.92/898

For comparison, Option 3 (300k, same config, 20-instance sample, entry above):
  Option 3 (300k)  tardiness=267.75+/-105.77  (already the worst of 4 methods compared then)
```

**Observation:** More training made Option 3 online worse (267.75 -> 327.60 raw tardiness), not better -- it now loses to every heuristic listed, including EDF, by an even larger margin than at 300k. This is the same "more online training hurts" pattern already found for Option 1 (2026-09-18 entry: 730.30 -> 788.20 weighted, 300k->900k) and confirmed independently unresolved by entropy regularization (2026-09-19 entry above). Two different action-space designs (Option 1's rule-selection, Option 3's ATC-primed priority scoring) both regress with more online training -- this is evidence of a genuine property of online training in this environment, not something specific to one design's architecture.

**Conclusion / next step:** The online case's "more training helps" assumption (which holds reliably offline -- every offline option improved monotonically from 300k to full-scale) does NOT hold online for either design tested so far. This is now a 2-for-2 pattern, not a single anomaly, and is a more fundamental open question than either individual result suggested -- worth investigating directly (e.g., tracking per-rule/per-job-score action distribution over the course of training, not just final entropy) before spending more compute on longer online training runs for any option. The 300k checkpoints remain the best available online results for both Option 1 (730.30 weighted) and Option 3 (267.75 raw, still behind ATC).

---

## 2026-09-19 (S2W9) -- `--ent-coef 0.01` test result: training-time entropy fixed, but the SPT-collapse itself was NOT -- earlier root-cause diagnosis was incomplete

**Config:** Option 1, online, same protocol as the entropy-collapse finding below (`--arrival-rate 9 --online-horizon 100 --online-max-jobs 1300 --job-size-distribution lognormal --reward-mode dense_tardiness --job-weight-min 1 --job-weight-max 6`), 900k timesteps, `--ent-coef 0.01` (the fix this run was testing), save-tag `online_lognormal_rho075_dense_weighted_entcoef`. Evaluated on the 50-instance randomized protocol (larger than the 20-instance sample the original finding used).

**Stats:**
```
Training-time entropy_loss (sampled across the run): -2.03 -> -1.46 -> -1.31 -> -1.34 -> -1.42 -> -1.51
  (stayed in this band throughout -- did NOT decay to near-zero the way the uncorrected 900k run did)

Eval (50 instances):
  Option 1 (900k, ent_coef=0.01)   tardiness=  269.20+/-116.80  weighted_tardiness=  798.46+/-346.65  late=22.90  scheduled=833.42/898
  SPT                              tardiness=  269.20+/-116.80  weighted_tardiness=  798.46+/-346.65  late=22.90  scheduled=833.42/898
  ATC                              tardiness=  234.36+/-114.67  weighted_tardiness=  648.16+/-338.30  late=23.80  scheduled=835.92/898

For comparison (2026-09-18 entry below, 20-instance sample):
  Option 1 (900k, ent_coef=0.0, uncorrected)  weighted=788.20+/-389.26  -- also identical to SPT
  Option 1 (300k, ent_coef=0.0)               weighted=730.30+/-306.91  -- genuinely mixed SPT+LST, best online Option 1 so far
```

**Observation:** The fix worked exactly as intended at the mechanism it targeted -- `entropy_loss` stayed in the -1.3 to -1.5 band for the whole 900k run instead of decaying to -0.019. But the deterministic (argmax) eval policy is, once again, numerically IDENTICAL to SPT -- same as the uncorrected 900k run this was meant to fix, and worse than the untouched 300k checkpoint. Maintaining training-time entropy did not prevent the deterministic policy from converging onto a single dominant rule. This means the earlier "root cause: zero entropy regularization" diagnosis was incomplete: the entropy decay observed in the original 900k run was a genuine, correctly-measured phenomenon, but it was a correlate of convergence, not the cause of the SPT-collapse itself -- a policy can retain healthy entropy over its full action distribution while still having SPT as the clear argmax-dominant choice in nearly every state it encounters. The real driver of "more online training converges toward SPT" remains unexplained.

**Conclusion / next step:** Revert to treating the 300k checkpoint (`weighted=730.30`, mixed-rule behaviour) as the best available online Option 1 result -- neither 900k variant (with or without entropy regularization) has beaten it. Do not pursue further `ent_coef` tuning for this specific problem; it's now been tested and shown not to address the actual mechanism. Genuinely open questions for a future session: (a) whether the online case's credit-assignment problem itself (not entropy) makes sustained multi-rule switching hard to maintain over long training runs, (b) whether this is specific to Option 1's small `Discrete(8)` action space rather than a general online-training pathology, (c) whether an explicit stochastic (not deterministic) eval policy would show different behaviour than what argmax reveals here.

---

## 2026-09-19 (S2W9) -- Option 1 randomized-instance full-scale (1.2M, dense+weighted): catastrophic collapse fixed, but still short of LST

**Config:** Option 1, `--randomize-instances`, `--reward-mode dense_tardiness --job-weight-min 1 --job-weight-max 6`, 1.2M timesteps (up from the 300k that produced the catastrophic-collapse result below), save-tag `offline_randomized_dense_weighted_fullscale`. Evaluated on the standard 50-instance randomized protocol (seeds 500000-500049).

**Stats:**
```
Option 1 (1.2M, dense+weighted)   tardiness=   38.88+/- 62.44  weighted_tardiness=  116.50+/-181.87  late=12.22  scheduled=99.00/100
EDF                                tardiness=   37.30+/- 60.43  weighted_tardiness=  109.36+/-175.73  late=12.22  scheduled=96.84/100
LST                                tardiness=   23.94+/- 56.02  weighted_tardiness=   68.62+/-159.10  late= 8.26  scheduled=97.76/100

For comparison, the same randomized-instance case at legacy reward, 300k (2026-09-18 entry below):
  Option 1 (legacy, 300k)          tardiness=1377.88+/-158.85  late=44.30  scheduled=99.22/100  (LPT-collapse, reward-hacking the flat completion bonus)
```

**Observation:** The combination of dense_tardiness reward, real job weights, and 4x more training timesteps fixes the catastrophic LPT-collapse found on 2026-09-18 (1377.88 -> 38.88 raw tardiness, a ~35x improvement) and brings the randomized-instance case to an almost-exact tie with EDF (38.88 vs 37.30 raw; 116.50 vs 109.36 weighted). It does not, however, close the gap to LST (68.62 weighted) the way the *fixed*-instance Option 1 result did earlier tonight -- the randomized-instance (must-generalize) case remains harder than the fixed-instance (can-memorize) case at the same reward design, consistent with the 2026-09-18 entry's original diagnosis that action-space size was necessary but not sufficient here.

**Conclusion / next step:** This is the full-scale, "for completeness" run queued after the user's priority ordering (best performers first, offline prioritized) -- treat it as confirming the direction (dense+weighted+action-space-reduction generalizes, doesn't just memorize one instance) rather than a new state-of-the-art number. LST remains the bar Option 1 hasn't cleared on the randomized-instance case; Options 2/3 (not yet run at this full randomized-instance+weighted+full-scale combination) are the natural next comparison point if further budget is available.

---

## 2026-09-18 (S2W9) -- Week-label correction + windowed Option 3 offline result

**Note on week labels:** recomputing `((2026-09-18 - 2026-07-20).days // 7) + 1` gives
`60 // 7 + 1 = 9` -> **S2W9**, not S2W10 as every entry/doc from today and yesterday
in this log has used. That was an arithmetic error made early in tonight's session
and repeated since. Per this file's "never edit past entries" rule, not going back to
fix already-written S2W10 labels -- just using the correct S2W9 from here on. If this
matters for citation/organization purposes, past S2W10 labels dated 2026-09-17/18
should be read as S2W9.

**Config:** Windowed Option 3 offline (DeepRM-style bounded action-space window,
`--window-size 15`, EDF-ordered + backlog scalar -- see
`Code/env/windowed_priority_gym_wrapper.py`), 300k timesteps (first-pass filter
scale, matching this session's own established precedent), `--reward-mode
dense_tardiness --job-weight-min 1 --job-weight-max 6`, save-tag
`window15_offline_dense_weighted`. Evaluated on the full 50-instance randomized
protocol (seeds 500000-500049), newly wired through
`Code/utils/results_log.py::append_eval_result()` (see this session's separate
CSV-persistence commit) -- this is the first result recorded there instead of only
in this file.

**Stats:**
```
Option 3 (window=15)   tardiness=   55.14+/- 70.66  weighted_tardiness=  105.86+/-137.35  late= 9.20  scheduled=98.22/100
EDF                    tardiness=   37.30+/- 60.43  weighted_tardiness=  109.36+/-175.73  late=12.22  scheduled=96.84/100
LST                    tardiness=   23.94+/- 56.02  weighted_tardiness=   68.62+/-159.10  late= 8.26  scheduled=97.76/100
ATC                    tardiness=  221.24+/-129.98  weighted_tardiness=  298.80+/-184.13  late=11.10  scheduled=94.68/100
```

**Observation:** Windowed Option 3 (weighted_tardiness=105.86) is roughly tied with
EDF (109.36) and clearly *worse* than LST (68.62) -- a real regression from the
unwindowed Option 3 result reported earlier this session (weighted_tardiness=13.00,
beating LST outright). Two confounds not yet separated: (1) the windowed run is only
300k timesteps (first-pass filter scale) vs. the unwindowed "beats LST" result's
larger/full-scale training, so this may just be an undertrained comparison, not a
windowing regression; (2) restricting the visible candidate set to the 15
EDF-nearest jobs structurally biases the learnable policy toward EDF-like behaviour
(the window is EDF-ordered by construction -- see this module's own design-choices
docstring), which is consistent with the observed near-tie with plain EDF.

**Conclusion / next step:** Do not treat windowing as validated or as an improvement
yet -- it needs a same-scale (matching timesteps) apples-to-apples comparison against
unwindowed Option 3 before either confound can be ruled out. Next: once the
currently-running windowed Option 3 *online* job (save-tag
`window15_online_lognormal_rho075_dense_weighted`) finishes, evaluate it the same
way; if both windowed results underperform their unwindowed counterparts at matched
timesteps, the EDF-ordering design choice (flagged as untested in the wrapper's own
docstring) is the first thing to revisit -- e.g. arrival-order or raw-priority-score
windowing instead of EDF-order, so the window doesn't pre-bias toward one heuristic.

**Update (same day): windowed Option 2 offline lands in the same place.** Same
protocol (`--window-size 15`, 300k timesteps, dense+weighted, 50-instance
randomized eval), save-tag `window15_offline_dense_weighted`:
```
Option 2 (window=15)   tardiness=   55.10+/- 68.54  weighted_tardiness=  111.08+/-154.73  late=15.30  scheduled=98.18/100
EDF                    tardiness=   37.30+/- 60.43  weighted_tardiness=  109.36+/-175.73  late=12.22  scheduled=96.84/100
LST                    tardiness=   23.94+/- 56.02  weighted_tardiness=   68.62+/-159.10  late= 8.26  scheduled=97.76/100
```
weighted_tardiness=111.08 -- within 2% of windowed Option 3's 105.86, and again an
almost-exact tie with plain EDF (109.36 for both). Two different learned policies
(raw-feature-only vs. ATC-primed) converging on the *same* number, and that number
matching EDF almost exactly, is a second independent data point for the EDF-ordering
confound in §ablation above (2026-09-18, S2W9 windowed offline entry) rather than an
architecture-specific fluke -- strengthens the case that the window's EDF-ordered
candidate selection, not the learned scoring network, is what's determining outcomes
at this training scale. Still not separated from the undertrained-vs-biased confound
(both windowed runs are 300k vs. unwindowed Option 3's larger full-scale training) --
same next step as above.

**Update (same day): windowed Option 3 online is a clear negative result, not just a
tie.** Save-tag `window15_online_lognormal_rho075_dense_weighted`, 300k timesteps,
arrival_rate=9 (rho~0.75), lognormal, dense+weighted, 50-instance randomized online
eval (seeds 500000-500049):
```
Option 3 (window=15)   tardiness=  309.74+/-117.71  weighted_tardiness=  952.74+/-370.59  late=28.16  scheduled=825.98/898
ATC                    tardiness=  234.36+/-114.67  weighted_tardiness=  648.16+/-338.30  late=23.80  scheduled=835.92/898
WSPT+BestFit           tardiness=  263.66+/-107.00  weighted_tardiness=  709.42+/-299.06  late=24.30  scheduled=836.88/898
EDF+BestFit            tardiness=  245.46+/-139.84  weighted_tardiness=  736.44+/-430.41  late=26.82  scheduled=834.90/898
SPT                    tardiness=  269.20+/-116.80  weighted_tardiness=  798.46+/-346.65  late=22.90  scheduled=833.42/898
EDF                    tardiness=  263.16+/-126.71  weighted_tardiness=  800.90+/-404.57  late=28.78  scheduled=832.62/898
LST                    tardiness=  304.32+/-172.91  weighted_tardiness=  918.92+/-551.65  late=31.12  scheduled=827.70/898
FCFS+FirstFit          tardiness=  323.02+/-154.46  weighted_tardiness=  987.50+/-511.52  late=33.44  scheduled=835.50/898
```
Unlike the offline case (where windowing produced a near-tie with EDF), windowed
Option 3 online is the **second-worst of all nine methods compared** -- only
FCFS+FirstFit (987.50), Tetris (1235.04), and LPT+WorstFit (2370.06) are worse; it
loses to every priority-rule heuristic including plain EDF and SPT, and ATC (648.16)
beats it by ~32%. ATC's number here (648.16) is consistent with the earlier
non-windowed online finding (644.20, different eval-seed set) -- ATC remains the
online case's best live policy across both action-space designs tested so far.

**Conclusion / next step:** Across all three windowed checkpoints trained tonight
(Option 3 offline, Option 2 offline, Option 3 online), none beat the best available
heuristic, and the online case is not just a tie but a clear regression below most
heuristics -- windowing has not reproduced or improved on the unwindowed Options 2/3
results at this (300k, first-pass-filter) training scale. Combined with the offline
EDF-matching pattern above, the most likely explanation remains the EDF-ordered
window-selection design (flagged as untested in
`Code/env/windowed_priority_gym_wrapper.py`'s own docstring): online arrivals
constantly reshuffle which jobs are EDF-nearest, so a bounded EDF-ordered window may
be *more* disruptive to a learned policy online than offline, consistent with online
being the worse of the two results here. Not treating windowing as a validated
direction based on tonight's results -- recommend either a same-scale unwindowed
comparison to properly isolate the undertrained-vs-biased confound, or revisiting the
window-selection ordering itself, before investing further training budget in this
design.

---

## 2026-09-18 (S2W10) -- Weighted retrain results (overnight queue, best performers first)

**Config:** Following the user's priority ("best performing models first"), weighted
retrain queue launched in order: Option 1 offline (dense+weighted), Option 3 offline
(dense+weighted), Option 1 offline randomized-instance (dense+weighted), ... Each
evaluated on the SAME weighted instance/distribution it was trained on.

**Stats:**
```
Option 3 offline, dense_tardiness + weighted (job_weight_range=(1,6)):
  tardiness=28.00  late=5  scheduled=100/100
  (vs. dense-only 56.00, vs. legacy 152.00/155.00 @ 300k/600k)
  EDF=16.00  LST=8.00  ATC=301.00 (much worse once weights matter)  WSPT+BestFit=1152.00
```

**Observation:** The two fixes compound: dense_tardiness alone took Option 3 from
152/155 (legacy) to 56.00; adding real weights on top takes it to 28.00 -- within 20 of
EDF, and now dramatically ahead of ATC/WSPT (both of which get noticeably worse once
weights are real, since a "good" unweighted job ordering can now be a bad weighted one).
This is the best "real RL" (non-hyper-heuristic) result of the entire session.

**Follow-up:** Option 1 offline, dense_tardiness + weighted: **tardiness=12.00, late=7,
scheduled=100/100 -- BEATS EDF (16.00) outright**, closing in on LST (8.00). First time
any RL-trained policy this session has beaten EDF. This is now the single best offline
result of the entire session, RL or heuristic, other than LST/CP-SAT themselves.

**Follow-up: randomized-instance results, dense+weighted (50 held-out instances each):**
```
Option 1 (dense+weighted, randomized): tardiness= 49.32+/- 70.74  late=13.72  scheduled=99.44/100
Option 3 (dense+weighted, randomized): tardiness= 98.82+/- 94.83  late=12.08  scheduled=99.28/100
EDF:          tardiness= 37.30+/- 60.43  late=12.22  scheduled=96.84/100
LST:          tardiness= 23.94+/- 56.02  late= 8.26  scheduled=97.76/100
ATC:          tardiness=221.24+/-129.98  late=11.10  scheduled=94.68/100
WSPT+BestFit: tardiness=1221.40+/-170.28 late=40.10  scheduled=94.20/100
```
Both a huge improvement over the legacy-mode disaster found earlier today
(Option 1 legacy randomized: 1377.88; the LPT-collapse). Option 1 dense+weighted
(49.32) is now close to EDF (37.30) and dramatically ahead of the weight-aware
heuristics (ATC/WSPT), continuing the same compounding pattern as the fixed-instance
results. Rule-choice diagnostic on the new Option 1 randomized+dense+weighted
checkpoint: EDF ~85%, LPT ~13%, ATC/idle ~1% each -- genuine multi-rule switching,
not a collapse (LPT's reappearance here, despite being the earlier collapse rule, is
plausible as a deliberately-used minority choice rather than the sole strategy -- not
investigated further).

**Follow-up: Option 1 online rho~0.75, dense+weighted -- the online breakthrough.**
```
Option 1 (dense+weighted): tardiness=136.00  late=16  scheduled=818/866
EDF:          147.00  late=14  scheduled=822/866
ATC:          167.00  late=16  scheduled=819/866
SPT:          177.00  late=17  scheduled=816/866
WSPT+BestFit: 178.00  late=18  scheduled=821/866
```
**First time Option 1 has beaten every heuristic online**, including ATC (previously the
benchmark Option 1 couldn't reach: 163 vs 120 under legacy/dense-only). Rule-choice
diagnostic: SPT ~84%, LST ~10%, forced-idle ~12% -- still SPT-dominant, but now
genuinely mixing in LST rather than the complete single-rule collapse found earlier
today (legacy AND dense-only-unweighted both gave 100% SPT, 0% anything else, exactly
163.00 both times). The combination of dense_tardiness + real weights appears to be what
was needed to unstick the online policy from its earlier local optimum, not either fix
alone.

**Follow-up: Option 3 online rho~0.75, dense_tardiness (unweighted) -- first Option 3
online result at all.**
```
Option 3 (dense, unweighted): tardiness=199.00  late=19  scheduled=772/864
EDF: 193.00  ATC: 120.00  SPT: 163.00  WSPT+BestFit: 192.00
```
Worse than ATC/SPT, roughly level with EDF/WSPT, and notably fewer jobs scheduled
(772 vs 796-799 for every heuristic) -- Option 3's raw-feature-priority-learning
approach doesn't transfer to the online case as readily as Option 1's rule selection
did, at least not from dense_tardiness alone. Consistent with the offline pattern where
Option 3 needed BOTH dense_tardiness AND real weights to clearly beat ATC (56.00 dense-
only vs 28.00 dense+weighted) -- the weighted version is running now.

**Follow-up: Option 2 offline, dense+weighted -- completes the priority queue.**
```
Option 2 (dense+weighted): tardiness=45.00  late=10  scheduled=99/100  (was 525.00 legacy)
EDF: 16.00   LST: 8.00   ATC: 301.00   WSPT+BestFit: 1152.00
```
Even Option 2 -- the weakest performer all session (raw-feature priority learning, no
ATC feature, no hyper-heuristic structure) -- improves ~12x with the two fixes combined,
decisively beating both weight-aware heuristics. Confirms the dense_tardiness+weighted
combination is a genuine, broad fix, not something specific to Options 1/3's designs.

**Summary table, offline fixed instance, dense+weighted (all three options + heuristics,
same instance/weights):**
```
LST                     8.00  (best)
Option 1 (hyper-heur)  12.00  <- beats EDF
EDF                    16.00
Option 2 (raw feature) 45.00
ATC                   301.00
Option 3 (ATC-primed)   -- see 28.00 entry above (unweighted-vs-weighted eval instances
                            differ slightly run to run; both entries stand as reported)
WSPT+BestFit         1152.00
```

**Follow-up: full-scale (1.2M timestep) Option 1 validation, dense+weighted -- the
session's capstone offline result.**
```
Option 1 (dense+weighted, 1.2M): tardiness=11.00  late=8  scheduled=100/100
Option 1 (dense+weighted, 300k): tardiness=12.00  (essentially the same -- stable, not a fluke)
LST:  8.00   EDF: 16.00   ATC: 301.00   WSPT+BestFit: 1152.00
```
Confirms the 300k result holds at 4x the training budget -- RL is now within 3 tardiness
units of the best classical heuristic (LST, which itself matches CP-SAT's proven-optimal
floor for the unweighted case) and solidly ahead of EDF. This is the strongest, most
validated RL result of the entire session: starting point was ~1300 tardiness (7 failed
mechanisms), ending point is 11.00 -- roughly a 118x improvement, via action-space
reduction (Option 1) + dense-tardiness reward + real job weights, each contributing a
distinct, separately-verified piece of the fix.

**Follow-up: Option 3 online rho~0.75, dense+weighted -- the one result that did NOT
improve, completing tonight's full queue.**
```
Option 3 (dense+weighted): tardiness=210.00  late=20  scheduled=818/866
Option 3 (dense, unweighted): tardiness=199.00  (SLIGHTLY WORSE with weights added)
EDF: 147.00   ATC: 167.00   SPT: 177.00   WSPT+BestFit: 178.00
```
Unlike every other combination tonight (Option 1 offline/online, Option 2 offline,
Option 3 offline all improved substantially with dense+weighted), **Option 3 online got
marginally worse**, and remains behind every heuristic. Honest, unresolved finding, not
investigated further tonight -- Option 3's raw-feature-priority-learning approach may
simply need more training online (it has had only one 300k-timestep pass, vs. Option 1's
multiple passes and the full-scale 1.2M offline validation), or there may be a genuine
architectural mismatch between its continuous-scoring design and the online case's
harder credit-assignment problem. Flagged for review rather than chased further
autonomously -- a case where more training vs. a different fix isn't obvious from the
data alone, the kind of judgment call worth a second opinion rather than guessing.

**Conclusion / next step:** Full priority queue (offline + online, all three options,
dense+weighted, plus the 1.2M full-scale validation) now complete. Session summary:
Option 1 offline beats EDF (11.00 vs 16.00) and Option 1 online beats every heuristic
including ATC (136.00 vs 147.00) -- both firsts this session. Option 2/3 offline both
improve 4-20x. Option 3 online is the one open thread, flagged above rather than
resolved. Results artifact updated to reflect all of this
(https://claude.ai/artifact/Y1AFGzyT6P6jUWmbcwT1uc).

**Important correction, same day: the online "beats ATC" claim doesn't fully survive a
proper multi-instance check.** Ran the newly-built online `--randomized-eval` mode (20
held-out instances, seeds 500000-500019, same rho~0.75/dense/weighted config) on the
Option 1 checkpoint the single-seed result above was based on:
```
ATC:      tardiness=229.95+/- 99.28  late=24.10  scheduled=831.60/893  <- best, on average
Option 1: tardiness=242.65+/-101.82  late=22.60  scheduled=829.45/893
EDF:      tardiness=254.10+/-128.67  late=28.70  scheduled=827.90/893
SPT:      tardiness=266.70+/-138.92  late=22.40  scheduled=829.70/893
```
Across 20 instances, **ATC is still slightly ahead on average** -- the single seed=0
instance the earlier "Option 1 beats everyone" claim was based on happened to be one
where Option 1 did unusually well (or ATC unusually poorly). Option 1 still clearly and
consistently beats EDF and SPT, and the gap to ATC (242.65 vs 229.95, ~5%) is far smaller
than before this session's fixes (was 163 vs 120, ~36%) -- so the real, defensible claim
is "closed most of the gap to ATC, not fully closed it," not "beats ATC." This is exactly
the single-instance-noise risk flagged earlier tonight as a real gap in the online eval
protocol -- now caught in practice, not just in principle. Results artifact corrected to
match.

**Same check run on Option 3's online result, to see if its regression was also noise --
it wasn't.** 20 held-out instances, same config:
```
ATC:      229.95+/- 99.28   EDF: 254.10+/-128.67   SPT: 266.70+/-138.92
Option 3: 267.75+/-105.77  (worst of the four, consistent with the single-instance result)
```
Option 3's online underperformance is real and confirmed, not an artifact of one bad
seed -- unlike Option 1's case, more rigorous evaluation did not overturn the earlier
finding here.

**Methodological correction (user-prompted: "do we need to update the heuristics to work
with weighted jobs?"): every "tardiness" number reported since job weights were
introduced was the WRONG metric.** `SchedulingEnv.tardiness[job]` stores raw, unweighted
`T_j = max(0, C_j-d_j)` -- no weight multiplication anywhere in the env. Every comparison
this session used `tardiness.sum()` (raw), not the actual objective the reward function
optimizes (`lambda_2 * sum(w_j * T_j)`, weighted). This matters specifically for WSPT/ATC,
which are *designed* to deliberately sacrifice a low-weight job's timeliness to protect
high-weight ones -- exactly the trade raw tardiness can't see. The heuristics themselves
need no code changes (EDF/SPT/LST/FCFS/LPT are correctly weight-blind by definition;
WSPT/ATC already correctly use weight, confirmed earlier today) -- the EVALUATION metric
was wrong, not the heuristics.

**Recomputed with the correct metric (weighted tardiness = sum(w_j*T_j)):**
```
Offline (fixed instance, single instance as always for this track):
  Option 3 (1.2M): weighted=13.00  <- BEATS LST (24.00) outright, not just "close"
  LST:             weighted=24.00
  Option 1 (1.2M): weighted=30.00  <- still beats EDF
  EDF:             weighted=46.00
  ATC:             weighted=341.00
  WSPT+BestFit:     weighted=3013.00

Online (rho~0.75, 20 held-out instances, mean):
  ATC:      weighted=644.20+/-303.43  <- still best (same conclusion as the raw-tardiness
                                          20-instance check -- a smaller 5-instance check
                                          run first gave a misleading reversal, itself
                                          another instance of the small-sample trap)
  Option 1: weighted=730.30+/-306.91
  EDF:      weighted=769.70+/-414.44
  SPT:      weighted=788.20+/-389.26
```

**Observation:** The offline picture genuinely improves under the correct metric -- Option
3 doesn't just approach LST, it beats it, a real (same single instance both ways, not a
sample-size artifact) and important upgrade to the session's headline result. The online
picture is qualitatively unchanged from the raw-tardiness 20-instance correction: ATC
still wins, Option 1 closes most but not all of the gap. Both raw and weighted tardiness
now get reported going forward (see the eval-script fix below) so this can't silently
recur.

**Conclusion / next step:** `eval_action_space_variant.py` updated to report
`weighted_tardiness` alongside raw tardiness whenever `--job-weight-min/max` is set.
Results artifact corrected to lead with weighted tardiness (the actual objective) for
every weighted comparison; raw tardiness kept as a secondary/historical reference where
useful. This is now the permanent convention for any future weighted-instance
evaluation in this project.

**Follow-up: full-scale (1.2M) Option 2 offline validation, weighted metric -- completes
the offline story with all three options now at or above LST.**
```
Offline, weighted tardiness, full scale (1.2M each):
  Option 3: 13.00  <- beats LST outright
  LST:      24.00
  Option 2: 24.00  <- TIES LST exactly (was 49.00 at 300k)
  Option 1: 30.00  <- beats EDF
  EDF:      46.00
```
Every RL option now matches or beats the best classical heuristics under the correct
objective, at full training scale. This is the cleanest possible summary of the whole
session's offline arc: starting point ~1300 (raw, unweighted, 7 failed mechanisms),
ending point 13.00-30.00 (weighted, correct metric) across three independently-designed
action-space reductions, all converging near or past the best heuristic.

**Follow-up: more training HURT Option 1 online, unlike every offline case tonight.**
Tested whether more training (900k vs. 300k) would close the remaining online gap to ATC,
mirroring the pattern that worked for every offline option (300k->1.2M monotonically
improved all three). It didn't -- it regressed:
```
Online, weighted tardiness, 20 held-out instances:
  Option 1 (900k): 788.20+/-389.26  -- IDENTICAL to SPT's numbers exactly (266.70 raw too)
  Option 1 (300k): 730.30+/-306.91  -- better, and genuinely mixed SPT+LST (not a collapse)
  ATC:              644.20+/-303.43  (still best)
  EDF:               769.70+/-414.44
```
The 900k checkpoint collapsed onto a PURE SPT policy (byte-identical results, not just
similar) -- worse than the 300k checkpoint's mixed rule-switching behavior. More
training didn't refine the switching strategy, it un-learned it. This is a genuinely
different failure mode from anything else tonight: more training was uniformly good
offline, and uniformly bad (so far, n=1) for this specific online case. Possible causes
not yet investigated: entropy decay over more updates locking onto the easiest-to-execute
single rule, or the online credit-assignment problem being harder to sustain multi-rule
behavior across a longer training run. Flagged for review rather than chased further --
another case where the direction to try next isn't obvious from the data alone. The
300k checkpoint remains the best online Option 1 result and is not superseded by this.

**Root cause found: zero entropy regularization, confirmed from the training log
itself.** `train_action_space_variant.py` never sets `ent_coef` on `MaskablePPO`, so it
uses SB3's stock default of `0.0` -- nothing in the loss counteracts the policy becoming
more deterministic over time. Sampled `entropy_loss` across the 900k run:
```
step ~2k:     -2.03   (high entropy, genuinely exploring)
step ~100k:   -1.07
step ~400k:   -0.55
step ~700k:   -0.29
step ~900k:   -0.019  (near-total collapse -- ~99%+ probability mass on one action)
```
A clean, monotonic decay to near-zero entropy -- exactly consistent with the SPT-only
collapse. At 300k the policy hadn't yet fully collapsed (entropy_loss still meaningfully
negative), which is why it retained the mixed SPT/LST behavior; by 900k it had. This
project has hit and fixed an analogous collapse before: `2026-08-09-pointer-network-
action-head.md` raised A2C's `ent_coef` from `0.0` to `0.01` for exactly this reason
("no forcing function pushed exploration ... at a curriculum transition"). Testing the
same fix here now: `--ent-coef 0.01`, online, dense+weighted.

---

## 2026-09-18 (S2W10) -- Full-scale Option 3 offline validation (superseded numbers corrected above)

**Full-scale (1.2M timestep) Option 3 offline validation, matching Option 1's treatment
-- new best RL result of the entire session.**
```
Option 3 (dense+weighted, 1.2M): tardiness=10.00  late=3  scheduled=99/100
Option 3 (dense+weighted, 300k): tardiness=28.00  (continued improving with more training)
Option 1 (dense+weighted, 1.2M): tardiness=11.00  (previous best)
LST: 8.00   EDF: 16.00   ATC: 301.00   WSPT+BestFit: 1152.00
```
Option 3 -- genuinely learned per-job priority scoring, not rule selection -- now
essentially matches LST/CP-SAT's proven-optimal floor for this instance (10.00 vs 8.00)
and clearly beats EDF. This is arguably the more significant of the two full-scale
results: Option 1's win is real but bounded by its own rule menu (Section 6.1 above);
Option 3's is a genuinely novel, continuously-learned scheduling policy independently
converging to near-optimal behavior.

---

## 2026-09-18 (S2W10) -- Windowed action space (prepared, not trained) + Option 1 curriculum integration

**Config:** Implementation-only entry (overnight autonomous work, alongside the weighted
retrain queue). `Code/env/windowed_priority_gym_wrapper.py` (new): DeepRM-style bounded
action-space window for Options 2/3 -- `Discrete(window_size+1)` instead of
`Discrete(max_jobs+1)`, EDF-ordered window selection, one backlog scalar. Matching
adapted network (`windowed_priority_pointer_policy.py`) and SB3 wrapper. 7 new regression
tests (`tests/test_windowed_priority_wrapper.py`), all passing. Deliberately NOT wired
into any training run -- window size/ordering are real design choices for review, not
just engineering (see the full write-up,
`2026-09-18-windowed-action-space-and-curriculum-integration.md`).

Separately: `train_optimized.py::make_env()` gained `action_mode="rule_selection"` (wraps
with `RuleSelectionGymSchedulingEnv`, i.e. Option 1) and `job_weight_range`, so Option 1
can now run through this file's real curriculum/Optuna machinery instead of only the
standalone `train_action_space_variant.py` trainer -- needed for a genuine full-scale
(curriculum, not flat-timestep) validation run. Smoke-tested via `--no-curriculum`; this
surfaced a real, unrelated pre-existing rough edge (`--no-curriculum` ignores
`--stage4-timesteps`, always runs a hardcoded 300k) which turned the intended tiny smoke
test into a full 300k-timestep run -- not a bug in tonight's changes, noted for later
rather than fixed mid-integration.

**Observation:** Both pieces are additive/backward-compatible (default behaviour
unchanged, verified). No training results yet from either -- this entry exists to record
what was implemented, not what it produced.

**Follow-up: the curriculum-integration smoke test surfaced a real problem, not just the
`--stage4-timesteps` rough edge.** Evaluated the resulting checkpoint (300k timesteps,
legacy reward, unweighted, offline, via the curriculum-integrated path): **tardiness=
1435.00, late=45** -- dramatically WORSE than the standalone trainer's matching 300k
result (35.00), not equivalent as expected. Root cause: `train_optimized.py` loads
`ppo_best_params.json` (Optuna-tuned hyperparameters) regardless of `action_mode` --
those hyperparameters were tuned for the OLD, huge placement action space
(`learning_rate=1.31e-5` in the log, genuinely tiny) and do not transfer to Option 1's
completely different `Discrete(8)` rule-selection space, the same
Eimer et al. (2023) "hyperparameters tuned at one scale/setting don't transfer to
another" failure mode this project has now hit a third time (previously A2C, S2W5; PPO's
tardiness-tuned search, S2W9). The curriculum-integration MECHANISM itself is not at
fault (no crash, no wrong action-space size, no wrong checkpoint path) -- it faithfully
reproduced whatever hyperparameters it was told to use, and those were wrong for this
action space.

**Conclusion / next step:** Do NOT launch the planned full-scale (1.9M) validation
through the curriculum-integrated path with the existing Optuna params -- it would very
likely reproduce this failure at 6x the cost. Re-running Optuna specifically for the
`rule_selection` action space is real, separate work, out of scope to start unsupervised
overnight. Instead, the full-scale validation uses the ALREADY-VALIDATED standalone
trainer (SB3 default hyperparameters, which have now taken Option 1 from 35.00 -> 19.00
-> 12.00 across three separate improvements tonight), extended to a longer timestep
budget, on the current best-known config (dense_tardiness + weighted). Windowed action
space (Section 1) still awaits the user's design-choice review separately.

---

## 2026-09-18 (S2W10) -- Job weights: randomized, no longer dead code for WSPT/ATC (user-directed, foundational fix)

**Config:** `Code/env/env_config.py::generate_env_config()` and
`Code/env/arrival_process.py::generate_poisson_arrivals()` both gained `job_weight_range`
(default `None`, unchanged -- every job weight stays 1.0 exactly as before, so the seed=0
fixed instance and every historic CP-SAT/EDF/LST/ATC reference number stay reproducible).
Passing e.g. `(1, 6)` draws each job's weight i.i.d. Uniform{1,...,5}. Threaded through
`train_action_space_variant.py`/`eval_action_space_variant.py`'s new
`--job-weight-min`/`--job-weight-max` flags and `train_optimized.py`'s two resamplers.
6 new regression tests (`tests/test_job_weights.py`).

**Stats (review-the-outputs sanity check, seed=0, 100-job fixed instance):**
```
                  unweighted (all w=1)          weighted (mean w=3.00)
EDF               tardiness=  16.00              tardiness=  16.00   (unchanged, doesn't use weight)
LST               tardiness=   8.00              tardiness=   8.00   (unchanged, doesn't use weight)
SPT               tardiness=1321.00              tardiness=1321.00  (unchanged, doesn't use weight)
WSPT+BestFit      tardiness=1321.00 (=SPT exactly)  tardiness=1152.00 (now genuinely differs from SPT)
ATC               tardiness= 106.00              tardiness= 301.00  (genuinely different job ordering)
```

**Observation:** Confirms the fix works exactly as intended: WSPT (duration/weight ratio)
was previously byte-identical to SPT (duration/weight=duration/1=duration for every job)
-- already flagged as dead code in `priority_rules.py::wspt_key`'s own docstring, now
genuinely live. ATC's weight-aware urgency term also now produces real, different
scheduling decisions. Rules that never referenced weight (EDF/LST/SPT) are correctly
unaffected. This was raised by the user as something they believed was already
implemented ("I got told we were randomising job weights") -- checked memory and this
session's own history, found no prior record of the request; treating this as the honest
answer either way, and as a standing process fix (see CLAUDE.md's new "Follow-through on
concrete requests" section) rather than litigating the history further.

**Conclusion / next step:** This is a foundational change to problem generation (the
objective function's `lambda_2*w_j*T_j` term was, in effect, degenerate for every result
this project has ever produced) -- per the user's explicit instruction, retrain the
currently-active action-space-reduction work (Options 1/2/3, offline/online,
dense_tardiness, randomized-instance) with real weights. Retraining the full multi-week
historical archive (RCPO/PPO-Lagrangian/original A2C-PPO sweeps) is out of scope for one
overnight session and not attempted here -- flagged explicitly rather than silently
narrowed.

---

## 2026-09-18 (S2W10) -- Randomized-instance Option 1: the action-space fix alone did NOT transfer; same collapse mechanism, worse rule

**Config:** Option 1, `--randomize-instances` (fresh random job set every episode via
`make_random_instance_resampler()`), legacy reward, 300k timesteps. Evaluated via the new
`--randomized-eval` mode (50 held-out instances, seeds
RANDOM_INSTANCE_SEED_CEILING..+49, matching `eval_rl_agent.py`'s own convention).
Motivated by the user noticing the project's long-standing randomized-instance PPO
results were still terrible (~1327 tardiness, tracked all the way back to 2026-09-14) and
asking whether the newly-diagnosed action-space-size bottleneck explains it.

**Stats:**
```
Option 1 (legacy, randomized, 300k): tardiness=1377.88+/-158.85  late=44.30  scheduled=99.22/100
EDF  (same 50 held-out instances):   tardiness=  37.30+/- 60.43  late=12.22  scheduled=96.84/100
LST  (same 50 held-out instances):   tardiness=  23.94+/- 56.02  late= 8.26  scheduled=97.76/100

Rule-choice diagnostic (5 held-out instances, logging every decision):
  LPT ~80%, WSPT ~13%, FCFS ~6%, idle (forced) ~1% -- consistent across all 5 checked instances
```

**Observation:** Surprising, and worth being honest about: shrinking the action space did
NOT by itself fix the randomized-instance case -- 1377.88 is back in the same catastrophic
band as every historic PPO failure. Diagnosed by re-running the same rule-choice logging
used for the online SPT-collapse finding (2026-09-17 action-space-reduction.md Section
5.3): the policy collapsed onto **LPT** (Longest-Processing-Time-first) as its dominant
choice (~80% of decisions) -- a rule already measured in
`2026-08-28-classical-heuristic-baselines.md` at 1100-1400 tardiness, matching this result
almost exactly. This is the same failure *mechanism* as the online case (policy-gradient
collapse onto a single rule with zero adaptive switching), just landing on a worse rule
here, and it happened on the genuinely harder task (must generalize across instances,
can't memorize one) at the same 300k-timestep budget that was enough for the *fixed*
instance. Action-space size was necessary but not sufficient here; exploration/training
budget is the evident remaining gap.

**Conclusion / next step:** Immediately launched the same config with
`--reward-mode dense_tardiness` (`offline_randomized_dense`) -- well-motivated given
dense_tardiness fixed an extremely similar-looking collapse on the fixed instance
(2026-09-18 entry below). If that doesn't fix it either, more timesteps and/or a higher
PPO entropy coefficient (to sustain exploration past the early LPT local optimum) are the
next levers, in that order.

**Follow-up, same day (user-prompted: "why would it pick literally the worst rule"):**
Measured `total_reward` per rule directly (not just tardiness) on the same held-out
instances, under `reward_mode="legacy"`. **LPT+WorstFit has the HIGHEST reward of all 7
rules on every instance checked** (e.g. seed=500000: LPT reward=328.21 vs LST's 288.50,
despite LPT's tardiness=1249 vs LST's 0.00) -- because LPT reliably schedules all 100/100
jobs, collecting the flat `+3.0`-per-completion and `+50`-all-done bonuses on every one,
which dwarfs the weak, capped tardiness penalty. **This is not an exploration failure at
all: the policy correctly found the actual reward-maximizing rule.** The reward function
was wrong, not the search. This is the 2026-09-17 dense-tardiness-reward defect, now
caught with direct numbers rather than just derived algebraically.

The SAME check was then run for the ONLINE rho~0.75 case, under both legacy and
dense_tardiness, to see if it explains the online SPT-collapse the same way:
```
Online, legacy:          SPT reward=3083.87 (highest, narrowly over ATC's 3034.30)
Online, dense_tardiness: EDF reward=-63.59 (highest) > ATC's -63.94 > ... > SPT's -65.67 (near worst)
```
**Under dense_tardiness, SPT is measurably NOT the best rule -- EDF/ATC score better.**
But the online dense_tardiness checkpoint (trained from scratch on this exact reward for
the full 300k timesteps) still converged onto SPT anyway. Conclusion: the offline
LPT-collapse and online SPT-collapse are two DIFFERENT failure modes that happened to look
similar. Offline was a reward-design problem (fixed by changing the reward). Online is a
genuine exploration/optimization failure -- the correct answer was reachable under the
reward it was actually trained on, and it still didn't find it. Next lever for the online
case specifically: more timesteps and/or a higher PPO entropy coefficient, not another
reward-mode change.

---

## 2026-09-18 (S2W10) -- Dense-tardiness reward and potential shaping retested against Option 1's fixed action space

**Config:** Option 1, 300k timesteps, four combinations: {offline fixed instance, online
rho~0.75 lognormal} x {`reward_mode=dense_tardiness`, `use_potential_shaping=True`}.
Both mechanisms were tested and ruled out earlier this session against the OLD (huge)
action space (`2026-09-17-dense-tardiness-reward.md`) -- untested against Option 1's
`Discrete(8)` action space until now. New `--reward-mode`/`--use-potential-shaping`
CLI flags added to `train_action_space_variant.py` for this.

**Stats:**
```
                        Offline (300k)      Online rho~0.75 (300k, 864 realized)
Legacy (baseline)          35.00                 163.00 (identical to SPT)
Dense_tardiness             16.00 (=EDF)         163.00 (IDENTICAL to legacy -- no change)
Potential shaping          148.00 (worse)        246.00 (worse)
```

**Observation:** Dense-tardiness reward is a real, substantial win offline -- 300k
timesteps with it beats even the 600k-timestep legacy-reward run (19.00), landing exactly
on EDF (16.00). This confirms the hypothesis raised this session: dense_tardiness was
correctly ruled out against the old action space, but the old action space was masking a
real interaction with the new one. Online, it made literally zero difference -- byte-
identical tardiness/late/scheduled to the legacy-reward run, both landing on SPT's exact
numbers. Potential shaping hurts in both settings at this budget -- not investigated
further here (not the promising lead).

**Follow-up (2026-09-18, dense_tardiness now default -- see policy decision below):**
Option 3 offline, `--reward-mode dense_tardiness`, 300k timesteps: **tardiness=56.00**
(vs. 152.00/155.00 at 300k/600k under legacy -- a 2.7x improvement, and now beats ATC's
106.00 too). Confirms Option 3's legacy-mode plateau was the same reward defect, not a
training-budget ceiling -- the first result this session where genuinely learned
(non-rule-selecting) RL beats a strong classical heuristic.

**Conclusion / next step:** Carry dense_tardiness forward as the default reward mode for
further offline Option 1/3 work; drop potential shaping for now. The online null result
is itself informative: whatever is causing Option 1 to converge onto SPT online isn't a
reward-shaping problem, which points back toward exploration/local-optimum causes (a
higher entropy coefficient, more training) or the "not much RL" ceiling problem
(`2026-09-17-action-space-reduction.md` Section 6.1) rather than the reward function --
consistent with this session's broader pattern of ruling out reward-side explanations one
at a time.

**Follow-up (2026-09-18): dense_tardiness fixes both the score AND the collapse mechanism.**
Same config, `--reward-mode dense_tardiness`, 300k timesteps, evaluated identically
(50 held-out instances): **tardiness=359.40+/-223.66** (vs. legacy's 1377.88 -- ~3.8x
better), late=24.18, scheduled=97.04/100. Still behind EDF (37.30) and LST (23.94). Rule-
choice diagnostic (same 5 instances as the legacy check): **ATC ~63%, LST ~22%, WSPT ~9%,
EDF ~4%, idle ~3%** -- genuine state-dependent switching among four rules, qualitatively
different from every prior collapse this session (online SPT-only, offline-randomized
LPT-only under legacy). Dense_tardiness fixed the underlying mechanism, not just the
number -- the remaining gap to EDF/LST is presumably an undertrained/uncalibrated
switching policy, not another collapse, though not yet verified further.

**Policy decision (2026-09-18, user-directed, following the LPT-reward-measurement finding
below):** `dense_tardiness` is now the DEFAULT reward mode for all new action-space-
reduction training, not an occasional A/B check -- the direct measurement that LPT
out-scores every genuinely-good rule under legacy reward on every held-out instance
checked (see below) is treated as decisive, not just suggestive. Caveat carried forward
honestly: this is necessary but not sufficient by itself -- the online SPT-collapse case
(same entry) shows a policy that already converged before finding the better answer isn't
automatically fixed by changing the reward mode alone; training budget/exploration may
still need separate attention there. The in-flight `option3_online_rho075` (legacy) run
was killed and relaunched as `option3_online_rho075_dense`; `option3_offline_dense`
(new, testing whether dense_tardiness also fixes Option 3's offline 300k/600k legacy
plateau) launched alongside it.

---

## 2026-09-17 (S2W10) -- Option 1 validation: offline 600k-timestep scale-up + online heavy-tailed comparison

**Config:** Four background jobs run concurrently: (1) Option 1, offline fixed instance,
600k timesteps (2x the earlier 300k comparison run, `--save-tag offline_600k`); (2)
Option 1, online, `job_size_distribution=lognormal`, `rho~0.25` (arrival_rate=3,
`--save-tag online_lognormal`); (3) Option 1, online, lognormal, `rho~0.75`
(arrival_rate=9, `--save-tag online_lognormal_rho075`); (4) retrospective CP-SAT oracle,
online, lognormal, `rho~0.75`, same instance as (3), 600s time limit, 2 search workers.
A 5th job (offline CP-SAT, 900s, 1 search worker, run under heavy contention from the
other four) is excluded from the results below -- see Observation.

**Stats:**
```
Offline fixed instance (300 evals timesteps=300k vs 600k):
  Option 1 (300k):  tardiness= 35.00  late=10  scheduled=100/100
  Option 1 (600k):  tardiness= 19.00  late=11  scheduled=100/100
  EDF:               tardiness= 16.00  late=10  scheduled= 98/100
  LST:                tardiness=  8.00  late= 7  scheduled= 99/100

Online, lognormal, rho~0.25 (309 realized, seed=0): Option 1 == every one of 9
  DEFAULT_HEURISTICS exactly (tardiness=36.00, late=2, scheduled=298/309) -- the
  no-differentiation regime already found in the heuristic-only sweep.

Online, lognormal, rho~0.75 (864 realized, seed=0):
  CP-SAT oracle (hindsight, UNKNOWN status): best_bound=105.0
  ATC:               tardiness=120.00  late=14  scheduled=799/864  (best live policy)
  Option 1:          tardiness=163.00  late=15  scheduled=799/864  (IDENTICAL to SPT)
  SPT:               tardiness=163.00  late=15  scheduled=799/864
  WSPT+BestFit:      tardiness=192.00  late=19  scheduled=796/864
  EDF:               tardiness=193.00  late=25  scheduled=797/864
  EDF+BestFit:       tardiness=235.00  late=23  scheduled=792/864
  Tetris:            tardiness=248.00  late=18  scheduled=799/864
  LST:               tardiness=275.00  late=28  scheduled=789/864
  FCFS+FirstFit:     tardiness=294.00  late=23  scheduled=793/864
  LPT+WorstFit:      tardiness=634.00  late=34  scheduled=759/864  (worst)
```

**Observation:** Offline: doubling timesteps (300k->600k) roughly halved tardiness again
(35->19), closing in on EDF (16) -- no sign of a plateau, so the full 1.9M-timestep run
is likely to close the gap further, possibly past EDF. Online rho~0.75 (the genuinely
differentiated regime found in the earlier heuristic sweep): Option 1, trained only
300k timesteps with no curriculum/tuning, beats 6 of 8 other heuristics but loses to ATC
(163 vs 120) and sits ~55% above the CP-SAT oracle's hindsight floor (105) vs. ATC's
~14%. Option 1's numbers are *exactly* identical to SPT's, not just close -- strong
evidence the learned policy converged to "always pick SPT" on this instance rather than
discovering a genuinely novel strategy, unlike the offline case where it clearly beats
every single-rule baseline. This is the most honest result of the session so far: RL
does not automatically dominate in the harder, literature-grounded online setting the
way it did on the offline fixed instance -- a real, literature-grounded heuristic (ATC)
still wins there at this training budget. The offline CP-SAT run (900s, 1 search worker,
run concurrently with the other four CPU-bound jobs) returned tardiness=500/best_bound=0,
worse than every heuristic -- attributed to running single-threaded under heavy
contention (CP-SAT leans heavily on parallel search workers at this job count), NOT
treated as a real update to the established offline floor (LST=8.0).

**Conclusion / next step:** Option 1 is confirmed as the right choice for a full
1.9M-timestep offline validation run (not yet done -- ~6hr estimated cost, deferred).
For the online case specifically, ATC remains the benchmark to beat -- Option 1 would
need more training/tuning (or a smarter action-space design) to surpass it, not just
match SPT. Re-running the online rho~0.75 comparison at a larger timestep budget (mirroring
the offline 300k->600k improvement) is a natural next step before concluding anything
final about RL vs. heuristics in the online heavy-tailed setting.

---

## 2026-09-17 (S2W10) -- Heavy-tailed (log-normal) job sizes for the online case

**Config:** `Code/env/arrival_process.py::generate_poisson_arrivals()` gained
`job_size_distribution="uniform"|"lognormal"` (default "uniform", unchanged). Under
"lognormal", duration and per-resource demand are drawn log-normal, mean-matched to the
uniform baseline's mean (`mu = ln(target_mean) - sigma^2/2`, an exact property of the
log-normal distribution) so `rho = arrival_rate/12` calibration stays valid -- only
variance/skew changes. Threaded through `train_optimized.py`'s `--job-size-distribution`
CLI flag.

**Stats:**
```
tests/test_heavy_tailed_arrivals.py (6 checks), seed=1, horizon=200, arrival_rate=2.0:
  uniform duration:   mean=5.13  std=2.56  max=10 (range ceiling)
  lognormal duration: mean=5.31  std=6.39  max=40 (4x range ceiling -- genuine elephants)
  max realized resource demand: 29 (machine_capacity_value=30 -- always schedulable by size)
```

**Observation:** Motivated by the user's own diagnosis that pushing `rho` toward/above 1
to force lateness "essentially just makes it offline" -- raw volume tests throughput, not
decision-making under genuine uncertainty about the future, which was the actual point of
the online case. Checked real-world grounding rather than picking a "realistic-sounding"
distribution by feel: Google's (Reiss et al. 2012, SoCC) and Microsoft Azure's (Cortez et
al. 2017, SOSP) published cluster-trace analyses both independently report heavily
right-skewed job-size/resource-demand distributions ("mostly mice, occasionally
elephants") -- log-normal is the standard parametric family for this. Mean-matching means
this isolates distribution *shape* as the only changed variable at any given nominal rho,
rather than confounding it with a load increase.

**Conclusion / next step:** Implemented and validated (mean-matching holds empirically,
genuine heavy tail confirmed, no job ever unschedulable by size alone). Follow-up
heuristic sweep (same day, user-prompted): at `rho~0.25` every one of 9
`DEFAULT_HEURISTICS` produced *identical* results (all failures traced to the same
11 arrive-too-late-to-finish jobs) -- heavy tails alone don't differentiate scheduling
skill at low load, only at `rho~0.75-1.0` do heuristics genuinely diverge (ATC best at
0.75: tardiness=120/864 scheduled=799; WSPT+BestFit best at 1.0: tardiness=333/1199
scheduled=1071; LPT+WorstFit worst at both). Option 1 is now training against both
`rho~0.25` and `rho~0.75` heavy-tailed online instances in the background for direct
comparison. Full writeup: `2026-09-17-heavy-tailed-arrivals.md`.

---

## 2026-09-17 (S2W10) -- Retrospective CP-SAT oracle for the online case + rho>1 testing findings

**Config:** `Code/baselines/exact_solver.py` gained `earliest_start` and
`enforce_single_start_per_tick` params on `solve()`, plus new
`solve_retrospective()`/`make_retrospective_config()` and a `--online` CLI mode.
No offline-case behaviour changed (`earliest_start=None`,
`enforce_single_start_per_tick=True` are both the prior defaults).

**Stats:**
```
Small scale (rho~0.04, 12 realized/9 completable arrivals, horizon=15):
  Retrospective CP-SAT oracle: tardiness=0.00, scheduled=9/9, status=OPTIMAL
  EDF / ATC (live online):     tardiness=0.00, scheduled=9/9   (exact match)

Deployed scale, rho~0.67 (arrival_rate=8, horizon=100, max_jobs=200):
  Retrospective CP-SAT oracle: status=UNKNOWN (NP-hard at this scale, matches
                                the offline case's own ~10-job proven-optimal limit)
  EDF: tardiness=0.00, scheduled=200/200   ATC: tardiness=0.00, scheduled=200/200

Deployed scale, rho~1.17 (arrival_rate=14, horizon=100), max_jobs=300 (WRONG --
see finding below) vs. max_jobs=2000 (correct):
  max_jobs=300:  all 300 arrivals land by tick 20 of 100 -- 80-tick uncontested
                 tail -> EDF/ATC both 0.00 tardiness, scheduled=300/300 (fake result)
  max_jobs=2000: 1411 realized arrivals sustained through tick 97 -- EDF:
                 tardiness=0.00 but scheduled=1136/1352 completable; ATC:
                 tardiness=0.00 but scheduled=1145/1352 completable
```

**Observation:** Two real findings, not just an implementation exercise. (1) The
retrospective oracle's first draft reused the offline model's
`AddAllDifferent(start)` unchanged, which does not hold for the online case
(`OnlineSchedulingEnv`'s Option 2 tick-advance relaxation allows same-tick
placements) -- this produced a proven-`INFEASIBLE` result at deployed scale via a
plain pigeonhole contradiction (more jobs than distinct tick values), caught by
noticing the status was `INFEASIBLE` (a proof) rather than `UNKNOWN` (a timeout)
and not accepting it at face value. (2) `generate_poisson_arrivals()` stops
drawing arrivals once `max_jobs` slots fill -- at high `arrival_rate` this
happens almost immediately, leaving a long arrival-free tail that lets any
heuristic fully drain its backlog and report a misleadingly perfect 0.00
tardiness. Once `max_jobs` is sized to sustain arrivals across the whole horizon,
rho>1 overload is real but shows up as **job abandonment**
(`jobs_scheduled` collapsing, ~15-18% of completable jobs never reached), not as
tardiness on the jobs that do get scheduled -- tardiness alone is the wrong metric
to read for rho>1 comparisons.

**Conclusion / next step:** Oracle mechanism is correct and validated; a
long-time-limit run at deployed scale for a real citable bound is future work
(matches the offline case's own eventual "final" CP-SAT protocol). Any future
rho>1 online comparison (Options 1/2/3, or PPO/A2C) must report
`jobs_scheduled`/completion-rate alongside tardiness or a policy that "solves"
overload by quietly abandoning the hardest jobs will look identical to one that
doesn't. Full writeup:
`2026-09-17-retrospective-cpsat-oracle-and-rho-testing.md`.

---

## 2026-09-17 (S2W10) -- Comprehensive reward redesign result: dense per-tick tardiness does NOT fix PPO either

**Config:** `Code/env/scheduling_env.py` gained `reward_mode="dense_tardiness"` -- an
exact, algebraically-verified per-tick decomposition of weighted tardiness (charges
`-lambda_2*w_j/horizon` every tick a job remains unfinished past its deadline,
reproducing the legacy lump-sum total exactly -- see
`2026-09-17-dense-tardiness-reward.md` and `tests/test_dense_tardiness_reward.py`),
with the flat `+3.0`/`+50` completion bonuses removed. Modelled directly on DeepRM
(Mao et al., 2016) and Decima (Mao et al., 2019)'s reward-design principle, in direct
response to the mathematically-proven defect in the legacy reward (the flat bonuses
always exceeding the bounded tardiness penalty for every `lambda_2` any search here
has found). Tested both a fast (300k, no-curriculum) and a full-scale (1.9M,
full curriculum) PPO run, same Optuna-tuned hyperparameters as every other offline
PPO result, `lambda_2` unchanged.

**Stats:**
```
                        tardiness   P95     late     scheduled
Legacy (full-scale):     1301.12   58.92   41.96/100  96.90/100
Dense  (full-scale):     1309.66   59.00   41.72/100  96.74/100
Legacy (300k quick):     1337.43   58.89   42.40/100  96.73/100
Dense  (300k quick):     1336.93   60.27   41.97/100  96.80/100
```

**Observation:** No idle-collapse (the honestly-flagged risk from removing the flat
bonuses never materialized -- jobs-scheduled stays healthy at ~96.7-96.9/100 in both
modes). But also **no improvement whatsoever** -- dense_tardiness and legacy are
statistically indistinguishable, both landing squarely in the same ~1290-1330 band
every other PPO variant has landed in tonight. This is despite the dense reward
visibly producing much better-behaved training dynamics (`value_loss` ~0.02-0.03 for
dense vs. ~30-46 for legacy, `explained_variance` similarly better) -- confirming the
literature's credit-assignment argument was directionally correct (a dense, every-
tick reward IS more learnable/predictable for the value function), but that improved
learnability did not translate into a better final *tardiness* policy.

**This is the seventh mechanism tried for PPO's tardiness tonight** (fixed reward
weight, four Lagrangian $\lambda_{\max}$ ceilings, a tardiness-directed
hyperparameter search, pointer-network architecture, and now a comprehensive,
literature-grounded reward redesign) -- all seven converge on the same band. The
reward-function hypothesis was well-reasoned, correctly and rigorously implemented
(verified algebraically exact, not just empirically plausible), and is now also ruled
out as the dominant bottleneck, alongside architecture and hyperparameters. A genuine,
structural difference from DeepRM/Decima that has NOT yet been tested: **action space
size**. DeepRM's action space is ~11 discrete choices (M=10 job slots + void); this
project's is `max_jobs*num_machines+1` (1000+ choices for the 100-job/10-machine
deployed scale). A combinatorially large action space is a well-known driver of
sample-inefficiency and poor convergence in policy-gradient methods independent of
reward design -- this was named as "out of scope" when the reward redesign was
planned, on the assumption the reward was the more likely culprit; given the reward
fix's negative result, this becomes the leading untested structural hypothesis.

**Conclusion / next step:** Do not pursue further reward-function variants for PPO's
tardiness without new evidence pointing back at the reward specifically -- seven
mechanisms is a strong signal this isn't where the problem lives. The action-space-
size hypothesis (and, more generally, PPO's clipped-surrogate optimization dynamics
on this specific problem, independent of any single input to it) is the recommended
next investigation, not attempted tonight -- flagged for the user's input before
committing more compute to it, given the scale of investigation already spent on this
question. `dense_tardiness` is kept in the codebase (opt-in, A/B-able, well-tested)
since it's a real, literature-grounded improvement in training dynamics even though
it didn't move the tardiness needle here -- worth re-testing on A2C and the online
case rather than discarding.

---

## 2026-09-16 (S2W9) -- Online case: first real campaign launched (3-rate sweep x 2 algorithms)

**Config:** User confirmed 4 scope decisions for the first real online campaign
(not just validation runs): (1) sweep 3 arrival rates rather than one fixed
target, given tonight's rho~0.7-vs-rho~0.92 finding; (2) single-stage
(`--no-curriculum`) -- online curriculum scaling is an unresolved design
question, not solved tonight; (3) baseline only for now (no importance-feature
or hotspot-penalty ablation yet); (4) both PPO and A2C.

Concrete choices made to operationalize this (not separately re-confirmed with
the user -- reasonable defaults consistent with tonight's established
patterns, flagged here for visibility): three rates spanning known-trivial to
known-discriminating (light rho~0.5 / moderate rho~0.8, the untested gap
between tonight's two data points / heavy rho~0.95, just past tonight's
discriminating rho~0.92), each with `max_jobs` sized to the rate's expected
arrival count plus ~25% headroom, `--randomize-instances` (fresh Poisson draw
every episode -- treated as the more natural "online" default over a single
repeated arrival sequence), single seed for this first pass (matching how the
offline campaign itself started single-seed before its later 3-seed rigor
pass), 300k timesteps each:

| Rate | nu | rho | max_jobs |
|---|---|---|---|
| Light | 6.0 | ~0.50 | 750 |
| Moderate | 9.6 | ~0.80 | 1150 |
| Heavy | 11.4 | ~0.95 | 1350 |

6 runs (`online_sweep_{ppo,a2c}_{light,moderate,heavy}`) launched concurrently.

**Bug hit and fixed during this campaign (before results below could be trusted):**
concurrent runs at different arrival rates shared the same canonical checkpoint
path (`path_suffix` never distinguished `arrival_rate`) -- verified via checkpoint
weight hashing that no actual corruption occurred this time (the runs happened to
finish at staggered times, purely by luck, since heavier configs take longer), but
fixed properly (`online_suffix` added to `path_suffix`) rather than relying on that
luck again. Separately, `eval_rl_agent.py`'s A2C loading path always built its
"what shape network do I construct" template env via bare `make_env()`, which reads
the shared `ENV_CONFIG_PATH` file -- for 2 of 6 evals (a2c moderate/heavy) this
didn't match the specific checkpoint being loaded (a different concurrent run had
last written that file), crashing with an action-mask shape mismatch. Fixed: the
template env now uses the same explicit `--online`/`--randomized-eval` dimension
overrides as the actual eval instances, not a blind shared-file read.

**Stats** (15 held-out instances per cell; single seed, not yet 3-seed-verified):
```
                  rho~0.50 (trivial)      rho~0.80                  rho~0.95
                  tardiness / late        tardiness / late           tardiness / late
EDF/LST/ATC       0.00 / 0.00             0.00 / 0.00                0.00 / 0.00
FCFS+FirstFit     0.00 / 0.00             0.47 / 0.33                63.93 / 21.53
PPO               0.00 / 0.00             21.73 / 3.07               392.93 / 30.60
A2C (pointer)     0.00 / 0.00             176.00 / 10.00             1031.00 / 59.20
```

**Observation -- a genuine, important reversal from the offline case.** Deadline-
aware heuristics (EDF/LST/ATC) stay *perfectly* on time even at rho~0.95; FCFS
degrades gracefully as load rises (deadline-blind, so this is expected). But
**A2C is now the WORSE learned policy, not the better one** -- at rho~0.80 A2C's
tardiness (176.00) is 8x PPO's (21.73); at rho~0.95 A2C's tardiness (1031.00,
matching the *scale* of offline PPO's worst results) is 2.6x PPO's (392.93).
This is the opposite ranking from every offline result this project has produced,
where A2C (pointer architecture) was consistently and substantially better than
PPO. Neither algorithm comes close to the heuristics' near-perfect online
performance, but PPO is clearly the less-bad learner here.

Plausible explanations, **not yet investigated, stated as open hypotheses only**:
A2C's hand-rolled training loop runs a single, unvectorized environment (no
parallel rollout collection, unlike PPO's `n_envs=2` here), which may cope worse
with the added stochasticity of dynamic arrivals than PPO's more diverse,
parallel experience; alternatively, A2C's global-context pooling in
`PointerActorCritic` (mean over all currently-revealed jobs) may be a worse fit
when the revealed-job population itself is constantly changing size and
composition, compared to the offline case's fixed job set. Both are hypotheses,
not conclusions -- no ablation has been run to distinguish them.

**Conclusion / next step:** This reverses the "A2C is just better" assumption
carried over from the offline campaign -- do not assume it holds for the online
case without re-testing. 3-seed verification at rho~0.80 and rho~0.95 is the
natural next step before treating this ranking as confirmed, followed by
investigating *why* (the two hypotheses above, and/or a direct training-dynamics
comparison akin to what's now also needed for the offline PPO-vs-A2C puzzle).

---

## 2026-09-16 (S2W9) -- Online case (dynamic arrivals): MDP design + core implementation

**Config:** N/A -- implementation, not a training run. Full design writeup:
`2026-09-16-online-arrival-mdp-design.md`.

**Stats:** `python -m tests.test_online_env` -- all 6 checks + 1 regression guard
pass.

**Observation:** Implemented `OnlineSchedulingEnv`/`OnlineGymSchedulingEnv`
(subclassing, not modifying, the offline classes), `generate_poisson_arrivals()`,
and wired `--online`/`--arrival-rate` into `train_optimized.py` and
`eval_rl_agent.py`. Found and fixed **three real bugs** during implementation,
one of which (the invalid-action-detection proxy misclassifying valid same-tick
placements as invalid, eventually truncating episodes) would have silently
corrupted every online training run without ever raising an exception -- see the
dated doc's Section 3 for full detail on all three. Small-scale smoke tests (both
flat-MLP and, via `build_vec_env`, the full PPO training path) pass; no real-scale
training run yet.

**Conclusion / next step:** Core environment is solid and tested. Remaining: the
importance-feature and hotspot-penalty ablations (both scoped, neither
implemented), a real deployment-scale training run, and combining with
pointer-network PPO once that lands (see next entry).

**Update, same day: first real (demo-scale) online training + eval run.**
`train_optimized.py --algo ppo --policy-type flat --no-curriculum --online
--arrival-rate 1.0` (300k timesteps, flat MLP -- pointer-PPO not yet tuned),
evaluated on 30 held-out Poisson instances against EDF/LST/ATC/FCFS:
```
PPO:           reward=3132.16  tardiness=0.00  late=0.00/100  scheduled=93.93/100
EDF/LST/ATC:   reward=3139.2-3139.3  tardiness=0.00  late=0.00/100  scheduled=93.93/100
```
**Important caveat**: `arrival_rate=1.0` gives $\rho=1/12\approx0.08$ -- a very
light load, not the calibrated $\rho\approx0.7$ target agreed for the real
experiment. Every method (including untrained-comparable heuristics) achieves
zero tardiness here, which is expected at this load, not evidence PPO has solved
the online case. This run's purpose was **pipeline validation** (train -> eval ->
baseline-comparison -> CSV logging all work correctly end-to-end for the online
case), which it confirms; it is not yet a meaningful performance result.
A real run needs `max_jobs` sized for the target rate (expected arrivals over
$H=100$ at $\rho\approx0.7$ is $\approx840$, not 100 -- see the arrival-rate
scale discussion) and, ideally, pointer-network PPO once that Optuna search
(next entry) completes and full training runs.

**Update, same day: real target-load run (rho~0.7, max_jobs=900) -- a genuinely
new finding.** Same flat-MLP PPO, 300k timesteps, `--arrival-rate 8.4 --max-jobs
900`, evaluated on 20 held-out instances:
```
PPO:           reward=5850.15  tardiness=0.10  P95=0.00  late=0.05/808  scheduled=808.05/808
EDF/LST/ATC/FCFS: reward=6190-6211  tardiness=0.00  late=0.00/808  scheduled=808.2-808.3/808
```
**Everyone -- PPO and every heuristic alike -- achieves essentially zero tardiness
at this load**, unlike the offline case where PPO was specifically deadline-blind.
This is a real, structurally-explainable finding, not noise: under Option 2's
relaxed tick-advance rule (many placements allowed per tick), the system's
effective per-tick service capacity is far higher than the offline single-decision
model implicitly assumed when $\rho=\nu/12$ was derived -- so a nominally
"moderately loaded" $\rho\approx0.7$ (computed the same way as the offline
resource-utilization bound) turns out not to meaningfully stress deadline-keeping
once multiple placements per tick are allowed. **This means the earlier assumption
that $\rho\approx0.7$ would be "interesting but stable" doesn't hold in practice
under the chosen relaxation** -- reaching a genuinely challenging online regime
likely needs a substantially higher $\nu$ (pushing $\rho$ toward or past 1) than
the offline-derived formula alone suggests, since that formula doesn't account for
Option 2's much higher effective throughput.

The informative gap that *does* show up is elsewhere: PPO's reward (5850) is
~350-360 below every heuristic's (~6200) despite near-identical tardiness/
completion -- meaning PPO is doing something less efficient on the *other* reward
terms (machine activation, hotspot, idling), not tardiness, at this load. This is
a genuinely different online-case story from the offline one, worth a real
ablation/investigation once more runs exist, not just this one data point.

**Conclusion / next step:** Re-run at a higher $\nu$ (e.g. targeting
$\rho\gtrsim0.9$-1.0) to find a load that actually stresses deadline-keeping under
the relaxed clock, and investigate the activation/hotspot/idle gap directly (e.g.
per-term reward breakdown, not just the aggregate) rather than assuming it's the
same tardiness story as offline.

**Update, same day: high-load run (rho~0.92, arrival_rate=11, max_jobs=1200) --
reproduces the offline case's core finding in the online setting.** Same
flat-MLP PPO, 300k timesteps, evaluated on 15 held-out instances:
```
PPO:           reward=3339.42  tardiness=287.93  P95=0.00  late=23.87/1028  scheduled=1027.93/~1031
EDF/LST/ATC:   reward=3371-3390  tardiness=0.00   P95=0.00  late=0.00/1032  scheduled=1029.8-1031.7
FCFS+FirstFit: reward=3402.92  tardiness=31.20   P95=0.00  late=12.87/1034  scheduled=1033.80
```
**This is the discriminating load the rho~0.7 run wasn't**: deadline-aware
heuristics (EDF/LST/ATC) still achieve *perfect* tardiness even at rho~0.92, while
PPO shows real, meaningful tardiness and even underperforms simple
deadline-blind FCFS. This is qualitatively the same pattern as the offline
case's PPO-vs-heuristics gap (`training-log.md`'s 2026-09-15 overnight entry) --
evidence that PPO's tardiness-blindness is a robust phenomenon across both the
offline and online formulations, not an offline-specific artifact. Single-seed,
single-run -- not yet 3-seed-verified the way the offline results were, so
treat as a strong signal rather than a confirmed result.

**Conclusion / next step:** rho~0.92 is the right ballpark for a genuinely
informative online arrival-rate target going forward (supersedes the earlier
rho~0.7 recommendation). Natural next steps once returning to this thread:
3-seed verification at this load, and testing whether the same interventions
that failed for offline PPO (Lagrangian, pointer architecture) also fail here --
though given tonight's pointer-PPO result (previous entries), that specific one
is not a promising avenue to repeat.

---

## 2026-09-16 (S2W9) -- Environment breakage found and fixed (`requirements.txt` UTF-16 encoding)

**Config:** N/A -- infrastructure, not a training run.

**Observation:** At the start of this autonomous session, `sb3_contrib`/
`stable_baselines3`/`torch` were not importable under any Python installation
discoverable on this machine, despite the previous session having produced
real checkpoints and eval results hours earlier. Root cause:
`requirements.txt` was UTF-16LE-encoded (with BOM) -- a known PowerShell
`Out-File`/`Set-Content` default-encoding gotcha when the encoding isn't
given explicitly. `pip install -r` does not auto-detect UTF-16 and appears to
silently no-op or partially fail rather than raising a clear parse error,
which is presumably how a prior session's environment drifted out of sync
with its own dependency list without anyone noticing at the time.

**Conclusion / next step:** Rewrote `requirements.txt` as UTF-8 and
reinstalled (`pip install -r requirements.txt`); all packages now import
correctly. If any future session hits an inexplicable "package not found"
error despite `requirements.txt` looking normal when opened in an editor,
check the file's actual byte encoding first
(`file requirements.txt` / check for a `fffe` BOM) before assuming the
package was never installed. Avoid `Out-File`/`Set-Content` without
`-Encoding utf8` when writing any file another tool (pip, etc.) will parse.

---

## 2026-09-16 (S2W9) -- A2C 3-seed rigor pass evaluated: shaped is solid, RCPO is unstable (new negative finding)

**Config:** Evaluated the 3-seed A2C checkpoints trained in the previous
session (`a2c_pointer_scheduling_optimized_{shaped,shaped_rcpo}_seed{1,2,3}.pt`,
`--hidden 32` per `a2c_pointer_best_params.json`) with the same
`eval_rl_agent.py --randomized-eval --num-jobs 100 --num-machines 10
--horizon 100 --max-jobs 100` protocol used for every other 3-seed set.

**Stats:**
```
a2c shaped:      seed1 reward=277.95 tardiness=28.48  late=9.66  sched=96.50/100
                 seed2 reward=262.08 tardiness=27.30  late=9.46  sched=91.14/100
                 seed3 reward=228.32 tardiness=31.20  late=9.96  sched=81.76/100

a2c shaped+rcpo: seed1 reward=154.40 tardiness=219.60  late=11.50  sched=60.04/100
                 seed2 reward=161.29 tardiness=171.22  late=11.04  sched=61.84/100
                 seed3 reward=271.48 tardiness=1299.66 late=42.16  sched=97.08/100
```

**Observation:** Plain A2C (shaped, no RCPO) confirms the strong tardiness
already suggested by the earlier single-run check (~28-29): consistent
across all 3 seeds, vastly better than any PPO variant's ~1290-1330. A2C+RCPO
is a genuinely new negative finding, however: high seed-to-seed variance
(tardiness 171-1300, a >7x spread) and, critically, **seed3 shows the exact
same reward-hacking signature already diagnosed for PPO-Lagrangian**
(2026-09-14-ppo-lagrangian-and-reward-structure.md): high completion
(97/100 scheduled) paired with tardiness back at the bad-PPO band, meaning
the Lagrangian-adapted cost is being paid off by scheduling jobs late rather
than by avoiding lateness. RCPO does not reliably preserve or improve on
A2C's baseline good tardiness -- if anything it makes results worse and far
less predictable across seeds.

**Conclusion / next step:** User flagged this as "not acceptable" and wants
it reviewed later -- not fixed unsupervised in this session, consistent with
the standing decision (training-log 2026-09-15 entries) that the Lagrangian
instability's real fix (two-critic architecture, separate reward/cost value
functions, Ray/Achiam/Amodei 2019) needs human design input rather than
further unsupervised hyperparameter tuning. Recorded here as a confirmed,
3-seed-verified finding, not a single-run anomaly: **RCPO's instability is
not PPO-specific** -- it also affects A2C's hand-rolled implementation, which
points toward the Lagrangian-mutates-the-reward-in-place mechanism itself
(shared by both algorithms' current RCPO implementation) as the common root
cause, rather than something specific to either algorithm's architecture.

---

## 2026-09-16 (S2W9) -- A2C randomized-instance brought to 3 seeds: confirms the known tardiness-transfer caveat

**Config:** Trained `a2c_pointer_randinst_seed{1,2,3}` (previously only 1
untracked seed existed) for parity with PPO's randomized-instance set --
same protocol as the fixed-instance 3-seed set, `--randomize-instances`
added. Evaluated the same way.

**Stats:**
```
seed1: reward=281.46 tardiness=1589.98 late=36.16 sched=98.52/100
seed2: reward=271.17 tardiness=1723.42 late=37.54 sched=97.84/100
seed3: reward=280.78 tardiness=1580.16 late=35.14 sched=98.58/100
```

**Observation:** Tight 3-seed agreement confirms (not just suggests, as the
single earlier seed did) the caveat already recorded in
`project_offline_training_campaign` memory and the 2026-08-20 entry: A2C's
excellent fixed-instance tardiness (~28) does not transfer to the
randomized-instance distribution at all -- it lands *worse* than every PPO
variant's ~1290-1330 band. Root cause remains as previously diagnosed: the
Optuna hyperparameters (`a2c_pointer_best_params.json`) were tuned against
the fixed instance only and never re-tuned for the harder randomized
distribution (Eimer et al. 2023's tuning-transfer failure mode, already hit
twice elsewhere in this project).

**Conclusion / next step:** Offline case's A2C results are now fully
3-seed-verified across both instance regimes, closing out that part of the
finalization checklist. A randomized-instance-specific Optuna search for A2C
would be the natural fix, but is out of scope for this session (not
requested; the priority is the online case and pointer-network PPO).

---

## 2026-09-16 (S2W9) -- Pointer-network PPO result: architecture does NOT fix PPO's tardiness -- the overnight session's leading hypothesis is refuted

**Config:** `ppo_pointer_fixed_seed{1,2,3}`, full protocol (Optuna-tuned params:
embed_dim=256, hidden=64, lambda_2=1.89; 1.9M timesteps curriculum; same fixed
instance as every other offline PPO/A2C result).

**Stats:**
```
seed1: reward=330.85  tardiness=1303.36 (P95=55.96)  late=45.48/100  scheduled=99.96/100
seed2: reward=328.23  tardiness=1619.24 (P95=65.07)  late=46.16/100  scheduled=100.0/100
seed3: reward=317.84  tardiness=1403.86 (P95=59.47)  late=45.36/100  scheduled=99.76/100

For comparison, same fixed instance:
  Flat-MLP PPO (reward-tuned):    ~1310-1320
  Flat-MLP PPO (tardiness-tuned): ~1289-1315
  Pointer-network PPO (this):     ~1303-1619
  A2C, same pointer architecture: ~28 (!)
```

**Observation:** This directly refutes the previous entry's leading hypothesis.
Pointer-network PPO is statistically indistinguishable from flat-MLP PPO --
if anything slightly worse and with higher seed-to-seed variance (seed2 at
1619 is the worst PPO tardiness result recorded this entire campaign, excluding
the lambda_max=50 collapse). **A2C achieves ~28 tardiness with the EXACT SAME
`PointerActorCritic` architecture** -- so architecture cannot be the
explanatory variable; the real differentiator must be something about PPO
itself (the clipped surrogate objective, multi-epoch minibatch reuse of
rollout data, GAE/advantage estimation, or an interaction between these and
this environment/reward's sparsity) rather than the network's capacity to
reason about job-to-job relationships.

**Conclusion / next step:** Five now-refuted-or-ineffective mechanisms for
PPO's tardiness: fixed reward weight, four Lagrangian lambda_max ceilings, a
tardiness-directed hyperparameter search, and now pointer-network
architecture. Every one of them left PPO in the same ~1290-1620 band while
A2C, sharing either the reward structure (flat MLP tests) or now the
architecture (this test), sits at ~28. The remaining plausible explanation is
something intrinsic to PPO's optimization procedure on this environment/
reward, not any single fixable input to it -- this needs a genuinely
different investigation (e.g. directly comparing PPO's and A2C's *training
dynamics* on identical architecture+reward, not just final performance) rather
than another reward/architecture variant. Recommended as the next real
investigation, not attempted further tonight given the scope already covered.

---

## 2026-09-16 (S2W9) -- Pointer-network PPO implemented and validated; full search + training launched

**Config:** New `Code/policies/pointer_ppo_policy.py` --
`PointerMaskableActorCriticPolicy`, an `sb3_contrib`-compatible wrapper
around `PointerActorCritic` (the same architecture A2C has used since
2026-08-09), giving `MaskablePPO` a drop-in `policy_class` alternative to
`"MlpPolicy"`. Wired into `train_optimized.py` (`--policy-type pointer` now
works for `--algo ppo`, previously silently ignored) and `optuna_tune.py`
(`objective_ppo` gained the same `policy_type` branch `objective_a2c`
already had). See the new dated doc,
`2026-09-16-pointer-network-ppo.md`, for the full design writeup
(SB3 policy-interface analysis, the optimizer-parameter-registration hazard
found and avoided, and the backward-compatibility fix needed for
`--policy-type`'s CLI default).

**Stats:** Two smoke tests (standalone `PointerMaskableActorCriticPolicy`
construction/rollout/save-load; full `build_vec_env` -> `get_attr` ->
`MaskablePPO` integration) both passed before any real-scale run. A 50-trial
Optuna search (`optuna_tune.py --algo ppo --policy-type pointer`, matching
every other architecture variant's search rigor) is running as of this
entry; full 3-seed fixed-instance + 3-seed randomized-instance training to
follow once it completes.

**Observation:** N/A yet -- implementation and validation only, no
tardiness result to report in this entry.

**Conclusion / next step:** This directly replaces the full 3-seed/full-budget
PPO-Lagrangian $\lambda_{\max}$ sweep the user originally requested (deprioritized
in favor of this, per the user's explicit choice: smoke tests + a 500k-step
diagnostic already showed no lambda value escapes the ~1290-1330 band, so
that compute goes toward the untested architecture variable instead). This
is the test of the overnight session's own leading hypothesis
(2026-09-15 entry above): if pointer-network PPO closes some of the gap to
A2C's tardiness, architecture is confirmed as the real bottleneck; if it
doesn't, the mystery deepens further and the two-critic Lagrangian
architecture becomes the more urgent next thing to try (with human design
input, as already decided).

---

## 2026-09-15 (S2W9) -- Overnight session conclusion: no reward/hyperparameter fix improves PPO's tardiness; architecture is the new leading hypothesis

**Config:** Full-scale validation of the tardiness-tuned Optuna search from
the previous entry: 3 seeds, `train_optimized.py --params-tag tardiness`,
same protocol as every other PPO run this session.

**Stats:**
```
seed 1: reward=266.19  tardiness=1315.06 (P95=58.98)  late=42.32/100  scheduled=97.06/100
seed 2: reward=264.71  tardiness=1289.04 (P95=57.35)  late=42.46/100  scheduled=96.84/100
seed 3: reward=264.72  tardiness=1304.88 (P95=59.50)  late=41.76/100  scheduled=96.90/100

Every PPO mechanism tried tonight, for comparison:
  Reward-tuned (fixed instance):      ~1310-1320
  Reward-tuned (randomized instance): ~1327
  PPO-Lagrangian lambda_max=8/12/18:  ~1291-1312
  PPO-Lagrangian lambda_max=50:       collapse (0 scheduled)
  Tardiness-tuned (this entry):       ~1289-1315
```

**Observation:** The tardiness-tuned search's perfect small-scale result
(previous entry) completely failed to transfer to the deployed 100-job
scale -- exactly the Eimer et al. (2023) failure mode this project has now
hit twice (previously A2C, S2W5; now PPO). More strikingly, **five
completely different mechanisms for fixing PPO's tardiness -- a fixed
weight, four different Lagrangian multiplier ceilings, and a from-scratch
hyperparameter search -- all converge on the same ~1290-1330 band**, with
the sole exception being the Lagrangian collapse at lambda_max=50 (a
different, worse failure). This consistency is hard to explain by "the
weight/mechanism was wrong" and points instead to something upstream of
reward weighting entirely: most plausibly the flat MlpPolicy architecture
itself, which (unlike A2C's `PointerActorCritic`, which gets 28.16-28.66
tardiness) has no explicit mechanism for reasoning about job-to-job
relationships or deadline ordering. Full analysis:
`2026-09-14-ppo-lagrangian-and-reward-structure.md` Section 12.

**Conclusion / next step (end of this autonomous overnight session):**
Every reward-formulation and hyperparameter-search avenue tried for PPO
tonight is now exhausted without success -- the most promising untried lead
is architecture, not reward: a pointer/attention-based PPO policy
(`sb3_contrib.MaskablePPO` supports custom feature extractors via
`policy_kwargs`, so this doesn't require a hand-rolled training loop the way
the two-critic PPO-Lagrangian rewrite would). Also still open: the two-critic
PPO-Lagrangian architecture itself (deliberately not attempted unsupervised,
needs human design input -- see the previous PPO-Lagrangian entries). All
five PPO variants trained tonight (fixed-instance, randomized-instance,
PPO-Lagrangian at 4 lambda_max values plus the collapse, tardiness-tuned)
are checkpointed and organized under
`rl_training/results_by_setting/*/checkpoints/` with matching eval plots,
and every finding is cross-referenced between this log and the two dated
research docs touched tonight
(`2026-09-14-ppo-lagrangian-and-reward-structure.md`,
`2026-08-28-pso-metaheuristic-baseline.md`).

---

## 2026-09-15 (S2W9) -- optimize_for="tardiness" extended to PPO; full-scale validation launched; PSO-tardiness-fitness rerun in progress (autonomous overnight session, continued)

**Config:** `Code/training/optuna_tune.py::objective_ppo` now supports
`optimize_for` ("reward"/"tardiness"/"pareto"), mirroring `objective_a2c`'s
existing mechanism exactly (same `TARDINESS_PENALTY_WEIGHT`-scalarized
composite score). `run_optimization()`'s suffix/dispatch logic (previously
`algorithm == "a2c"`-gated) extended to apply to PPO. Also added
`--fitness {reward,tardiness}` to `Code/baselines/pso.py` (negated total
tardiness as the fitness value when `tardiness`), plus `--num-jobs/
--num-machines/--horizon/--max-jobs` overrides to both `pso.py` and (earlier
this session) `eval_rl_agent.py` so either can run safely while a training
run is concurrently active.

**Stats:**
```
PPO Optuna search, optimize_for=tardiness, 50 trials (tuning scale: 20 jobs,
5 machines, horizon 30, make_tuning_env's default):
  Best trial (42): composite_score=107.66, lambda_2=1.927 (~unchanged from
    the reward-tuned 1.9315), but layer_size=512, activation=relu,
    learning_rate=0.000196 (~15x the reward-tuned value), n_steps=1024,
    batch_size=512, n_epochs=10, ent_coef=0.029 (~4x larger) -- everything
    EXCEPT lambda_2 differs substantially.
  Top ~10 trials by score: mean_tardiness=0.0, mean_late_jobs=0.0 exactly
    (perfect on-time completion) AT THIS TUNING SCALE.
```

**Observation:** The tardiness-directed search did NOT converge on a larger
`lambda_2` the way the earlier ad hoc Lagrangian sweep's reasoning predicted
-- it found a *different overall configuration* (bigger network, much higher
learning rate, more entropy) achieves perfect tardiness at the small tuning
scale with essentially the *same* lambda_2 as before. This is a genuinely
different, more promising hypothesis than "lambda_2 was too low": maybe the
original reward-tuned PPO's network/learning-rate/entropy settings simply
couldn't learn a sequencing-sensitive policy well at all, independent of how
tardiness was weighted. **Critical caveat, not yet resolved**: this is
measured at the small 20-job tuning scale, not the deployed 100-job
instance -- this project's own history (Eimer et al. 2023, cited in
`objective_a2c`'s docstring; the S2W5 tardiness-retuning episode) already
demonstrated that small-scale-tuned hyperparameters can fail to transfer.
Full grounding: `2026-09-14-ppo-lagrangian-and-reward-structure.md` Section
11.

**Conclusion / next step:** Launched the actual validation this needs: 3
seeds of full-scale PPO training (`--params-tag tardiness`, same protocol as
every other PPO run this session --n-envs 5 --vec-backend dummy
--torch-threads 5 --stage4-timesteps 1600000) on the real 100-job fixed
instance, run concurrently with a tardiness-fitness PSO rerun (`pso.py
--fitness tardiness`, real fixed instance + 10 held-out, swarm=20/
iterations=40) that was already in progress -- both use mostly single-core/
light CPU individually, confirmed to coexist without issue. Results pending.

---

## 2026-09-15 (S2W9) -- PPO-Lagrangian lambda_max sweep: ruled out as a hyperparameter fix, autonomous overnight session

**Config:** Fast smoke tests (real `train_optimized.py` pipeline, full 4-stage
curriculum, reduced `--stage4-timesteps`) at `lambda_max in {8, 12, 18}`
(100k stage-4 steps each) plus one longer diagnostic (`lambda_max=15`, 500k
stage-4 steps), per the user's request to verify with small excerpts before
committing another multi-hour run, done autonomously overnight after the
user went to sleep.

**Stats:**
```
lambda_max=8,  100k steps: tardiness=1312.62  late=42.40/100  scheduled=96.90/100
lambda_max=12, 100k steps: tardiness=1291.16  late=42.42/100  scheduled=96.82/100
lambda_max=18, 100k steps: tardiness=1291.64  late=42.50/100  scheduled=96.84/100

lambda_max=15, 500k steps -- episode_cost trend across stage 4 (lambda held
constant at 15.0 throughout, so this isolates the training-time question):
  t=300k-349k: avg_cost=14.38   t=549k-599k: avg_cost=16.54
  t=349k-399k: avg_cost=15.20   t=599k-649k: avg_cost=16.77
  t=399k-449k: avg_cost=15.63   t=649k-699k: avg_cost=16.89
  t=449k-499k: avg_cost=16.07   t=699k-749k: avg_cost=17.12
  t=499k-549k: avg_cost=16.37   t=749k-801k: avg_cost=17.27
```

**Observation:** None of 8/12/18 improved on the original (uncontrolled)
PPO's tardiness (~1315-1327) or on each other -- all indistinguishable
within this training budget, despite lambda sitting at 90%+ of its cap for
essentially the entire stage. The longer run rules out "just needs more
training time": cost rose smoothly and monotonically over the full 500k-step
window at a *constant* lambda (no confound from the multiplier itself
changing), the opposite of convergence. Full mechanistic discussion:
`2026-09-14-ppo-lagrangian-and-reward-structure.md` Sections 9-10.

**Conclusion / next step:** `lambda_max` is not the lever -- every value
tried sits on the same failure spectrum (no effect -> slow degradation ->
full collapse at 50) rather than having a working middle ground. The
mechanism is architectural: mutating the environment's raw reward in place
(shared with A2C's RCPO) forces PPO's single value function to track a
shifting target, corrupting the advantage estimates the policy gradient
needs. The real fix is the two-critic PPO-Lagrangian architecture (separate
reward/cost value functions and advantages, Ray/Achiam/Amodei 2019) rather
than more hyperparameter search -- scoped but deliberately NOT attempted
autonomously overnight (a genuinely new, hand-rolled training loop with
substantial unverified design surface, better done with a human in the
loop). Pausing PPO-Lagrangian here; continuing with the lower-risk queued
follow-ups (PSO-with-tardiness-fitness rerun, `optimize_for="tardiness"`
extended to PPO) that reuse existing, already-validated infrastructure.

---

## 2026-09-15 (S2W9) -- PPO-Lagrangian result: all 3 seeds collapsed to zero jobs scheduled

**Config:** Same as the fixed-instance PPO entry below, plus `--use-rcpo`
(PPO-Lagrangian, `Code/policies/ppo_lagrangian.py`), `alpha=0,
lambda_init=1.9315 (warm-started from PPO's tuned lambda_2), lambda_lr=0.01,
lambda_max=50, update_every=5 episodes`.

**Stats:**
```
seed 1: reward=-50.50  tardiness=0.00  late=0/100  scheduled=0/100  (std=0 across 50 held-out runs)
seed 2: reward=-50.50  tardiness=0.00  late=0/100  scheduled=0/100  (std=0)
seed 3: reward=-50.50  tardiness=0.00  late=0/100  scheduled=0/100  (std=0)

lambda trajectory (all seeds similar): hit lambda_max=50 at ~18% of training
(timestep ~345k/1.9M) and stayed pinned there; mean sampled cost ROSE over
the remaining 82% of training (0.96 -> 15.88 -> 18.15 -> 25.86), not fell.
```

**Observation:** Complete, deterministic idle-collapse (zero variance across
all 50 held-out instances x 3 seeds) -- not partial degradation, not noise.
Tardiness=0 is trivial and worthless here: unscheduled jobs are never late by
definition. Full mechanistic diagnosis (runaway lambda-vs-abandonment
feedback loop) and proposed fixes (tighter `lambda_max`; possibly the
two-critic PPO-Lagrangian architecture instead of modifying the reward in
place) recorded in `2026-09-14-ppo-lagrangian-and-reward-structure.md`
Section 8. This is the exact instability risk that doc's Section 5 flagged
*before* this run, based on Stooke, Achiam & Abbeel (2020)'s critique of
plain Lagrangian ascent -- now confirmed empirically, not just cited as a
risk.

**Conclusion / next step:** PPO-Lagrangian as implemented is not usable as-is
-- it did not close the gap to the proven floor (tardiness=8.0, S2W9's
2026-09-14 entry below), it produced a worse outcome than the original
reward-tuned PPO. Two concrete follow-ups identified, not yet tried: (1)
retry with a much tighter `lambda_max` (~5-10 instead of 50); (2) implement
the standard two-critic PPO-Lagrangian architecture (separate cost value
function/advantage, `A = A_reward - lambda*A_cost`) instead of mutating the
environment's raw reward, which may itself be contributing to the
instability by constantly moving the value function's target.

---

## 2026-09-14 (S2W9) -- Randomized-instance PPO (3 seeds) matches fixed-instance's tardiness failure; CP-SAT proves the real floor is 8.0; PPO-Lagrangian launched

**Config:** Same as the fixed-instance entry below, plus `--randomize-instances`
(fresh random job set every episode). `--seed` now also determines the stream
of training instances (via seeding the global numpy RNG the resampler reads),
not just algorithmic randomness -- appropriate here since there's no single
"correct" instance to hold fixed in this mode (see conversation this session
for the full reasoning).

**Stats:**
```
Randomized-instance PPO, 50-held-out-instance protocol:
  seed 1: reward=267.18  tardiness=1327.62 (P95=58.60)  late=42.82/100  scheduled=96.84/100
  seed 2: reward=269.54  tardiness=1327.22 (P95=58.75)  late=41.86/100  scheduled=96.94/100
  seed 3: reward=267.64  tardiness=1327.30 (P95=59.10)  late=42.96/100  scheduled=96.68/100

CP-SAT, real fixed instance (seed=0, 100 jobs), 900s budget, 15 search workers:
  status=OPTIMAL, objective=8.0, best_bound=8.0, solve_time=21.98s
  reward=339.09  tardiness=8.00  late=5/100 (schedules ALL 100 jobs)
  (reference) EDF: tardiness=16.00 late=10   LST: tardiness=8.00 late=7
```

**Observation:**
1. Randomized-instance training lands at essentially the same tardiness
   failure as the fixed-instance run (~1327 vs ~1315). This is exactly what
   the reward-structure analysis (`2026-09-14-ppo-lagrangian-and-reward-
   structure.md`) predicts: the root cause is that `lambda_2=1.93 < 3.0` (the
   flat per-job completion bonus), independent of which instance(s) are
   trained on. Instance diversity was never the problem, so randomizing
   instances doesn't fix it -- consistent with, and now further evidence for,
   that doc's diagnosis.
2. **CP-SAT solved the real deployed instance to proven optimality** (not
   just a small synthetic one -- see `exact_solver.py --fixed-instance`,
   and the new `--num-search-workers` flag that made this fast) in 22
   seconds: true minimum tardiness is 8.0, achieved while scheduling all 100
   jobs. This settles a question raised earlier this session ("how do we
   know 42 late jobs isn't just because some were impossible") definitively
   for this instance: they weren't impossible. It also disproves a possible
   "completion vs. punctuality tradeoff" explanation -- CP-SAT gets both
   simultaneously, so PPO's ~97 scheduled + ~1315 tardiness isn't a forced
   trade-off, it's just not deadline-aware at all. Full writeup and the
   decisive framing: `2026-09-14-ppo-lagrangian-and-reward-structure.md`
   Section 7.

**Conclusion / next step:** PPO-Lagrangian (`Code/policies/ppo_lagrangian.py`,
extending RCPO to PPO via an SB3 callback, see the dated doc) launched
immediately after this -- 3 seeds, same protocol, warm-started from PPO's own
tuned `lambda_2`. Smoke-tested separately before the real run: multiplier
correctly climbed from 1.93 toward 3+ within a few thousand steps and
propagated to all sub-envs. Success criterion is now concrete: how close does
it get to the proven floor of 8.0, not just "better than before."

---

## 2026-09-14 (S2W9) -- Final offline-case PPO (fixed instance, 3 seeds): completes jobs but is not deadline-aware; ENV_CONFIG_PATH hazard recurred and is now actually fixed

**Config:** PPO, MlpPolicy, default reward-tuned Optuna params (`ppo_best_params.json`,
unchanged), curriculum training with `--stage4-timesteps 1600000` (scaled up from the
200k default after measuring real throughput -- see below), 3 seeds (1, 2, 3), each
`--n-envs 5 --vec-backend dummy --torch-threads 5` for parallel rollout collection
(new this session -- see `Code/training/train_optimized.py::build_vec_env()`).
Evaluated via `eval_rl_agent.py --randomized-eval --num-jobs 100 --num-machines 10
--horizon 100 --max-jobs 100` (new explicit-override flags, see Observation 3 below).

**Stats:**
```
Per-stage throughput (dummy-backend vectorized rollout, this machine, 16 cores):
  Stage 1 (15 jobs,  h=20 ):  630 fps
  Stage 2 (30 jobs,  h=40 ):  389 fps
  Stage 3 (60 jobs,  h=60 ):  246 fps
  Stage 4 (100 jobs, h=100):  162-172 fps

PPO (fixed instance, seed=0), 50-held-out-instance protocol:
  seed 1: reward=269.33  tardiness=1318.38 (P95=59.32)  late=42.44/100  scheduled=97.14/100
  seed 2: reward=267.12  tardiness=1320.54 (P95=58.68)  late=42.36/100  scheduled=97.06/100
  seed 3: reward=268.61  tardiness=1309.78 (P95=58.78)  late=42.12/100  scheduled=96.90/100

Same instance, classical heuristics (from eval_results.csv, unchanged from 2026-08-28):
  EDF   tardiness=37.30  late=12.22  scheduled=96.84
  LST   tardiness=23.94  late=8.26   scheduled=97.76
  SPT/WSPT (no deadline signal) tardiness=1150.86  late=37.74  scheduled=92.08
  FCFS+FirstFit (no deadline signal) tardiness=1299.52  late=42.06  scheduled=96.86
```

**Observation:**
1. **This PPO checkpoint has not learned deadline-aware scheduling.** All 3 seeds
   agree tightly (tardiness 1310-1320, late 42.1-42.4), which rules out seed noise
   as the explanation. It lands squarely in the same tardiness/late-jobs band as
   heuristics that ignore deadlines entirely (SPT, FCFS, Tetris -- see the
   2026-08-28 classical-heuristics entry), not anywhere near EDF or LST. This is
   the first same-instance PPO-vs-heuristics comparison at this training scale in
   this project (the only pre-existing PPO checkpoint, `ppo_scheduling.zip`, has an
   incompatible legacy action-space size and can't be loaded against the current
   wrapper). Likely cause (not yet verified): the default reward-tuned Optuna
   params' `lambda_2` (tardiness weight) may not sufficiently penalize tardiness
   relative to `lambda_1`/idle-penalty for this architecture -- a `--params-tag
   tardiness`-tuned rerun (if such a params file exists / is worth producing) would
   directly test this.
2. **The new per-machine utilisation graphs already earn their keep.** EDF's
   placement rule concentrates almost all load on machine 0 (visible for the first
   time -- the old single-aggregated-line plot averaged this away); this PPO
   checkpoint spreads load across far more machines. So: PPO load-balances better
   but is not deadline-aware, while EDF is deadline-aware but load-imbalanced --
   a real trade-off the new graphs surface that the old ones couldn't.
3. **Re-hit and finally fixed the 2026-08-28 `ENV_CONFIG_PATH` concurrency hazard**
   (this exact log's prior entry describes it and explicitly deferred fixing it).
   Ran eval against these checkpoints while a second training run (randomized-
   instance set, see below) was concurrently active; the first eval attempt read
   `ENV_CONFIG_PATH` mid-write by the new run's early curriculum stage, producing
   a fully plausible-looking but wrong result (jobs_scheduled ~14-30 instead of
   ~97). Fixed for real this time: `build_vec_env()`/`make_env()` now only write
   `ENV_CONFIG_PATH` on the training run's FINAL curriculum stage, and
   `eval_rl_agent.py --randomized-eval` gained `--num-jobs/--num-machines/
   --horizon/--max-jobs` overrides so it never needs to touch the shared file at
   all when the caller already knows the target dimensions.
4. **Parallel rollout collection (`n_envs`-way `DummyVecEnv`) works and was
   correctly re-derived after a real bug**: an earlier version of this session's
   `build_vec_env()` gave each of the `n_envs` parallel workers a *different* job
   instance (and coupled that to the `--seed` CLI flag meant only for algorithmic
   randomness) -- silently turning "3 seeds of the same fixed-instance task" into
   3-5x mixed-instance training. A first (buggy) 3-seed run completed under that
   code before the bug was caught; its results (reward ~268-270, tardiness
   ~1300-1330, late ~42/100) are numerically close to this corrected run's, by
   apparent coincidence -- read the buggy run's numbers as invalid regardless of
   the similarity (mixed-instance training, not the intended comparison).
   `SubprocVecEnv` vs `DummyVecEnv` was also benchmarked directly on this machine:
   `dummy` was at least as fast as `subproc` for this cheap environment (IPC
   overhead offsets `subproc`'s parallelism gain at this scale), so `dummy` was
   used for the real run.

**Conclusion / next step:** Randomized-instance set (`--randomize-instances`, same
3 seeds) launched immediately after this one finished, per the user's explicit
request for both settings as final data (not just whichever already performed
better). A CP-SAT best-objective-bound run on this same fixed instance (new
`Code/baselines/exact_solver.py --fixed-instance` flag, added this session) is
planned once both training sets finish, to get a genuine provable tardiness floor
rather than only heuristic comparisons. Worth testing directly: does a
`--params-tag tardiness`-tuned PPO run close the gap to EDF/LST, or is the flat
MlpPolicy architecture itself the limiting factor (cf. A2C+pointer's much better
28.66 tardiness in the 2026-08-28 entry)?

---

## 2026-08-28 (S2W6) -- jobs_scheduled measured for the full baseline roster; corrects two earlier ad hoc figures

**Config:** N/A (measurement pass, prompted by a request to add jobs-scheduled to the results review tables). All numbers below use the official 50-held-out-instance protocol (`num_jobs=100, num_machines=10, horizon=100`, seeds 500000-500049) -- the same one `eval_results.csv` uses everywhere else.

**Stats:**
```
Classical heuristics (never tracked before tonight):
  EDF            96.84/100      LPT+FirstFit  100.00/100
  LST            97.76/100      FCFS+FirstFit  96.86/100
  SPT/WSPT       92.08/100      Tetris         96.80/100
  Random         97.14/100

RL checkpoints, corrected:
  Pointer + shaping (Phase 8):        89.48/100   (previously reported 98.5/100)
  Pointer + RCPO (original, buggy):   55.24/100   (previously reported 49.8/100)
  Pointer + RCPO (fixed constraint):  93.12/100   (unchanged, already measured at horizon=100)
  Pointer + Pareto-knee:              99.38/100   (unchanged, from the official eval_rl_agent.py run)
```

**Observation:** The two corrected RL figures came from an earlier same-night ad hoc diagnostic script that called `generate_env_config(..., horizon=110)` -- `generate_env_config`'s own default, not the deployed instance's actual `horizon=100` -- while every other number on this page uses `horizon=100`. The qualitative story is unchanged (shaped schedules almost everything, the original buggy RCPO checkpoint abandons roughly half), but the precise figures were measured under a slightly different instance distribution than everything else and are corrected here rather than left standing. `LPT+FirstFit` reaching exactly 100/100 while `SPT`/`WSPT` bottom out at 92.08/100 is a new data point, not previously measured for this roster -- consistent with the completion-rate-varies-by-priority-rule pattern CP-SAT's comparison (`2026-08-28-exact-solver-baseline.md`) already found on small instances.

**Conclusion / next step:** No action needed -- this is a precision correction, not a new finding. Flagging per CLAUDE.md's rule that an error, once found, should be corrected in a new entry rather than silently left in place.

---

## 2026-08-28 (S2W6) -- Pareto-knee point collapses into the same reward-hacking pattern once fully trained -- a negative result closing tonight's Pareto arc

**Config:** Trained trial 12 from the full-scale Pareto front (`lambda_2=10.981`, the front's "knee": `reward=170.88, tardiness_norm=2.47` at 30k tuning timesteps) to full convergence (500k timesteps, standard curriculum, `--use-potential-shaping`) via `train_optimized.py --params-tag paretoknee`. Evaluated identically to every other checkpoint (fixed instance + 50 held-out). See `Future/research/2026-08-28-multi-objective-optuna-pareto.md` Section 8.

**Stats:**
```
                              reward   tardiness  late_jobs  jobs_scheduled
Pareto-knee, fully trained    303.08   733.56     23.84      99.38/100
Trial 12 @ 30k timesteps      170.88   ~247 (tardiness_norm x H)   --       --
Phase 8 shaped (no RCPO)      254.75   28.66      9.56       98.5/100
Pointer + RCPO (fixed)        268.77   28.16      9.74       93.1/100
```

**Observation:** The "balanced" trade-off trial 12 showed at 30k timesteps did not survive full training -- at convergence it fully collapsed into the same reward-hacking pattern as everything else tonight (highest reward of any RL checkpoint this project has produced, 20-25x every other RL checkpoint's tardiness), via high throughput (99.38/100 scheduled, not abandonment) rather than lateness avoidance. `lambda_2=10.98` here is higher than the historical reward-tuned `3.85`, yet still insufficient at full convergence.

**Conclusion / next step:** A second confound beyond instance scale: training budget also has to match deployment for a Pareto-selected point to mean anything. Section 6b's full-scale front matched instance dimensions but not training budget, and that was not sufficient. A trustworthy multi-objective search for a real deployment `lambda_2` would need trials trained at (or close to) the full ~500k-timestep budget -- a much larger compute commitment, explicitly flagged rather than attempted tonight. This closes tonight's Pareto investigation arc on an honest negative note: the technique works and reveals real structure (Sections 4-6b), but this project doesn't yet have a validated way to turn that structure into a better deployed checkpoint.

---

## 2026-08-28 (S2W6) -- Silent CSV column-misalignment bug found and fixed in eval_results.csv

**Config:** N/A (data-integrity bug, discovered while evaluating the Pareto-knee checkpoint). See `Code/utils/results_log.py`.

**Stats:** N/A -- this is a bug report, not an experiment result.

**Observation:** Adding `jobs_scheduled_mean`/`heuristic_jobs_scheduled_mean` (etc.) to `EVAL_RESULT_FIELDS` earlier tonight changed the column order `csv.DictWriter` writes new rows in, but `eval_results.csv`'s on-disk header (written once, long before tonight) was never updated to match -- `DictWriter` writes values in *fieldnames* order regardless of the file's actual header row. Every row appended since that schema change had correct values sitting under the wrong column names (`heuristic_name` silently held `jobs_scheduled_mean`'s value, etc.). Caught when reading the pareto-knee eval results back by column name produced obvious garbage (`heuristic_name: "100.0"`).

**Conclusion / next step:** Repaired the file by reconstructing every row from its own actual field count (20 = pre-tonight schema, 24 = post-tonight schema), verified against known-good historical numbers before replacing the corrupted file. Root-cause fixed in `append_eval_result()`: it now detects an on-disk header mismatch and self-heals by rewriting every row positionally against its own current header before appending -- generic protection against this exact class of bug recurring on any future field addition, verified with both the real file and a synthetic schema-drift test. Checked which rows were actually affected: exactly 4 of 142 (all from the pareto-knee eval, the run this bug was caught while reading) -- every other write-up finalized tonight (Stage A, RCPO refix, Stage B/C, all three Pareto studies) was read and finalized before the schema change even happened, so none of them are affected.

---

## 2026-08-28 (S2W6) -- Closing the loop: Pareto front confirmed at the actual full deployment scale

**Config:** Same `optimize_for="pareto"` mode, 15 trials (reduced budget given the hour), `num_jobs=100, num_machines=10, horizon=100` -- the exact deployed instance dimensions, not a proxy. See `Future/research/2026-08-28-multi-objective-optuna-pareto.md` Section 6b.

**Stats:**
```
5 of 15 trials non-dominated, reward -89.94 (tardiness=0) to 257.05 (tardiness_norm=19.37)
Historical reward-tuned lambda_2=3.8529 falls between trial 11 (5.685) and trial 7 (1.434) on this front
```

**Observation:** Confirms the 60-job proxy-scale finding robustly at the real production scale -- a wide, real trade-off exists here too. Caveat: every trial trains for only 30,000 timesteps (vs. ~500,000 for a full curriculum run), so absolute reward/tardiness values here are not comparable to `eval_results.csv`'s fully-trained checkpoints -- the existence and shape of the trade-off is the finding.

**Conclusion / next step:** This closes tonight's Pareto investigation arc (small scale -> null result -> intermediate scale -> real front -> full scale -> confirmed). A full deployment-budget (500k-timestep) multi-objective study is the natural next step for informing an actual production `lambda_2` choice, but that is a much larger compute commitment left for a future session with the user's input on budget.

---

## 2026-08-28 (S2W6) -- Deployment-scale Pareto rerun confirms the trade-off is scale-dependent, and finds a real bug along the way

**Config:** Same `optimize_for="pareto"` mode, 40 trials, but `num_jobs=60, num_machines=8, horizon=60` (new `--tuning-num-jobs`/`--tuning-num-machines`/`--tuning-horizon` CLI flags) instead of the default (20, 5, 30) every prior study used. See `Future/research/2026-08-28-multi-objective-optuna-pareto.md` Section 6.

**Stats:**
```
Small scale (20/5/30):  1 of 40 trials non-dominated -- degenerate front, no trade-off
Larger scale (60/8/60): 10 of 40 trials non-dominated -- reward ranges -117.05 (0 tardiness) to 195.39 (7.15 tardiness_norm)
```

**Observation:** Confirms the small-scale null result's own hypothesis directly: the reward/tardiness misalignment is scale-dependent, not a fixed reward-function property -- it's essentially absent at 20-job scale and severe at 60-job scale. `lambda_2` correlates with front position but noisily (not monotonic), meaning other hyperparameters materially shape the trade-off too, information a single scalarized search would never surface. Also found and fixed a real bug enabling this: `make_tuning_env()` hardcoded `max_jobs=30` regardless of `num_jobs`, silently crashing (cryptic "index N out of bounds") for any `num_jobs > 30` -- never hit before since every prior study used `num_jobs=20`. Fixed to `max(30, num_jobs)`.

**Conclusion / next step:** This is the clearest demonstration yet that this project's reward-hacking pattern (Phase 6, PSO tonight, every RCPO run implicitly) is a scale phenomenon, driven by throughput bonuses accumulating over more placements per episode as `num_jobs` grows against an `O(1)`-per-job tardiness penalty. Recommended next step for a future session: rerun at full deployment scale (100 jobs, horizon=100) to find the front that would actually inform a production `lambda_2` choice.

---

## 2026-08-28 (S2W6) -- Multi-objective Optuna Pareto front collapses to one point at tuning scale -- an informative null result

**Config:** New `optimize_for="pareto"` mode (`Code/training/optuna_tune.py`), 40 trials, A2C pointer, standard small tuning env (20 jobs, 5 machines, horizon=30). See `Future/research/2026-08-28-multi-objective-optuna-pareto.md`.

**Stats:**
```
17 of 40 trials reached exactly zero mean tardiness
Best reward among zero-tardiness trials:     107.65  <- the sole Pareto-front point
Best reward among nonzero-tardiness trials:  106.10  <- strictly worse
```

**Observation:** The front collapsed to a single non-dominated trial because nothing had to be traded away -- the highest-reward trial in the entire population also happens to have zero tardiness. No reward/tardiness trade-off exists to characterize at this scale.

**Conclusion / next step:** This means the reward/tardiness misalignment this project keeps finding (Phase 6, PSO earlier tonight) is absent or far weaker at small tuning scale, and only emerges at deployment scale (100 jobs) where throughput bonuses accumulate over many more placements per episode. Corroborates Eimer et al. (2023)'s tuning/deployment mismatch warning at the level of the Pareto front's shape, not just which point gets picked. Next step (not attempted tonight, cost-prohibitive at deployment-scale per-trial timesteps): rerun `optimize_for="pareto"` at a tuning scale closer to deployment (e.g. 60-100 jobs).

---

## 2026-08-28 (S2W6) -- Stage B: PSO independently rediscovers reward hacking via a non-gradient search method

**Config:** SPV-encoded PSO (`Code/baselines/pso.py`), `swarm_size=15, iterations=30`, fitness = total episode reward, fixed instance + 15 held-out instances (seeds 500000-500014). See `Future/research/2026-08-28-pso-metaheuristic-baseline.md` for full grounding/methodology.

**Stats:**
```
                       reward   tardiness  late_jobs  wall_clock/instance
PSO (fixed instance)   337.37   963.00     39         208.5s
PSO (15 held-out mean) 329.68   1012.40    40.93      182.9s
EDF (same held-out)    286.00   50.33      14.20      ~0.4s
```

**Observation:** PSO beats EDF on reward by ~15% on every instance, with ~20x the tardiness. Fitness curve is monotonically improving by construction and converges well -- this is PSO succeeding at optimizing exactly what it was told to (raw episode reward), not failing to search. Since PSO evaluates candidates by directly replaying them through the unmodified reward function (no RL training dynamics involved at all), finding the same reward/tardiness misalignment Phase 6 found via Optuna-tuned RL, and Phase 7 grounded in Skalse et al. (2022)'s formal reward-hacking definition, is much stronger evidence the misalignment is a property of the reward function's own weighting (throughput bonuses outweighing the tardiness penalty at lambda_2=1.0) than an RL-specific training artifact.

**Conclusion / next step:** Confirms PSO works correctly as a baseline and adds a third, independent line of evidence for the reward-hacking finding. Not yet tried: rerunning PSO with fitness = negative tardiness directly, to get an achievable-tardiness reference point at full instance scale (complementing CP-SAT's small-instance-only exact optimum from Stage C).

---

## 2026-08-28 (S2W6) -- Apparent RL scale-generalization failure was a self-inflicted test confound, not a real finding

**Config:** N/A (methodology correction). Following up on `2026-08-28-exact-solver-baseline.md`'s open question ("does the RL policy generalize to a much smaller instance scale"), ran the RCPO-refixed pointer checkpoint on `num_jobs=10, horizon=15` instances (padding `max_jobs=100` to match the trained obs/action space).

**Stats:**
```
First attempt (default deadline_range=(10,110), unscaled for horizon=15):
  RL reward=-2 to -8, jobs_scheduled=0-2/10   <- looked catastrophic
Controlled retest (deadline_range=(2, horizon), proportionally scaled):
  RL matches or beats EDF on all 5 seeds tested, jobs_scheduled=9-10/10
```

**Observation:** `Code/env/env_config.py::generate_env_config`'s `deadline_range` defaults to `(10, 110)` regardless of the `horizon` argument -- at `horizon=15` this produces deadlines like 99, drastically exceeding the horizon. The RL policy consumes *normalised* deadlines (`deadline/horizon`); heuristics compare raw deadlines directly. So the same malformed instance pushed the RL policy's observations far outside its training distribution while leaving EDF/LST completely unaffected -- the "catastrophic failure" was a property of the test instance, not the policy.

**Conclusion / next step:** No scale-generalization failure found once properly controlled -- the policy holds up reasonably at 10x smaller scale than training. Recording the failed-then-corrected attempt in full (not just the clean final numbers) because the artifact itself is the useful finding: **`generate_env_config`'s `deadline_range` not scaling with `horizon` is a footgun for any future cross-scale evaluation.** Worth fixing generate_env_config itself eventually (e.g. default `deadline_range` proportional to `horizon`) -- not done tonight, flagged for a future session.

---

## 2026-08-28 (S2W6) -- RCPO rerun with the fixed constraint + achievable alpha: a real, modest win

**Config:** A2C pointer, potential-based shaping ON, `--use-rcpo --rcpo-alpha 0.2866` (achievable, anchored to Phase 8's held-out tardiness converted into `C(tau)` units) with the fixed `episode_cost` (this session's earlier entry) instead of `alpha=0.0` on the buggy constraint (Phase 10). `lambda_init=3.8529`, `lambda_max=50.0`, `lambda_lr=0.01`, `update_every=5` -- unchanged from Phase 10. See `Future/research/2026-08-21-rcpo-constrained-tardiness.md` Section 7 for the full writeup.

**Stats:**
```
                                  reward   tardiness  late_jobs  jobs_scheduled  idle_steps
This rerun (fixed, alpha=0.2866)  268.77   28.16      9.74       93.1/100        7.9   (50 held-out)
Phase 8 (shaped, no RCPO)         254.75   28.66      9.56       98.5/100       11.6   (50 held-out, from earlier diagnostic)
Phase 10 (buggy, alpha=0)         135.67   19.84      3.20       49.8/100       61.2   (50 held-out)
EDF                                284.95   37.30      12.22      --              --
LST                                288.28   23.94       8.26      --              --
```

**Observation:** Job abandonment dropped by ~85% (49.8->93.1 jobs scheduled) with no new failure mode taking its place -- confirms the constraint fix worked as intended. Reward beat Phase 8 by 5.5% with essentially flat tardiness, a genuine (if modest) win for the RCPO mechanism once both fixes are applied together. Still does not beat EDF on reward, or LST on anything -- Stage A's finding that LST is the real bar to beat still stands.

**Conclusion / next step:** This is the first RCPO result on this environment that can be read at face value without a hidden gaming strategy. Keep this checkpoint as the project's best pointer configuration going forward. The remaining ~7% unscheduled-job gap is not further investigated tonight -- plausible next step is checking whether a stricter (lower) achievable alpha pushes jobs_scheduled closer to 100/100 without reward collapsing again, now that the free-abandonment loophole is closed.

---

## 2026-08-28 (S2W6) -- Stage C: CP-SAT exact solver confirms EDF/LST are tardiness-optimal but leave jobs unscheduled

**Config:** OR-Tools CP-SAT (`Code/baselines/exact_solver.py`), 5 small held-out instances (`num_jobs=10, num_machines=3, horizon=15`, seeds 500000-500004), 60s time limit, compared against `EDF`/`LST` on the same instances. See `Future/research/2026-08-28-exact-solver-baseline.md` for the full formulation, a structural finding about the environment (single global decision clock, derived and verified while building this), and scope/limitations.

**Stats:**
```
seed     CP-SAT reward   CP-SAT tard   EDF reward   EDF tard   LST reward   LST tard
500000   75.13           0.00          20.50        0.00       20.50        0.00
500001   74.03           0.00          20.57        0.00       20.74        0.00
500002   74.77           0.00          77.00        0.00       77.22        0.00
500003   74.00           0.00          77.00        0.00       77.00        0.00
500004   73.63           0.00          21.86        0.00       21.86        0.00
```
All solves OPTIMAL in ~0.05s; CP-SAT's objective matched an independently-computed env replay on every instance.

**Observation:** Every method gets zero tardiness on every instance, yet CP-SAT beats both heuristics on reward by ~3.5x on 3 of 5 seeds. Checked directly on seed 500000: EDF schedules only 9 of 10 jobs -- greedy resource-packing gets it stuck even though a fully-completing, still-on-time schedule exists (CP-SAT requires every job scheduled, so it always finds one). This is a different failure mode from RCPO's job abandonment (2026-08-28 entry above): here the heuristics *fail* to complete every job through no-lookahead myopia, rather than *choosing* to skip jobs to game a constraint.

**Conclusion / next step:** Confirms EDF/LST are already tardiness-optimal on small instances -- their remaining gap to CP-SAT is entirely a completion-rate gap. Recommend tracking jobs-scheduled alongside reward/tardiness/late-jobs by default in future evals (this is now the second time this session a hidden completion-rate gap explained a reward discrepancy that looked like something else at first glance). Not done tonight: running the RL checkpoints on these same small instances (padding `max_jobs` to match their trained size) to see where they land between the heuristics and the CP-SAT oracle.

---

## 2026-08-28 (S2W6) -- Discovered hazard: training silently corrupts the shared eval instance file if run concurrently with eval/baseline work

**Config:** N/A (infrastructure finding, not an experiment). Discovered while smoke-testing the new PSO baseline (`Code/baselines/pso.py`) against `EDF` on "the fixed instance" while the Priority-1 RCPO retrain (with the fixed `episode_cost`, see the entry below) was running concurrently in the background.

**Stats:**
```
EDF on "the fixed instance" (rl_training/models/env_config.npz):
  earlier this session (no training running): reward=289.38 tardiness=16.00 late_jobs=10
  mid-way through this session's background RCPO retrain: reward=137.00 tardiness=0.00 late_jobs=0
  env_config.npz contents at that point: num_jobs=30, horizon=40 (a curriculum stage's instance, not the deployed 100-job/horizon=100 one)
```

**Observation:** `Code/training/train_optimized.py`'s per-stage env-construction helper calls `np.savez(ENV_CONFIG_PATH, **config)` (around line 109) on *every* curriculum stage transition, unconditionally overwriting the same shared file every eval/heuristic/PSO script reads as "the fixed instance." Every prior eval this project has run assumed this file always holds the final, full-scale (100 jobs, horizon=100, seed=0) deployment instance -- true whenever no training is concurrently running, but silently false while a curriculum training run is in progress: the file transiently holds whatever stage the training loop is currently on (20/40/60/100 jobs across horizons 20/40/60/100), with no error or warning to any process reading it at the wrong moment. This produced a fully plausible-looking but wrong `EDF` result (137.00/0.00/0) that would have been logged as genuine if I hadn't cross-checked against this session's earlier, known-correct EDF numbers.

**Conclusion / next step:** Treating this as a hard operational rule for the rest of this project, not just tonight: **never run an eval/heuristic/PSO/exact-solver script that reads `ENV_CONFIG_PATH` concurrently with an active `train_optimized.py` (or any script that calls its env-construction helper) run.** Archived `env_config.npz` copies under `rl_training/models/archive/*/` are unaffected (copied once, at the end of a completed run) and remain a reliable source to restore from if the live file is caught mid-corruption. Not fixing the underlying `train_optimized.py` behavior tonight (would need to distinguish "save for later eval" from "save for this stage's own env construction," e.g. only writing `ENV_CONFIG_PATH` after the final curriculum stage) -- flagging it as a real fix worth making, but out of scope for tonight's priority list, which already serializes training and eval so the bug can't bite again.

---

## 2026-08-28 (S2W6) -- RCPO's "best-ever tardiness" (2026-08-21 entry below) was bought by abandoning jobs, not scheduling them better

**Config:** No new training. Diagnostic re-run of the existing checkpoints from
the 2026-08-21 RCPO entry (`a2c_pointer_scheduling_optimized_shaped.pt` vs.
`a2c_pointer_scheduling_optimized_shaped_rcpo.pt`) on 10 held-out instances
(seeds >= `RANDOM_INSTANCE_SEED_CEILING`), instrumented to also record jobs
actually scheduled, idle steps taken, and machines activated per episode --
not just the top-line reward/tardiness/late-jobs numbers every prior eval
reported.

**Stats:**
```
              reward   tardiness  late_jobs  jobs_scheduled  idle_steps  machines_active
shaped        286.70   14.00      7.60       98.5 / 100      11.6        7.0
rcpo          113.04    0.70      0.20       49.8 / 100      61.2        4.8
```

**Observation:** The 2026-08-21 entry below reported RCPO as achieving the
project's best-ever tardiness/late-jobs. That is numerically true but was
read in isolation, without checking *how* it was achieved. `SchedulingEnv`
only ever writes `tardiness[j]` inside `step()` when job `j` is actually
placed (`Code/env/scheduling_env.py`) -- a job that is never scheduled
contributes exactly 0 to both the tardiness metric and RCPO's constraint
cost `C(tau)`, forever. Once the Lagrange multiplier climbed toward its
`lambda_max=50` ceiling chasing an unreachable `alpha=0` target (as already
diagnosed on 08-21), refusing to schedule a job that might end up late
became cheaper under that inflated penalty than scheduling it -- the RCPO
policy schedules only half the jobs (49.8/100 vs. shaped's 98.5/100) and
idles for the majority of the episode (61.2 vs. 11.6 steps) instead.
Eval always scores every method under a fixed, shared `lambda_1=lambda_2=
lambda_3=1.0` rubric (`Code/evaluation/eval_rl_agent.py::make_env()`,
hardcoded regardless of training-time lambda values), so the ~2.5x reward
gap is not a scoring artefact -- most of this reward function's magnitude
comes from the throughput shaping terms (`+3.0` per valid placement, `+50`
for finishing all jobs; `SchedulingEnv.step()`), and a policy that abandons
half the jobs forfeits nearly all of that regardless of how clean its
tardiness looks on the jobs it does commit to.

**Conclusion / next step:** This is not "the reward function is wrong" --
it is the RCPO *constraint* being incompletely specified: `C(tau)` should
charge something for a job still unscheduled at episode end (e.g. treat it
as maximally late, deadline-relative) rather than letting non-completion be
a free way to satisfy the constraint. Any future RCPO rerun (including the
already-flagged achievable-`alpha` rerun below) should fix this constraint
definition first -- otherwise a less strict `alpha` may just produce a
milder version of the same abandonment strategy rather than genuinely
better-scheduled jobs. Flagging this as a required fix, not an optional
refinement, before RCPO results are compared against anything else on
reward terms again.

---

## 2026-08-21 (S2W5) -- RCPO constrained tardiness (pointer): best-ever tardiness/late-jobs, but reward collapses -- multiplier saturated at its ceiling

**Config:** A2C pointer, potential-based shaping ON (Phase 8 config), `--use-rcpo`
(`Code/policies/a2c_policy.py::MaskableA2C`, `alpha=0.0`, `lambda_init=3.8529`
warm-started from the Phase 8 reward-tuned `lambda_2`, `lambda_lr=0.01`,
`lambda_max=50.0`, `update_every=5` episodes). All other hyperparameters
unchanged from `a2c_pointer_best_params.json`. See
`Future/research/2026-08-21-rcpo-constrained-tardiness.md` for the full CMDP
formulation. Checkpoint: `a2c_pointer_scheduling_optimized_shaped_rcpo.pt`,
archived at `rl_training/models/archive/2026-08-21_S2W5_a2c_pointer_s4-200000_shaped_rcpo`.

**Stats:**
```
                        reward     tardiness   late_jobs
EDF (fixed instance)     289.38        16.00       10.00
Phase 8 (fixed λ=3.85)   253.94         9.00        6.00
RCPO (adaptive λ)        124.47        10.00        3.00

                        reward (mean±std)   tardiness (mean±std)   late_jobs (mean±std)
EDF (50 held-out)        284.95±3.65           37.30±60.43            12.22±14.58
Phase 8 (50 held-out)    254.75±9.14           28.66±57.56             9.56±13.45
RCPO (50 held-out)       135.67±15.10          19.84±17.99             3.20±2.12

lambda(0) = 3.8529 -> lambda(final) = 49.37 (of a lambda_max ceiling of 50.0),
still climbing at the end of training. Mean episode cost at the final logged
update: ~3.7 (of an alpha target of 0.0) -- i.e. the constraint was still
being violated when training ended; the multiplier never reached an interior
equilibrium, it saturated against its projection bound.
```

**Observation:** Two things are true simultaneously, and both matter:

1. **On tardiness and late-jobs specifically, RCPO is the best result in the
   project so far, on both axes at once.** 19.84 held-out tardiness beats
   Phase 8's 28.66 (and EDF's 37.30) with less than a third of Phase 8's
   variance (std 17.99 vs 57.56) -- i.e. not just a lower average but a much
   more *reliably* low tardiness outcome. Late-jobs held-out (3.20) is under
   half of Phase 8's (9.56) and a quarter of EDF's (12.22). This is exactly
   the kind of result the CMDP reformulation was meant to produce: letting
   the penalty weight find its own level rather than guessing one fixed
   constant ahead of time found a policy on a part of the reward-tardiness
   Pareto front no fixed-lambda_2 search this project has run has reached.
2. **But reward roughly halved (254.75 -> 135.67 held-out), and the
   multiplier saturated at its projection ceiling rather than converging to
   an interior value.** `alpha=0.0` asks the constraint to drive weighted
   normalised tardiness to *exactly* zero -- for a stochastic scheduling
   problem with finite machine capacity, some tardiness is essentially
   unavoidable on a busy instance, so `E[C(tau)] > alpha` stays true
   indefinitely and the projected-ascent update keeps pushing `lambda`
   upward with nothing to stop it except the `lambda_max=50` bound we chose
   (see the dated doc's Section 4 grounding for that bound -- it was reused
   from `optuna_tune.py`'s `TARDINESS_PENALTY_WEIGHT` anchor, not derived
   for this specific run). At `lambda ~= 49`, the tardiness penalty
   dominates the `+3` placement / `+50` completion bonuses badly enough that
   the policy appears to be leaving many jobs unscheduled rather than risk
   any lateness -- consistent with late-jobs dropping to 3/100 at the cost
   of overall reward, rather than genuinely better scheduling throughput.

**Conclusion / next step:** This is a genuine result, not a bug -- the
"CMDP with `alpha=0`" formulation (grounded in Tessler et al. [1]'s Section
5.2 pattern, see the dated doc) behaves exactly as the theory predicts for a
target that is asymptotically unreachable: the multiplier saturates at
whatever ceiling is imposed rather than settling at an interior saddle
point. The result is real evidence that *adaptive* tardiness weighting can
reach a better tardiness/reliability trade-off than any fixed weight tried
this project -- but `alpha=0` was too strict a target for this environment,
and reward is being sacrificed further than necessary as a side effect of
hitting the projection bound rather than a deliberate trade-off. Follow-up
(not yet run): repeat with a less strict, still-grounded `alpha` (e.g.
anchored at Phase 8's own achieved held-out tardiness of ~28.66, or a
fraction of EDF's ~37.30) so the multiplier has an achievable target to
converge toward instead of climbing to its ceiling -- this should recover
more of the sacrificed reward while keeping most of the tardiness gain.
Proceeding next to the `flat`-architecture RCPO run for the A/B comparison,
per the agreed experiment ordering, before deciding whether to rerun with a
revised `alpha`.

[1] Tessler, Mankowitz, Mannor, ICLR 2019, arXiv:1805.11074.

---

## 2026-08-21 (S2W5) -- RCPO constrained tardiness (flat): perfect on the fixed instance, catastrophic held-out -- confirms flat is the architecture that memorizes

**Config:** Identical RCPO setup to the pointer run above, but `policy_type="flat"`
(`MaskableActorCritic`, one weight row per action index), warm-started at
`lambda_init=5.8464` (this architecture's own reward-tuned `lambda_2` from
`a2c_flat_best_params.json`). Checkpoint:
`a2c_flat_scheduling_optimized_shaped_rcpo.pt`, archived at
`rl_training/models/archive/2026-08-21_S2W5_a2c_flat_s4-200000_shaped_rcpo`.

**Stats:**
```
                        reward     tardiness   late_jobs      (fixed instance)
EDF                      289.38        16.00       10.00
Phase 8 flat (fixed λ)   262.98      1252.00       26.00
RCPO flat (adaptive λ)   198.50         0.00        0.00    <- perfect

                        reward (mean±std)   tardiness (mean±std)   late_jobs (mean±std)   (50 held-out)
EDF                      284.95±3.65           37.30±60.43            12.22±14.58
RCPO flat (adaptive λ)   192.92±1.02          570.62±89.14            24.02±3.34    <- worse than EDF, worse than every prior flat result

lambda(0) = 5.8464 -> lambda(final) = 20.59, still slowly climbing but NOT
saturated against the lambda_max=50.0 ceiling the way pointer's run was --
mean episode cost per update near the end was ~1.2-2.2, above the alpha=0.0
target but converging, not stuck at the projection bound.
```

**Observation:** The fixed-instance number looks like the best result this
project has ever produced -- literally zero tardiness, zero late jobs. It
is not. Evaluated on 50 held-out instances, the same checkpoint scores
570.62 mean tardiness and 24.02 late-jobs -- worse than EDF, worse than
every other flat-architecture result logged this session (including Phase
8 flat's already-bad 1252/26 *fixed*-instance numbers). This is the
starkest fixed-vs-held-out generalization gap recorded in this project to
date, and it lands on exactly the architecture already flagged as
overfitting-prone: `MaskableActorCritic` assigns one weight row per
`(job-slot, machine)` action index (see `a2c_policy.py`), so it has a
direct parametric route to memorizing "this specific job slot always goes
to this specific machine at this specific time" rather than learning
transferable job-feature-based placement rules, unlike the pointer
architecture's shared job/machine encoders. The 2026-08-20 randomized-
instance generalization entry already found the pointer/flat asymmetry in
generalization quality; RCPO's harder-driving adaptive penalty (pushed by
`alpha=0.0`, same as the pointer run) appears to have pushed the flat
network to fully exploit that memorization route rather than learn
anything transferable, making the asymmetry far more visible than the
Phase 8 fixed-lambda comparison did.

**Conclusion / next step:** RCPO's benefit found in the pointer run above
does **not** transfer to the flat architecture -- for flat, it produced the
project's worst-ever held-out result behind a perfect-looking but
meaningless fixed-instance number. Combined with the pointer result, this
is now a second, independent piece of evidence (on top of 2026-08-20's
entry) that generalization quality is primarily an *architecture* property
(pointer's shared encoders vs. flat's per-index weights), not a reward-
formulation property -- no reward-shaping or constraint mechanism tried
this project (potential-based shaping, tardiness-focused Optuna, RCPO) has
made the flat architecture generalize. Recommendation going forward:
treat pointer + potential-based shaping as the only architecture worth
further reward-side experimentation on; flat should only be kept as the
fixed-instance-only A/B baseline it already serves as. Next step for RCPO
specifically (pointer only): rerun with a less strict `alpha` per the
follow-up flagged in the pointer entry above, to test whether recovering
reward also affects the held-out generalization gap.

---

## 2026-08-20 (S2W5) -- Randomized-instance generalization: the fixed-instance shaping win is real, not memorization

**Config:** A2C, both `flat` and `pointer`. Implemented Experiment 2 in full:
`SchedulingEnv.set_jobs()` + `GymSchedulingEnv`'s new `job_resampler` let a gym
env draw a fresh random job set every episode
(`Code/training/train_optimized.py::make_random_instance_resampler()`,
`--randomize-instances`); `eval_rl_agent.py --randomized-eval` builds 50
held-out instances at seeds >= 500,000 (disjoint from any training seed by
construction) so both the model and EDF get evaluated on genuinely unseen
instances, not the single fixed one every prior entry used. Two things tested:
(A) train fresh under `--randomize-instances --use-potential-shaping` with a
newly re-tuned Optuna search on that distribution (50 trials/architecture,
`--randomize-instances --use-potential-shaping`); (B) evaluate the *existing*
2026-08-19 fixed-instance-trained+shaped pointer checkpoint (the one that beat
EDF, tardiness=9.0) on the same 50 held-out instances, to directly test
whether that win was genuine or fixed-instance memorization.

**Stats:**
```
                                          reward            tardiness          late_jobs
EDF, held-out (50 instances)             284.95 (3.65)     37.30 (60.43)      12.22 (14.58)

pointer, fixed-inst-trained+shaped,
  on the ONE fixed instance (2026-08-19) 253.94            9.00               6
pointer, fixed-inst-trained+shaped,
  on 50 HELD-OUT instances (new)         254.75 (9.14)     28.66 (57.56)      9.56 (13.45)

pointer, RANDINST-trained+shaped+retuned,
  on the fixed instance                  338.67            733.00             38
pointer, RANDINST-trained+shaped+retuned,
  on 50 held-out instances               334.995 (14.43)   732.50 (125.04)    39.44 (4.30)

flat, RANDINST-trained+shaped+retuned,
  on the fixed instance                  264.11            1396.00            43
flat, RANDINST-trained+shaped+retuned,
  on 50 held-out instances               264.41 (3.02)     1311.32 (148.70)   42.22 (3.66)
(figures in parens are std across the 50 episodes/instances; 0.0/blank for the
single fixed instance, since a deterministic model on one fixed instance has
no variance to measure)
```

**Observation, part 1 -- the real answer to the open question:** The
2026-08-19 fixed-instance-trained pointer+shaping model **generalizes**: on 50
instances it never saw during training, tardiness only rises from 9.00 to
28.66 (not collapsing), and it *still beats EDF's own held-out tardiness*
(28.66 vs. 37.30) and late-jobs (9.56 vs. 12.22). This directly answers the
question flagged in the 2026-08-19 entry and, further back, in
`2026-08-09-pointer-network-action-head.md` Section 9/10: the tardiness win
was not fixed-instance memorization. Plausible reason: potential-based
shaping's urgency signal (`Φ(s) = -Σ urgency_j(t)`) is a function of job
*features* (slack relative to deadline), not job *identity* -- so even
training on one fixed instance, the gradient signal it provides is inherently
general, and the pointer network's shared encoders (designed exactly to
generalize from features, per the original 2026-08-09 design doc) picked that
up rather than only memorizing per-slot lookups.

**Observation, part 2 -- deliberately training for generalization did worse,
not better:** Training fresh with `--randomize-instances` (jobs re-sampled
every episode, forcing generalization by construction, with hyperparameters
re-tuned on that same distribution) produced a *dramatically worse* result:
tardiness 732-733 on both the fixed instance and the held-out set -- roughly
25x worse than the fixed-instance-trained model's held-out tardiness (28.66).
This is a genuinely counter-intuitive negative result: the "textbook correct"
way to force generalization did worse than a model that happened to
generalize well despite fixed-instance training. Leading hypothesis, not yet
verified: every episode presenting a *different* job set the whole way through
training makes the learning problem itself much higher-variance within the
same finite timestep budget (each curriculum stage never gets to consolidate
around consistent job identities), and/or the fresh Optuna search's own
30k-timestep-per-trial budget was too short to properly assess hyperparameter
quality under this harder, higher-variance distribution (an instance of the
same tuning/testing mismatch risk Eimer et al. (2023) warn about, just
manifesting differently here). Both `flat` and `pointer` randinst-trained
models show the same pattern (fixed-instance and held-out performance track
each other closely -- i.e. *these* models generalize consistently too, just to
a worse policy), which supports "harder optimization problem," not "failed to
generalize," as the explanation.

**Conclusion / next step:** The best configuration found in this project to
date, across every experiment this session, is **pointer + potential-based
shaping + fixed-instance training + the original reward-tuned hyperparameters**
(2026-08-19's checkpoint) -- it beats EDF on tardiness/late-jobs on both the
instance it trained on AND 50 unseen ones. Not recommending
`--randomize-instances` for future runs based on this evidence; the
`--randomize-instances`/`--randomized-eval` infrastructure stays in the
codebase as a permanent, reusable capability (useful for the generalization
*test*, which is how this entry's key finding was actually established) even
though the *training* variant underperformed here. Per the user's agreed
ordering, proceeding next to Experiment 5 (RCPO-style constrained
optimization), on a dedicated git branch given how structurally different it
is from everything built so far.

---

## 2026-08-19 (S2W5) -- Potential-based shaping ablation: pointer beats EDF on tardiness

**Config:** A2C, both `flat` and `pointer`. Added `--use-potential-shaping` to
`Code/training/train_optimized.py`, threading `use_potential_shaping=True` and
`shaping_gamma=params["gamma"]` (this run's own tuned discount factor, not a
separate default) into every `make_env()` call. Everything else identical to
the S2W4 baseline: same reward-tuned Optuna params (`params_tag=None`, *not*
the S2W5 tardiness-tuned ones), same 200k stage-4 budget -- shaping is the
only changed variable, isolating it cleanly against the S2W4 baseline table.
Shaping itself (`Code/env/scheduling_env.py::_compute_potential()`) was
already implemented and sign-checked before this session; this is the first
time it has been run through the full 4-stage curriculum. Per Ng, Harada &
Russell (1999), this shaping is provably policy-invariant -- it changes
*training dynamics* (credit assignment), not which policy is optimal at
convergence.

**Stats:**
```
Deterministic eval, 50 episodes, stage-4 instance (horizon=100, num_jobs=100, seed=0):
                              total_reward   total_tardiness   late_jobs (/100)
EDF (fresh)                    289.38          16.00              10.00
a2c_pointer (S2W4 baseline)    270.23        1227.00             32.00
a2c_pointer (shaped)           253.94           9.00              6.00
a2c_flat (S2W4 baseline)       231.84         866.00             27.00
a2c_flat (shaped)              262.98        1252.00             26.00
```

**Observation:** Pointer + shaping is the best tardiness/late-jobs result in
this project's history by a wide margin, and the first RL result to actually
**beat EDF** on both (9.00 vs. EDF's 16.00 tardiness; 6 vs. 10 late jobs) --
at a reward cost of only 16.3 points (270.23->253.94, still well above every
non-tardiness-tuned flat result). Flat's result is mixed: reward improved
(231.84->262.98) but tardiness got worse (866->1252), late-jobs roughly flat
(27->26) -- shaping helped pointer far more than flat. Plausible reason:
pointer's shared job/machine encoders let the same per-step urgency signal
generalise across every job slot at once, where flat's per-index weights only
get that signal for the specific slots sampled in a given rollout -- consistent
with the architectural argument in
`2026-08-09-pointer-network-action-head.md` for why parameter sharing should
matter more once the learning *signal* itself (not just the objective) is
improved. Not yet verified against the trials/training curves in detail.

**Conclusion / next step:** This is the strongest positive result of the
project so far and the first genuine candidate for "RL beats the heuristic."
Per the user's prioritised order (literature review Section 7 + discussion),
proceeding next to Experiment 2 (randomized-instance generalization) to test
whether pointer+shaping's advantage survives when the job set isn't the same
fixed, memorized instance -- that is still the real test of the pointer
network's design claim, and now also the real test of whether this tardiness
win is genuine scheduling skill or another form of fixed-instance
overfitting. RCPO-style constrained optimization (Experiment 5, on a separate
git branch) queued after that.

---

## 2026-08-17 (S2W5) -- Doubling stage-4 training time: mixed, not a clean win

**Config:** A2C, both `flat` and `pointer`, same Optuna-tuned hyperparameters and
seed=0 fixed instance as the 2026-08-10 (S2W4) entry below -- only change is stage
4's timestep budget, doubled from 200k to 400k (added as `--stage4-timesteps` on
`Code/training/train_optimized.py`). Motivation: the S2W4 entry noted pointer's
stage-4 reward was still climbing (212.21->227.66), not plateaued, at 200k --
this tests whether that trend continues with more budget. Checkpoints archived to
`rl_training/models/archive/2026-08-17_S2W5_a2c_{pointer,flat}_s4-400000/` (see
`Code/utils/results_log.py`, added this session so runs stop overwriting each
other's checkpoints); eval summaries now also appended to
`rl_training/results/eval_results.csv`.

**Stats:**
```
Deterministic eval, 50 episodes, stage-4 instance (horizon=100, num_jobs=100, seed=0),
fresh EDF baseline (identical to S2W4's, as expected -- EDF and the instance are both
deterministic):
                         total_reward   total_tardiness   late_jobs (/100)
EDF (fresh)              289.38         16.00              10.00

                         S2W4 (200k stage4)      S2W5 (400k stage4)        delta
a2c_flat_optimized       231.84 / 866 / 27       259.17 / 1546 / 30        reward +27.3, tardiness +680, late +3
a2c_pointer_optimized    270.23 / 1227 / 32      249.47 / 1103 / 34        reward -20.8, tardiness -124, late +2
(columns: total_reward / total_tardiness / late_jobs)
```

**Observation:** Doubling stage-4 training time did **not** reliably help, and in
one case made things worse. Flat's reward improved (+27.3) but its tardiness
nearly doubled (866->1546) and late-jobs also rose -- more training time let it
find a *higher-reward* policy that is *more* tardy, consistent with the
already-documented reward/tardiness misalignment (the O(1) tardiness term is
small relative to the flat +3 placement / +50 completion bonuses) simply
getting more room to express itself with more optimization steps. Pointer's
result contradicts the motivating hypothesis outright: reward *fell*
(270.23->249.47) despite more training, though its tardiness improved modestly
(1227->1103). Neither architecture moved meaningfully closer to EDF on the
metric that matters most (tardiness/late-jobs); if anything flat moved further
away.

**Why EDF wins by so much on tardiness (user question, worth recording):** EDF
directly sorts by deadline at every decision -- it *is* a deadline-minimization
rule by construction, with the classical result that earliest-deadline-first is
optimal for minimizing maximum lateness on a single machine (J.R. Jackson,
"Scheduling a Production Line to Minimize Maximum Tardiness," Management
Science Research Project, UCLA, 1955); our setting is multi-machine and
resource-constrained so that exact optimality guarantee doesn't transfer, but
it explains why EDF is such a strong, tightly-targeted baseline in general.
This is training-log context, not a formal claim for this environment -- if
it becomes load-bearing for a written-up conclusion, it belongs in a dated
doc's numbered References instead. The RL agents, by contrast, are optimizing
a *reward* that only weakly encodes tardiness (`lambda_2 * T_j/H`, capped well
below 1 per job) relative to the dominant placement/completion bonuses -- so
"maximize reward" and "minimize tardiness" are related but not the same
objective here, and this entry's flat result is a direct demonstration of that
gap widening, not narrowing, under more optimization.

**Also fixed (unrelated bug, found while reading eval output with the user):**
`Code/evaluation/eval_rl_agent.py`'s per-episode eval function was named
`run_ppo()` and unconditionally printed "Running PPO episode..." even when
evaluating A2C (flat or pointer) -- a leftover from when it was PPO-only. This
caused a real mix-up mid-session (an A2C flat eval run's console output was
mistaken for a PPO run). Renamed to `run_model()`, print now generic. No PPO
run has actually happened this session; PPO+pointer integration remains
explicitly deferred (`Future/research/2026-08-09-pointer-network-action-head.md`
Section 9).

**Conclusion / next step:** Training-time alone is not the lever that closes
the tardiness gap -- this result argues *for* prioritizing Experiment 3
(tardiness-focused reward retuning: re-search `lambda_2` and the placement/
completion bonus weights specifically against tardiness/late-jobs) over further
training-time increases. Proceeding to Experiment 2 (randomized-instance
generalization) next per the already-agreed order, with Experiment 3 next after
that.

---

## 2026-08-17 (S2W5) -- Tardiness-focused Optuna retuning: helped at tuning scale, hurt at full scale

**Config:** A2C, both `flat` and `pointer`. Added `--optimize-for tardiness` to
`Code/training/optuna_tune.py`: Optuna's fitness metric for ranking trials
becomes `mean_reward - 50 * mean_tardiness_normalised` (was: `mean_reward`
alone) -- training still uses the same env reward as before (agent still has
to actually complete jobs to score well), only the *trial-selection* criterion
changed, the same way a model can be trained on one loss but selected on a
different validation metric. Weight of 50 chosen to match the environment's
own +50 completion bonus (documented as a calibration point, not a derived
optimum, in the code). 50-trial search per architecture on the existing small
tuning env (20 jobs, 5 machines, horizon 30, unchanged from prior studies),
then full 4-stage curriculum training (200k stage-4 budget, matching S2W4's
original, not S2W5's stage4-bump entry above) using each architecture's new
`*_tardiness_best_params.json`, then deterministic eval on the stage-4 fixed
instance exactly as previous entries.

**Stats:**
```
Optuna search (tuning env, 20 jobs/horizon 30, best trial of 50):
                    lambda_1   lambda_2   idle_pen   invalid_pen   other notable
pointer (reward)    0.66       3.85       1.78       7.84          hidden=32,  lr=1.47e-05
pointer (tardiness) 1.52       2.99       1.68       3.04          hidden=128, lr=6.84e-05
flat (reward)       0.59       5.85       1.26       4.06          lr=1.37e-04
flat (tardiness)    0.67       1.55       1.49       8.85          lr=9.85e-05
Both tardiness-tuned best trials: mean_tardiness=0.0, mean_late_jobs=0.0 on the
tuning env (10 eval episodes) -- a clean result at that scale.

Deterministic eval, 50 episodes, stage-4 instance (horizon=100, num_jobs=100, seed=0):
                              total_reward   total_tardiness   late_jobs (/100)
EDF (fresh)                   289.38          16.00              10.00
a2c_flat (S2W4 baseline)       231.84         866.00             27.00
a2c_pointer (S2W4 baseline)    270.23        1227.00             32.00
a2c_flat (tardiness-tuned)     270.39         998.00             31.00
a2c_pointer (tardiness-tuned)  328.21        1699.00             44.00
```

**Observation:** Both search's best trials achieved *zero* tardiness on the
small tuning env, but neither transferred to the full 100-job curriculum --
if anything both got worse on the metric this experiment specifically targeted.
Flat: reward improved (231.84->270.39) but tardiness rose (866->998) and so did
late-jobs (27->31). Pointer: reward hit the best score seen in this project's
history (328.21, beating even S2W5's stage4-bump result), but tardiness also
hit its worst-ever value (1699, vs. 1227 at baseline) and late-jobs rose to 44 --
the worst of every RL variant measured so far. Verified this isn't a params-file
mismatch: both training runs' logged "Loaded best hyperparameters from:
...tardiness_best_params.json" and printed penalty values match the Optuna
output exactly, and eval used matching `--embed-dim 128 --hidden 128` for
pointer.

**Why the tuning-env result didn't transfer (hypothesis, not yet verified):**
the tuning env (20 jobs, 5 machines, horizon 30) is a comparatively easy
packing problem -- reaching zero tardiness there may not require genuinely
tardiness-robust hyperparameters, just "any reasonably competent policy," since
there's little resource contention at that scale. Consistent with this: for
*both* architectures, the tardiness-optimized search actually picked a *lower*
`lambda_2` than the reward-optimized search did (pointer 3.85->2.99, flat
5.85->1.55) -- if the small env rewarded raising lambda_2 to fight tardiness,
we'd expect the opposite. Instead the search reached for other levers (pointer:
4x larger hidden layer, ~5x higher learning rate, much lower invalid_penalty;
flat: much lower lambda_2, higher invalid_penalty) that happened to work at 20
jobs but apparently don't scale to a 4-stage curriculum ending at 100 jobs --
plausibly because a much bigger, faster-learning pointer network is more prone
to overfitting/instability across curriculum-stage transitions (per the
architectural argument in `2026-08-09-pointer-network-action-head.md`), and a
much lower flat lambda_2 simply under-penalises tardiness once the problem gets
harder and slack is scarcer. Not yet verified against the trials data in
detail -- if this becomes load-bearing for a written conclusion it needs a
proper dated write-up, not just this log entry.

**Conclusion / next step:** Reward-penalty tuning targeting tardiness, as
implemented here, is a **negative result at deployment scale** -- worth
recording clearly rather than quietly dropping, per this project's rule about
reporting negative results honestly. The likely fix is not to abandon the
approach but to fix *what's being tuned against*: evaluate Optuna trials'
composite score on a harder/larger instance (closer to stage-4 scale) instead
of the current small 20-job tuning env, even though that makes each trial
slower. Flagging as a follow-up rather than doing it in this same session, so
it can be scoped and run deliberately. Also notable across this entry and the
one above it: the two highest-ever reward scores in this project's history
(259-328) have now both come paired with the two worst-ever tardiness scores
(1546, 1699) -- reward and tardiness are not just weakly correlated at this
point, they may be actively trading off against each other as either training
time or reward-side hyperparameters are pushed harder, which is worth keeping
in mind for Experiment 2 (randomized-instance generalization) as well.

---

## 2026-08-17 (S2W5) -- Literature review: reward hacking / HPO generalization explain this session's results

**Config:** N/A (literature review, not a training run). Full write-up:
`Future/research/2026-08-17-literature-review-improving-rl-agent.md`.

**Observation:** The two entries directly above this one (stage-4 timestep bump;
tardiness-focused Optuna retuning) both showed the same shape: pushing
optimization harder raised reward while also raising tardiness, sometimes to
record-worst levels in the same run. Literature search found this is a named,
studied phenomenon, not specific to this codebase: Skalse et al. (2022) define
reward hacking formally and show non-trivial proxy/true-reward pairs are
essentially always hackable; Pan, Bhatia & Steinhardt (ICLR 2022) empirically
show hacking *increases* with agent capability (model size, training duration)
via sharp phase transitions -- a close match to this session's data. Separately,
Eimer, Lindauer & Raileanu (2023) explain *why* the tardiness-focused Optuna
search didn't transfer: hyperparameter landscapes can overfit to the tuning
seed/environment, and recommend testing on a held-out environment before
trusting a tuned result -- this project's 20-job tuning env vs. 100-job
deployment scale is exactly that mismatch.

**Conclusion / next step:** Five concrete follow-ups identified, in priority
order (full reasoning in the dated doc, Section 7): (1) finally run Experiment
4 (potential-based shaping, already implemented, never run full-curriculum) --
raised in priority since related literature independently corroborates dense
per-step tardiness signals; (2) redesign the tardiness Optuna search to
evaluate trials on a larger/harder instance, not the small tuning env; (3) add
a tardiness sanity check to trial/checkpoint selection, not just reward; (4)
try a genuine multi-objective Optuna study instead of a hand-weighted scalar;
(5) prototype constrained RL (Tessler et al.'s RCPO) as a structurally
different alternative to hand-tuning `lambda_2` at all. None implemented yet
this session -- this entry is the research basis for a follow-up
implementation pass.

---

## 2026-08-10 (S2W4) -- Fresh Optuna + full-curriculum reruns post-bugfix: RL closes the gap to EDF

**Config:** A2C, both `flat` (`MaskableActorCritic`) and `pointer`
(`PointerActorCritic`) architectures, full `train_optimized.py` curriculum
(50k/100k/150k/200k = 500k timesteps), using fresh Optuna-tuned hyperparameters
(independent 50-trial studies per architecture, tuning env `horizon=30,
num_jobs=20, num_machines=5`) -- built on top of the bug fixes and reward
rescale in the entry immediately below and detailed in
`Future/research/2026-08-09-fixed-instance-bugfix-and-reward-rescale.md`.
Notably: `lambda_2` (tardiness weight) converged to 5.85 (flat) / 3.85
(pointer) in both studies, well above the old `[0.5, 2.0]` search range's
ceiling -- expected, since tardiness is now `T_j/H` (O(1)) instead of raw `T_j`
and needed more relative weight to matter. Pointer's tuned architecture:
`embed_dim=128, hidden=32` (hidden below the untuned default of 64).

**Stats:**
```
Training, first200->last200 mean episode reward per stage:
                       stage1(15j/h20)     stage2(30j/h40)     stage3(60j/h60)     stage4(100j/h100)
a2c_flat_optimized     57.75  -> 88.16     129.40 -> 135.10    133.21 -> 132.75    187.53 -> 180.85
a2c_pointer_optimized  52.83 -> 39.27      132.46 -> 129.65    137.85 -> 136.67    212.21 -> 227.66

Deterministic eval, 50 episodes, stage-4 instance (horizon=100, num_jobs=100, seed=0),
rescaled reward, freshly-measured EDF baseline (std=0.00 throughout -- both sides
deterministic on this one fixed, repeated instance):
                       total_reward   total_tardiness   late_jobs (/100)
EDF (fresh)            289.38         16.00              10.00
a2c_flat_optimized     231.84         866.00             27.00
a2c_pointer_optimized  270.23         1227.00            32.00
```

**Observation:** RL reward at stage 4 is now within **7% (pointer) / 20%
(flat)** of EDF -- compare against every prior fixed-instance entry in this
log, where stage 4 was deeply negative (e.g. `-21.8 -> 50.7` for the best prior
result, `flat_fixed_full`, itself against a *differently-scaled, not directly
comparable* EDF reference of `+275.5`). This is the first entry in this log
where RL is reward-competitive with EDF rather than losing by an order of
magnitude or landing on the wrong sign. However, RL's tardiness (866-1227) and
late-jobs (27-32) remain far worse than EDF's (16.0 / 10) -- the now-O(1)
tardiness term is small relative to the flat `+3.0` placement / `+50`
completion bonuses, so a policy can score well on reward while still routinely
scheduling jobs late. Pointer beat flat here (270.23 vs. 231.84, and still
improving at the end of stage 4 training: 212.21->227.66 vs. flat's plateaued
187.53->180.85) -- a reversal of the previous pointer-vs-flat comparison two
entries below, plausibly because this run gave each architecture its own
properly-tuned hyperparameters instead of reusing the flat head's incidental
defaults for the pointer network.

**Also discovered (documented, not fixed this session):** the saved
`training_rewards.csv`'s `timestep` column resets to ~0 at every curriculum
stage boundary (`MaskableA2C.train()`'s step counter `t` is local per call, not
cumulative across the curriculum loop's repeated `agent.train()` calls) -- the
stage table above was built using episode-index segments split at the reset
points, not by filtering on `timestep` directly, after that filtering
approach silently produced an empty "stage 4" bucket. The saved
`training_rewards.png` plots for these runs visibly wrap on the x-axis as a
result; this is a logging/plotting issue, not a training-correctness issue.
Also hit and fixed in passing: `train_optimized.py` crashed immediately on
Windows when its output was piped/redirected, due to Greek-subscript
characters in a print statement that cp1252 (the default Windows console
codepage) can't encode -- replaced with plain ASCII (`lambda_1` etc.); and
`eval_rl_agent.py` had no way to evaluate a checkpoint saved under a
non-default path or a pointer network built with non-default `embed_dim`/
`hidden` -- added `--model-path`/`--embed-dim`/`--hidden` overrides, needed to
evaluate these `train_optimized.py` checkpoints at all.

**Conclusion / next step:** The session's opening hypothesis -- that RL losing
badly to EDF on an instance it trains on repeatedly was primarily an
optimization/implementation-bug problem, not a capability or generalization
gap -- is well supported by this result. Two follow-ups, both explicitly
separate from what this session's scope covered: (1) if tardiness/late-jobs
specifically (not just total reward) matters for the paper's claims, that
needs its own deliberate retuning (e.g. `lambda_2` higher still, or reweighting
the placement/completion bonuses) rather than assuming today's reward parity
already implies it; (2) the real test of the pointer network's actual design
claim (generalizing across job identity, not memorizing one fixed instance)
is still the deferred randomized-per-episode-instance experiment -- see
`Future/research/2026-08-09-fixed-instance-bugfix-and-reward-rescale.md`
Section 7.

---

## 2026-08-09 (S2W3) -- Three more bugs found and fixed (machine-activation ordering, numpy-bool dead code, mask/step time mismatch), tardiness term normalised

**Config:** N/A (bug fixes + reward-formula change, not a training-hyperparameter
change). Full detail and derivations:
`Future/research/2026-08-09-fixed-instance-bugfix-and-reward-rescale.md`.

**Bugs found and fixed (all in `Code/env/scheduling_env.py` /
`Code/env/gym_scheduling_wrapper.py`):**

1. **`machine_active` flipped before the feasibility check, never rolled back.**
   `SchedulingEnv.step()` used to set `machine_active[machine] = 1` before
   checking `is_feasible()`, and didn't undo it if that same action then failed
   feasibility -- so an infeasible attempt on a never-used machine permanently
   marked it "active" without the `-lambda1` activation penalty ever being
   charged on the real first successful use. Fixed by moving the mutation to
   after the feasibility check passes.
2. **`if ym is True:` never actually fires -- numpy bool identity bug.** While
   writing the regression test for fix #1, found that `reward()`'s activation
   penalty branch checks `if ym is True:`, but every caller passes a numpy
   `bool_` (from `self.machine_active[machine] == 0`), and
   `np.bool_(True) is True` is `False` (an identity check against a different
   object, not an equality check). **This means the `-lambda1` activation
   penalty has never actually fired, in the project's entire history**,
   independent of bug #1 -- confirmed by a direct regression test
   (`tests/test_bugfixes.py`) before vs. after: reward for a real first
   placement read `3.0` (bug present) vs. the mathematically correct `2.0`
   (`-lambda1 + 3.0` flat bonus) after the fix. Fixed by using plain truthiness
   (`if ym:`), which is correct for both a Python bool and a numpy bool_.
3. **Mask/execution time-index mismatch at `t == horizon`.**
   `GymSchedulingEnv.get_action_mask()` used a clamped
   `t = min(env.time, horizon-1)`, but the real check inside
   `SchedulingEnv.step()`/`is_feasible()` used the uncapped `env.time`. At
   `env.time == horizon` (reachable -- episodes only end once `time > horizon`),
   a duration-1 job could read as feasible in the mask but fail the real check,
   landing in the invalid-action branch -- which does not advance time, so a
   policy trusting the mask could get stuck repeating the same invalid action
   forever with no way for the episode to end on its own. Fixed by uncapping the
   mask's `t` to match `step()` exactly (safe: `is_feasible()` short-circuits on
   `t+duration > horizon` before any array indexing, and every job has
   `duration >= 1`, so no out-of-bounds read is possible). `_get_obs()`'s
   *separate* clamp (a real array index into `capacity[m, r, t_idx]`) was left
   untouched. Added a belt-and-suspenders per-episode invalid-action counter to
   `GymSchedulingEnv` that sets `truncated=True` past a fixed cap, as a general
   safety net against this *class* of bug independent of this specific
   instance.

**Reward-formula change:** tardiness is now normalised by the episode's own
horizon (`T_j/H` instead of raw `T_j`) in `SchedulingEnv.reward()`. Raw
tardiness is unbounded and scales with `horizon` (`T_j <= H-10` given
`deadline_range=(10,110)`), while every other reward term is a fixed O(1)
constant regardless of curriculum stage -- across `horizon in {20,40,60,100}`,
this let the tardiness term's achievable magnitude grow ~5x from the first to
the last curriculum stage while one A2C model/value-head is reused across all 4
stages with no reset. `T_j/H < 1` provably for any job that is ever actually
scheduled, putting tardiness on the same O(1) footing as everything else. **Old
reward numbers throughout this log (everything above this entry) are NOT
directly comparable to anything measured after this point** -- both RL and EDF
go through the same `reward()`, so *relative* RL-vs-EDF comparisons stay valid,
but absolute magnitudes shifted (e.g. the `+275.5` EDF stage-4 reference two
entries below was measured pre-fix and needs re-measuring, not reuse).

**Also fixed (not a bug, but a fairness/diagnosability gap):** A2C's
`select_action()`/`act()` previously had no deterministic/greedy mode -- always
sampled, even during evaluation -- while PPO's eval already used
`model.predict(..., deterministic=True)`. Added a `deterministic` flag
(argmax when `True`), threaded through to `eval_rl_agent.py` and
`optuna_tune.py`'s A2C eval loops. Also added per-stage model checkpointing to
`train_rl_agent.py`/`train_optimized.py` (previously only a single save after
the entire curriculum), so a stage-3/4 regression can be diagnosed without
rerunning from scratch.

**Stats:**
```
Regression test (tests/test_bugfixes.py), all 4 checks PASS:
  Issue A: infeasible attempt -> machine_active stays 0 (was silently flipping to 1)
  Issue A + numpy-bool bug: real first placement reward == 2.0 (was 3.0, activation
    penalty silently never charged)
  Issue C: t=horizon-1 mask=1 (correct); t=horizon mask=0 AND step() agrees (both were
    previously divergent: mask said feasible, step() said invalid)
  Issue D: tardiness term at H=20 -> 0.90 (was raw 18.0); at H=100 -> 0.98 (was raw 98.0)
    -- both now O(1) as proven, vs. a previous ~5x spread across curriculum stages

Smoke runs (--smoke-test, 300 timesteps/stage, full 4-stage curriculum):
  A2C flat: completes end-to-end, 4/4 stage checkpoints saved, no crash
  A2C pointer: completes end-to-end, 4/4 stage checkpoints saved, no crash
  Eval plumbing (A2C deterministic + EDF): both run cleanly against the rescaled
    reward, e.g. one post-smoke-training eval episode: A2C=193.6, EDF=289.4
    (undertrained model from a 300-step/stage smoke run -- not a real comparison,
    plumbing check only)
```

**Observation:** The numpy-bool `is True` bug (#2) is the most significant
finding here: it means the activation-cost term of the reward function has been
dead code for the project's entire history, so every prior training-log entry's
`lambda_1` was effectively `0` in practice regardless of its configured value.
This does not bias RL-vs-EDF comparisons specifically (both share the same
`reward()`), but it does mean `lambda_1` was never actually doing anything in
any run logged above, including every Optuna trial that "tuned" it.

**Conclusion / next step:** All three bugs and the reward rescale are
prerequisites for every experiment from this point forward, same as the
2026-08-09 capacity-leak/dict.get() entry below was for its generation of
experiments. Next: rerun Optuna against this fixed, rescaled environment
(separately for the flat and pointer A2C architectures -- see
`Future/research/2026-08-09-fixed-instance-bugfix-and-reward-rescale.md`), then
full-curriculum reruns and a freshly-measured EDF baseline for an apples-to-apples
comparison. Results to be logged in a follow-up entry once those complete.

---

## 2026-08-09 (S2W3) -- Pointer network vs. flat baseline, both with ent_coef/reward-norm/masked-entropy fixes

**Config:** A2C, full `train_rl_agent.py` curriculum (375k timesteps), both runs on
top of the capacity-leak and eager-`dict.get()` fixes (previous entries) AND the
`a2c_policy.py` fixes made while implementing Solution 2 (`ent_coef` 0.0 -> 0.01,
running-std reward normalisation added, masked-entropy-in-loss bug fixed --
`Future/research/2026-08-09-pointer-network-action-head.md` Section 7.3).
**flat_fixed_full**: `policy_type="flat"` (`MaskableActorCritic`, unchanged
architecture). **pointer_full**: `policy_type="pointer"`
(`PointerActorCritic`, `embed_dim=128`, `hidden=64`, defaults -- not tuned).

**Stats:**
```
                    stage1(15j/h20)      stage2(30j/h40)      stage3(60j/h60)       stage4(100j/h100)
                    first200->last200    first200->last200    first200->last200     first200->last200
flat_fixed_full     68.8  -> 86.3        81.5  -> 139.2         32.9 -> 104.4        -21.8  -> 50.7
pointer_full        58.7  -> 94.3        92.7  -> 101.4         34.9 -> 25.0        -131.2  -> -63.3
```

**Observation:** Two findings, and they point in different directions.

First -- **the `ent_coef`/reward-normalisation/masked-entropy fixes alone (i.e. even
on the unchanged flat architecture) mostly fix the stage-3/4 regression** documented
in the entry above. Compare `flat_fixed_full` here against `baseline_fixed` in the
entry above (same architecture, same curriculum, only these three fixes differ):
stage 3 goes from a flat -251.9/-236.2 plateau to an *improving* 32.9->104.4 trend;
stage 4 goes from a flat -1264.2/-1299.2 plateau to an *improving* -21.8->50.7 trend
that ends positive. So zero exploration pressure and an unnormalised, scale-jumping
reward target were doing most of the damage in that regression -- not primarily the
lack of parameter sharing across job-slot identity, as Section 7.2 of the research
doc hypothesised.

Second -- **contrary to that same hypothesis, the pointer network does *worse* than
the now-fixed flat baseline** on this specific comparison, in both stage 3
(25.0 vs. 104.4 final) and stage 4 (-63.3 vs. 50.7 final), despite clearly beating
the *original* (unfixed) flat baseline by a wide margin. Leading explanation, on
reflection: **this curriculum does not actually test what the pointer network is
designed for.** Every stage reuses the exact same fixed job set (`seed=0` throughout,
each stage just truncates to a longer prefix of the same underlying jobs) for
thousands of repeated episodes -- stage 3 alone is ~1,638 episodes over the same 60
jobs. A flat per-index head has no need to generalise from job *features* to do well
here; given working exploration and correctly-scaled gradients (this entry's two
fixes), it can simply memorise the optimal action for each of the ~30-40 "newly
unmasked" fixed job slots directly, which is a strictly easier optimisation target
than the pointer network's indirect, shared-weight-through-an-attention-score
parameterisation -- and the pointer network's extra indirection (encoder -> scorer ->
pooling) may need more tuning (learning rate, embed_dim, or more timesteps) to match
a direct lookup table's convergence speed on a fixed, repeated instance. The pointer
network's actual hypothesised advantage -- generalising to a job it has never seen
the specific index of, from its features alone -- is never exercised by a curriculum
where "new" job slots are still the same fixed jobs seen thousands of times.

**Conclusion / next step:** Don't read this as the pointer-network idea being wrong;
read it as this curriculum being the wrong experiment to test it with. The real test
needs per-episode (or per-stage) *randomised* job sets, so a flat per-index head
genuinely cannot memorise and must generalise from features to do well -- exactly the
condition the pointer network's shared encoders are designed for. Until that
experiment is run, the honest claim is: the cheap hygiene fixes (entropy bonus,
reward normalisation, consistent masking in the loss) recovered most of the
stage-3/4 regression on their own, and the pointer network's benefit on a
fixed-instance curriculum is currently negative, not positive -- its value proposition
remains theoretically sound (Section 7.2 of the research doc) but is unproven by this
run. See `Future/research/2026-08-09-pointer-network-action-head.md` Section 8/10.

---

## 2026-08-09 (S2W3) -- A2C full curriculum, capacity-leak fixed: stages 1-2 solved, stages 3-4 regress hard

**Config:** A2C (`Policies/a2c_policy.py`, `policy_type` not yet added -- still the flat
`MaskableActorCritic`). Full `train_rl_agent.py` curriculum (15/30/60/100 jobs,
horizon 20/40/60/100, 50k/75k/100k/150k timesteps, 375k total). Three variants run in
parallel from the same fixed environment: **baseline_fixed** (no Solution-1 change,
`idle_penalty=0.5`), **solution1a** (`restrict_idle=True`, `idle_penalty=0.5`),
**solution1b** (`idle_penalty=50`, idle not restricted). All three built on top of two
bug fixes made just before this run (see 2026-08-09 entry below): the
`SchedulingEnv.reset()` capacity leak, and an eager-`dict.get()` crash in
`a2c_policy.py`'s rollout loop that previously made A2C untrainable outright.

**Stats:**
```
                    stage1(15j/h20)      stage2(30j/h40)      stage3(60j/h60)       stage4(100j/h100)
                    first200->last200    first200->last200    first200->last200     first200->last200
baseline_fixed      57.7  -> 77.6        107.5 -> 107.6        -251.9 -> -236.2      -1264.2 -> -1299.2
solution1a          71.6  -> 86.5        128.5 -> 129.6        -218.3 -> -213.0      -1214.4 -> -1191.3
solution1b          20.2  -> 69.9        116.6 -> 114.5        -346.5 -> -350.2      -1491.0 -> -1504.7
```
(`first200`/`last200` = mean episode reward over the first/last 200 episodes of that
stage -- a stalled-vs-improving check, not just a single-point snapshot.)

Reference point: an EDF heuristic (earliest-deadline-first, existing code in
`test_env.py`/`eval_rl_agent.py`) run on the exact stage-4 job set gets **+275.5**
total reward with only 16.0 cumulative tardiness and 98/100 jobs scheduled -- i.e. the
stage-4 problem itself is close to fully solvable by a simple greedy rule.

**Observation:** With the capacity-leak fix alone, stages 1-2 now train to strong,
stable positive reward for all three variants (previously: 100% idle collapse, per
every prior entry in this log) -- confirming that bug was a major, possibly dominant,
confound in the "PPO/A2C fundamentally fails" conclusions reached before 2026-08-09.
`solution1a` (idle restricted) modestly outperforms `baseline_fixed` throughout, and
`solution1b` (idle_penalty=50) modestly *under*performs baseline in every stage --
consistent with idle-penalty magnitude being a secondary factor, not the primary lever.

But **stages 3 and 4 collapse hard immediately on transition and never recover**:
first200 vs. last200 within each stage are statistically indistinguishable (e.g.
baseline stage3: -251.9 -> -236.2 over 1,639 episodes; stage4: -1264.2 -> -1299.2 over
1,485 episodes) -- a flat plateau, not a slow recovery in progress. Given the EDF
reference shows the stage-4 problem is easy, and given `max_jobs=100` padding means job
slots 30-59 are masked out (never sampled, never gradient-updated) throughout stages
1-2 and only "activate" in stage 3 (same for slots 60-99 in stage 4), the leading
hypothesis is architectural: `MaskableActorCritic.policy_head = Linear(256, 1001)`
gives every `(job_slot, machine)` index its own independent weight row with zero
parameter sharing across job identity, so newly-unmasked slots start every stage
transition from scratch with no transferred knowledge, on top of a reward-scale jump
(stage1-2 rewards live in roughly [0,130]; stage3-4 jump to [-1500,+300] with no reward
normalization anywhere in the pipeline) that likely destabilizes the value function's
bootstrapped targets right when it can least afford it. Two additional gaps noted
while investigating: this hand-rolled A2C has `ent_coef=0.0` (zero exploration bonus,
unconditionally, so no forcing function pushes exploration of newly-unmasked slots),
and running all three variants concurrently with unfixed relative output paths caused
them to overwrite each other's `rl_training/models/a2c_scheduling.pt` -- only the
final one to finish survived on disk (see `Code/utils/paths.py`, planned).

**Conclusion / next step:** This is evidence for, not against, the pointer-network plan
already in motion (`Future/research/2026-08-09-pointer-network-action-head.md`): a
shared job encoder / machine encoder (rather than per-index weights) means a job slot's
score is a function of its *features*, learned from every job seen so far regardless of
slot index -- so newly-unmasked slots at a curriculum transition are scored
correctly from the first step, with no separate "curriculum learning fix" needed. Two
cheap, architecture-independent additions are being folded in alongside it: reward
normalization (keep the value function's target distribution roughly stationary across
stages) and `ent_coef > 0` for this A2C implementation (currently always exactly zero).

---

## 2026-08-09 (S2W3) -- Two pre-existing bugs found and fixed before the above run

**Config:** N/A (bug fixes, not a training-hyperparameter change).

**Bug 1 -- `SchedulingEnv.reset()` capacity leak.** `reset()` restored every timestep's
capacity from `self.capacity[:, :, 0]`, but that slice is itself mutated by `step()`
whenever a job starts at time 0 (true for most episodes), so it was never actually the
original capacity after episode one. Since a single `SchedulingEnv` instance is reused
for hundreds/thousands of episodes per curriculum stage, capacity leaked downward
permanently and never recovered. Confirmed directly: 5 episodes each scheduling one job
at t=0 dropped machine capacity from `[30,30,30,30]` to `[18,6,21,21]`, permanently.
Fixed by storing a pristine `self.machine_capacity` vector separately and having
`reset()` restore from that, not from the mutable per-timestep array.

**Stats:**
```
20k-timestep smoke test, stage-1-sized problem (15 jobs, horizon 20), before vs after fix:
PPO:  100% idle, reward ~= -24   ->   0% idle, reward = +94.5
A2C:  100% idle                 ->   0% idle, reward = +79.5
```

**Bug 2 -- eager `dict.get()` default crash in `a2c_policy.py`.** `train()`'s rollout
loop had `mask = info.get("action_mask", self.env.get_action_mask())` -- `dict.get()`
evaluates its default argument unconditionally, so this called
`self.env.get_action_mask()` every step regardless of whether `"action_mask"` was
already in `info` (it always was). That call crashed under the installed gymnasium
version (1.2.3), which removed automatic attribute forwarding through
`Wrapper.__getattr__` (`Monitor` has no `get_action_mask` of its own). This made every
A2C training run fail immediately, independent of anything else in this log. Fixed by
using `info["action_mask"]` directly, since `GymSchedulingEnv` always populates it.

**Observation:** Bug 1 in particular reframes a large portion of this log's prior
"policy collapse" entries: they may be partially or fully explained by an environment
data bug rather than (only) the PPO-clipped-objective / large-action-space mechanism
argued in `2026-07-24-idle-action-policy-collapse.md`. That diagnosis isn't invalidated
-- see the entry above, where collapse re-emerges at larger curriculum stages even with
this bug fixed -- but every collapse result *before* this fix should be read as
confounded, not as clean evidence for the clipped-objective hypothesis specifically.

**Conclusion / next step:** Both fixes are prerequisites for every experiment from this
point forward (Solutions 1, 2, 3 in `Future/research/2026-08-09-pointer-network-
action-head.md`) and are already included in the run logged above.

---

## 2026-07-24 (S2W1) -- Bigger network + fewer PPO epochs (result pending)

**Config:** PPO. Curriculum: `num_jobs` now varies per stage (15/30/60/100, all
`<= horizon`), `max_jobs=100` fixed via job-slot padding. `ent_coef=0.05` (from 0.01).
Reward: hotspot double-count removed, placement bonus `+1` (from `+0.1`),
`idle_penalty=0.5` (from `1.0`). `policy_kwargs=dict(net_arch=dict(pi=[256,256],
vf=[256,256]), activation_fn=nn.Tanh)` (from SB3 default `[64,64]`). `n_epochs=4`
(from `10`).

**Stats:** not yet run with this configuration -- to be filled in once training
completes.

**Observation:** --

**Conclusion / next step:** See
`Future/research/2026-07-24-idle-action-policy-collapse.md` Section 6 for what to
check for in this run's stats (non-zero `entropy_loss`/`approx_kl` past the first few
iterations; `ep_rew_mean` not an exact multiple of `-idle_penalty`). If collapse
persists despite these changes, that favours the deferred fix in Section 5
(factorizing the action space into `MultiDiscrete([job, machine])`) over further
hyperparameter tuning.

---

## 2026-07-24 (S2W1) -- Reward reweighting + higher ent_coef (still collapsed)

**Config:** PPO. `ent_coef=0.05` (from `0.01`). Hotspot penalty double-count removed.
Placement bonus `+1` (from `+0.1`). `idle_penalty=0.5` (from `1.0`). Curriculum
`num_jobs` made reachable-within-horizon (previously stuck at env_config's default of
100 regardless of horizon).

**Stats:**
```
rollout/ep_len_mean       21
rollout/ep_rew_mean       -10.5
time/total_timesteps      51,200
train/approx_kl           0.0
train/clip_fraction       0
train/entropy_loss        0
train/explained_variance  0.998
train/policy_gradient_loss 7.04e-09
train/value_loss          0.263
```

**Observation:** `ep_rew_mean / ep_len_mean = -0.5`, exactly `-idle_penalty`. Still
fully collapsed onto the idle action -- just at the new, lower idle-penalty scale.
Raising `ent_coef` and making the reward more favourable toward placement did not
prevent the collapse; only the constant it collapsed to changed.

**Conclusion / next step:** The cause is more likely structural (network capacity
and/or the flattened ~1000-way action space interacting badly with PPO's clipped
objective) than a reward-magnitude problem. Led to the literature review in
`Future/research/2026-07-24-idle-action-policy-collapse.md` and the network-size /
`n_epochs` changes in the entry above.

---

## 2026-07-24 (S2W1) -- Initial run (collapsed onto idle)

**Config:** PPO. `ent_coef=0.01`. `idle_penalty=1.0`. Hotspot penalty double-counted
(bug, since fixed). Placement bonus `+0.1`. Curriculum `num_jobs` fixed at
env_config's default of 100 regardless of horizon, so the `+50` completion bonus was
unreachable except in the final (horizon=100) curriculum stage.

**Stats:**
```
rollout/ep_len_mean       21
rollout/ep_rew_mean       -21
time/total_timesteps      14,336
train/approx_kl           0.0
train/clip_fraction       0
train/entropy_loss        0
train/policy_gradient_loss 3.56e-09
```

**Observation:** `ep_rew_mean / ep_len_mean = -1.0`, exactly `-idle_penalty`. Policy
had already collapsed onto idling by this point, this early in stage 1.

**Conclusion / next step:** First appearance of the collapse. Initial hypothesis:
insufficient exploration (`ent_coef` too low) and the completion bonus being
unreachable early in the curriculum. Both addressed in the next entry.
