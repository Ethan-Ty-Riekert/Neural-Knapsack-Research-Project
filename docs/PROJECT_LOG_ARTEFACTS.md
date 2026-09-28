# Project Log Artefacts — Curated Selection

A curated, diverse selection of artefacts from the Neural-Knapsack Research Project
(Curtin University Year 3, RL for cloud-resource scheduling / dynamic bin-packing),
spanning the full ~9-week arc from the first training attempts (2026-07-24) to the
most recent diagnostic work (2026-09-22). Every table below is copied directly from
the project's own logs — nothing is summarised-then-lost; the numbers here are the
actual numbers. Where an entry has an image, its file path is given so it can be
copied straight into your log; every other result is reproduced in full below, no
need to open another file. Entries are ordered chronologically so they read as a
development arc.

---

## 1. Literature-grounded diagnosis of an early training failure (2026-07-24)

**Caption:** The project's very first training runs collapsed onto always picking
"idle."

**Result:**
- `ep_rew_mean` locked onto an exact multiple of `-idle_penalty` (the policy always
  chose the one action that gets punished least, not the task).
- Policy entropy and KL divergence both flatlined at exactly zero.
- Reweighting the reward and raising the entropy coefficient changed *which* constant
  the policy collapsed to, not *whether* it collapsed — a diagnostic clue, not a fix.
- Diagnosis: matches a documented failure mode of PPO's clipped objective in large
  discrete action spaces (Hsu, Mendler-Dünner & Hardt, 2020).

**Why this matters as evidence of learning:** This is the project's first instance of
a habit that recurs throughout its history: when a result looks wrong, go to the
literature before touching hyperparameters. The two cheap fixes tried first (bigger
network, fewer PPO epochs) were correctly read as inconclusive rather than declared a
win, and the structurally stronger fix (factorising the action space) was deliberately
deferred rather than jumped to — sequencing effort by cost/risk instead of reaching
for the most sophisticated tool immediately.

---

## 2. Three bugs behind the project's first non-negative result (2026-08-09)

**Caption:** While writing a regression test for an unrelated bug, a second bug was
found in the reward function: `if ym is True:`, where `ym` is a numpy `bool_`, not a
Python `bool` — and `np.bool_(True) is True` evaluates to `False`. The
machine-activation penalty had never fired once in the project's history, regardless
of what value Optuna had "tuned" it to. Two more bugs were found the same session: a
mask/step time-index mismatch at `t == horizon` (could strand a policy repeating an
invalid action forever), and a reward-formula fix (tardiness normalised to `T_j/H`,
provably `< 1`, instead of raw `T_j`, which scaled ~5x across curriculum stages).

**Result — first result in the project's history that isn't strictly negative,**
after fixing all three plus retuning:

| Method | Total reward | Total tardiness | Late jobs (/100) |
|---|---|---|---|
| EDF (heuristic) | 289.38 | 16.00 | 10.00 |
| A2C flat (optimised) | 231.84 | 866.00 | 27.00 |
| A2C pointer (optimised) | 270.23 | 1227.00 | 32.00 |

**Why this matters as evidence of learning:** This is a textbook language-level
gotcha (identity vs. equality comparison across numpy and Python types), only found
because a regression test was being written for something else entirely — direct
evidence that writing tests as a matter of course, not just when something looks
broken, pays off. The harder, more honest part: every `lambda_1` value ever tuned by
Optuna up to this point — including the "best" one — had been tuned against a
completely inert term, so every prior conclusion had to be treated as provisional
rather than quietly kept. That kind of retroactive humility about your own past
results is a harder skill than finding the bug itself.

---

## 3. Pointer-network action head: an architecture choice grounded in two papers, not intuition (2026-08-09)

**Caption:** After finding that the flat `Linear(256, 1001)` policy head gives every
`(job, machine)` pair its own independent weight — so newly-unmasked job slots start
from scratch with no transferred knowledge — a `PointerActorCritic` was designed with
shared job/machine encoders and an attention-style compatibility score per pair,
following Vinyals et al. (2015) and Kool, van Hoof & Welling (2019).

**Result:** The very first comparison was negative — the pointer network did *worse*
than the (now-bug-fixed) flat baseline. Rather than abandon the design, this was
diagnosed as testing the wrong thing: every curriculum stage reused the same fixed
`seed=0` job set for thousands of episodes, letting the flat head simply memorise the
optimal action per slot — a strictly easier target than the pointer network's shared,
indirect parameterisation. The real test (randomised per-episode job sets) was
explicitly deferred rather than treated as answered — see artefact 4 for how that test
eventually turned out.

**Why this matters as evidence of learning:** This shows the project's core standard
in action — every architecture choice must trace to a citation or formal derivation,
not just "this worked better." Just as important is what happened when the first
result contradicted the hypothesis: it was diagnosed rather than either forced through
or quietly dropped, correctly separating "wrong experiment" from "wrong idea" — a
genuine research-methodology skill, not just an engineering one.

*(Code: `Code/policies/pointer_policy.py`)*

---

## 4. Potential-based reward shaping: the first RL result to beat the heuristic baseline (2026-08-19)

**Image:** `Results/v1_legacy_reward/reduced_budget_2026-09-04/figures/gantt_pointer_shaped.png`

**Caption:** After several results tied tardiness improvements to reward hacking,
potential-based shaping (Ng, Harada & Russell, 1999 — provably policy-invariant, so it
cannot change what the optimal policy *is*, only how quickly it's found) was applied
on top of the existing reward-tuned hyperparameters, isolated as the single changed
variable against an otherwise-identical baseline.

**Result:**

| Method | Total reward | Total tardiness | Late jobs |
|---|---|---|---|
| EDF (heuristic) | — | 16.00 | 10 |
| Pointer, pre-shaping | 270.23 | — | — |
| Pointer + potential-based shaping | 253.94 | 9.00 | 6 |

Tardiness and late-jobs both beat EDF, at a reward cost of only ~16 points
(270.23 → 253.94).

**Why this matters as evidence of learning:** This wasn't reached by throwing random
techniques at the wall — it was the literature review's top-priority recommendation,
implemented as the very next step, isolated as the only changed variable (proper
single-variable experimental control). The result is also reported with its actual
limits intact: the flat architecture's version of the same fix was mixed, not
uniformly good, so the finding was written up as "shaping's benefit is
architecture-dependent" rather than "shaping works" — the more defensible, more useful
claim.

---

## 5. RCPO's "best-ever tardiness" was reward hacking via job abandonment — caught, admitted, and fixed (2026-08-21 → 2026-08-28)

**Image:** `EthanTravelDocs/portfolio-artefacts/figures/rcpo_timeline.png`

**Caption:** A constrained-optimisation method (RCPO — Tessler, Mankowitz & Mannor,
2019) initially produced the best tardiness result in the project's history — until a
diagnostic re-run revealed the checkpoint was scheduling only about half its jobs,
achieving "low tardiness" by refusing risky-looking jobs rather than scheduling them
better. The constraint's cost function had no penalty term for a job that was never
scheduled at all.

**Result — before the fix** (pointer architecture, held-out evaluation):

| Method | Reward | Tardiness | Late jobs |
|---|---|---|---|
| EDF | 284.95 ± 3.65 | 37.30 ± 60.43 | 12.22 ± 14.58 |
| Shaped pointer, no RCPO | 254.75 ± 9.14 | 28.66 ± 57.56 | 9.56 ± 13.45 |
| RCPO (alpha=0, unfixed constraint) | 135.67 ± 15.10 | **19.84 ± 17.99** | **3.20 ± 2.12** |

Looked like the best result in the project — until jobs-actually-scheduled was
checked: the RCPO checkpoint scheduled only **~55 of 100 jobs** per episode, vs. the
shaped checkpoint's **~89 of 100** — low tardiness bought by abandonment, not skill.

**Result — after the fix** (unscheduled jobs now charged their worst-case cost;
target `alpha` set to an achievable 0.2866 instead of an unreachable 0):

| Method | Reward | Tardiness | Late jobs |
|---|---|---|---|
| RCPO, fixed constraint, achievable alpha | **268.77** | 28.16 | 9.74 |
| Shaped pointer, no RCPO | 254.75 | 28.66 | 9.56 |
| EDF | 284.95 | 37.30 | 12.22 |
| LST (strongest heuristic) | 288.28 | 23.94 | 8.26 |

Jobs scheduled recovered to **93.1/100** (abandonment cut by ~85%), reward improved
5.5% over the shaped baseline with tardiness essentially unchanged — a genuine, if
modest, win once the loophole was closed.

**Why this matters as evidence of learning:** This is the project's clearest single
example of methodological rigor catching a result that looked great on the watched
metric but was wrong on the metric that mattered. The fix wasn't patched in quietly —
the original write-up carries an explicit, dated correction notice rather than being
silently edited, matching the project's own append-only documentation rule ("if a
conclusion turns out wrong, say so in a new entry, don't rewrite history"). Publishing
a correction to your own prior claim, with the mechanism of the error spelled out, is
a stronger demonstration of research integrity than the original result would have
been on its own.

---

## 6. Building a real baseline suite instead of a strawman (2026-08-28)

**Code:** `EthanTravelDocs/portfolio-artefacts/code-samples/priority_rules.py`
(source: `Code/baselines/priority_rules.py`)

**Caption:** Six classical job-priority dispatching rules (EDF, SPT, LST, FCFS, LPT,
WSPT) composed with three machine-placement rules, plus Tetris's multi-resource
scorer — 23 named baselines total, each grounded in a citation (e.g. the ATC composite
index from Vepsalainen & Morton, 1987).

**Result:** Running this suite for the first time revealed that **LST, not EDF, is
the strongest classical heuristic on every metric** — something ten prior project
phases had never checked, because EDF had been assumed as "the" reference point
without ever being tested against the wider rule set.

**Why this matters as evidence of learning:** It's easy to compare a new method
against a weak or convenient baseline and call it progress. This shows the opposite
instinct: after ten phases of comparing everything to EDF, the comparison set was
deliberately widened, and the result overturned an assumption the whole project had
been implicitly running on. Correctly implementing several non-trivial scheduling
rules from primary sources is also straightforward evidence of careful,
literature-faithful engineering.

---

## 7. The exact-solver (CP-SAT) baseline reveals a gap invisible to every previous metric (2026-08-28)

**Caption:** A CP-SAT exact solver (Google OR-Tools) was built as a provably-optimal
reference point on small instances (`num_jobs=10, num_machines=3, horizon=15`),
compared against EDF and LST — Stage A's two strongest heuristics — on the same five
held-out seeds.

**Result:**

| Seed | CP-SAT reward | CP-SAT tardiness | EDF reward | EDF tardiness | LST reward | LST tardiness |
|---|---|---|---|---|---|---|
| 500000 | 75.13 | 0.00 | 20.50 | 0.00 | 20.50 | 0.00 |
| 500001 | 74.03 | 0.00 | 20.57 | 0.00 | 20.74 | 0.00 |
| 500002 | 74.77 | 0.00 | 77.00 | 0.00 | 77.22 | 0.00 |
| 500003 | 74.00 | 0.00 | 77.00 | 0.00 | 77.00 | 0.00 |
| 500004 | 73.63 | 0.00 | 21.86 | 0.00 | 21.86 | 0.00 |

All three methods reach **zero tardiness on every instance**, yet CP-SAT beats both
heuristics on reward by a wide margin on 3 of 5 seeds. Checked directly on seed
500000: EDF schedules only **9 of 10 jobs**, not 10 — it gets greedily stuck even
though a globally-optimal ordering exists that completes all 10 on time (which CP-SAT
finds, because its "every job must be scheduled" constraint forces it to).

**Why this matters as evidence of learning:** This demonstrates the value of an
unambiguous ground truth. Every comparison up to this point had implicitly trusted
that "zero tardiness" meant "good scheduling" — the exact solver showed that metric
alone hides a completion-rate failure even the strongest heuristics were exhibiting
undetected. Recognising this as "the same lesson twice in one night" (alongside
artefact 5's RCPO finding, discovered independently the same session) rather than two
unrelated incidents shows the kind of pattern-recognition across separate
investigations that turns isolated bug-fixes into a generalisable engineering
principle.

---

## 8. A gradient-free search method independently rediscovers the project's central reward-hacking finding (2026-08-28)

**Caption:** A Particle Swarm Optimisation (PSO) baseline, optimising directly against
the same episode reward every RL policy is trained on, was run on the fixed instance
plus 15 held-out instances.

**Result:**

| | Reward | Tardiness | Late jobs | Wall-clock / instance |
|---|---|---|---|---|
| PSO (fixed instance) | 337.37 | 963.00 | 39 | 208.5s |
| PSO (15 held-out, mean) | 329.68 | 1012.40 | 40.93 | 182.9s |
| EDF (same 15 held-out, mean) | 286.00 | 50.33 | 14.20 | ~0.4s |

PSO beats EDF on reward by ~15%, but with **~20x the tardiness** and 3x the late-jobs
rate — the identical reward/tardiness misalignment pattern found earlier via Optuna
hyperparameter search, now reproduced by a completely different, gradient-free
optimisation method with no neural network and no policy gradient involved at all.

**Why this matters as evidence of learning:** A single method finding a strange result
could be a bug in that method. A second, structurally unrelated method landing on the
*same* exploit is much stronger evidence the misalignment is a property of the reward
function itself, not an artefact of one training algorithm. Deliberately building this
cross-validation, rather than treating the RL result as case-closed, shows an
understanding of how to strengthen a causal claim through independent replication — a
core scientific-method skill applied here to a machine-learning pipeline.

---

## 9. Multi-objective (Pareto) tuning: a null result that only appeared once scale was corrected (2026-08-28)

**Caption:** Genuine multi-objective (NSGA-II) Optuna tuning was implemented to act on
a formal critique of single-weight reward scalarisation (Roijers et al., 2013), and
run at three different tuning scales to see how the reward/tardiness trade-off's shape
changed.

**Result:**

| Tuning scale | Non-dominated Pareto points | Reward range | Tardiness (normalised) range |
|---|---|---|---|
| 20 jobs (project's usual tuning scale) | 1 (front collapsed to a single point) | — | 17 of 40 trials hit exactly zero |
| 60 jobs | 10 | −117.05 to 195.39 | 0 to 7.15 |
| 100 jobs (real deployment scale) | 5 | −89.94 to 257.05 | 0 to 19.37 |

**Closing negative result:** the front's "knee" point (`lambda_2=10.98`) trained to
full 500k-step convergence collapsed into reward hacking anyway — reward 303.08 (the
highest of any RL checkpoint in the project) against tardiness 733.56 (20-25x every
other checkpoint), despite genuinely high throughput (99.38/100 jobs scheduled).

**Why this matters as evidence of learning:** The first result (a single-point front)
could easily have been reported as "no trade-off exists here" and closed out. Instead
it was treated as suspicious given everything else the project had found about reward
hacking, and re-tested at increasing scale until the true trade-off appeared —
corroborating a separate paper's warning about tuning/deployment-scale mismatch
(Eimer, Lindauer & Raileanu, 2023) at the level of the Pareto front's own shape. This
is statistical/experimental care: not trusting a convenient null result without first
checking whether the experimental scale itself could be hiding the effect.

---

## 10. A full-suite comparison table, not just an RL-vs-one-baseline number (2026-09-04)

**Image:** `Results/v1_legacy_reward/reduced_budget_2026-09-04_v2/figures/comparison_tardiness.png`

**Caption:** A single evaluation run comparing six methods side by side on 8 held-out
instances (45 jobs / 6 machines / horizon 50), each reported as mean ± standard
deviation, not a single seed's number.

**Result:**

| Method | Reward | Tardiness | Late jobs | Jobs scheduled |
|---|---|---|---|---|
| EDF | 174.65 ± 18.60 | 14.88 ± 17.37 | 7.38 ± 7.00 | 44.88/45 ± 0.33 |
| LST | 181.82 ± 0.30 | 9.12 ± 15.05 | 4.62 ± 6.59 | 45.00/45 ± 0.00 |
| Pointer + shaping | 119.36 ± 24.96 | 10.25 ± 16.14 | 4.75 ± 6.36 | 40.38/45 ± 3.12 |
| Pointer + RCPO | 93.64 ± 5.88 | 156.88 ± 42.99 | 12.75 ± 2.22 | 36.75/45 ± 1.85 |
| Pointer + Pareto-knee | 175.95 ± 0.77 | 263.00 ± 27.43 | 19.25 ± 3.31 | 45.00/45 ± 0.00 |
| PSO | 179.21 ± 0.52 | 146.62 ± 18.51 | 17.75 ± 1.56 | 45.00/45 ± 0.00 |

**Why this matters as evidence of learning:** Reporting a mean with an explicit
standard deviation across multiple held-out instances, for every method in the same
table, is basic but frequently-skipped experimental hygiene — it lets a reader judge
whether an apparent difference between two methods is a real effect or noise. The
variance column alone tells an important story: PSO and EDF are extremely stable
across instances while Pointer+RCPO is not, a distinction a bare mean would have
hidden entirely.

---

## 11. Choosing a statistical convention properly, not by assertion (2026-09-13)

**Image:** `EthanTravelDocs/portfolio-artefacts/figures/example_machine_utilisation_ppo.png`

**Caption:** Asked to redesign a machine-utilisation figure with a "top 95%, excluding
outliers" envelope, two candidate statistical definitions were compared and justified
before implementation: a plain 95th-percentile estimator (Hyndman & Fan, 1996) vs. a
Tukey IQR fence (Tukey, 1977).

**Result:** The percentile method was chosen — but with an explicit caveat attached in
the same documentation that introduces it: with only 10 machines, a 95th-percentile
line sits within *one order statistic* of the true maximum, so it should not be read
as an aggressive outlier trim at this sample size.

**Why this matters as evidence of learning:** This is a small design decision that
could easily have been implemented by guessing at a `numpy` percentile call and moving
on. Instead it was treated with the same rigor demanded of every design decision in
the project: name the candidates, cite the definitions, and — critically — state the
caveat that undermines the chosen method's intuitive framing at this specific sample
size, rather than letting a reader misread the figure. Flagging your own method's
limitation in the documentation that introduces it is a mark of genuine statistical
maturity.

---

## 12. Seven failed fixes, then the real root cause: action-space size (2026-09-17)

**Image:** `EthanTravelDocs/portfolio-artefacts/figures/offline_tardiness_comparison.png`
**Code:** `EthanTravelDocs/portfolio-artefacts/code-samples/rule_selection_gym_wrapper.py`
(source: `Code/env/rule_selection_gym_wrapper.py`)

**Caption:** Seven separate mechanisms (reward reweighting, four Lagrangian constraint
ceilings, a tardiness-directed hyperparameter search, and a full dense-reward
redesign) all failed to move PPO's offline tardiness out of the ~1290–1330 band, next
to a proven-optimal floor of 8.00. Comparing the project's action-space size (1000+
discrete choices) against DeepRM/Decima's (~11) surfaced the actual bottleneck.
"Option 1" — an RL policy that selects *which classical rule to apply* each tick
(`Discrete(8)`) instead of a raw `(job, machine)` pair — was the fix.

**Result — offline, fixed instance (100 jobs, 10 machines, horizon 100):**

| Method | Tardiness | Late jobs | Jobs scheduled |
|---|---|---|---|
| CP-SAT (proven optimal) | 8.00 | — | — |
| LST | 8.00 | 7 | 99/100 |
| EDF | 16.00 | 10 | 98/100 |
| **Option 1 — 600k steps** | **19.00** | 11 | 100/100 |
| Option 1 — 300k steps | 35.00 | 10 | 100/100 |
| ATC | 106.00 | 14 | 96/100 |
| Option 3 (learned priority + ATC feature) | 152.00 | 8 | 100/100 |
| Option 2 (learned priority, raw features) | 525.00 | 16 | 100/100 |
| Historic PPO (7 prior fix attempts) | ~1290–1330 | — | — |
| SPT (worst-case reference) | 1321.00 | 43 | 92/100 |

A **~68x tardiness improvement** over the historic PPO band — the single biggest
result of the project.

**Result — online, heavy-tailed arrivals, rho ≈ 0.75 (the genuinely hard regime):**

| Method | Tardiness | Late jobs | Scheduled |
|---|---|---|---|
| CP-SAT (hindsight bound) | 105.00 | — | — |
| ATC (best live policy) | 120.00 | 14 | 799/864 |
| **Option 1** | 163.00 | 15 | 799/864 |
| SPT | 163.00 | 15 | 799/864 |
| WSPT+BestFit | 192.00 | 19 | 796/864 |
| EDF | 193.00 | 25 | 797/864 |
| Tetris | 248.00 | 18 | 799/864 |
| LST | 275.00 | 28 | 789/864 |
| LPT+WorstFit (worst) | 634.00 | 34 | 759/864 |

Option 1's numbers here are *identical* to SPT's — it loses to ATC and shows every
sign of having converged onto imitating one classical rule rather than discovering a
new strategy.

**Why this matters as evidence of learning:** This is the project's strongest
demonstration of persistence paired with honest failure-reporting: six prior attempts
are described as failures in the same document that reports the eventual win, not
edited out in hindsight. It also shows composition-over-inheritance software design
(the wrapper reuses the existing, already-validated baseline `choose()` functions
rather than reimplementing placement logic) and intellectual honesty about the
result's own limits — the online loss to ATC is reported in the same breath as the
offline win, with an explicitly named ceiling ("its ceiling is bounded by whatever the
best achievable combination of its 7 rules is") rather than the result being oversold
as a general breakthrough.

---

## 13. A self-directed engineering efficiency audit (2026-09-20)

**Caption:** A deliberate code-reading pass through the training/environment hot path,
independent of any specific training result.

**Result:** Two concrete performance bugs, each spot-checked against source before
being written up:
- `SchedulingEnv.get_state()` is called on every single RL step across every training
  run the project has ever done, deep-copying four arrays — and its return value is
  unconditionally discarded two lines later. A free, zero-behaviour-change fix.
- Optuna's configured `MedianPruner` cannot actually prune — both PPO and A2C
  objectives report their pruning metric only after a trial's full compute is already
  spent (or, for A2C, never report one at all).

**Why this matters as evidence of learning:** This artefact is different in kind from
the others — it is not chasing a specific result, it is a proactive
engineering-quality review of the project's own infrastructure. Every claim is stated
as "verified directly" against the source rather than asserted from memory or
suspicion, and each finding comes with a concrete, actionable fix. Noticing that a
fixed compute budget is being spent inefficiently, not just that a model isn't
performing well, is a professional-engineering habit distinct from, and complementary
to, the research-results work everywhere else in this log.

---

## 14. Correcting your own diagnostic claim, precisely (2026-09-21 → 2026-09-22)

**Image:** `EthanTravelDocs/portfolio-artefacts/figures/online_more_training_hurts.png`

**Caption:** After confirming three separate times that more training regresses the
online policy, a diagnostic tool was built to inspect the policy's actual per-state
probability distribution, not just aggregate training curves.

**Result:**

| Metric | Value |
|---|---|
| Decision margin (top1 − top2 probability), across 5,591 sampled decisions | mean 1.0000, std 0.0000, min 1.0000, max 1.0000 — exactly 1.0 on every single decision |
| Value estimate V(s) | mean −6.81, std 1.44, range [−13.12, −1.99] — a real spread, not collapsed |
| corr(chosen-rule probability, congestion) | +0.073 (essentially none) |
| corr(value estimate, congestion) | +0.306 (weak, unexplained) |

**SB3's own per-state `entropy_loss` across the run:**

| Step | 4,096 | 90,000 | 175,000 | 233,472 | 262,144 | 348,160 | 462,848 | 548,864 | 605,000 | 720,000 | 835,000 | 900,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy_loss | −2.040 | −1.040 | −0.196 | −0.118 | −0.160 | −0.100 | −0.066 | −0.018 | −0.0013 | −0.0003 | −0.00008 | −0.00006 |

Entropy declines smoothly and continuously for the whole run, reaching near-total
saturation only in roughly the final third — and on closer reading, this precise
trajectory showed an *earlier* diagnostic entry had itself conflated two different
entropy metrics (per-state saturation vs. cross-state action diversity). A follow-up
entry verified the correction's own precision with fresh data before letting it stand.

**Why this matters as evidence of learning:** This is three linked entries, each
correcting or sharpening the one before it, all left in the record rather than merged
into a single "clean" final answer — a direct enactment of the project's own
append-only rule, applied here to the writer's *own* recent claim, not just an old
one. Being willing to publicly narrow or correct a diagnostic conclusion you made two
days earlier, and to verify the correction itself with fresh data rather than just
asserting it, is a mature research habit that is easy to state as a value and
genuinely hard to practice consistently.

---

## 15. The recurring lesson: reading your own project's history for a pattern (2026-08-28, reflective)

**Caption:** Stepping back after twelve project phases, the project's own narrative
log identifies a pattern across its history.

**Result — the pattern, quoted directly:** Three separate training failures (idle
collapse, curriculum-stage collapse, pointer-underperforms-flat) were each *initially*
read as evidence about the RL method or architecture, and each time turned out to be
fully or partly caused by an environment or training-loop bug instead — a capacity
leak, a crashing default-argument call, the numpy-bool identity bug (artefact 2), and
a mask/step time-index mismatch. The standing rule that came out of this: "before
drawing a conclusion from a training result, especially a negative one, write a
regression test that isolates the specific mechanism first."

**Why this matters as evidence of learning:** This is the project's own explicit
lessons-learned reflection, and it draws a genuinely actionable conclusion rather than
a vague "be more careful." Recognising a *recurring* failure mode across a project's
history — not just fixing each bug as it appears, but naming the meta-pattern they
share and turning it into a standing process rule — is exactly the kind of
higher-order reflection a professional-development log is asking for: it shows the
project didn't just accumulate results, it accumulated a better way of working.

---

*All figures and tables above are real, unedited project output. Every number is
copied directly from the project's training log, dated research write-ups, or results
files — nothing here is fabricated or rounded for presentation.*
