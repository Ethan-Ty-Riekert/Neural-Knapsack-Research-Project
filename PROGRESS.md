# Progress Notes

A narrative account of how this project got to its current state: what was tried, what
broke, how it was diagnosed, and what changed as a result. This is the "story" version —
for the authoritative, detailed record (exact stats, hyperparameters, regression tests),
see:

- **`Future/research/training-log.md`** — chronological, terse, one entry per run/change.
  This is the source of truth for numbers; this document does not repeat them in full.
- **`Future/research/YYYY-MM-DD-<topic>.md`** — dated deep-dives for investigations big
  enough to need their own write-up, each with a numbered References section.

Update this file when a phase of work concludes (not after every run) — a few sentences
placing the new training-log entries in context, plus a link to the entries/docs that
back it up. Do not duplicate stats tables here; link to them.

## Phase 0 — PPO collapses onto idling (2026-07-24, S2W1)

First real training attempts (`MaskablePPO` on the flattened `(job, machine)` action
space) collapsed onto always picking the idle action — `ep_rew_mean` locked onto an exact
multiple of `-idle_penalty`, entropy and KL flatlined at zero. Reward reweighting and a
higher entropy coefficient changed *which* constant it collapsed to, not whether it
collapsed, which pointed at something structural rather than a reward-magnitude problem.

That led to a literature review (`Future/research/2026-07-24-idle-action-policy-collapse.md`):
PPO's clipped objective is a documented failure mode for exactly this symptom in large
discrete action spaces (Hsu, Mendler-Dünner & Hardt, 2020), and FJSP-scheduling RL papers
generally avoid flattening `(job, machine)` into one `Discrete` index for this reason. Two
low-risk fixes were tried first (bigger network, fewer PPO epochs); factorizing the action
space was identified as the structurally stronger fix but deliberately deferred pending
whether the cheaper fixes were enough.

## Phase 1 — Two bugs were the real cause, not the algorithm (2026-08-09, S2W3)

Before evaluating whether Phase 0's fixes worked, two environment/training bugs were
found that had been confounding every result up to this point:

- `SchedulingEnv.reset()` was restoring capacity from an array that `step()` itself
  mutated, so capacity leaked downward permanently across episodes within a stage.
- A2C's rollout loop called `dict.get()` with an eagerly-evaluated default that crashed
  under the installed `gymnasium` version — A2C had never actually been trainable.

Fixing just the capacity leak took PPO/A2C from 100% idle collapse to real positive
reward on a stage-1-sized smoke test. This reframed the Phase 0 diagnosis: the clipped-
objective/large-action-space story isn't wrong, but every result before this fix should
be read as confounded by these bugs, not as clean evidence for it.

## Phase 2 — Curriculum stages 3–4 still collapse; pointer network proposed (2026-08-09, S2W3)

With both bugs fixed, a full 4-stage curriculum run showed stages 1–2 training cleanly,
but stages 3–4 collapsing immediately on transition and plateauing — flat, not slowly
recovering, over ~1,500+ episodes each. Leading hypothesis: the flat policy head
(`Linear(256, 1001)`) gives every `(job_slot, machine)` pair its own independent weight
row, so job slots that only unmask at a later curriculum stage start from scratch with no
transferred knowledge, compounded by a reward scale that jumps an order of magnitude
between stages with no normalization anywhere in the pipeline.

This motivated `Code/policies/pointer_policy.py` — a `PointerActorCritic` using shared
job/machine encoders and an attention-style compatibility score per `(job, machine)` pair
(Vinyals et al. 2015; Kool, van Hoof & Welling 2019), so a slot's score is a function of
its *features*, not its index. Full design rationale and literature review:
`Future/research/2026-08-09-pointer-network-action-head.md`.

**First comparison was a negative/confounding result, and that turned out to matter.**
Once `ent_coef` was raised off zero, reward normalization was added, and a masked-entropy
bug in the loss was fixed, the *flat* architecture alone recovered most of the stage-3/4
regression — meaning those hygiene fixes, not parameter sharing, explained most of the
earlier collapse. Worse, the pointer network did *worse* than the now-fixed flat baseline
on this same curriculum. Diagnosis: every stage reuses the same fixed `seed=0` job set for
thousands of episodes, so a flat per-index head can simply memorize the optimal action per
slot — a strictly easier target than the pointer network's indirect, shared-weight
parameterization. The curriculum wasn't exercising the property the pointer network is
actually designed for (generalizing from job features, not memorizing job identity), so
this result was read as "wrong experiment," not "wrong idea" — the real test (randomized
per-episode job sets) was explicitly deferred rather than treated as answered.

## Phase 3 — Three more bugs, tardiness rescale, and the first non-negative result (2026-08-09 → 2026-08-10, S2W3 → S2W4)

While writing a regression test for the ordering of a `machine_active` flag update,
found a numpy-bool identity bug (`if ym is True:` where `ym` is a numpy `bool_`, and
`np.bool_(True) is True` is `False`) — meaning **the activation-cost penalty had never
fired once in the project's history**, independent of every `lambda_1` value ever "tuned"
by Optuna. A third bug (mask/step time-index mismatch at `t == horizon`) could leave a
policy trusting the mask stuck repeating an invalid action with no way for the episode to
end. All three, plus a reward-formula change (tardiness normalized to `T_j/H`, provably
`< 1`, instead of raw `T_j`, which scaled ~5x across curriculum stages) are detailed with
derivations in `Future/research/2026-08-09-fixed-instance-bugfix-and-reward-rescale.md`.

Rerunning Optuna (separately per architecture) and the full curriculum on top of all of
this produced the first result in the project's history that isn't strictly negative:

| | total_reward | total_tardiness | late_jobs (/100) |
|---|---|---|---|
| EDF (heuristic) | 289.38 | 16.00 | 10.00 |
| A2C flat (optimized) | 231.84 | 866.00 | 27.00 |
| A2C pointer (optimized) | 270.23 | 1227.00 | 32.00 |

Pointer reward is within ~7% of EDF (flat ~20%), and pointer now beats flat — a reversal
of Phase 2's comparison, plausibly because each architecture finally got its own properly
tuned hyperparameters instead of reusing the flat head's defaults. But tardiness and
late-jobs are still far worse than EDF for both: reward-competitive is not the same claim
as tardiness-competitive, and the O(1) tardiness term is currently small relative to the
flat placement/completion bonuses. Full stats: `Future/research/training-log.md`,
2026-08-10 entry.

## Phase 4 — Standards codified, three follow-up directions scoped (2026-08-13, S2W4)

A project-level `CLAUDE.md` was added to make the standard this project already holds
itself to explicit for future sessions: every design decision needs a citation or a formal
derivation, not just an empirical "this worked better" — matching the standard already set
by the tardiness-boundedness argument and the Ng/Harada/Russell (1999) policy-invariance
justification for potential-based shaping.

Three directions were scoped to close the remaining tardiness/late-jobs gap and to
actually test the pointer network's design claim (see
`2026-08-09-pointer-network-action-head.md` §9–10 for the full reasoning): (1) tardiness-
focused reward retuning, (2) randomized-instance generalization, (3) potential-based
shaping ablation. Order chosen: a cheap training-time check first, then generalization,
then tardiness retuning, then shaping.

## Phase 5 — More training time doesn't close the tardiness gap (2026-08-17, S2W5)

Cheapest experiment first: doubled stage 4's timestep budget (200k → 400k) for both
architectures, on the reasoning that pointer's stage-4 reward was still climbing (not
plateaued) at the old budget. Result was mixed, not a clean win — full stats and the
"why does EDF beat RL by so much on tardiness" explanation are in
`Future/research/training-log.md`'s 2026-08-17 entry. Headline: flat's reward improved
(+27.3) but its tardiness nearly *doubled* (866 → 1546) and late-jobs rose too — more
optimization steps gave the reward/tardiness misalignment more room to express itself,
not less. Pointer's reward actually *fell* (270.23 → 249.47), contradicting the
motivating "still improving" hypothesis outright. Conclusion: training time is not the
lever that closes the tardiness gap — if anything it argues for prioritizing tardiness-
focused retuning (direction 1) over further training-time increases. Proceeding to
randomized-instance generalization (direction 2) next per the already-agreed order.

Also this session: `rl_training/models/` checkpoints and `eval_rl_agent.py`'s summary
stats were previously silently overwritten by the next run with no history kept. Added
`Code/utils/results_log.py` — every `train_optimized.py` run now archives its checkpoints
to a dated/tagged folder under `rl_training/models/archive/`, and every eval run appends
its summary row to `rl_training/results/eval_results.csv` — so results accumulate across
runs instead of only living in this file's prose or a human's memory of stdout.

## Phase 6 — Tardiness-focused reward retuning: negative result, and why (2026-08-17, S2W5)

Direct response to Phase 5's conclusion: added `optuna_tune.py --optimize-for tardiness`,
which changes Optuna's trial-*selection* metric to `mean_reward - 50 × mean_tardiness_normalised`
instead of raw reward (training itself is unchanged — same env reward, same anti-idle-collapse
pressure). Both architectures' best trials hit *zero* tardiness on the small 20-job tuning
env. Full stats in `Future/research/training-log.md`'s second 2026-08-17 entry. It did not
transfer: full-curriculum training with these hyperparameters made both reward and tardiness
go up simultaneously — pointer hit the best reward ever recorded in this project (328.21)
*and* the worst tardiness ever recorded (1699); flat's tardiness also rose (866 → 998)
despite its reward improving. Combined with Phase 5, the two highest-reward RL results in
this project's history are now also its two worst-tardiness results — reward and tardiness
appear to be actively trading off against each other under harder optimization, not just
weakly correlated.

Leading hypothesis: the 20-job tuning environment is easy enough that zero tardiness there
doesn't require genuinely tardiness-robust hyperparameters — consistent with both searches
picking a *lower* `lambda_2` (the tardiness weight itself) than the reward-tuned baseline,
and instead reaching for other levers (a 4x larger pointer network, very different
`invalid_penalty`) that apparently don't scale to the full 100-job curriculum. Flagged
follow-up: evaluate Optuna trials on a harder/larger instance closer to stage-4 scale,
rather than the current small tuning env — not done this session, scoped for later.

## Phase 7 — Literature review: Phases 5–6's pattern is a named, studied problem (2026-08-17, S2W5)

User asked directly what the research literature says about diagnosing/improving an RL
agent. Full review: `Future/research/2026-08-17-literature-review-improving-rl-agent.md`.
Short version: Phases 5 and 6's "push optimization harder → reward up, tardiness up too"
pattern matches reward hacking as formally defined by Skalse et al. (2022) and empirically
characterized by Pan, Bhatia & Steinhardt (ICLR 2022, capability-driven phase transitions
in proxy/true-reward divergence) — not a bug specific to this codebase. Separately, Eimer,
Lindauer & Raileanu (2023) explain *why* Phase 6's tardiness-tuned hyperparameters didn't
transfer (tuning/testing environment mismatch) and recommend exactly the fix already
flagged at the end of Phase 6. Five prioritized next actions came out of this, the top two
being: finally run the already-implemented potential-based shaping (Experiment 4, never
run full-curriculum), and redesign the tardiness Optuna search to evaluate on a
harder/larger instance. None implemented yet — this phase is research, not a code change.

## Phase 8 — Potential-based shaping: pointer beats EDF on tardiness (2026-08-19, S2W5)

Direct follow-through on Phase 7's top recommendation. Added `train_optimized.py
--use-potential-shaping`, ran the full curriculum with shaping on top of the *same*
reward-tuned hyperparameters as the S2W4 baseline (shaping isolated as the only changed
variable). Full stats in `Future/research/training-log.md`'s 2026-08-19 entry. Headline:
**pointer + shaping is the best result in this project's history** — total_tardiness 9.00
and late_jobs 6, both *better than EDF* (16.00 / 10), at a reward cost of only 16.3 points
(270.23 → 253.94). This is the first RL result that actually beats the heuristic on the
metric that mattered most throughout Phases 5–6. Flat's result was mixed (reward improved,
tardiness got worse) — shaping's benefit is architecture-dependent, plausibly because
pointer's shared encoders let the per-step urgency signal generalise across every job slot
at once, where flat's per-index weights don't.

Per the user's explicit ordering, proceeding next to Experiment 2 (randomized-instance
generalization) — both to finally test the pointer network's actual design claim (per
Phase 2) and to check whether this tardiness win is genuine scheduling skill or another
fixed-instance artifact. RCPO-style constrained optimization (on a dedicated git branch)
is queued after that.

## Phase 9 — Generalization confirmed; the best config is still fixed-instance (2026-08-20, S2W5)

Built the actual Experiment 2 infrastructure: per-episode job randomization
(`GymSchedulingEnv`'s `job_resampler`, `--randomize-instances`) and a held-out evaluation
mode (`--randomized-eval`, 50 instances at seeds ≥500,000, disjoint from training by
construction). Two results, full stats in `Future/research/training-log.md`'s 2026-08-20
entry:

1. **The Phase 8 result is real, not memorization.** Evaluating the existing fixed-instance
   pointer+shaping checkpoint (tardiness 9.00 on the instance it trained on) on 50 unseen
   held-out instances gives tardiness 28.66 — some degradation, but it still **beats EDF's
   own held-out tardiness** (28.66 vs. 37.30) and late-jobs (9.56 vs. 12.22). This directly
   answers the question open since Phase 2: the pointer network's design claim (generalizing
   from job features via potential-based shaping's urgency signal) holds up.
2. **Deliberately training for generalization did much worse.** A fresh run with
   `--randomize-instances` (jobs resampled every episode, hyperparameters re-tuned on that
   distribution) produced tardiness ~732 on both the fixed instance and held-out set — ~25x
   worse than the fixed-instance-trained model's held-out result. Both architectures
   generalize *consistently* under this training regime (fixed vs. held-out performance
   track closely) — they're just consistently worse, suggesting the per-episode-changing
   distribution makes the learning problem itself much harder within the same timestep
   budget, not a generalization failure per se.

**Current best configuration project-wide: pointer + potential-based shaping + the original
fixed-instance training + reward-tuned hyperparameters** (Phase 8's checkpoint) — beats EDF
on tardiness/late-jobs both on its training instance and on 50 unseen ones.
`--randomize-instances`/`--randomized-eval` stay in the codebase as reusable capabilities
(the held-out eval is how this phase's key finding was established at all), but
`--randomize-instances` is not recommended for future training runs based on this evidence.

Proceeding next to Experiment 5 (RCPO-style constrained optimization) on a dedicated git
branch, per the user's explicit instruction given how structurally different it is.

## Phase 10 — RCPO: best-ever tardiness for pointer, worst-ever generalization gap for flat (2026-08-21, S2W5)

Set up the `rcpo-constrained-optimization` git branch (per the user's explicit instruction)
and replaced the fixed `lambda_2` tardiness weight with a Lagrange multiplier that adapts
during training toward a constraint `E[tardiness cost] <= alpha`, following Tessler,
Mankowitz, Mannor (ICLR 2019). Full formalization, grounding, and results in
`Future/research/2026-08-21-rcpo-constrained-tardiness.md`; run-by-run stats in
`Future/research/training-log.md`'s two 2026-08-21 entries.

Motivation: every fixed-`lambda_2` search this project ran (Optuna reward-tuned, Optuna
tardiness-tuned) hit the same wall — a single weight picked before training can't reliably
target a specific tardiness level. RCPO lets the weight find its own level during training
instead, using the same tardiness term already in the reward, now tracked separately as a
constraint cost the multiplier reacts to.

Two very different results depending on architecture, both instructive:

- **Pointer:** best tardiness/late-jobs result in the project's history, on both metrics at
  once, with much lower variance than any prior result (held-out tardiness 19.84 vs. Phase
  8's 28.66 and EDF's 37.30; late-jobs 3.20 vs. 9.56 vs. 12.22). But reward roughly halved
  (254.75 → 135.67 held-out), because the constraint target (`alpha=0`, "drive tardiness to
  zero") isn't actually achievable in this environment, so the multiplier climbed to its
  ceiling (49.37 of 50.0) instead of settling at a stable value — a textbook case of an
  overly strict constraint target, not a flaw in the mechanism itself.
- **Flat:** looked like the best result ever on the fixed training instance — literally
  zero tardiness, zero late jobs. Evaluated on 50 held-out instances, the same checkpoint
  scored 570.62 tardiness / 24.02 late-jobs — worse than EDF, worse than every other flat
  result this project has produced. This is the sharpest fixed-vs-held-out generalization
  gap seen yet, and it lands on the architecture already flagged (Phase 9, and the pointer-
  vs-flat comparisons since Phase 2) as prone to memorizing rather than generalizing —
  `MaskableActorCritic`'s one-weight-row-per-action-index parameterization gives it a direct
  route to memorize "this job slot goes to this machine," which RCPO's harder-driving
  adaptive penalty seems to have let it fully exploit.

Taken together with Phase 9, this is now two independent pieces of evidence that
generalization quality in this project is an *architecture* property (pointer's shared
encoders vs. flat's per-index weights), not something any reward-shaping or constraint
technique tried so far can fix. Flat is no longer being carried forward as a candidate for
further reward-side experiments — only as the fixed-instance A/B baseline it already is.

Flagged, not-yet-run follow-up: rerun RCPO on the pointer architecture with a less strict
`alpha` (grounded at an achievable target, e.g. Phase 8's own ~28.66 held-out tardiness)
instead of 0, to test whether the multiplier can reach an interior equilibrium and recover
reward while keeping most of the tardiness gain.

## Recurring lesson

Three separate rounds of this project's history (idle collapse, stage-3/4 collapse,
pointer-underperforms-flat) were each initially read as evidence about the RL
method/architecture, and each time turned out to be — fully or partly — confounded by an
environment or training-loop bug (capacity leak, `dict.get()` crash, numpy-bool identity
bug, mask/step mismatch). The working habit that has come out of this: before drawing a
conclusion from a training result, especially a negative one, write a regression test that
isolates the specific mechanism first (see `tests/test_bugfixes.py` for the pattern).
