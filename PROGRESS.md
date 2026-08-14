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

## Phase 0 — PPO collapses onto idling (2026-07-24)

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

## Phase 1 — Two bugs were the real cause, not the algorithm (2026-08-09)

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

## Phase 2 — Curriculum stages 3–4 still collapse; pointer network proposed (2026-08-09)

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

## Phase 3 — Three more bugs, tardiness rescale, and the first non-negative result (2026-08-09 → 2026-08-10)

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

## Where things stand now (2026-08-13)

A project-level `CLAUDE.md` was added to make the standard this project already holds
itself to explicit for future sessions: every design decision needs a citation or a formal
derivation, not just an empirical "this worked better" — matching the standard already set
by the tardiness-boundedness argument and the Ng/Harada/Russell (1999) policy-invariance
justification for potential-based shaping.

Three directions are on the table to close the remaining tardiness/late-jobs gap and to
actually test the pointer network's design claim (see the discussion in
`Future/research/training-log.md`'s latest entries and
`2026-08-09-pointer-network-action-head.md` §9–10 for the full reasoning):

1. **Tardiness-focused reward retuning** — re-search `lambda_2` and the placement/
   completion bonus weights specifically against tardiness/late-jobs, not total reward.
   Fast, contained, stays on the fixed instance.
2. **Randomized-instance generalization experiment** — train on per-episode randomized
   job sets instead of fixed `seed=0`. This is the still-deferred real test of whether the
   pointer network generalizes from features (its actual design claim) rather than
   memorizing, per Phase 2. Higher effort, may regress before improving.
3. **Potential-based shaping ablation** — already implemented and sign-checked
   (`use_potential_shaping` flag), just never run through the full curriculum. Cheap,
   orthogonal, can combine with either of the above.

Not yet decided which to run first — to be logged as a new `training-log.md` entry (and a
new dated doc if it turns into its own investigation) once it is.

## Recurring lesson

Three separate rounds of this project's history (idle collapse, stage-3/4 collapse,
pointer-underperforms-flat) were each initially read as evidence about the RL
method/architecture, and each time turned out to be — fully or partly — confounded by an
environment or training-loop bug (capacity leak, `dict.get()` crash, numpy-bool identity
bug, mask/step mismatch). The working habit that has come out of this: before drawing a
conclusion from a training result, especially a negative one, write a regression test that
isolates the specific mechanism first (see `tests/test_bugfixes.py` for the pattern).
