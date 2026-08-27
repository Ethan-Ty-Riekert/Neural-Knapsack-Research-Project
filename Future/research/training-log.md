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
