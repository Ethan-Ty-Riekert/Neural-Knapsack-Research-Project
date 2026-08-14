# Training Log

Chronological record of training runs, what changed since the previous entry, the
resulting stats, and what was concluded from them. This is a running log of results
across the project, separate from the dated deep-dive write-ups in this folder (which
investigate one specific problem in depth and are linked from the relevant entry below).

Newest entries at the top. Each entry should be appended, not edited retroactively --
if a conclusion turns out to be wrong, say so in a later entry rather than rewriting
history.

## Template for new entries

```
## YYYY-MM-DD -- <short description>

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

## 2026-08-10 -- Fresh Optuna + full-curriculum reruns post-bugfix: RL closes the gap to EDF

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

## 2026-08-09 -- Three more bugs found and fixed (machine-activation ordering, numpy-bool dead code, mask/step time mismatch), tardiness term normalised

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

## 2026-08-09 -- Pointer network vs. flat baseline, both with ent_coef/reward-norm/masked-entropy fixes

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

## 2026-08-09 -- A2C full curriculum, capacity-leak fixed: stages 1-2 solved, stages 3-4 regress hard

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

## 2026-08-09 -- Two pre-existing bugs found and fixed before the above run

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

## 2026-07-24 -- Bigger network + fewer PPO epochs (result pending)

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

## 2026-07-24 -- Reward reweighting + higher ent_coef (still collapsed)

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

## 2026-07-24 -- Initial run (collapsed onto idle)

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
