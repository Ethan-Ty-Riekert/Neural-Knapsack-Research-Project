# Design & Comparison: Action-Space and Reward Interventions for Idle-Collapse

**Date:** 2026-08-09
**Environment:** `Code/scheduling_env.py`, `Code/gym_scheduling_wrapper.py`,
`Code/Policies/{a2c_policy.py, pointer_policy.py, ppo_policy.py}`
**Status:** Solutions 1 and 3 implemented and tested to full curriculum length.
Solution 2 implemented and run full-curriculum against the flat baseline (Section 8)
-- result is mixed: the entropy/reward-normalisation/masking fixes bundled in
alongside it did most of the observed improvement, and the pointer architecture
itself underperformed the fixed flat baseline on this curriculum, for reasons
explained in Section 8 (this curriculum doesn't exercise generalisation, which is
what the architecture is actually for). PPO port explicitly deferred (Section 9).

---

## 1. Motivation

`Future/research/2026-07-24-idle-action-policy-collapse.md` diagnosed a policy-collapse
failure mode in `MaskablePPO`/A2C training on this environment: the policy converges
onto the always-idle action within the first few thousand timesteps, citing Hsu et
al. (2020)'s finding that PPO's clipped objective can trap a policy on a
near-deterministic, suboptimal action in large discrete action spaces (this
environment's flattened `Discrete(max_jobs * num_machines + 1)` is ~1,001-wide). That
document identified two candidate fixes: increase network capacity / reduce PPO
epochs (adopted at the time), and factorize the flattened `(job, machine)` action into
`MultiDiscrete([job, machine])` (deferred).

A follow-up session (2026-08-07) re-tested reward-magnitude tuning (idle penalty
0.5-2.5, entropy coefficient 0.001-0.3) and reconfirmed collapse regardless of
magnitude, but drew this from runs of only 2,000-30,000 timesteps against a
375,000-timestep curriculum design, and concluded "PPO fundamentally fails" -- an
overreach given the training budget used. This document reports on three follow-up
interventions run properly (full curriculum length), plus two significant
pre-existing bugs found while investigating them, and a literature-grounded pointer
network architecture that generalizes the deferred `MultiDiscrete` idea.

---

## 2. Two pre-existing bugs found and fixed before any of the below

Full detail in `Future/research/training-log.md`'s 2026-08-09 entries; summarised
here since they materially affect how every prior "collapse" result in this project
should be read.

1. **`SchedulingEnv.reset()` capacity leak.** `reset()` restored every timestep's
   capacity from `self.capacity[:, :, 0]`, which is itself mutated by `step()`
   whenever a job starts at time 0 (true for most episodes) -- so it was never
   actually the original capacity after episode one. Since a single `SchedulingEnv`
   instance is reused for hundreds/thousands of episodes per curriculum stage,
   capacity leaked downward permanently and never recovered, eventually making every
   job infeasible on every machine regardless of policy quality -- producing the
   exact `-idle_penalty * episode_length` signature this project used throughout to
   diagnose "policy collapse." Fixed by storing a pristine `machine_capacity` vector
   separately from the mutable per-timestep array.
2. **Eager `dict.get()` default crash in `a2c_policy.py`.** `mask =
   info.get("action_mask", self.env.get_action_mask())` evaluates its default
   argument unconditionally, so it called `self.env.get_action_mask()` every step
   regardless of whether `"action_mask"` was already present (it always was). That
   call crashed under the installed gymnasium version (1.2.3), which removed
   automatic attribute forwarding through `Wrapper.__getattr__`. This made every A2C
   training run fail immediately, independent of anything else in this document.

**Implication:** bug 1 in particular means every "policy collapse" result recorded
*before* 2026-08-09 should be read as confounded by an environment data bug, not
necessarily as clean evidence for the PPO-clipped-objective / large-action-space
mechanism the July 24 document argued for. That mechanism isn't invalidated -- collapse
re-emerges at larger curriculum stages even with both bugs fixed (Section 3) -- but the
*small-scale* collapse results predating this fix are not reliable evidence for it
specifically.

---

## 3. Solution 1: full-curriculum results on the flat architecture

Three full 375k-timestep curriculum runs (`Code/train_rl_agent.py --algo a2c`), both
bugs above fixed, on top of `MaskableActorCritic` (the original flat, one-weight-per-
action-index head):

- **baseline_fixed**: no other change (`idle_penalty=0.5`).
- **solution1a**: idle masked out whenever a non-idle action is feasible
  (`GymSchedulingEnv(restrict_idle=True)`).
- **solution1b**: idle penalty raised to a dominant magnitude (`idle_penalty=50`).

```
                    stage1(15j/h20)      stage2(30j/h40)      stage3(60j/h60)       stage4(100j/h100)
                    first200->last200    first200->last200    first200->last200     first200->last200
baseline_fixed      57.7  -> 77.6        107.5 -> 107.6        -251.9 -> -236.2      -1264.2 -> -1299.2
solution1a          71.6  -> 86.5        128.5 -> 129.6        -218.3 -> -213.0      -1214.4 -> -1191.3
solution1b          20.2  -> 69.9        116.6 -> 114.5        -346.5 -> -350.2      -1491.0 -> -1504.7
```

Reference: an EDF heuristic on the exact stage-4 job set gets **+275.5** (16.0 total
tardiness, 98/100 jobs scheduled) -- the stage-4 problem itself is close to fully
solvable by a simple greedy rule.

**Finding 1 -- the capacity-leak fix alone resolves the originally-documented
collapse.** All three variants train to strong, stable positive reward through
stages 1-2 (previously: 100% idle collapse at every scale tested). `idle_penalty`
magnitude is a secondary factor: `solution1a` (idle restricted) modestly outperforms
`baseline_fixed` throughout; `solution1b` (`idle_penalty=50`) modestly
*underperforms* baseline in every stage -- consistent with idle-penalty magnitude
never having been the primary lever, matching this project's own July 24 finding that
tuning it "only changed which constant the reward converges to."

**Finding 2 -- a second, distinct collapse emerges at curriculum-stage transitions,
independent of idle handling.** All three variants collapse hard immediately at the
stage2->stage3 transition and never recover: `first200` vs. `last200` within stage 3
(1,639 episodes) and stage 4 (1,485 episodes) are statistically indistinguishable in
every variant -- a flat plateau, not a slow recovery in progress. Given the EDF
reference shows stage 4 is easy, and given `max_jobs=100` padding means job slots
30-59 (respectively 60-99) are masked out -- never sampled, never gradient-updated --
throughout stages 1-2 (respectively 1-3) and only "activate" when their stage
unmasks them, the leading hypothesis is architectural, not a training-budget or
reward-magnitude problem: see Section 4.

---

## 4. Why naive independent `MultiDiscrete` was rejected

Before settling on the pointer-network design below, a `MultiDiscrete([job,
machine])` action space (independent job-head and machine-head, sampled from a
shared trunk in one forward pass) was considered and rejected. `sb3_contrib`'s masked
`MultiDiscrete` support masks each dimension independently based on the state --
there is no way to express "mask machine 3 only when job 7 was the job actually
sampled," because feasibility here is a *joint* property (a machine's fit depends on
the specific job's resource vector). An independent two-head policy would let the
agent pick a job and a machine that individually look reasonable but don't actually
fit together, or would require a coarser marginal mask that under- or over-restricts
the action space. What's needed instead is a mechanism that scores every `(job,
machine)` **pair** jointly, so the job choice is genuinely a function of which
machines are currently good fits for it -- not chosen first and reconciled with
machines second (autoregressive job-then-machine selection has the same problem in a
milder form: the job head still doesn't see per-machine detail when it commits).

---

## 5. Literature review

1. Vinyals, Fortunato, Jaitly, **"Pointer Networks"** (NeurIPS 2015). Foundational
   query/key compatibility-as-logit mechanism for selecting one element of a
   variable-size input set -- the lineage this design is drawn from.
2. Kool, van Hoof, Welling, **"Attention, Learn to Solve Routing Problems!"**
   (ICLR 2019, arXiv:1803.08475). Encoder = stacked multi-head self-attention over
   node tokens (no positional encoding, invariant to input order); decoder =
   scaled dot-product compatibility between a context query and node embeddings,
   clipped to `[-C, C]` via `C * tanh(.)` (`C = 10`) before masking infeasible nodes
   with `-inf` and softmax; trained with REINFORCE + a greedy-rollout baseline. The
   clip-then-mask-then-softmax pattern and the `C=10` clipping constant are adopted
   directly in `pointer_policy.CompatibilityScorer`.
3. Song et al., **"Flexible Job Shop Scheduling via Dual Attention Network Based
   RL"** (arXiv:2305.05119, "DANIEL"). The closest structural analog: separate
   operation/job embeddings (10-dim raw features in their paper) and machine
   embeddings (8-dim), a pairwise decision network that concatenates `[job_embed,
   machine_embed, global_context, pair_features]` through an MLP to score every
   feasible `(operation, machine)` pair, masked softmax over feasible pairs only,
   trained with PPO+GAE. Confirms "encode job and machine separately, then score
   pairs jointly" as the right shape for this exact class of problem (two-entity
   joint selection, not sequential routing) -- directly validating the design
   rejected-alternative discussion in Section 4. DANIEL's own reward
   (`r_t = max(C̄(t)) - max(C̄(t+1))`, a lower-bound-based potential difference) is
   itself potential-based in spirit, independently supporting Section 6's approach.
   Its dual-attention *inter-entity* message passing (jobs attending to other jobs
   via precedence, machines attending to competing machines) is **not** adopted in
   this iteration -- see Section 9.
4. Hu et al., **"Attend2Pack: Bin Packing through Deep Reinforcement Learning with
   Attention"** (arXiv:2107.04333). Self-attention item encoder + pointer-style
   query/key/value masked selection, action-space decomposition (sequence policy +
   placement policy), REINFORCE with a learned baseline and "prioritized
   oversampling" for hard episodes. Confirms masking-after-scoring (not
   before) as standard practice, and is the closest bin-packing-specific analog
   given this project's roots in vector bin packing.
5. **"Graph Neural Networks for Job Shop Scheduling Problems: A Survey"**
   (arXiv:2406.14096, 2024). Recency/context citation; GNN message-passing between
   job and machine nodes (as in DANIEL's dual attention blocks) is the natural next
   architectural step beyond this iteration's independent (non-message-passing)
   encoders -- see Section 9.
6. Ng, Harada, Russell, **"Policy Invariance Under Reward Transformations: Theory
   and Application to Reward Shaping"** (ICML 1999). Proves that
   `shaped_reward = reward + γΦ(s') - Φ(s)` has the same set of optimal policies as
   the unshaped reward, for any bounded potential function `Φ`. Basis for Section 6.

---

## 6. Solution 3: potential-based reward shaping

**Why not just raise the idle penalty further** (the natural first idea, and one this
project already tested): Section 3's `solution1b` result (`idle_penalty=50`)
underperforms the plain baseline in every curriculum stage, matching the July 24
document's earlier finding that reward-magnitude tuning only changes which constant
the reward collapses to, not whether it collapses. Mechanically, PPO/A2C's
near-deterministic early convergence in a large discrete action space isn't a
rational comparison of magnitudes -- the policy locks onto whichever action is
*easiest to discover* (idle, which needs no multi-step credit assignment) within the
first few gradient updates, and once probability mass collapses onto it, gradient
signal for the alternative vanishes regardless of how bad that action's true reward
is. A larger penalty can just as easily produce a *different* degenerate collapse
(e.g. repeatedly attempting one fixed invalid placement) rather than a good policy.

**Potential-based shaping is a different, provably safe tool for the same goal**
(reward progress toward urgent jobs before their deadline penalty fires) --
Ng/Harada/Russell's guarantee means it reshapes the *gradient*, not the *optimum*,
unlike magnitude tuning. Implemented in `SchedulingEnv` behind an opt-in constructor
flag (`use_potential_shaping`, off by default -- separately toggleable from Solutions
1 and 2 so it can be ablated independently):

```
Φ(s) = -Σ_{j in remaining_jobs} urgency_j(t),   urgency_j(t) = 1 / (max(slack_j(t), 0) + 1)
slack_j(t) = deadline_j - t - duration_j
shaped_reward = reward + shaping_gamma * Φ(s') - Φ(s)
```

Note this refines the originally-sketched formula (a raw `1/(slack+1)` term):
clamping `slack` to be non-negative before the reciprocal keeps `urgency_j` bounded in
`(0, 1]` and monotonic for all slack values, including already-late jobs (`slack <
0`) -- the unclamped version blows up and flips sign as `slack -> -1`, which would
incorrectly *reward* being very late. `Φ(s)` is bounded in `[-|remaining_jobs|, 0]`.
Completing a job removes its (negative) contribution entirely; idling while an urgent
job's slack shrinks makes `Φ` more negative, so the shaping term specifically
penalises idling *more* when jobs are close to deadline than when there's slack to
spare, and rewards scheduling a job *before* that happens.

Sign-check (5-job smoke test, `Code/scheduling_env.py`): scheduling the
most-imminent-deadline job first yields a net-positive shaped reward including a
shaping bonus for removing its urgency term from `Φ`; subsequent idle steps show `Φ`
drifting further negative as remaining jobs' slack shrinks. Mechanism confirmed
correct; full-curriculum evaluation against the unshaped reward is follow-up work
(Section 10).

---

## 7. Solution 2: pointer/attention network architecture

### 7.1 Design

`Code/Policies/pointer_policy.py`, a drop-in replacement for
`MaskableActorCritic.policy_head`/`value_head` -- same `forward(obs) -> (logits,
value)` interface, same output shape `(max_jobs*num_machines+1,)`, so
`masked_softmax`, `select_action`, `RolloutBuffer`, and `eval_rl_agent.py` all work
unchanged:

- `JobEncoder` / `MachineEncoder`: 2-layer MLPs with **shared weights across every
  job/machine slot** (an `nn.Linear` applied to a `(B, J, F)` tensor broadcasts
  identically over the slot dimension) -- the key structural difference from
  `MaskableActorCritic.policy_head`, which gives every `(job_slot, machine)` index
  its own independent weight row.
- `CompatibilityScorer`: scaled dot-product between job and machine embeddings
  (query/key projections), `10 * tanh(.)` clipped (Kool et al.), **left unmasked** --
  masking stays entirely in `a2c_policy.masked_softmax`, unchanged.
- Flat obs vector reshaped back into `(time, machine_feats, job_feats)` *inside* the
  module (not via a `Dict` observation space), matching
  `GymSchedulingEnv._get_obs()`'s exact layout -- chosen to keep `GymSchedulingEnv`,
  `eval_rl_agent.py`, and `MaskablePPO`'s `MlpPolicy` (PPO port out of scope, Section
  9) completely untouched. Verified by direct unit test that the
  `(job, machine) -> job*num_machines+machine` flatten order matches
  `GymSchedulingEnv`'s action encoding exactly, since a transposed reshape here would
  silently misalign logits with the mask without necessarily crashing.
- Idle logit and value estimate both come from a masked mean-pooled global context
  (job pooling excludes padded/already-scheduled slots via the existing
  `scheduled_flag` feature, so padding never dilutes the pooled representation).

### 7.2 Why this specifically targets the Section 3 curriculum-transition collapse

The flat head's `Linear(256, 1001)` gives every `(job_slot, machine)` index an
independent weight row; slots 30-99 (respectively 60-99) receive literally zero
gradient throughout stages 1-2 (respectively 1-3), since they're always masked out
by padding. When a stage unmasks them for the first time, those rows are still near
their random initialisation. The pointer network's job/machine encoders instead learn
a function of *features* (duration, deadline, resource vector), shared across every
slot at every stage -- so a job appearing in slot 45 for the first time in stage 3 is
scored by encoder weights already trained on thousands of examples from slots 0-29,
because the feature *distributions* are identical across slot indices; only the
index differs, which is exactly what a per-index weight row wrongly treats as
meaningful and a features-based encoder correctly ignores. This is the same
underlying flaw (no parameter sharing across the action-index space) implicated in
both the original idle-collapse (July 24 doc) and the curriculum-transition collapse
(Section 3) -- one architectural fix addresses both symptoms.

### 7.3 Two additional fixes bundled into `a2c_policy.py` while implementing this

1. **Masked-entropy bug fix.** The training loss previously recomputed
   entropy/log-probs via an *unmasked* softmax (a stale comment said "unmasked
   entropy for simplicity"), inconsistent with the masked distribution actually
   sampled during rollout collection. Fixed by storing the per-step action mask in
   `RolloutBuffer` and using `masked_softmax` in the loss computation too.
2. **`ent_coef` raised from `0.0` to `0.01`, and reward normalisation added.**
   This hand-rolled A2C previously had *zero* entropy bonus, unconditionally -- no
   forcing function pushed exploration of newly-unmasked action-space regions at a
   curriculum transition. Separately, no reward normalisation existed anywhere in
   the pipeline, despite episode-reward scale growing sharply with problem size
   (stage 1-2: roughly `[0, 130]`; stage 3-4: `[-1500, +300]`, Section 3) -- the
   value function's bootstrapped TD targets shift abruptly at every transition with
   nothing to keep them on a stationary scale. `RunningMeanStd` (Welford-style
   online mean/variance) now divides rewards by their running std before they enter
   the returns/advantage computation (mean is deliberately not subtracted, matching
   SB3's `VecNormalize(norm_reward=True)` convention, since centering would distort
   the sign of terminal completion/tardiness rewards); raw reward is still used for
   episode-reward logging/plots. Both are cheap, architecture-independent, and
   layered on top of the pointer network in the comparison below.

---

## 8. Results: pointer network vs. flat baseline (full curriculum, both with Section 7.3's fixes)

Full detail in `Future/research/training-log.md`'s "Pointer network vs. flat
baseline" entry; summary here.

```
                    stage1(15j/h20)      stage2(30j/h40)      stage3(60j/h60)       stage4(100j/h100)
                    first200->last200    first200->last200    first200->last200     first200->last200
flat_fixed_full     68.8  -> 86.3        81.5  -> 139.2         32.9 -> 104.4        -21.8  -> 50.7
pointer_full        58.7  -> 94.3        92.7  -> 101.4         34.9 -> 25.0        -131.2  -> -63.3
```

**Two findings, in tension with each other:**

1. **The Section 7.3 hygiene fixes (entropy bonus, reward normalisation, consistent
   masked-entropy loss) -- applied to the unchanged flat architecture -- already
   resolve most of Section 3's stage-3/4 regression.** `flat_fixed_full`'s stage 3/4
   both switch from a flat, non-recovering plateau (Section 3's `baseline_fixed`) to
   a clearly improving trend that ends positive. This means zero exploration
   pressure and an unnormalised, scale-jumping value target -- not primarily the
   lack of parameter sharing this document's Section 7.2 argued for -- were the
   dominant cause of that specific regression.
2. **The pointer network underperforms the now-fixed flat baseline on this
   comparison**, in both stage 3 (25.0 vs. 104.4 final) and stage 4 (-63.3 vs. 50.7
   final), while still comfortably beating the *original, unfixed* flat baseline.

**Why, and why this doesn't retract Section 7.2's argument:** this curriculum reuses
the exact same fixed job set (`seed=0`) for every episode of every stage -- stage 3
alone repeats the same 60 jobs over ~1,638 episodes. A flat per-index head has no
need to generalise from job *features* under these conditions: given working
exploration and correctly-scaled gradients, it can simply memorise the right action
for each of the ~30-40 "newly unmasked" (but fixed, repeated) job slots directly,
which is a strictly easier optimisation target than the pointer network's indirect,
shared-weight parameterisation through an attention-style score. The pointer
network's actual hypothesised advantage -- scoring a job correctly from its features
alone, without having seen *that specific slot index* before -- is never exercised by
a curriculum where every "new" slot is still the same fixed job seen thousands of
times. **This curriculum tests memorisation-with-repetition, not
generalisation-to-novel-instances, and the pointer network is not designed to win at
the former.**

**Honest bottom line:** don't read this as the pointer-network idea being wrong; read
it as this curriculum being the wrong experiment to test it with. The value
proposition in Section 7.2 remains theoretically sound and literature-supported, but
is currently unproven -- on the one experiment run so far, it underperforms a
correctly-tuned simpler baseline. See Section 10 for the actual test this needs.

---

## 9. Explicitly deferred

- **PPO port.** `MaskablePPO`'s `MlpPolicy` would need a custom `ActorCriticPolicy`
  (or `policy_kwargs`-injected features extractor) to plug in the same
  encoder/scorer; real engineering work, out of scope for this iteration.
  `Code/Policies/ppo_policy.py` is untouched.
- **Inter-entity message passing** (DANIEL's dual attention blocks, ref. [3]; GNN
  survey, ref. [5]). The current encoders are fully independent -- no job-to-job or
  machine-to-machine context sharing. A job's embedding doesn't currently know
  anything about which other jobs are competing for the same machines.
- **Variable-size generalisation claims.** The padding/masking scheme still fixes
  `max_jobs` at construction time and the encoder design *should* generalise better
  to newly-unmasked slots (Section 7.2's argument), but this has not yet been
  evaluated for zero-shot transfer to an unseen `max_jobs` or a `num_jobs`
  distribution outside the fixed curriculum. Don't claim "generalises to
  variable-size problems" without an actual cross-stage/cross-instance eval.
- **Solution 3 full-curriculum evaluation.** The shaping term is implemented and
  sign-checked (Section 6) but not yet run through the full curriculum, alone or
  combined with the pointer network.

## 10. How to tell if it worked / next steps

Section 8's result means the original test design was wrong, not necessarily the
architecture. The actual test the pointer network's design argument (Section 7.2)
needs:

- **Randomise the job set per episode (or at least per curriculum stage), instead of
  reusing `seed=0` throughout.** `env_config.generate_env_config(seed=...)` already
  supports this -- `make_env()` in `train_rl_agent.py` would need to draw a fresh
  seed per episode/reset rather than fixing `seed=0` for the whole run. Under that
  condition, a flat per-index head cannot memorise per-slot behaviour (each episode's
  job identities differ), so its performance should degrade relative to a
  feature-based encoder that can generalise -- if the pointer network's design
  argument is right, this is where it should show a real, currently-unproven
  advantage rather than a currently-observed disadvantage.
- Tune the pointer network's own hyperparameters (`embed_dim`, `hidden`, learning
  rate) rather than reusing `MaskableA2C`'s defaults tuned incidentally for the flat
  head -- `optuna_tune.py --algo a2c` now targets `PointerActorCritic` (Section 6 of
  this doc's implementation notes / `training-log.md`) and should be run properly
  once the randomised-instance experiment above is in place.
- Run Solution 3 (potential-based shaping) as an ablation on top of both policies
  under the randomised-instance setup; if it helps either, that's independent
  evidence for Section 6's argument, since it's a reward-level change orthogonal to
  the architecture question.
- `training-log.md` gets one entry per run, per that file's existing convention
  (already done for the runs completed so far).

## References

[1] O. Vinyals, M. Fortunato, N. Jaitly, "Pointer Networks," NeurIPS 2015.

[2] W. Kool, H. van Hoof, M. Welling, "Attention, Learn to Solve Routing Problems!,"
ICLR 2019. arXiv:1803.08475.

[3] W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job Shop Scheduling via Dual
Attention Network Based Reinforcement Learning," arXiv:2305.05119, 2023.

[4] R. Hu, J. Yang, A. Ferber, et al., "Attend2Pack: Bin Packing through Deep
Reinforcement Learning with Attention," arXiv:2107.04333, 2021.

[5] "Graph Neural Networks for Job Shop Scheduling Problems: A Survey,"
arXiv:2406.14096, 2024.

[6] A. Y. Ng, D. Harada, S. Russell, "Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping," ICML 1999.
