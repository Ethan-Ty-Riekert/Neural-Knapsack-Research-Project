# RL improvement plan after the uniform v2 protocol

**Date:** 2026-10-06 (S2W12). **Status:** proposed, awaiting user approval. Paper due ~2026-10-09.
Evidence: `training-log.md` entries of 2026-10-06.

## 1. Where we are

| | Best heuristic | Best RL (test J) | Status |
|---|---|---|---|
| off_tf05 (weighted) | LST 39,536 | Option 3w PPO 31,291 (-20.9%) | RL wins |
| off_tf05_w1 (w = 1) | LST 13,098 | Option 1 A2C 13,318 (+1.7%) | tie |
| on_rho095 (online) | EDF+Consolidate 21,986 | Option 0 A2C 44,321 (+102%) | RL loses |

Diagnosed so far:
- **Tuned worse than defaults** (Option 2 offline: 74,554 tuned vs 32,672 defaults). Cause: 100k-step trials
  favour slow-learning settings, and the defaults were never a trial (observed, 3 seeds).
- **Online, training does not improve on the initial policy**; several designs get worse as they train.
  Option 1 keeps settling on one fixed rule (SPT, WSPT, ATC...) although EDF+Consolidate is in its menu.
- **Discounting is not the main cause:** gamma = 1 (training on exactly J) did not fix it (observed, 3 seeds).
- **Seed-to-seed spread is huge online** (31k to 171k validation J for one design), and offline it is small.

## 2. Already running (started 2026-10-06 19:35)

- **R1** Default hyperparameters for every option x algorithm x preset, work-conserving, 300k x 3 seeds.
- **R2** The same with free idling (user request).
- About 10 h; done ~05:00-06:00 Wednesday. Tells us (a) tuned vs default per cell and (b) whether free
  idling helps or causes the earlier degenerate idling (Option 3 idled 6,000+ times per episode).

## 3. Proposed changes (each uniform across all options)

### P1. Input-dependent critic (the main online fix) -- recommended
**Problem (hypothesis, fits all the evidence):** online, a run's return depends mostly on the random arrival
sequence, which the critic cannot predict from F_t. Advantage estimates are therefore mostly noise, so the
policy gradient is mostly noise: the policy drifts, collapses onto a rule, and seeds disagree. Offline there
are no arrivals after t = 0, so this noise does not exist -- matching "offline learns, online does not".
This is the "input-driven environment" problem of Mao et al. (ICLR 2019).
**Fix:** let the **critic only** see the future arrival sequence (asymmetric actor-critic, Pinto et al. 2018).
Mao et al. prove that a baseline may depend on the exogenous input sequence without biasing the policy
gradient, because the policy's actions cannot change the arrivals. **The policy still sees exactly F_t,
so your MDP is unchanged**; only the training signal gets less noisy. Offline the critic already sees
everything, so offline results are unaffected in principle.
**Cost:** ~half a day to implement and test (critic input = the arrival rows the policy cannot see).

### P2. Observation fixes (one change, no new information)
- **Capacity look-ahead window**: each machine's free capacity per resource for the next K = p_max ticks
  (9 offline, 40 online), computed from F_t, so the state and Markov property are unchanged. +360 numbers
  offline, +1,600 online (+13%). Lets the agent see when capacity frees up (needed for sensible idling).
- **Consistent scaling**: job requirements scaled by machine capacity C = 30 (so "fits" is A/C <= R/C),
  durations, weights and deadlines by fixed generator constants instead of each instance's own maximum.
- **Total capacity**: constant (30 on every resource, machine and instance), so it is not needed in the
  state; remaining capacity is already given as a fraction of it. No change.
**Cost:** ~2 h including tests.

### P3. Fix the tuning
- Defaults as trial 0 of every study (`study.enqueue_trial`), so tuning can never pick something worse than
  the defaults on validation.
- Trials as long as the final runs (300k), with pruning checks at 75k / 150k / 225k: removes the
  short-trial bias by construction.
- 12 trials per study instead of 20 (to fit the deadline; trial 0 = defaults).

### P4. More training steps
Offline Options 2 and 4 were still improving at 300k; online only Option 3w was. Final runs at **1M steps**.
Offline this is cheap (up to ~70 min per run). Online the slow designs (Options 0, 2, 4 PPO) take ~3-5 h
per run, which is the time risk (see section 5).

### Not recommended now
- **Packing the job slots** (explained below): ~2.5x smaller online observation, but medium effort and it
  needs an overflow rule. Future work.
- **Starting from the best heuristic (imitation warm start)**: strong practical fix for rule collapse, but it
  makes "RL learned its own policy" harder to claim. Optional, if P1 fails.

## 4. What "packing" means

The online observation has one row for every job that can **ever** appear in the episode: 1,235 rows.
At a given tick most rows are empty: jobs that have not arrived yet (zeros) or are finished. On average
only ~155 jobs are waiting or running (peak ~490). Packing = put just the waiting and running jobs in
the first rows and drop the rest, like showing only the people currently in the queue instead of a list
of everyone who will visit all day. Same information (your MDP only uses J_t and P_t); ~2.5x smaller input.
The catch: actions refer to row numbers, so every design must translate "row 3" back to the real job, and a
fixed maximum (e.g. 512 rows) needs a rule for the rare case where more jobs are waiting.

## 5. Schedule (estimate)

| When | What |
|---|---|
| Tue night | R1 + R2 running (already) |
| Wed morning | Implement P1 + P2 + P3, tests, short pilot of P1 online |
| Wed afternoon -> Thu | Tuning v3 (12 x 300k per study) + finals (1M x 3 seeds) |
| Thu evening / Fri morning | Results archive, Appendix E, results section |

**Risk:** the full set (5 options x 2 algorithms x tuning + 1M finals) may not finish by Friday. Ways to cut
time if needed: (a) keep only one idling mode, chosen from R1 vs R2; (b) PPO only for v3 (PPO beat A2C in
13 of 15 cells; A2C rows from v2 stay in the paper); (c) 1M steps offline only, 300k online.

## References

1. Mao, H., Venkatakrishnan, S. B., Schwarzkopf, M., Alizadeh, M. Variance Reduction for Reinforcement
   Learning in Input-Driven Environments. ICLR 2019. arXiv:1807.02264.
2. Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., Abbeel, P. Asymmetric Actor Critic for
   Image-Based Robot Learning. RSS 2018. arXiv:1710.06542.
3. Mao, H., Alizadeh, M., Menache, I., Kandula, S. Resource Management with Deep Reinforcement Learning
   (DeepRM). HotNets 2016. (capacity look-ahead image)
4. Akiba, T. et al. Optuna. KDD 2019. arXiv:1907.10902.
5. Andrychowicz, M. et al. What Matters in On-Policy Reinforcement Learning? ICLR 2021. arXiv:2006.05990.
