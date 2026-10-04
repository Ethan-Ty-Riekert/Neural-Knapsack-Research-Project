# Design + Result: Dense Per-Tick Tardiness Reward (`reward_mode="dense_tardiness"`)

**Date:** 2026-09-17 (S2W10)
**Environment:** `Code/env/scheduling_env.py`, `Code/env/online_scheduling_env.py`,
`Code/training/train_optimized.py`, `tests/test_dense_tardiness_reward.py`
**Status:** Implemented, algebraically verified correct, tested at both quick
(300k-timestep) and full (1.9M-timestep) scale on offline PPO. **Result: no
improvement over the legacy reward** -- see Section 5.

---

## 1. Motivation

The user challenged this project's "RL doesn't beat a strong heuristic here"
conclusion, citing literature reporting RL beating heuristic baselines by 20%+ on
comparable dynamic scheduling workloads, and identified the reward function as the
likely root cause. That prompted a direct mathematical audit of
`SchedulingEnv.reward()`/`step()`, which found a provable defect: the tardiness
penalty is bounded (`T_j/horizon < 1`, so the penalty is capped below `lambda_2`),
while `step()` adds a flat `+3.0`-per-completion and `+50`-for-finishing-everything
bonus, independent of lateness. Every Optuna search this project has ever run
converged on `lambda_2 ~= 1.9` -- below 3.0 -- meaning scheduling any job, however
late, was always reward-positive versus abandoning it, for every configuration ever
actually trained.

## 2. Research grounding

Read DeepRM (Mao, Alizadeh, Menache, Kandula, 2016, HotNets) in full and searched
further RL-for-scheduling literature (see the earlier plan-mode discussion this
session for the full search trail). DeepRM's reward is dense (paid every tick, for
every job in the system) and tied directly to the true objective -- no flat,
lateness-independent bonus. Decima (Mao et al., SIGCOMM 2019, 19-45% improvement
over baselines) follows the same principle.

## 3. Design

`T_j = C_j - d_j` is, by definition, the number of ticks a job is unfinished past
its deadline. Charging `-lambda_2 * w_j / horizon` at every such tick reproduces
`-lambda_2 * sum_j(w_j*T_j/horizon)` **exactly** -- the same total the legacy
lump-sum charge computes, just paid across time instead of once at scheduling time.
The `/horizon` normalization matters and is not optional: without it, spreading the
same total over more possible ticks at larger horizons would silently reintroduce
the horizon-scaling miscalibration a previous session's fix (`T_j/horizon`) already
solved for the lump-sum case.

The flat `+3.0`/`+50` bonuses are dropped under this mode -- no longer needed
(finishing a job, especially before its deadline, directly stops the per-tick
charge before it can start), and removing them removes the exact mechanism proven
to make lateness not matter.

**Named, explicit risk**: the `+3.0` bonus's original purpose was countering an
unrelated idle-collapse failure mode (`2026-07-24-idle-action-policy-collapse.md`).
Removing it was validated empirically (Section 5), not assumed safe.

## 4. Two real implementation bugs found and fixed by the test suite before any
training run, not just reasoned about

`tests/test_dense_tardiness_reward.py` was written to verify the exactness claim
algebraically (comparing `episode_cost` between a legacy and a dense run on an
identical scripted scenario), not just check it runs. It caught two real bugs:

1. **Scale mismatch**: the first implementation charged raw `-lambda_2*w_j` per
   tick (unnormalized), which would have reintroduced the exact horizon-scaling
   problem the `/horizon` normalization exists to prevent. Fixed by dividing the
   per-tick charge by `horizon` too, keeping the *total* charged per job identical
   in scale to the legacy lump sum, just distributed differently across time.
2. **Early-termination undercounting**: the offline case ends an episode the
   instant `remaining_jobs` empties (every job scheduled) -- but a job scheduled
   recently, with a multi-tick duration, may still have ticks left to run past its
   deadline when that happens, and those ticks would never get charged (no further
   `step()`/`step_idle()` calls occur once `done=True`). Fixed with
   `_finalize_dense_tardiness_running_jobs()`, called whenever `done` becomes True,
   charging exactly the remaining un-elapsed late-ticks in one lump sum. Proven
   safe to call unconditionally (a no-op when the episode instead ends at the
   horizon, since `is_feasible()` guarantees every scheduled job is complete by
   then).

After both fixes, `episode_cost` under `dense_tardiness` matched the legacy lump-sum
total to floating-point precision on the test scenario -- the exactness claim is
verified, not just argued.

## 5. Result

Full-scale (1.9M timesteps, full curriculum, same Optuna-tuned hyperparameters,
same fixed instance as every other offline PPO result):

```
                     tardiness   P95     late       scheduled
Legacy (full-scale):  1301.12   58.92   41.96/100   96.90/100
Dense  (full-scale):  1309.66   59.00   41.72/100   96.74/100
```

Statistically indistinguishable, both squarely inside the ~1290-1330 band every
other PPO variant has produced this session. No idle-collapse (the named risk in
Section 3 never materialized -- jobs-scheduled stayed healthy in both modes). The
dense reward *did* produce visibly better-behaved training diagnostics
(`value_loss` ~0.02-0.03 vs. legacy's ~30-46, i.e. a much more predictable value
target, consistent with the literature's credit-assignment argument) -- but that
improved learnability did not translate into a better final tardiness policy.

## 6. Conclusion

This is the **seventh** mechanism tried for PPO's tardiness this session (fixed
reward weight, four Lagrangian $\lambda_{\max}$ ceilings, a tardiness-directed
hyperparameter search, pointer-network architecture, and now this reward redesign).
All seven land in the same band. The reward-function hypothesis was correctly
reasoned and rigorously implemented -- and is now also ruled out as the dominant
bottleneck, not just under-tested. A genuine structural difference from DeepRM/
Decima not yet tested: DeepRM's action space is ~11 discrete choices (M=10 job
slots + void) versus this project's `max_jobs*num_machines+1` (1000+ at deployed
scale) -- a combinatorially larger action space is a well-known independent driver
of policy-gradient sample-inefficiency. Flagged as the leading untested structural
hypothesis, not attempted here.

`reward_mode="dense_tardiness"` is kept in the codebase (opt-in, A/B-able,
algebraically tested) rather than discarded -- it's a real, literature-grounded
improvement in training dynamics even though it didn't move tardiness here, and is
worth re-testing on A2C and the online case before any final verdict on it either.

## 7. References

1. Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). "Resource Management
   with Deep Reinforcement Learning." *HotNets 2016*.
2. Mao, H., et al. (2019). "Learning Scheduling Algorithms for Data Processing
   Clusters." *SIGCOMM 2019*.
