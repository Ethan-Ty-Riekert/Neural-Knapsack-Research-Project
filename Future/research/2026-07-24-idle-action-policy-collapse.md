# Investigation: Policy Collapse onto the Idle Action in MaskablePPO Training

**Date:** 2026-07-24
**Environment:** `Code/scheduling_env.py` (static, offline resource-constrained scheduling),
`Code/gym_scheduling_wrapper.py`, trained via `sb3_contrib.MaskablePPO` in `Code/train_rl_agent.py`.
**Status:** Two of three candidate causes addressed (Section 4). Third cause (Section 5) deferred pending
evaluation of the first two.

## 1. Observed symptom

Two independent training runs, before and after an initial round of reward-shaping and entropy-coefficient
fixes, exhibited the same failure signature. Run 1 (`ent_coef=0.01`, `idle_penalty=1.0`, hotspot penalty
double-counted, placement bonus `+0.1`):

```
rollout/ep_len_mean       21
rollout/ep_rew_mean       -21
train/approx_kl           0.0
train/clip_fraction       0
train/entropy_loss        0
train/policy_gradient_loss 3.56e-09
```

Run 2, after raising `ent_coef` to `0.05`, fixing the double-counted hotspot penalty, raising the
placement bonus to `+1`, and lowering `idle_penalty` to `0.5` (total_timesteps=51,200):

```
rollout/ep_len_mean       21
rollout/ep_rew_mean       -10.5
train/approx_kl           0.0
train/clip_fraction       0
train/entropy_loss        0
train/explained_variance  0.998
train/policy_gradient_loss 7.04e-09
train/value_loss          0.263
```

## 2. Diagnostic reasoning

In both runs, `ep_rew_mean / ep_len_mean` equals `-idle_penalty` exactly (`-21/21 = -1.0`;
`-10.5/21 = -0.5`). Since `SchedulingEnv.step_idle()` returns a flat `-idle_penalty` and nothing else,
this is only possible if the policy is selecting the idle action on effectively every step across the
trailing 100-episode window SB3 averages over. `entropy_loss`, `approx_kl`, and `policy_gradient_loss`
all being numerically zero corroborates this: once a categorical policy places ~100% probability mass on
a single action, its entropy and its gradient both underflow to zero in float32, so PPO has no further
learning signal to escape the state. `explained_variance = 0.998` with near-zero `value_loss` is
consistent with this too -- a policy that behaves identically every episode produces a trivially
predictable return, which the critic fits almost perfectly.

The reward-shaping and entropy-coefficient changes made between the two runs did not prevent the
collapse; they only changed which constant the reward converges to. This indicates the cause is
structural, not simply a matter of reward magnitudes.

## 3. Literature review

### 3.1 PPO's clipped objective is a documented source of this exact failure mode

Hsu, Mendler-Dünner, and Hardt (2020) [1] identify a failure mode of standard PPO with discrete Softmax
policies in which the clipped surrogate objective -- designed to keep policy updates conservative -- can
trap the policy on a suboptimal, near-deterministic action early in training, since the same clipping
that provides stability also prevents the large corrective probability shifts needed to escape a bad
early commitment. Related analysis of PPO in large discrete action spaces notes that this failure
mode "becomes increasingly problematic as the number of actions increases," because the probability of a
given rollout ever sampling the optimal action shrinks as the action space grows. Our flattened
`Discrete(max_jobs * num_machines + 1)` action space is ~1,001-wide, well into the regime this concerns.

### 3.2 `sb3_contrib`'s `MaskablePPO` has reported numerical fragility at comparable action-space scale

A `stable-baselines3-contrib` GitHub issue [2] reports `MaskablePPO` failing outright (categorical
distribution "Simplex constraint" validation errors from float32 softmax precision loss) on action spaces
of approximately 1,400. Our action space (~1,001) does not hit this crash, but sits in the same
numerically fragile neighbourhood, which is consistent with `entropy_loss`/`approx_kl` collapsing to
exactly `0.0` rather than merely a small value.

### 3.3 Job-shop-scheduling RL literature factorizes (job, machine) rather than flattening it

Multiple recent flexible job-shop-scheduling (FJSP) papers [3, 4] decompose the scheduling decision into
two smaller sub-decisions -- a job-selection action and a machine-selection action -- rather than a single
flattened `job * machine` index, precisely to avoid the combinatorial blow-up of the joint action space.
This is directly relevant: `GymSchedulingEnv` currently flattens the environment's native `(job, machine)`
action pair (see `scheduling_env.py`'s `step(action: Tuple[int, int])`) into one `Discrete` index, which is
the action-space shape the FJSP literature has moved away from for this exact reason.

## 4. Interventions adopted (this session)

1. **Larger policy/value network.** `train_rl_agent.py` previously constructed `MaskablePPO("MlpPolicy",
   env, ...)` with no `policy_kwargs`, so it used Stable-Baselines3's default architecture: two 64-unit
   hidden layers for both the actor and critic. Against an ~840-dimensional observation and a ~1,001-way
   action space, a 64-unit bottleneck has very little capacity to differentiate between actions, plausibly
   contributing to (though not fully explaining, per Section 3.1) the speed and totality of the collapse.
   `policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.Tanh)` was added.
2. **Fewer PPO epochs per rollout.** `n_epochs` lowered from 10 to 4. Ten gradient passes over a single
   rollout's data increases the risk of overfitting to (and locking in on) whatever that rollout happened
   to contain -- compounding the collapse risk described in Section 3.1.

These were chosen first because they are low-risk, localized changes (hyperparameters and network size,
not a change to the action-space semantics) and can be evaluated quickly before committing to a larger
restructuring.

## 5. Intervention deferred: factorized action space

Per Section 3.3, the structurally strongest fix is to stop flattening `(job, machine)` into one
`Discrete` action and instead expose it as `MultiDiscrete([num_jobs + 1, num_machines])` (or an
autoregressive two-head policy: select job, then select machine conditioned on the job), which
`sb3_contrib`'s `MaskablePPO` supports natively for masked `MultiDiscrete` spaces. This would reduce what
PPO has to search from ~1,001 joint options to two much smaller, largely independent decisions (~101 and
~10).

This was not implemented in this session because it changes the action-space contract across
`gym_scheduling_wrapper.py` (action space definition, `step()` decode logic, `get_action_mask()`) and the
hand-rolled `MaskableActorCritic` in `Policies/a2c_policy.py` (which currently assumes a single flat
categorical action and would need a second head plus autoregressive/independent masking logic). It is
recorded here, and in `Future/README.md`, so it isn't lost, and should be picked up if Section 4's changes
turn out to be insufficient.

## 6. How to tell if Section 4's changes worked

On the next training run, check (via the live plot / `training_rewards.csv` from `plotting_utils.py`) for:
- `ep_rew_mean` values that are *not* an exact multiple of `-idle_penalty` (currently `-0.5`) divided by
  episode length -- i.e. evidence of non-idle actions actually being taken and scored.
- `entropy_loss` and `approx_kl` remaining measurably non-zero for longer into training, rather than
  flatlining at `0.0` within the first handful of iterations.
- `ep_rew_mean` trending upward over the course of a curriculum stage rather than sitting flat.

If the collapse persists despite these changes, that is evidence favouring Section 5 (the action space
itself, not the network size or epoch count, is the dominant cause) per Hsu et al.'s finding that this
failure mode is fundamentally about the clipped objective's interaction with a large discrete action
space, not just undercapacity.

## References

[1] C. C.-Y. Hsu, C. Mendler-Dünner, and M. Hardt, "Revisiting Design Choices in Proximal Policy
Optimization," arXiv preprint arXiv:2009.10897, 2020. [Online]. Available:
https://arxiv.org/abs/2009.10897

[2] Stable-Baselines-Team, "MaskablePPO Masking Doesn't Work with Big Action Space," GitHub Issue #247,
stable-baselines3-contrib. [Online]. Available:
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/247

[3] "A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem,"
Expert Systems with Applications, 2022. [Online]. Available:
https://www.sciencedirect.com/science/article/abs/pii/S0957417422010624

[4] "An efficient deep reinforcement learning environment for flexible job-shop scheduling," arXiv
preprint, 2025. [Online]. Available: https://arxiv.org/html/2509.07019
