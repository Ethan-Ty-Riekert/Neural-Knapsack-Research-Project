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
