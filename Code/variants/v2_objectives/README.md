# Variant v2: reward = the selected objectives

**Status:** active. Build steps 1–5 are implemented and tested (2026-09-29).
**Definition and proofs:** `Future/research/2026-09-28-v2-objective-formal-definition.md`
**Decisions:** `Future/research/2026-09-28-objective-redesign-discussion.md`
**Code:** `Code/core/objectives.py`, switched on with `reward_mode="objective"` in the environments.

## The objective (default)

J = Σ_{finished j} w_j T_j + Σ_{dropped j} w_j (max(0, H − d_j) + H)

This is weighted lateness, where a dropped job counts as if it finished at the horizon plus a further H ticks late. The reward is
−J / (number of jobs), charged as the costs happen:
- each tick a job is overdue;
- the moment a job can no longer start (latest start H − P_j) it counts as dropped.

Option: `--objectives tardiness,late_count` adds λ_U · Σ w_j U_j (weighted late-job count, the SLA violation
rate).

Removed compared with v1: the +3/+50 bonuses, the machine-activation, hotspot and idle penalties.

## Build steps

| Step | What | Status |
|---|---|---|
| 1 | v2 reward: lateness, drops, late count; offline and online; proofs checked by `tests/test_objective_reward.py` | done |
| 2 | Heuristics, PSO and CP-SAT under v2 through `run.py`; CP-SAT solves the same J, with optional jobs | done |
| 3 | Energy objective (`linear` = active machine-ticks, or `specpower_ml110g5`) and the `Consolidate` placement rule | done |
| 4 | Difficulty presets (`Code/core/difficulty.py`): offline deadline tightness, online load, tight online deadlines | done |
| 5 | RL training and evaluation under v2 (`--reward-mode objective`, `--difficulty`), with drop-risk shaping | done |

## Presets

The presets are the same instances as v1 (`python run.py --list`), so v1 and v2 numbers are always on identical
instances. Example:

```
python run.py --variant v2_objectives --preset off_c_15 --method EDF
python run.py --variant v2_objectives --preset off_c_15 --method "cpsat,LST,EDF,pso"      # ranked comparison table
python run.py --variant v2_objectives --preset on_rho095 --method heuristics            # all heuristics
python run.py --variant v2_objectives --preset off_tf05 --method EDF+Consolidate --objectives tardiness,energy
python run.py --variant v2_objectives --preset on_rho075 --method rl-train:1 --timesteps 300000 --checkpoint-tag mytag
python run.py --variant v2_objectives --preset on_rho075 --method rl-eval:1 --checkpoint-tag mytag
```

Difficulty presets (15 held-out instances each, weights 1–5): `off_tf02`, `off_tf05` and `off_tf08` (offline, loose to
tight deadlines), `on_rho050`, `on_rho075`, `on_rho095` and `on_rho110` (online load), and `on_rho075_tight` (online, tight
deadlines). Comparison tables are saved in `Results/v2_objectives/comparisons/`.

For heuristics, PSO and CP-SAT, shaping is off, so reward = −J / (number of jobs) exactly, and `objective_J` is
reported with every run.
