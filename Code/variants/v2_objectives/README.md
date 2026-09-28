# Variant v2: reward = the selected objectives

**Status:** in progress. Build step 1 (lateness + drops + late count) is done and verified.
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
| 1 | v2 reward: lateness, drops, late count; offline and online; proofs checked by tests | **done** |
| 2 | Run heuristics, PSO and CP-SAT under v2 through `run.py` and compare | next |
| 3 | Energy objective (active machine-ticks) and an energy-aware heuristic | |
| 4 | Difficulty settings (load, deadline tightness) | |
| 5 | Retrain RL under v2 | |

## Presets

The presets are the same instances as v1 (`python run.py --list`), so v1 and v2 numbers are always on identical
instances. Example:

```
python run.py --variant v2_objectives --preset off_c_15 --method EDF
python run.py --variant v2_objectives --preset on_r_50 --method ATC --objectives tardiness,late_count
```

For heuristics, PSO and CP-SAT, shaping is off, so reward = −J / (number of jobs) exactly, and `objective_J` is
reported with every run.
