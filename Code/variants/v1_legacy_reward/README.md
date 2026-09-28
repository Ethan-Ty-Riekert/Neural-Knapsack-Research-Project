# Variant v1: legacy reward

**Status:** frozen. It covers everything produced from 2026-07-24 to 2026-09-28, and it must keep
reproducing the recorded results.
**Results:** [`Results/v1_legacy_reward/`](../../../Results/v1_legacy_reward/)

## Problem

- **Offline:** every job is known at t = 0. One placement per tick (a tick is a decision point, not
  wall-clock time), so time advances after every placement. A job can start at t only if t + P_j <= H
  and every resource fits.
- **Online:** Poisson arrivals, several placements per tick, and time advances only on idle.
- Instances come from `Code/core/env_config.py::generate_env_config` (offline) and
  `Code/core/arrival_process.py::generate_poisson_arrivals` (online).

## Reward

`Code/core/scheduling_env.py`, `reward_mode="legacy"`, with lambda_1 = lambda_2 = lambda_3 = 1:

| Event | Reward |
|---|---|
| Valid placement | +3 − lambda_2 · w_j · T_j / H − lambda_3 · Δθ − lambda_1 (first use of the machine) |
| Last job placed | +50 |
| Idle tick | −0.5 |
| Invalid action | −5 |

Some runs in this era used `reward_mode="dense_tardiness"` (2026-09-17). It charges tardiness per tick
instead and drops the +3/+50 bonuses. Those runs are labelled as such in the training log.

**Known defects** (why v2 exists): the +3/+50 bonuses outweigh the bounded tardiness term, dropped jobs
are nearly free, and the activation, hotspot and idle terms don't correspond to any stated objective. See
`Future/research/2026-09-28-objective-redesign-discussion.md` sections 1-2.

## Presets (`python run.py --list`)

The names match the protocol keys in `Results/v1_legacy_reward/ALL_RESULTS_*/scripts/results_data.py`.

| Preset | Instances | Notes |
|---|---|---|
| `off_c_fixed` | seed 0 | 100 jobs / 10 machines / H = 100 |
| `off_c_15` | seeds 500000-014 | PSO study |
| `off_c_50` | seeds 500000-049 | official offline generalisation protocol |
| `off_c_small` | seeds 500000-004 | 10 jobs / 3 machines / H = 15 (CP-SAT) |
| `off_c_rb2` | seeds 500000-007 | 45 jobs / 6 machines / H = 50 |
| `off_r_fixed`, `off_r_50` | as above | weights w_j in {1..5} |
| `on_r_20`, `on_r_50` | seeds 500000-... | rate 9 (rho ~0.75), log-normal sizes, H = 100, weights {1..5} |

**Watch the horizon:** `generate_env_config`'s default is H = 110, but every 100-job protocol uses
H = 100. At H = 110 EDF fits every job in, and the reward jumps from 286 to 346.

## Reproduction checks (2026-09-28)

| Command | Expected |
|---|---|
| `run.py --preset off_c_15 --method EDF` | reward 286.00, tardiness 50.33, late 14.20, scheduled 97.2 |
| `run.py --preset off_c_small --method cpsat` | reward 74.31, tardiness 0 (all 5 optimal) |
