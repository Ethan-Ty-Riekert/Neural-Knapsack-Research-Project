# off_tf02_w1: leaderboards

Top 10 methods on each metric. Full ranking by J: [`tables/off_tf02_w1.md`](../tables/off_tf02_w1.md). Archive overview: [`HIGHLIGHTS.md`](../HIGHLIGHTS.md).

## J (lower is better)

| rank | method | family | J |
|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 0 +/- 0 |
| 2 | WLST+BestFit | Heuristic | 0 +/- 0 |
| 3 | WLST+WorstFit | Heuristic | 0 +/- 0 |
| 4 | WLST+Consolidate | Heuristic | 0 +/- 0 |
| 5 | WEDF+FirstFit | Heuristic | 0 +/- 0 |
| 6 | WEDF+BestFit | Heuristic | 0 +/- 0 |
| 7 | WEDF+WorstFit | Heuristic | 0 +/- 0 |
| 8 | WEDF+Consolidate | Heuristic | 0 +/- 0 |
| 9 | MDC+FirstFit | Heuristic | 0 +/- 0 |
| 10 | MDC+BestFit | Heuristic | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_objective_J.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_objective_J.png) |

## on-time rate (higher is better)

| rank | method | family | on-time rate | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 2 | WLST+BestFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 3 | WLST+WorstFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 4 | WLST+Consolidate | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 5 | WEDF+FirstFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 6 | WEDF+BestFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 7 | WEDF+WorstFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 8 | WEDF+Consolidate | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 9 | MDC+FirstFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |
| 10 | MDC+BestFit | Heuristic | 1.000 +/- 0.000 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_on_time_rate.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_on_time_rate.png) |

## late jobs (lower is better)

| rank | method | family | late jobs | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 2 | WLST+BestFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 3 | WLST+WorstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 4 | WLST+Consolidate | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 5 | WEDF+FirstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 6 | WEDF+BestFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 7 | WEDF+WorstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 8 | WEDF+Consolidate | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 9 | MDC+FirstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 10 | MDC+BestFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_late_jobs.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_late_jobs.png) |

## weighted late jobs (lower is better)

| rank | method | family | weighted late jobs | J |
|---|---|---|---|---|
| 1 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.0 +/- 0.0 | 0 +/- 0 |
| 2 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 0.0 +/- 0.0 | 0 +/- 0 |
| 3 | PPO Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 0.0 +/- 0.0 | 0 +/- 0 |

| top 5 |
|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_weighted_late_jobs.png) |

## weighted tardiness (lower is better)

| rank | method | family | weighted tardiness | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 2 | WLST+BestFit | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 3 | WLST+WorstFit | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 4 | WLST+Consolidate | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 5 | WEDF+FirstFit | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 6 | WEDF+BestFit | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 7 | WEDF+WorstFit | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 8 | WEDF+Consolidate | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 9 | MDC+FirstFit | Heuristic | 0 +/- 0 | 0 +/- 0 |
| 10 | MDC+BestFit | Heuristic | 0 +/- 0 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_weighted_tardiness.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_weighted_tardiness.png) |

## max tardiness (lower is better)

| rank | method | family | max tardiness | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 2 | WLST+BestFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 3 | WLST+WorstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 4 | WLST+Consolidate | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 5 | WEDF+FirstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 6 | WEDF+BestFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 7 | WEDF+WorstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 8 | WEDF+Consolidate | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 9 | MDC+FirstFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |
| 10 | MDC+BestFit | Heuristic | 0.0 +/- 0.0 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_max_tardiness.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_max_tardiness.png) |

## p95 tardiness (lower is better)

Every method has the same value here, so there is no ranking.

## mean wait (lower is better)

| rank | method | family | mean wait | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 2 | WLST+BestFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 3 | WLST+WorstFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 4 | WLST+Consolidate | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 5 | WEDF+FirstFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 6 | WEDF+BestFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 7 | WEDF+WorstFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 8 | WEDF+Consolidate | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 9 | MDC+FirstFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |
| 10 | MDC+BestFit | Heuristic | 49.50 +/- 0.00 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_mean_wait.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_mean_wait.png) |

## mean flow time (lower is better)

| rank | method | family | mean flow time | J |
|---|---|---|---|---|
| 1 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 54.54 +/- 0.00 | 0 +/- 0 |
| 2 | WLST+FirstFit | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 3 | WLST+BestFit | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 4 | WLST+WorstFit | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 5 | WLST+Consolidate | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 6 | WEDF+FirstFit | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 7 | WEDF+BestFit | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 8 | WEDF+WorstFit | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 9 | WEDF+Consolidate | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |
| 10 | MDC+FirstFit | Heuristic | 54.54 +/- 0.27 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_mean_flow_time.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_mean_flow_time.png) |

## makespan (lower is better)

| rank | method | family | makespan | J |
|---|---|---|---|---|
| 1 | MDC+FirstFit | Heuristic | 102.3 +/- 1.4 | 0 +/- 0 |
| 2 | MDC+BestFit | Heuristic | 102.3 +/- 1.4 | 0 +/- 0 |
| 3 | MDC+WorstFit | Heuristic | 102.3 +/- 1.4 | 0 +/- 0 |
| 4 | MDC+Consolidate | Heuristic | 102.3 +/- 1.4 | 0 +/- 0 |
| 5 | WLST+FirstFit | Heuristic | 102.6 +/- 1.5 | 0 +/- 0 |
| 6 | WLST+BestFit | Heuristic | 102.6 +/- 1.5 | 0 +/- 0 |
| 7 | WLST+WorstFit | Heuristic | 102.6 +/- 1.5 | 0 +/- 0 |
| 8 | WLST+Consolidate | Heuristic | 102.6 +/- 1.5 | 0 +/- 0 |
| 9 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 102.6 +/- 0.0 | 0 +/- 0 |
| 10 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 103.0 +/- 0.6 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_makespan.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_makespan.png) |

## utilisation of active machines (higher is better)

| rank | method | family | utilisation of active machines | J |
|---|---|---|---|---|
| 1 | MDC+Consolidate | Heuristic | 0.650 +/- 0.021 | 0 +/- 0 |
| 2 | WLST+Consolidate | Heuristic | 0.644 +/- 0.018 | 0 +/- 0 |
| 3 | WEDF+Consolidate | Heuristic | 0.640 +/- 0.019 | 0 +/- 0 |
| 4 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 0.629 +/- 0.006 | 0 +/- 0 |
| 5 | MDC+BestFit | Heuristic | 0.629 +/- 0.021 | 0 +/- 0 |
| 6 | WLST+BestFit | Heuristic | 0.627 +/- 0.018 | 0 +/- 0 |
| 7 | MDC+FirstFit | Heuristic | 0.625 +/- 0.020 | 0 +/- 0 |
| 8 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.624 +/- 0.001 | 0 +/- 0 |
| 9 | WLST+FirstFit | Heuristic | 0.623 +/- 0.015 | 0 +/- 0 |
| 10 | PPO Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 0.623 +/- 0.003 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_mean_active_utilisation.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_mean_active_utilisation.png) |

## active machine-ticks (lower is better)

| rank | method | family | active machine-ticks | J |
|---|---|---|---|---|
| 1 | MDC+Consolidate | Heuristic | 166 +/- 9 | 0 +/- 0 |
| 2 | WLST+Consolidate | Heuristic | 167 +/- 10 | 0 +/- 0 |
| 3 | WEDF+Consolidate | Heuristic | 169 +/- 11 | 0 +/- 0 |
| 4 | MDC+BestFit | Heuristic | 172 +/- 10 | 0 +/- 0 |
| 5 | WLST+BestFit | Heuristic | 172 +/- 10 | 0 +/- 0 |
| 6 | MDC+FirstFit | Heuristic | 173 +/- 11 | 0 +/- 0 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 173 +/- 2 | 0 +/- 0 |
| 8 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 173 +/- 0 | 0 +/- 0 |
| 9 | WLST+FirstFit | Heuristic | 173 +/- 10 | 0 +/- 0 |
| 10 | WEDF+BestFit | Heuristic | 174 +/- 11 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_active_machine_ticks.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_active_machine_ticks.png) |

## energy (SPECpower) (lower is better)

| rank | method | family | energy (SPECpower) | J |
|---|---|---|---|---|
| 1 | MDC+Consolidate | Heuristic | 142 +/- 8 | 0 +/- 0 |
| 2 | WLST+Consolidate | Heuristic | 143 +/- 9 | 0 +/- 0 |
| 3 | WEDF+Consolidate | Heuristic | 144 +/- 10 | 0 +/- 0 |
| 4 | MDC+BestFit | Heuristic | 146 +/- 9 | 0 +/- 0 |
| 5 | WLST+BestFit | Heuristic | 146 +/- 9 | 0 +/- 0 |
| 6 | MDC+FirstFit | Heuristic | 147 +/- 9 | 0 +/- 0 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 147 +/- 1 | 0 +/- 0 |
| 8 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 147 +/- 0 | 0 +/- 0 |
| 9 | WLST+FirstFit | Heuristic | 147 +/- 9 | 0 +/- 0 |
| 10 | WEDF+BestFit | Heuristic | 147 +/- 10 | 0 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf02_w1/top5_energy_specpower.png) | ![top 10](../figures/leaderboards/off_tf02_w1/top10_energy_specpower.png) |

Spread (+/-): std across seeds for multi-seed RL rows, otherwise std across instances.
