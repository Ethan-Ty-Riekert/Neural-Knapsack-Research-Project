# off_tf08_w1: leaderboards

Top 10 methods on each metric. Full ranking by J: [`tables/off_tf08_w1.md`](../tables/off_tf08_w1.md). Archive overview: [`HIGHLIGHTS.md`](../HIGHLIGHTS.md).

## J (lower is better)

| rank | method | family | J |
|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 129805 +/- 12950 |
| 2 | WLST+BestFit | Heuristic | 129805 +/- 12950 |
| 3 | WLST+WorstFit | Heuristic | 129805 +/- 12950 |
| 4 | WLST+Consolidate | Heuristic | 129805 +/- 12950 |
| 5 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 129805 +/- 0 |
| 6 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 129805 +/- 0 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 130583 +/- 119 |
| 8 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 130671 +/- 0 |
| 9 | WEDF+FirstFit | Heuristic | 130919 +/- 12957 |
| 10 | WEDF+BestFit | Heuristic | 130919 +/- 12957 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_objective_J.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_objective_J.png) |

## on-time rate (higher is better)

| rank | method | family | on-time rate | J |
|---|---|---|---|---|
| 1 | ATC+FirstFit | Heuristic | 0.243 +/- 0.039 | 210328 +/- 14586 |
| 2 | ATC+BestFit | Heuristic | 0.243 +/- 0.039 | 210328 +/- 14586 |
| 3 | ATC+Consolidate | Heuristic | 0.243 +/- 0.039 | 210328 +/- 14586 |
| 4 | PPO Opt3 ATC-prior score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 0.029 +/- 0.001 | 173184 +/- 458 |
| 5 | PPO Opt2 priority score windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.023 +/- 0.000 | 168539 +/- 1 |
| 6 | PPO Opt2 priority score windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.017 +/- 0.000 | 152000 +/- 19 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 0.011 +/- 0.001 | 130583 +/- 119 |
| 8 | WEDF+FirstFit | Heuristic | 0.010 +/- 0.001 | 130919 +/- 12957 |
| 9 | WEDF+BestFit | Heuristic | 0.010 +/- 0.001 | 130919 +/- 12957 |
| 10 | WEDF+WorstFit | Heuristic | 0.010 +/- 0.001 | 130919 +/- 12957 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_on_time_rate.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_on_time_rate.png) |

## late jobs (lower is better)

| rank | method | family | late jobs | J |
|---|---|---|---|---|
| 1 | ATC+FirstFit | Heuristic | 75.7 +/- 3.9 | 210328 +/- 14586 |
| 2 | ATC+BestFit | Heuristic | 75.7 +/- 3.9 | 210328 +/- 14586 |
| 3 | ATC+Consolidate | Heuristic | 75.7 +/- 3.9 | 210328 +/- 14586 |
| 4 | PPO Opt3 ATC-prior score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 97.1 +/- 0.1 | 173184 +/- 458 |
| 5 | PPO Opt2 priority score windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 97.7 +/- 0.0 | 168539 +/- 1 |
| 6 | PPO Opt2 priority score windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 98.3 +/- 0.0 | 152000 +/- 19 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 98.9 +/- 0.1 | 130583 +/- 119 |
| 8 | WEDF+FirstFit | Heuristic | 99.0 +/- 0.1 | 130919 +/- 12957 |
| 9 | WEDF+BestFit | Heuristic | 99.0 +/- 0.1 | 130919 +/- 12957 |
| 10 | WEDF+WorstFit | Heuristic | 99.0 +/- 0.1 | 130919 +/- 12957 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_late_jobs.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_late_jobs.png) |

## weighted late jobs (lower is better)

| rank | method | family | weighted late jobs | J |
|---|---|---|---|---|
| 1 | PPO Opt3 ATC-prior score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 97.1 +/- 0.1 | 173184 +/- 458 |
| 2 | PPO Opt2 priority score windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 97.7 +/- 0.0 | 168539 +/- 1 |
| 3 | PPO Opt2 priority score windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 98.3 +/- 0.0 | 152000 +/- 19 |
| 4 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 98.9 +/- 0.1 | 130583 +/- 119 |
| 5 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 99.0 +/- 0.0 | 129805 +/- 0 |
| 6 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 99.0 +/- 0.0 | 129805 +/- 0 |
| 7 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 99.0 +/- 0.0 | 130671 +/- 0 |

| top 5 |
|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_weighted_late_jobs.png) |

## weighted tardiness (lower is better)

| rank | method | family | weighted tardiness | J |
|---|---|---|---|---|
| 1 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 3283 +/- 0 | 129805 +/- 0 |
| 2 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 3283 +/- 0 | 129805 +/- 0 |
| 3 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 3283 +/- 0 | 130671 +/- 0 |
| 4 | WLST+FirstFit | Heuristic | 3283 +/- 171 | 129805 +/- 12950 |
| 5 | WLST+BestFit | Heuristic | 3283 +/- 171 | 129805 +/- 12950 |
| 6 | WLST+WorstFit | Heuristic | 3283 +/- 171 | 129805 +/- 12950 |
| 7 | WLST+Consolidate | Heuristic | 3283 +/- 171 | 129805 +/- 12950 |
| 8 | MDC+FirstFit | Heuristic | 3283 +/- 171 | 161279 +/- 15663 |
| 9 | MDC+BestFit | Heuristic | 3283 +/- 171 | 161279 +/- 15663 |
| 10 | MDC+WorstFit | Heuristic | 3283 +/- 171 | 161279 +/- 15663 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_weighted_tardiness.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_weighted_tardiness.png) |

## max tardiness (lower is better)

| rank | method | family | max tardiness | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 53.9 +/- 1.5 | 129805 +/- 12950 |
| 2 | WLST+BestFit | Heuristic | 53.9 +/- 1.5 | 129805 +/- 12950 |
| 3 | WLST+WorstFit | Heuristic | 53.9 +/- 1.5 | 129805 +/- 12950 |
| 4 | WLST+Consolidate | Heuristic | 53.9 +/- 1.5 | 129805 +/- 12950 |
| 5 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 53.9 +/- 0.0 | 129805 +/- 0 |
| 6 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 53.9 +/- 0.0 | 129805 +/- 0 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 55.9 +/- 0.6 | 130583 +/- 119 |
| 8 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 56.5 +/- 0.0 | 130671 +/- 0 |
| 9 | WEDF+FirstFit | Heuristic | 57.1 +/- 1.3 | 130919 +/- 12957 |
| 10 | WEDF+BestFit | Heuristic | 57.1 +/- 1.3 | 130919 +/- 12957 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_max_tardiness.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_max_tardiness.png) |

## p95 tardiness (lower is better)

| rank | method | family | p95 tardiness | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 52.3 +/- 1.5 | 129805 +/- 12950 |
| 2 | WLST+BestFit | Heuristic | 52.3 +/- 1.5 | 129805 +/- 12950 |
| 3 | WLST+WorstFit | Heuristic | 52.3 +/- 1.5 | 129805 +/- 12950 |
| 4 | WLST+Consolidate | Heuristic | 52.3 +/- 1.5 | 129805 +/- 12950 |
| 5 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 52.3 +/- 0.0 | 129805 +/- 0 |
| 6 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 52.3 +/- 0.0 | 129805 +/- 0 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 52.6 +/- 0.1 | 130583 +/- 119 |
| 8 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 53.0 +/- 0.0 | 130671 +/- 0 |
| 9 | WEDF+FirstFit | Heuristic | 53.2 +/- 1.7 | 130919 +/- 12957 |
| 10 | WEDF+BestFit | Heuristic | 53.2 +/- 1.7 | 130919 +/- 12957 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_p95_tardiness.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_p95_tardiness.png) |

## mean wait (lower is better)

| rank | method | family | mean wait | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 49.50 +/- 0.00 | 129805 +/- 12950 |
| 2 | WLST+BestFit | Heuristic | 49.50 +/- 0.00 | 129805 +/- 12950 |
| 3 | WLST+WorstFit | Heuristic | 49.50 +/- 0.00 | 129805 +/- 12950 |
| 4 | WLST+Consolidate | Heuristic | 49.50 +/- 0.00 | 129805 +/- 12950 |
| 5 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 49.50 +/- 0.00 | 129805 +/- 0 |
| 6 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 49.50 +/- 0.00 | 129805 +/- 0 |
| 7 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 49.50 +/- 0.00 | 130671 +/- 0 |
| 8 | WEDF+FirstFit | Heuristic | 49.50 +/- 0.00 | 130919 +/- 12957 |
| 9 | WEDF+BestFit | Heuristic | 49.50 +/- 0.00 | 130919 +/- 12957 |
| 10 | WEDF+WorstFit | Heuristic | 49.50 +/- 0.00 | 130919 +/- 12957 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_mean_wait.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_mean_wait.png) |

## mean flow time (lower is better)

| rank | method | family | mean flow time | J |
|---|---|---|---|---|
| 1 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 54.54 +/- 0.00 | 129805 +/- 0 |
| 2 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 54.54 +/- 0.00 | 129805 +/- 0 |
| 3 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 54.54 +/- 0.00 | 130671 +/- 0 |
| 4 | PPO Opt2 priority score windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 54.54 +/- 0.00 | 152000 +/- 19 |
| 5 | PPO Opt2 priority score windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 54.54 +/- 0.00 | 168539 +/- 1 |
| 6 | PPO Opt3 ATC-prior score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 54.54 +/- 0.00 | 173184 +/- 458 |
| 7 | WLST+FirstFit | Heuristic | 54.54 +/- 0.27 | 129805 +/- 12950 |
| 8 | WLST+BestFit | Heuristic | 54.54 +/- 0.27 | 129805 +/- 12950 |
| 9 | WLST+WorstFit | Heuristic | 54.54 +/- 0.27 | 129805 +/- 12950 |
| 10 | WLST+Consolidate | Heuristic | 54.54 +/- 0.27 | 129805 +/- 12950 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_mean_flow_time.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_mean_flow_time.png) |

## makespan (lower is better)

| rank | method | family | makespan | J |
|---|---|---|---|---|
| 1 | WLST+FirstFit | Heuristic | 102.6 +/- 1.5 | 129805 +/- 12950 |
| 2 | WLST+BestFit | Heuristic | 102.6 +/- 1.5 | 129805 +/- 12950 |
| 3 | WLST+WorstFit | Heuristic | 102.6 +/- 1.5 | 129805 +/- 12950 |
| 4 | WLST+Consolidate | Heuristic | 102.6 +/- 1.5 | 129805 +/- 12950 |
| 5 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 102.6 +/- 0.0 | 129805 +/- 0 |
| 6 | PPO Opt2 priority score windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 102.6 +/- 0.0 | 129805 +/- 0 |
| 7 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget | RL | 104.4 +/- 1.0 | 130583 +/- 119 |
| 8 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 105.0 +/- 0.0 | 130671 +/- 0 |
| 9 | WEDF+FirstFit | Heuristic | 105.6 +/- 1.3 | 130919 +/- 12957 |
| 10 | WEDF+BestFit | Heuristic | 105.6 +/- 1.3 | 130919 +/- 12957 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_makespan.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_makespan.png) |

## utilisation of active machines (higher is better)

| rank | method | family | utilisation of active machines | J |
|---|---|---|---|---|
| 1 | MDC+Consolidate | Heuristic | 0.654 +/- 0.021 | 161279 +/- 15663 |
| 2 | ATC+Consolidate | Heuristic | 0.650 +/- 0.025 | 210328 +/- 14586 |
| 3 | WLST+Consolidate | Heuristic | 0.645 +/- 0.021 | 129805 +/- 12950 |
| 4 | WEDF+Consolidate | Heuristic | 0.640 +/- 0.021 | 130919 +/- 12957 |
| 5 | MDC+FirstFit | Heuristic | 0.636 +/- 0.022 | 161279 +/- 15663 |
| 6 | WLST+BestFit | Heuristic | 0.634 +/- 0.019 | 129805 +/- 12950 |
| 7 | MDC+BestFit | Heuristic | 0.633 +/- 0.023 | 161279 +/- 15663 |
| 8 | WLST+FirstFit | Heuristic | 0.631 +/- 0.019 | 129805 +/- 12950 |
| 9 | ATC+FirstFit | Heuristic | 0.630 +/- 0.024 | 210328 +/- 14586 |
| 10 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 0.630 +/- 0.000 | 129805 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_mean_active_utilisation.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_mean_active_utilisation.png) |

## active machine-ticks (lower is better)

| rank | method | family | active machine-ticks | J |
|---|---|---|---|---|
| 1 | MDC+Consolidate | Heuristic | 166 +/- 9 | 161279 +/- 15663 |
| 2 | ATC+Consolidate | Heuristic | 166 +/- 7 | 210328 +/- 14586 |
| 3 | WLST+Consolidate | Heuristic | 167 +/- 10 | 129805 +/- 12950 |
| 4 | WEDF+Consolidate | Heuristic | 169 +/- 11 | 130919 +/- 12957 |
| 5 | WLST+BestFit | Heuristic | 170 +/- 10 | 129805 +/- 12950 |
| 6 | MDC+FirstFit | Heuristic | 171 +/- 10 | 161279 +/- 15663 |
| 7 | WLST+FirstFit | Heuristic | 171 +/- 10 | 129805 +/- 12950 |
| 8 | MDC+BestFit | Heuristic | 171 +/- 10 | 161279 +/- 15663 |
| 9 | ATC+FirstFit | Heuristic | 171 +/- 10 | 210328 +/- 14586 |
| 10 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 172 +/- 0 | 129805 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_active_machine_ticks.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_active_machine_ticks.png) |

## energy (SPECpower) (lower is better)

| rank | method | family | energy (SPECpower) | J |
|---|---|---|---|---|
| 1 | MDC+Consolidate | Heuristic | 142 +/- 8 | 161279 +/- 15663 |
| 2 | ATC+Consolidate | Heuristic | 142 +/- 7 | 210328 +/- 14586 |
| 3 | WLST+Consolidate | Heuristic | 143 +/- 9 | 129805 +/- 12950 |
| 4 | WEDF+Consolidate | Heuristic | 144 +/- 9 | 130919 +/- 12957 |
| 5 | WLST+BestFit | Heuristic | 145 +/- 9 | 129805 +/- 12950 |
| 6 | MDC+FirstFit | Heuristic | 145 +/- 9 | 161279 +/- 15663 |
| 7 | WLST+FirstFit | Heuristic | 145 +/- 9 | 129805 +/- 12950 |
| 8 | MDC+BestFit | Heuristic | 145 +/- 9 | 161279 +/- 15663 |
| 9 | ATC+FirstFit | Heuristic | 146 +/- 8 | 210328 +/- 14586 |
| 10 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 146 +/- 0 | 129805 +/- 0 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/off_tf08_w1/top5_energy_specpower.png) | ![top 10](../figures/leaderboards/off_tf08_w1/top10_energy_specpower.png) |

Spread (+/-): std across seeds for multi-seed RL rows, otherwise std across instances.
