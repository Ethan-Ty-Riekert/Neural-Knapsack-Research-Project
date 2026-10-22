# on_rho050: leaderboards

Top 10 methods on each metric. Full ranking by J: [`tables/on_rho050.md`](../tables/on_rho050.md). Archive overview: [`HIGHLIGHTS.md`](../HIGHLIGHTS.md).

## J (lower is better)

| rank | method | family | J |
|---|---|---|---|
| 1 | WLST+Consolidate | Heuristic | 4085 +/- 2414 |
| 2 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 4087 +/- 2 |
| 3 | LST+Consolidate | Heuristic | 4087 +/- 2413 |
| 4 | MDC+Consolidate | Heuristic | 4087 +/- 2413 |
| 5 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 4088 +/- 1 |
| 6 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 4089 +/- 1 |
| 7 | WEDF+Consolidate | Heuristic | 4089 +/- 2415 |
| 8 | WEDF+BestFit | Heuristic | 4090 +/- 2414 |
| 9 | ATC+Consolidate | Heuristic | 4090 +/- 2414 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 4090 +/- 3 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_objective_J.png) | ![top 10](../figures/leaderboards/on_rho050/top10_objective_J.png) |

## on-time rate (higher is better)

| rank | method | family | on-time rate | J |
|---|---|---|---|---|
| 1 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.978 +/- 0.000 | 4087 +/- 2 |
| 2 | WLST+Consolidate | Heuristic | 0.978 +/- 0.007 | 4085 +/- 2414 |
| 3 | LST+Consolidate | Heuristic | 0.978 +/- 0.007 | 4087 +/- 2413 |
| 4 | MDC+Consolidate | Heuristic | 0.978 +/- 0.007 | 4087 +/- 2413 |
| 5 | LST+BestFit | Heuristic | 0.978 +/- 0.007 | 4092 +/- 2414 |
| 6 | MDC+BestFit | Heuristic | 0.978 +/- 0.007 | 4092 +/- 2414 |
| 7 | WLST+BestFit | Heuristic | 0.978 +/- 0.007 | 4098 +/- 2416 |
| 8 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 0.978 +/- 0.000 | 4088 +/- 1 |
| 9 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.978 +/- 0.000 | 4089 +/- 1 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 0.978 +/- 0.000 | 4090 +/- 3 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_on_time_rate.png) | ![top 10](../figures/leaderboards/on_rho050/top10_on_time_rate.png) |

## late jobs (lower is better)

| rank | method | family | late jobs | J |
|---|---|---|---|---|
| 1 | WLST+Consolidate | Heuristic | 10.6 +/- 3.3 | 4085 +/- 2414 |
| 2 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 10.6 +/- 0.0 | 4087 +/- 2 |
| 3 | LST+Consolidate | Heuristic | 10.6 +/- 3.3 | 4087 +/- 2413 |
| 4 | MDC+Consolidate | Heuristic | 10.6 +/- 3.3 | 4087 +/- 2413 |
| 5 | LST+BestFit | Heuristic | 10.6 +/- 3.3 | 4092 +/- 2414 |
| 6 | MDC+BestFit | Heuristic | 10.6 +/- 3.3 | 4092 +/- 2414 |
| 7 | WLST+BestFit | Heuristic | 10.6 +/- 3.3 | 4098 +/- 2416 |
| 8 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 10.6 +/- 0.0 | 4088 +/- 1 |
| 9 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 10.7 +/- 0.0 | 4089 +/- 1 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 10.7 +/- 0.0 | 4090 +/- 3 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_late_jobs.png) | ![top 10](../figures/leaderboards/on_rho050/top10_late_jobs.png) |

## weighted late jobs (lower is better)

| rank | method | family | weighted late jobs | J |
|---|---|---|---|---|
| 1 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 32.1 +/- 0.0 | 4087 +/- 2 |
| 2 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 32.1 +/- 0.0 | 4088 +/- 1 |
| 3 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 32.1 +/- 0.0 | 4090 +/- 3 |
| 4 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 32.1 +/- 0.0 | 4090 +/- 3 |
| 5 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 32.2 +/- 0.0 | 4089 +/- 1 |
| 6 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 32.2 +/- 0.0 | 4091 +/- 1 |

| top 5 |
|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_weighted_late_jobs.png) |

## weighted tardiness (lower is better)

| rank | method | family | weighted tardiness | J |
|---|---|---|---|---|
| 1 | WLST+Consolidate | Heuristic | 278 +/- 115 | 4085 +/- 2414 |
| 2 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 278 +/- 0 | 4087 +/- 2 |
| 3 | LST+Consolidate | Heuristic | 278 +/- 115 | 4087 +/- 2413 |
| 4 | MDC+Consolidate | Heuristic | 278 +/- 115 | 4087 +/- 2413 |
| 5 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 278 +/- 0 | 4088 +/- 1 |
| 6 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 278 +/- 0 | 4089 +/- 1 |
| 7 | WEDF+Consolidate | Heuristic | 278 +/- 115 | 4089 +/- 2415 |
| 8 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 278 +/- 0 | 4090 +/- 3 |
| 9 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 278 +/- 0 | 4090 +/- 3 |
| 10 | ATC+Consolidate | Heuristic | 278 +/- 115 | 4090 +/- 2414 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_weighted_tardiness.png) | ![top 10](../figures/leaderboards/on_rho050/top10_weighted_tardiness.png) |

## max tardiness (lower is better)

| rank | method | family | max tardiness | J |
|---|---|---|---|---|
| 1 | WLST+Consolidate | Heuristic | 21.9 +/- 6.4 | 4085 +/- 2414 |
| 2 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 21.9 +/- 0.0 | 4087 +/- 2 |
| 3 | LST+Consolidate | Heuristic | 21.9 +/- 6.4 | 4087 +/- 2413 |
| 4 | MDC+Consolidate | Heuristic | 21.9 +/- 6.4 | 4087 +/- 2413 |
| 5 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 21.9 +/- 0.0 | 4088 +/- 1 |
| 6 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 21.9 +/- 0.0 | 4089 +/- 1 |
| 7 | WEDF+Consolidate | Heuristic | 21.9 +/- 6.4 | 4089 +/- 2415 |
| 8 | WEDF+BestFit | Heuristic | 21.9 +/- 6.4 | 4090 +/- 2414 |
| 9 | ATC+Consolidate | Heuristic | 21.9 +/- 6.4 | 4090 +/- 2414 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 21.9 +/- 0.0 | 4090 +/- 3 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_max_tardiness.png) | ![top 10](../figures/leaderboards/on_rho050/top10_max_tardiness.png) |

## p95 tardiness (lower is better)

| rank | method | family | p95 tardiness | J |
|---|---|---|---|---|
| 1 | WLST+Consolidate | Heuristic | 0.0 +/- 0.0 | 4085 +/- 2414 |
| 2 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.0 +/- 0.0 | 4087 +/- 2 |
| 3 | LST+Consolidate | Heuristic | 0.0 +/- 0.0 | 4087 +/- 2413 |
| 4 | MDC+Consolidate | Heuristic | 0.0 +/- 0.0 | 4087 +/- 2413 |
| 5 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 0.0 +/- 0.0 | 4088 +/- 1 |
| 6 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.0 +/- 0.0 | 4089 +/- 1 |
| 7 | WEDF+Consolidate | Heuristic | 0.0 +/- 0.0 | 4089 +/- 2415 |
| 8 | WEDF+BestFit | Heuristic | 0.0 +/- 0.0 | 4090 +/- 2414 |
| 9 | ATC+Consolidate | Heuristic | 0.0 +/- 0.0 | 4090 +/- 2414 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 0.0 +/- 0.0 | 4090 +/- 3 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_p95_tardiness.png) | ![top 10](../figures/leaderboards/on_rho050/top10_p95_tardiness.png) |

## mean wait (lower is better)

| rank | method | family | mean wait | J |
|---|---|---|---|---|
| 1 | SPT+Consolidate | Heuristic | 0.03 +/- 0.04 | 4137 +/- 2423 |
| 2 | FCFS+Consolidate | Heuristic | 0.03 +/- 0.04 | 4099 +/- 2420 |
| 3 | MDC+Consolidate | Heuristic | 0.03 +/- 0.05 | 4087 +/- 2413 |
| 4 | ATC+Consolidate | Heuristic | 0.03 +/- 0.05 | 4090 +/- 2414 |
| 5 | LST+Consolidate | Heuristic | 0.03 +/- 0.05 | 4087 +/- 2413 |
| 6 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.03 +/- 0.00 | 4089 +/- 1 |
| 7 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 0.03 +/- 0.00 | 4091 +/- 1 |
| 8 | WSPT+Consolidate | Heuristic | 0.03 +/- 0.05 | 4137 +/- 2441 |
| 9 | WEDF+Consolidate | Heuristic | 0.03 +/- 0.05 | 4089 +/- 2415 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 0.03 +/- 0.00 | 4090 +/- 3 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_mean_wait.png) | ![top 10](../figures/leaderboards/on_rho050/top10_mean_wait.png) |

## mean flow time (lower is better)

| rank | method | family | mean flow time | J |
|---|---|---|---|---|
| 1 | SPT+Consolidate | Heuristic | 5.50 +/- 0.30 | 4137 +/- 2423 |
| 2 | FCFS+Consolidate | Heuristic | 5.50 +/- 0.30 | 4099 +/- 2420 |
| 3 | MDC+Consolidate | Heuristic | 5.51 +/- 0.30 | 4087 +/- 2413 |
| 4 | ATC+Consolidate | Heuristic | 5.51 +/- 0.29 | 4090 +/- 2414 |
| 5 | LST+Consolidate | Heuristic | 5.51 +/- 0.30 | 4087 +/- 2413 |
| 6 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 5.51 +/- 0.00 | 4089 +/- 1 |
| 7 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 5.51 +/- 0.00 | 4091 +/- 1 |
| 8 | WSPT+Consolidate | Heuristic | 5.51 +/- 0.30 | 4137 +/- 2441 |
| 9 | WEDF+Consolidate | Heuristic | 5.51 +/- 0.30 | 4089 +/- 2415 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 5.51 +/- 0.00 | 4090 +/- 3 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_mean_flow_time.png) | ![top 10](../figures/leaderboards/on_rho050/top10_mean_flow_time.png) |

## makespan (lower is better)

| rank | method | family | makespan | J |
|---|---|---|---|---|
| 1 | WLST+Consolidate | Heuristic | 127.0 +/- 7.9 | 4085 +/- 2414 |
| 2 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 127.0 +/- 0.0 | 4087 +/- 2 |
| 3 | LST+Consolidate | Heuristic | 127.0 +/- 7.9 | 4087 +/- 2413 |
| 4 | MDC+Consolidate | Heuristic | 127.0 +/- 7.9 | 4087 +/- 2413 |
| 5 | WEDF+Consolidate | Heuristic | 127.0 +/- 7.9 | 4089 +/- 2415 |
| 6 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 127.0 +/- 0.0 | 4089 +/- 1 |
| 7 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 127.0 +/- 0.0 | 4088 +/- 1 |
| 8 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 127.0 +/- 0.0 | 4090 +/- 3 |
| 9 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 127.0 +/- 0.0 | 4090 +/- 3 |
| 10 | WEDF+BestFit | Heuristic | 127.0 +/- 7.8 | 4090 +/- 2414 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_makespan.png) | ![top 10](../figures/leaderboards/on_rho050/top10_makespan.png) |

## utilisation of active machines (higher is better)

| rank | method | family | utilisation of active machines | J |
|---|---|---|---|---|
| 1 | LPT+Consolidate | Heuristic | 0.860 +/- 0.018 | 4091 +/- 2412 |
| 2 | LST+Consolidate | Heuristic | 0.854 +/- 0.017 | 4087 +/- 2413 |
| 3 | MDC+Consolidate | Heuristic | 0.854 +/- 0.017 | 4087 +/- 2413 |
| 4 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.853 +/- 0.001 | 4089 +/- 1 |
| 5 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 0.853 +/- 0.000 | 4090 +/- 3 |
| 6 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.853 +/- 0.000 | 4090 +/- 3 |
| 7 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.852 +/- 0.002 | 4087 +/- 2 |
| 8 | WLST+Consolidate | Heuristic | 0.851 +/- 0.018 | 4085 +/- 2414 |
| 9 | EDF+Consolidate | Heuristic | 0.849 +/- 0.019 | 4097 +/- 2411 |
| 10 | FCFS+Consolidate | Heuristic | 0.849 +/- 0.018 | 4099 +/- 2420 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_mean_active_utilisation.png) | ![top 10](../figures/leaderboards/on_rho050/top10_mean_active_utilisation.png) |

## active machine-ticks (lower is better)

| rank | method | family | active machine-ticks | J |
|---|---|---|---|---|
| 1 | LPT+Consolidate | Heuristic | 815 +/- 60 | 4091 +/- 2412 |
| 2 | LST+Consolidate | Heuristic | 823 +/- 58 | 4087 +/- 2413 |
| 3 | MDC+Consolidate | Heuristic | 823 +/- 58 | 4087 +/- 2413 |
| 4 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 824 +/- 1 | 4089 +/- 1 |
| 5 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 825 +/- 2 | 4087 +/- 2 |
| 6 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 826 +/- 0 | 4090 +/- 3 |
| 7 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 826 +/- 0 | 4090 +/- 3 |
| 8 | WLST+Consolidate | Heuristic | 827 +/- 61 | 4085 +/- 2414 |
| 9 | FCFS+Consolidate | Heuristic | 829 +/- 61 | 4099 +/- 2420 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 831 +/- 1 | 4088 +/- 1 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_active_machine_ticks.png) | ![top 10](../figures/leaderboards/on_rho050/top10_active_machine_ticks.png) |

## energy (SPECpower) (lower is better)

| rank | method | family | energy (SPECpower) | J |
|---|---|---|---|---|
| 1 | LPT+Consolidate | Heuristic | 708 +/- 52 | 4091 +/- 2412 |
| 2 | LST+Consolidate | Heuristic | 713 +/- 51 | 4087 +/- 2413 |
| 3 | MDC+Consolidate | Heuristic | 713 +/- 51 | 4087 +/- 2413 |
| 4 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 714 +/- 0 | 4089 +/- 1 |
| 5 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 714 +/- 2 | 4087 +/- 2 |
| 6 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert | RL | 715 +/- 0 | 4090 +/- 3 |
| 7 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 715 +/- 0 | 4090 +/- 3 |
| 8 | WLST+Consolidate | Heuristic | 716 +/- 53 | 4085 +/- 2414 |
| 9 | FCFS+Consolidate | Heuristic | 717 +/- 53 | 4099 +/- 2420 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 719 +/- 0 | 4088 +/- 1 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho050/top5_energy_specpower.png) | ![top 10](../figures/leaderboards/on_rho050/top10_energy_specpower.png) |

Spread (+/-): std across seeds for multi-seed RL rows, otherwise std across instances.
