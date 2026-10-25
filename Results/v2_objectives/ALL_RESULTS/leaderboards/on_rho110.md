# on_rho110: leaderboards

Top 10 methods on each metric. Full ranking by J: [`tables/on_rho110.md`](../tables/on_rho110.md). Archive overview: [`HIGHLIGHTS.md`](../HIGHLIGHTS.md).

## J (lower is better)

| rank | method | family | J |
|---|---|---|---|
| 1 | MDC+Consolidate | Heuristic | 106463 +/- 87144 |
| 2 | WEDF+Consolidate | Heuristic | 107519 +/- 73473 |
| 3 | EDF+Consolidate | Heuristic | 114142 +/- 68444 |
| 4 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 114316 +/- 2425 |
| 5 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle MDC expert | RL | 115590 +/- 5612 |
| 6 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 116522 +/- 7803 |
| 7 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget EDFL expert | RL | 118619 +/- 10202 |
| 8 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 122087 +/- 7505 |
| 9 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 127456 +/- 6187 |
| 10 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget MDC expert | RL | 131174 +/- 22367 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_objective_J.png) | ![top 10](../figures/leaderboards/on_rho110/top10_objective_J.png) |

## on-time rate (higher is better)

| rank | method | family | on-time rate | J |
|---|---|---|---|---|
| 1 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 0.935 +/- 0.001 | 158739 +/- 5231 |
| 2 | ATC+Consolidate | Heuristic | 0.935 +/- 0.013 | 159315 +/- 88420 |
| 3 | ATC+BestFit | Heuristic | 0.933 +/- 0.012 | 191577 +/- 108044 |
| 4 | WMDD+Consolidate | Heuristic | 0.930 +/- 0.015 | 141071 +/- 71452 |
| 5 | ATC+FirstFit | Heuristic | 0.930 +/- 0.012 | 205642 +/- 98055 |
| 6 | WMDD+BestFit | Heuristic | 0.928 +/- 0.014 | 173914 +/- 93418 |
| 7 | WMDD+FirstFit | Heuristic | 0.925 +/- 0.013 | 182580 +/- 97529 |
| 8 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 0.924 +/- 0.001 | 127456 +/- 6187 |
| 9 | COVERT+Consolidate | Heuristic | 0.924 +/- 0.016 | 150616 +/- 88508 |
| 10 | COVERT+BestFit | Heuristic | 0.924 +/- 0.014 | 187862 +/- 102012 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_on_time_rate.png) | ![top 10](../figures/leaderboards/on_rho110/top10_on_time_rate.png) |

## late jobs (lower is better)

| rank | method | family | late jobs | J |
|---|---|---|---|---|
| 1 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 69.0 +/- 1.4 | 158739 +/- 5231 |
| 2 | ATC+Consolidate | Heuristic | 69.4 +/- 13.5 | 159315 +/- 88420 |
| 3 | ATC+BestFit | Heuristic | 70.9 +/- 13.5 | 191577 +/- 108044 |
| 4 | WMDD+Consolidate | Heuristic | 74.2 +/- 15.9 | 141071 +/- 71452 |
| 5 | ATC+FirstFit | Heuristic | 74.5 +/- 13.6 | 205642 +/- 98055 |
| 6 | WMDD+BestFit | Heuristic | 76.9 +/- 15.1 | 173914 +/- 93418 |
| 7 | WMDD+FirstFit | Heuristic | 79.0 +/- 14.2 | 182580 +/- 97529 |
| 8 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 80.3 +/- 0.6 | 127456 +/- 6187 |
| 9 | COVERT+Consolidate | Heuristic | 80.5 +/- 17.9 | 150616 +/- 88508 |
| 10 | COVERT+BestFit | Heuristic | 80.7 +/- 16.0 | 187862 +/- 102012 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_late_jobs.png) | ![top 10](../figures/leaderboards/on_rho110/top10_late_jobs.png) |

## weighted late jobs (lower is better)

| rank | method | family | weighted late jobs | J |
|---|---|---|---|---|
| 1 | WMDD+Consolidate | Heuristic | 181.0 +/- 45.0 | 141071 +/- 71452 |
| 2 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 181.7 +/- 3.5 | 158739 +/- 5231 |
| 3 | ATC+Consolidate | Heuristic | 183.8 +/- 38.6 | 159315 +/- 88420 |
| 4 | ATC+BestFit | Heuristic | 189.3 +/- 41.2 | 191577 +/- 108044 |
| 5 | WMDD+BestFit | Heuristic | 193.9 +/- 42.1 | 173914 +/- 93418 |
| 6 | ATC+FirstFit | Heuristic | 198.9 +/- 40.6 | 205642 +/- 98055 |
| 7 | WMDD+FirstFit | Heuristic | 201.9 +/- 42.3 | 182580 +/- 97529 |
| 8 | WSPT+Consolidate | Heuristic | 223.9 +/- 41.5 | 339583 +/- 134907 |
| 9 | COVERT+Consolidate | Heuristic | 226.2 +/- 53.6 | 150616 +/- 88508 |
| 10 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 228.1 +/- 2.0 | 127456 +/- 6187 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_weighted_late_jobs.png) | ![top 10](../figures/leaderboards/on_rho110/top10_weighted_late_jobs.png) |

## weighted tardiness (lower is better)

| rank | method | family | weighted tardiness | J |
|---|---|---|---|---|
| 1 | WMDD+Consolidate | Heuristic | 3245 +/- 1254 | 141071 +/- 71452 |
| 2 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 3571 +/- 114 | 158739 +/- 5231 |
| 3 | ATC+Consolidate | Heuristic | 3604 +/- 1430 | 159315 +/- 88420 |
| 4 | MDC+Consolidate | Heuristic | 3763 +/- 1913 | 106463 +/- 87144 |
| 5 | WMDD+BestFit | Heuristic | 3824 +/- 1456 | 173914 +/- 93418 |
| 6 | COVERT+Consolidate | Heuristic | 3867 +/- 1565 | 150616 +/- 88508 |
| 7 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 3874 +/- 89 | 127456 +/- 6187 |
| 8 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 4018 +/- 93 | 114316 +/- 2425 |
| 9 | WMDD+FirstFit | Heuristic | 4019 +/- 1558 | 182580 +/- 97529 |
| 10 | WEDF+Consolidate | Heuristic | 4031 +/- 1856 | 107519 +/- 73473 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_weighted_tardiness.png) | ![top 10](../figures/leaderboards/on_rho110/top10_weighted_tardiness.png) |

## max tardiness (lower is better)

| rank | method | family | max tardiness | J |
|---|---|---|---|---|
| 1 | LST+Consolidate | Heuristic | 48.5 +/- 13.7 | 144386 +/- 97469 |
| 2 | EDF+Consolidate | Heuristic | 52.5 +/- 14.3 | 114142 +/- 68444 |
| 3 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle EDFL expert | RL | 53.2 +/- 1.8 | 116522 +/- 7803 |
| 4 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget EDFL expert | RL | 53.3 +/- 1.3 | 118619 +/- 10202 |
| 5 | LST+BestFit | Heuristic | 54.6 +/- 15.3 | 178161 +/- 110493 |
| 6 | LST+FirstFit | Heuristic | 55.1 +/- 14.0 | 187603 +/- 117291 |
| 7 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 59.2 +/- 1.1 | 122087 +/- 7505 |
| 8 | FCFS+Consolidate | Heuristic | 59.9 +/- 12.9 | 154636 +/- 98058 |
| 9 | WEDF+Consolidate | Heuristic | 60.7 +/- 15.0 | 107519 +/- 73473 |
| 10 | EDF+BestFit | Heuristic | 60.8 +/- 16.4 | 155351 +/- 109761 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_max_tardiness.png) | ![top 10](../figures/leaderboards/on_rho110/top10_max_tardiness.png) |

## p95 tardiness (lower is better)

| rank | method | family | p95 tardiness | J |
|---|---|---|---|---|
| 1 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle ATC expert | RL | 4.6 +/- 0.4 | 158739 +/- 5231 |
| 2 | ATC+Consolidate | Heuristic | 4.7 +/- 3.5 | 159315 +/- 88420 |
| 3 | WMDD+Consolidate | Heuristic | 5.0 +/- 3.5 | 141071 +/- 71452 |
| 4 | ATC+BestFit | Heuristic | 5.7 +/- 4.5 | 191577 +/- 108044 |
| 5 | COVERT+Consolidate | Heuristic | 6.1 +/- 3.9 | 150616 +/- 88508 |
| 6 | WMDD+BestFit | Heuristic | 6.3 +/- 3.8 | 173914 +/- 93418 |
| 7 | ATC+FirstFit | Heuristic | 6.5 +/- 4.3 | 205642 +/- 98055 |
| 8 | COVERT+BestFit | Heuristic | 6.9 +/- 4.6 | 187862 +/- 102012 |
| 9 | WMDD+FirstFit | Heuristic | 7.0 +/- 4.1 | 182580 +/- 97529 |
| 10 | COVERT+FirstFit | Heuristic | 7.5 +/- 5.3 | 180812 +/- 108182 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_p95_tardiness.png) | ![top 10](../figures/leaderboards/on_rho110/top10_p95_tardiness.png) |

## mean wait (lower is better)

| rank | method | family | mean wait | J |
|---|---|---|---|---|
| 1 | SPT+Consolidate | Heuristic | 4.35 +/- 0.95 | 391147 +/- 161979 |
| 2 | SPT+BestFit | Heuristic | 4.42 +/- 0.94 | 408295 +/- 157784 |
| 3 | WSPT+Consolidate | Heuristic | 4.70 +/- 0.99 | 339583 +/- 134907 |
| 4 | SPT+FirstFit | Heuristic | 4.71 +/- 0.86 | 445610 +/- 154209 |
| 5 | WSPT+BestFit | Heuristic | 4.79 +/- 0.98 | 367881 +/- 141653 |
| 6 | WSPT+FirstFit | Heuristic | 5.17 +/- 0.98 | 409159 +/- 146537 |
| 7 | PPO Opt2 priority score +Consolidate windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle +late-count objective 1M-step budget [lambda x1] | RL | 5.53 +/- 0.25 | 261489 +/- 38879 |
| 8 | PPO Opt2 priority score +Consolidate windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle +late-count objective 1M-step budget [lambda x4] | RL | 5.60 +/- 0.77 | 323156 +/- 90827 |
| 9 | SPT+WorstFit | Heuristic | 5.95 +/- 0.88 | 644925 +/- 189011 |
| 10 | ATC+FirstFit | Heuristic | 6.15 +/- 1.24 | 205642 +/- 98055 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_mean_wait.png) | ![top 10](../figures/leaderboards/on_rho110/top10_mean_wait.png) |

## mean flow time (lower is better)

| rank | method | family | mean flow time | J |
|---|---|---|---|---|
| 1 | SPT+Consolidate | Heuristic | 9.82 +/- 1.05 | 391147 +/- 161979 |
| 2 | SPT+BestFit | Heuristic | 9.89 +/- 1.05 | 408295 +/- 157784 |
| 3 | WSPT+Consolidate | Heuristic | 10.17 +/- 1.09 | 339583 +/- 134907 |
| 4 | SPT+FirstFit | Heuristic | 10.18 +/- 1.00 | 445610 +/- 154209 |
| 5 | WSPT+BestFit | Heuristic | 10.25 +/- 1.10 | 367881 +/- 141653 |
| 6 | WSPT+FirstFit | Heuristic | 10.64 +/- 1.09 | 409159 +/- 146537 |
| 7 | PPO Opt2 priority score +Consolidate windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle +late-count objective 1M-step budget [lambda x1] | RL | 11.00 +/- 0.25 | 261489 +/- 38879 |
| 8 | PPO Opt2 priority score +Consolidate windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle +late-count objective 1M-step budget [lambda x4] | RL | 11.07 +/- 0.77 | 323156 +/- 90827 |
| 9 | SPT+WorstFit | Heuristic | 11.42 +/- 0.99 | 644925 +/- 189011 |
| 10 | ATC+FirstFit | Heuristic | 11.62 +/- 1.33 | 205642 +/- 98055 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_mean_flow_time.png) | ![top 10](../figures/leaderboards/on_rho110/top10_mean_flow_time.png) |

## makespan (lower is better)

| rank | method | family | makespan | J |
|---|---|---|---|---|
| 1 | LPT+Consolidate | Heuristic | 140.9 +/- 7.6 | 1112071 +/- 557022 |
| 2 | LPT+BestFit | Heuristic | 142.0 +/- 7.4 | 1008368 +/- 508233 |
| 3 | LPT+FirstFit | Heuristic | 142.0 +/- 8.3 | 1051760 +/- 433452 |
| 4 | LPT+WorstFit | Heuristic | 145.9 +/- 7.0 | 1161351 +/- 417931 |
| 5 | RandomRule+FirstFitConsolidate | Heuristic | 148.2 +/- 9.1 | 241365 +/- 116668 |
| 6 | RandomRule+FirstFit | Heuristic | 148.7 +/- 9.0 | 268708 +/- 147638 |
| 7 | LST+Consolidate | Heuristic | 149.9 +/- 10.5 | 144386 +/- 97469 |
| 8 | LST+BestFit | Heuristic | 150.4 +/- 10.6 | 178161 +/- 110493 |
| 9 | WLST+Consolidate | Heuristic | 150.8 +/- 10.4 | 139904 +/- 104537 |
| 10 | WLST+FirstFit | Heuristic | 150.8 +/- 10.0 | 156726 +/- 99272 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_makespan.png) | ![top 10](../figures/leaderboards/on_rho110/top10_makespan.png) |

## utilisation of active machines (higher is better)

| rank | method | family | utilisation of active machines | J |
|---|---|---|---|---|
| 1 | LPT+Consolidate | Heuristic | 0.986 +/- 0.004 | 1112071 +/- 557022 |
| 2 | LPT+BestFit | Heuristic | 0.984 +/- 0.005 | 1008368 +/- 508233 |
| 3 | LPT+FirstFit | Heuristic | 0.983 +/- 0.005 | 1051760 +/- 433452 |
| 4 | LST+Consolidate | Heuristic | 0.979 +/- 0.005 | 144386 +/- 97469 |
| 5 | WLST+Consolidate | Heuristic | 0.978 +/- 0.005 | 139904 +/- 104537 |
| 6 | MDC+Consolidate | Heuristic | 0.978 +/- 0.005 | 106463 +/- 87144 |
| 7 | LST+BestFit | Heuristic | 0.977 +/- 0.006 | 178161 +/- 110493 |
| 8 | MDC+BestFit | Heuristic | 0.977 +/- 0.005 | 138364 +/- 93785 |
| 9 | COVERT+Consolidate | Heuristic | 0.977 +/- 0.005 | 150616 +/- 88508 |
| 10 | LST+FirstFit | Heuristic | 0.976 +/- 0.005 | 187603 +/- 117291 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_mean_active_utilisation.png) | ![top 10](../figures/leaderboards/on_rho110/top10_mean_active_utilisation.png) |

## active machine-ticks (lower is better)

| rank | method | family | active machine-ticks | J |
|---|---|---|---|---|
| 1 | LPT+Consolidate | Heuristic | 1368 +/- 68 | 1112071 +/- 557022 |
| 2 | LPT+FirstFit | Heuristic | 1377 +/- 69 | 1051760 +/- 433452 |
| 3 | LPT+BestFit | Heuristic | 1380 +/- 71 | 1008368 +/- 508233 |
| 4 | LST+Consolidate | Heuristic | 1387 +/- 71 | 144386 +/- 97469 |
| 5 | LST+BestFit | Heuristic | 1393 +/- 67 | 178161 +/- 110493 |
| 6 | WLST+Consolidate | Heuristic | 1394 +/- 73 | 139904 +/- 104537 |
| 7 | MDC+Consolidate | Heuristic | 1395 +/- 75 | 106463 +/- 87144 |
| 8 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 1395 +/- 2 | 122087 +/- 7505 |
| 9 | LST+FirstFit | Heuristic | 1395 +/- 70 | 187603 +/- 117291 |
| 10 | WLST+BestFit | Heuristic | 1396 +/- 67 | 152726 +/- 87886 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_active_machine_ticks.png) | ![top 10](../figures/leaderboards/on_rho110/top10_active_machine_ticks.png) |

## energy (SPECpower) (lower is better)

| rank | method | family | energy (SPECpower) | J |
|---|---|---|---|---|
| 1 | LPT+Consolidate | Heuristic | 1273 +/- 67 | 1112071 +/- 557022 |
| 2 | LPT+FirstFit | Heuristic | 1279 +/- 67 | 1051760 +/- 433452 |
| 3 | LPT+BestFit | Heuristic | 1281 +/- 69 | 1008368 +/- 508233 |
| 4 | LST+Consolidate | Heuristic | 1286 +/- 69 | 144386 +/- 97469 |
| 5 | LST+BestFit | Heuristic | 1291 +/- 67 | 178161 +/- 110493 |
| 6 | MDC+Consolidate | Heuristic | 1292 +/- 72 | 106463 +/- 87144 |
| 7 | WLST+Consolidate | Heuristic | 1292 +/- 71 | 139904 +/- 104537 |
| 8 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle | RL | 1292 +/- 1 | 122087 +/- 7505 |
| 9 | LST+FirstFit | Heuristic | 1292 +/- 68 | 187603 +/- 117291 |
| 10 | WLST+BestFit | Heuristic | 1293 +/- 66 | 152726 +/- 87886 |

| top 5 | top 10 |
|---|---|
| ![top 5](../figures/leaderboards/on_rho110/top5_energy_specpower.png) | ![top 10](../figures/leaderboards/on_rho110/top10_energy_specpower.png) |

Spread (+/-): std across seeds for multi-seed RL rows, otherwise std across instances.
