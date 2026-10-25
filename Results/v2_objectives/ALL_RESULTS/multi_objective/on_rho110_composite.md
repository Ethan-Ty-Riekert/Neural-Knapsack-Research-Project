# on_rho110: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 28.3 m_T * weighted tardiness + 384 m_U * weighted late jobs + 76.3 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | MDC+Consolidate | 106,463 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 114,316 | +7.4% |
| J + linear tardiness (x0.25) | MDC+Consolidate | 133,090 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 142,746 | +7.3% |
| J + linear tardiness (x0.5) | MDC+Consolidate | 159,716 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 171,177 | +7.2% |
| J + linear tardiness (x1) | MDC+Consolidate | 212,969 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 228,037 | +7.1% |
| J + linear tardiness (x2) | MDC+Consolidate | 319,475 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 341,758 | +7.0% |
| J + linear tardiness (x4) | WMDD+Consolidate | 508,423 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm sta (3 seeds) | 562,989 | +10.7% |
| J + late jobs (x0.25) | MDC+Consolidate | 133,078 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 139,001 | +4.5% |
| J + late jobs (x0.5) | WEDF+Consolidate | 159,620 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 163,687 | +2.5% |
| J + late jobs (x1) | WMDD+Consolidate | 210,583 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 213,058 | +1.2% |
| J + late jobs (x2) | WMDD+Consolidate | 280,095 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm sta (3 seeds) | 298,306 | +6.5% |
| J + late jobs (x4) | WMDD+Consolidate | 419,118 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm sta (3 seeds) | 437,872 | +4.5% |
| J + energy (x0.25) | MDC+Consolidate | 133,067 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 141,018 | +6.0% |
| J + energy (x0.5) | MDC+Consolidate | 159,671 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 167,721 | +5.0% |
| J + energy (x1) | MDC+Consolidate | 212,878 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 221,125 | +3.9% |
| J + energy (x2) | MDC+Consolidate | 319,292 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 327,935 | +2.7% |
| J + energy (x4) | MDC+Consolidate | 532,120 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 541,553 | +1.8% |
| J + late jobs + energy (x1) | MDC+Consolidate | 319,338 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 319,867 | +0.2% |
| all four terms (x1) | WMDD+Consolidate | 413,196 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 431,920 | +4.5% |
