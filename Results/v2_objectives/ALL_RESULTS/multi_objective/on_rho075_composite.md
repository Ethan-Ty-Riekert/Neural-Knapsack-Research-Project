# on_rho075: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 14.5 m_T * weighted tardiness + 117.4 m_U * weighted late jobs + 5.5 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | WLST+Consolidate | 5,990 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 6,045 | +0.9% |
| J + linear tardiness (x0.25) | WLST+Consolidate | 7,490 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 7,552 | +0.8% |
| J + linear tardiness (x0.5) | WLST+Consolidate | 8,991 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 9,058 | +0.7% |
| J + linear tardiness (x1) | WLST+Consolidate | 11,993 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 12,070 | +0.6% |
| J + linear tardiness (x2) | WLST+Consolidate | 17,996 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 18,094 | +0.5% |
| J + linear tardiness (x4) | WLST+Consolidate | 30,002 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 30,143 | +0.5% |
| J + late jobs (x0.25) | WLST+Consolidate | 7,487 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 7,544 | +0.8% |
| J + late jobs (x0.5) | WLST+Consolidate | 8,985 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 9,042 | +0.6% |
| J + late jobs (x1) | LST+Consolidate | 11,942 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 12,038 | +0.8% |
| J + late jobs (x2) | MDC+Consolidate | 17,822 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 18,031 | +1.2% |
| J + late jobs (x4) | MDC+Consolidate | 29,576 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 30,017 | +1.5% |
| J + energy (x0.25) | WLST+Consolidate | 7,479 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 7,530 | +0.7% |
| J + energy (x0.5) | WLST+Consolidate | 8,967 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 9,015 | +0.5% |
| J + energy (x1) | WLST+Consolidate | 11,945 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 11,985 | +0.3% |
| J + energy (x2) | WLST+Consolidate | 17,900 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 17,925 | +0.1% |
| J + energy (x4) | WLST+Consolidate | 29,811 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 29,805 | -0.0% |
| J + late jobs + energy (x1) | MDC+Consolidate | 17,881 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 17,978 | +0.5% |
| all four terms (x1) | MDC+Consolidate | 23,887 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 24,002 | +0.5% |
