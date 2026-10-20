# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 17.6 m_T * weighted tardiness + 176 m_U * weighted late jobs + 17.6 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 23,144 | +5.3% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 27,497 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 28,756 | +4.6% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 33,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 34,369 | +4.1% |
| J + linear tardiness (x1) | EDF+Consolidate | 44,030 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 45,594 | +3.6% |
| J + linear tardiness (x2) | EDF+Consolidate | 66,074 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 68,045 | +3.0% |
| J + linear tardiness (x4) | EDF+Consolidate | 110,161 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 112,947 | +2.5% |
| J + late jobs (x0.25) | EDF+Consolidate | 27,471 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 28,606 | +4.1% |
| J + late jobs (x0.5) | EDF+Consolidate | 32,955 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 34,069 | +3.4% |
| J + late jobs (x1) | EDF+Consolidate | 43,923 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 44,994 | +2.4% |
| J + late jobs (x2) | EDF+Consolidate | 65,860 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 66,845 | +1.5% |
| J + late jobs (x4) | WMDD+Consolidate | 104,633 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 110,545 | +5.7% |
| J + energy (x0.25) | EDF+Consolidate | 27,492 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 28,641 | +4.2% |
| J + energy (x0.5) | EDF+Consolidate | 32,997 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 34,137 | +3.5% |
| J + energy (x1) | EDF+Consolidate | 44,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 45,131 | +2.6% |
| J + energy (x2) | MDC+Consolidate | 65,968 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 67,118 | +1.7% |
| J + energy (x4) | MDC+Consolidate | 109,290 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 111,092 | +1.6% |
| J + late jobs + energy (x1) | EDF+Consolidate | 65,945 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 66,981 | +1.6% |
| all four terms (x1) | EDF+Consolidate | 87,989 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (2 seeds) | 89,432 | +1.6% |
