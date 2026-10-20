# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 17.6 m_T * weighted tardiness + 176 m_U * weighted late jobs + 17.6 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 23,666 | +7.6% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 27,497 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 29,316 | +6.6% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 33,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 34,966 | +5.9% |
| J + linear tardiness (x1) | EDF+Consolidate | 44,030 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 46,266 | +5.1% |
| J + linear tardiness (x2) | EDF+Consolidate | 66,074 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 68,866 | +4.2% |
| J + linear tardiness (x4) | EDF+Consolidate | 110,161 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 114,066 | +3.5% |
| J + late jobs (x0.25) | EDF+Consolidate | 27,471 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 29,100 | +5.9% |
| J + late jobs (x0.5) | EDF+Consolidate | 32,955 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 34,535 | +4.8% |
| J + late jobs (x1) | EDF+Consolidate | 43,923 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 45,404 | +3.4% |
| J + late jobs (x2) | EDF+Consolidate | 65,860 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 67,142 | +1.9% |
| J + late jobs (x4) | WMDD+Consolidate | 104,633 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 110,619 | +5.7% |
| J + energy (x0.25) | EDF+Consolidate | 27,492 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 29,164 | +6.1% |
| J + energy (x0.5) | EDF+Consolidate | 32,997 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 34,661 | +5.0% |
| J + energy (x1) | EDF+Consolidate | 44,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 45,657 | +3.7% |
| J + energy (x2) | MDC+Consolidate | 65,968 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 67,649 | +2.5% |
| J + energy (x4) | MDC+Consolidate | 109,290 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 111,632 | +2.1% |
| J + late jobs + energy (x1) | EDF+Consolidate | 65,945 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 67,396 | +2.2% |
| all four terms (x1) | EDF+Consolidate | 87,989 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (3 seeds) | 89,996 | +2.3% |
