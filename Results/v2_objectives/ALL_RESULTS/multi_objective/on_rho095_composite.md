# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 17.6 m_T * weighted tardiness + 176 m_U * weighted late jobs + 17.6 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 24,921 | +13.3% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 27,497 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 30,946 | +12.5% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 33,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 36,909 | +11.8% |
| J + linear tardiness (x1) | EDF+Consolidate | 44,030 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 48,836 | +10.9% |
| J + linear tardiness (x2) | EDF+Consolidate | 66,074 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 72,688 | +10.0% |
| J + linear tardiness (x4) | EDF+Consolidate | 110,161 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 120,393 | +9.3% |
| J + late jobs (x0.25) | EDF+Consolidate | 27,471 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 30,671 | +11.7% |
| J + late jobs (x0.5) | EDF+Consolidate | 32,955 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 36,359 | +10.3% |
| J + late jobs (x1) | EDF+Consolidate | 43,923 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 47,735 | +8.7% |
| J + late jobs (x2) | EDF+Consolidate | 65,860 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 70,488 | +7.0% |
| J + late jobs (x4) | WMDD+Consolidate | 104,633 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 115,993 | +10.9% |
| J + energy (x0.25) | EDF+Consolidate | 27,492 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 30,501 | +10.9% |
| J + energy (x0.5) | EDF+Consolidate | 32,997 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 36,020 | +9.2% |
| J + energy (x1) | EDF+Consolidate | 44,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 47,057 | +6.9% |
| J + energy (x2) | MDC+Consolidate | 65,968 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 69,130 | +4.8% |
| J + energy (x4) | MDC+Consolidate | 109,290 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 113,277 | +3.6% |
| J + late jobs + energy (x1) | EDF+Consolidate | 65,945 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 69,809 | +5.9% |
| all four terms (x1) | EDF+Consolidate | 87,989 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (8 seeds) | 93,662 | +6.4% |
