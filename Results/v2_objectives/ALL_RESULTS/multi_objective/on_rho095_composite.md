# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 17.6 m_T * weighted tardiness + 176 m_U * weighted late jobs + 17.6 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 24,921 | +13.3% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 27,497 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 31,239 | +13.6% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 33,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 37,234 | +12.8% |
| J + linear tardiness (x1) | EDF+Consolidate | 44,030 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 49,223 | +11.8% |
| J + linear tardiness (x2) | EDF+Consolidate | 66,074 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 73,202 | +10.8% |
| J + linear tardiness (x4) | EDF+Consolidate | 110,161 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 121,160 | +10.0% |
| J + late jobs (x0.25) | EDF+Consolidate | 27,471 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 30,928 | +12.6% |
| J + late jobs (x0.5) | EDF+Consolidate | 32,955 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 36,612 | +11.1% |
| J + late jobs (x1) | EDF+Consolidate | 43,923 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 47,979 | +9.2% |
| J + late jobs (x2) | EDF+Consolidate | 65,860 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 70,714 | +7.4% |
| J + late jobs (x4) | WMDD+Consolidate | 104,633 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 116,185 | +11.0% |
| J + energy (x0.25) | EDF+Consolidate | 27,492 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 30,591 | +11.3% |
| J + energy (x0.5) | EDF+Consolidate | 32,997 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 36,261 | +9.9% |
| J + energy (x1) | EDF+Consolidate | 44,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 47,318 | +7.5% |
| J + energy (x2) | MDC+Consolidate | 65,968 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 69,392 | +5.2% |
| J + energy (x4) | MDC+Consolidate | 109,290 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 113,539 | +3.9% |
| J + late jobs + energy (x1) | EDF+Consolidate | 65,945 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 70,053 | +6.2% |
| all four terms (x1) | EDF+Consolidate | 87,989 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (6 seeds) | 94,032 | +6.9% |
