# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 17.6 m_T * weighted tardiness + 176 m_U * weighted late jobs + 17.6 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 21,827 | -0.7% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 27,497 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 27,407 | -0.3% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 33,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 32,986 | -0.1% |
| J + linear tardiness (x1) | EDF+Consolidate | 44,030 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 44,146 | +0.3% |
| J + linear tardiness (x2) | EDF+Consolidate | 66,074 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 66,464 | +0.6% |
| J + linear tardiness (x4) | EDF+Consolidate | 110,161 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 111,101 | +0.9% |
| J + late jobs (x0.25) | EDF+Consolidate | 27,471 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 27,313 | -0.6% |
| J + late jobs (x0.5) | EDF+Consolidate | 32,955 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 32,799 | -0.5% |
| J + late jobs (x1) | EDF+Consolidate | 43,923 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 43,771 | -0.3% |
| J + late jobs (x2) | EDF+Consolidate | 65,860 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 65,715 | -0.2% |
| J + late jobs (x4) | WMDD+Consolidate | 104,633 | PPO Opt3 ATC-prior score +Consolidate windowed +EDF imitation warm sta (3 seeds) | 106,796 | +2.1% |
| J + energy (x0.25) | EDF+Consolidate | 27,492 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 27,338 | -0.6% |
| J + energy (x0.5) | EDF+Consolidate | 32,997 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 32,849 | -0.4% |
| J + energy (x1) | EDF+Consolidate | 44,008 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 43,871 | -0.3% |
| J + energy (x2) | MDC+Consolidate | 65,968 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 65,915 | -0.1% |
| J + energy (x4) | MDC+Consolidate | 109,290 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 110,002 | +0.7% |
| J + late jobs + energy (x1) | EDF+Consolidate | 65,945 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 65,815 | -0.2% |
| all four terms (x1) | EDF+Consolidate | 87,989 | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ah (1 seeds) | 88,133 | +0.2% |
