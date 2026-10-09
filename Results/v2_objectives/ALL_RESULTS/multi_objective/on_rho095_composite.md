# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 17.6 m_T * weighted tardiness + 176 m_U * weighted late jobs + 17.6 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 22,000 | +0.1% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 27,497 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 27,514 | +0.1% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 33,008 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 33,029 | +0.1% |
| J + linear tardiness (x1) | EDF+Consolidate | 44,030 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 44,057 | +0.1% |
| J + linear tardiness (x2) | EDF+Consolidate | 66,074 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 66,115 | +0.1% |
| J + linear tardiness (x4) | EDF+Consolidate | 110,161 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 110,230 | +0.1% |
| J + late jobs (x0.25) | EDF+Consolidate | 27,471 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 27,590 | +0.4% |
| J + late jobs (x0.5) | EDF+Consolidate | 32,955 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 33,180 | +0.7% |
| J + late jobs (x1) | EDF+Consolidate | 43,923 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 44,359 | +1.0% |
| J + late jobs (x2) | EDF+Consolidate | 65,860 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 66,718 | +1.3% |
| J + late jobs (x4) | WMDD+Consolidate | 104,633 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 111,436 | +6.5% |
| J + energy (x0.25) | EDF+Consolidate | 27,492 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 27,623 | +0.5% |
| J + energy (x0.5) | EDF+Consolidate | 32,997 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 33,246 | +0.8% |
| J + energy (x1) | EDF+Consolidate | 44,008 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 44,492 | +1.1% |
| J + energy (x2) | MDC+Consolidate | 65,968 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 66,984 | +1.5% |
| J + energy (x4) | MDC+Consolidate | 109,290 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 111,968 | +2.5% |
| J + late jobs + energy (x1) | EDF+Consolidate | 65,945 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 66,851 | +1.4% |
| all four terms (x1) | EDF+Consolidate | 87,989 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 88,909 | +1.0% |
