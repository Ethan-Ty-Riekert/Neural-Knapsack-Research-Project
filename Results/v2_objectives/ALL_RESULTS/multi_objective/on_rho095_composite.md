# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 17.6 m_T * weighted tardiness + 176 m_U * weighted late jobs + 17.6 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 24,921 | +13.3% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 27,497 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 31,339 | +14.0% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 33,008 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 37,758 | +14.4% |
| J + linear tardiness (x1) | EDF+Consolidate | 44,030 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 50,595 | +14.9% |
| J + linear tardiness (x2) | EDF+Consolidate | 66,074 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 76,269 | +15.4% |
| J + linear tardiness (x4) | EDF+Consolidate | 110,161 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 127,618 | +15.8% |
| J + energy (x0.25) | EDF+Consolidate | 27,492 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 30,591 | +11.3% |
| J + energy (x0.5) | EDF+Consolidate | 32,997 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 36,261 | +9.9% |
| J + energy (x1) | EDF+Consolidate | 44,008 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 47,600 | +8.2% |
| J + energy (x2) | MDC+Consolidate | 65,968 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 70,280 | +6.5% |
| J + energy (x4) | MDC+Consolidate | 109,290 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 115,640 | +5.8% |
