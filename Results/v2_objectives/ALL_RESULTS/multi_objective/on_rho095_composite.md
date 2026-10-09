# on_rho095: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 15.8 m_T * weighted tardiness + 146 m_U * weighted late jobs + 153 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | EDF+Consolidate | 21,986 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 23,114 | +5.1% |
| J + linear tardiness (x0.25) | EDF+Consolidate | 26,934 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 28,851 | +7.1% |
| J + linear tardiness (x0.5) | EDF+Consolidate | 31,881 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 34,589 | +8.5% |
| J + linear tardiness (x1) | EDF+Consolidate | 41,776 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 46,064 | +10.3% |
| J + linear tardiness (x2) | EDF+Consolidate | 61,565 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 69,014 | +12.1% |
| J + linear tardiness (x4) | EDF+Consolidate | 101,143 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 114,915 | +13.6% |
| J + energy (x0.25) | MDC+Consolidate | 69,721 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 72,140 | +3.5% |
| J + energy (x0.5) | MDC+Consolidate | 116,797 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 121,167 | +3.7% |
| J + energy (x1) | MDC+Consolidate | 210,949 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 219,220 | +3.9% |
| J + energy (x2) | MDC+Consolidate | 399,252 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 415,326 | +4.0% |
| J + energy (x4) | MDC+Consolidate | 775,858 | A2C Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 802,685 | +3.5% |
