# off_tf05: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 15.8 m_T * weighted tardiness + 146 m_U * weighted late jobs + 153 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | WLST+FirstFit | 26,670 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 28,061 | +5.2% |
| J + linear tardiness (x0.25) | WLST+FirstFit | 33,328 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 34,371 | +3.1% |
| J + linear tardiness (x0.5) | WLST+FirstFit | 39,985 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 40,682 | +1.7% |
| J + linear tardiness (x1) | WLST+FirstFit | 53,300 | A2C Opt0 full action space pointer look-ahead fixed scaling arrival-aw (3 seeds) | 52,363 | -1.8% |
| J + linear tardiness (x2) | WLST+FirstFit | 79,930 | A2C Opt0 full action space pointer look-ahead fixed scaling arrival-aw (3 seeds) | 75,031 | -6.1% |
| J + linear tardiness (x4) | WLST+FirstFit | 133,190 | A2C Opt0 full action space pointer look-ahead fixed scaling arrival-aw (3 seeds) | 120,366 | -9.6% |
| J + late jobs (x0.25) | WLST+FirstFit | 33,358 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 33,496 | +0.4% |
| J + late jobs (x0.5) | WLST+FirstFit | 40,046 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 38,931 | -2.8% |
| J + late jobs (x1) | WLST+FirstFit | 53,421 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 48,171 | -9.8% |
| J + late jobs (x2) | COVERT+BestFit | 69,335 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 64,131 | -7.5% |
| J + late jobs (x4) | COVERT+BestFit | 88,835 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 90,659 | +2.1% |
| J + energy (x0.25) | WLST+Consolidate | 33,071 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 34,645 | +4.8% |
| J + energy (x0.5) | WLST+Consolidate | 39,472 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 41,229 | +4.5% |
| J + energy (x1) | WLST+Consolidate | 52,274 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 54,398 | +4.1% |
| J + energy (x2) | WLST+Consolidate | 77,877 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 80,735 | +3.7% |
| J + energy (x4) | WLST+Consolidate | 129,083 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 133,410 | +3.4% |
| J + late jobs + energy (x1) | WLST+Consolidate | 79,024 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 74,643 | -5.5% |
| all four terms (x1) | WLST+Consolidate | 105,654 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 97,529 | -7.7% |
