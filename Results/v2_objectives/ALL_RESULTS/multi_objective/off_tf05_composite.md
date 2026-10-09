# off_tf05: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 15.8 m_T * weighted tardiness + 146 m_U * weighted late jobs + 153 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | WLST+FirstFit | 26,670 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 28,061 | +5.2% |
| J + linear tardiness (x0.25) | WLST+FirstFit | 33,328 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 34,371 | +3.1% |
| J + linear tardiness (x0.5) | WLST+FirstFit | 39,985 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 40,682 | +1.7% |
| J + linear tardiness (x1) | WLST+FirstFit | 53,300 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 52,117 | -2.2% |
| J + linear tardiness (x2) | WLST+FirstFit | 79,930 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 74,336 | -7.0% |
| J + linear tardiness (x4) | WLST+FirstFit | 133,190 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 118,772 | -10.8% |
| J + late jobs (x0.25) | WLST+FirstFit | 33,358 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 33,496 | +0.4% |
| J + late jobs (x0.5) | WLST+FirstFit | 40,046 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 38,931 | -2.8% |
| J + late jobs (x1) | WLST+FirstFit | 53,421 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 47,453 | -11.2% |
| J + late jobs (x2) | COVERT+BestFit | 69,335 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (1 seeds) | 62,380 | -10.0% |
| J + late jobs (x4) | COVERT+BestFit | 88,835 | PPO Opt3 ATC-prior score look-ahead fixed scaling arrival-aware critic (3 seeds) | 87,569 | -1.4% |
| J + energy (x0.25) | WLST+Consolidate | 33,071 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 34,645 | +4.8% |
| J + energy (x0.5) | WLST+Consolidate | 39,472 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 41,229 | +4.5% |
| J + energy (x1) | WLST+Consolidate | 52,274 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 54,398 | +4.1% |
| J + energy (x2) | WLST+Consolidate | 77,877 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 80,735 | +3.7% |
| J + energy (x4) | WLST+Consolidate | 129,083 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (3 seeds) | 133,410 | +3.4% |
| J + late jobs + energy (x1) | WLST+Consolidate | 79,024 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 73,991 | -6.4% |
| all four terms (x1) | WLST+Consolidate | 105,654 | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic  (2 seeds) | 96,137 | -9.0% |
