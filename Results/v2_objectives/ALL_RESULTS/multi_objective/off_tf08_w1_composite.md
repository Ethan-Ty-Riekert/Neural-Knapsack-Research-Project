# off_tf08_w1: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 39.5 m_T * weighted tardiness + 1311.2 m_U * weighted late jobs + 758.7 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | WLST+FirstFit | 129,805 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 129,805 | +0.0% |
| J + linear tardiness (x0.25) | WLST+FirstFit | 162,224 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 162,224 | +0.0% |
| J + linear tardiness (x0.5) | WLST+FirstFit | 194,642 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 194,642 | +0.0% |
| J + linear tardiness (x1) | WLST+FirstFit | 259,479 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 259,479 | +0.0% |
| J + linear tardiness (x2) | WLST+FirstFit | 389,153 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 389,153 | +0.0% |
| J + linear tardiness (x4) | WLST+FirstFit | 648,500 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 648,500 | +0.0% |
| J + late jobs (x0.25) | WLST+FirstFit | 162,257 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 162,257 | +0.0% |
| J + late jobs (x0.5) | WLST+FirstFit | 194,709 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 194,709 | +0.0% |
| J + late jobs (x1) | WLST+FirstFit | 259,614 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 259,614 | +0.0% |
| J + late jobs (x2) | WLST+FirstFit | 389,423 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 389,423 | +0.0% |
| J + late jobs (x4) | WMDD+BestFit | 601,052 | PPO Opt2 priority score windowed look-ahead fixed scaling arrival-awar (3 seeds) | 648,485 | +7.9% |
| J + energy (x0.25) | LST+Consolidate | 161,291 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 162,351 | +0.7% |
| J + energy (x0.5) | LST+Consolidate | 192,777 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 194,896 | +1.1% |
| J + energy (x1) | LST+Consolidate | 255,749 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 259,988 | +1.7% |
| J + energy (x2) | LST+Consolidate | 381,693 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 390,171 | +2.2% |
| J + energy (x4) | LST+Consolidate | 633,582 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 650,536 | +2.7% |
| J + late jobs + energy (x1) | LST+Consolidate | 385,558 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 389,797 | +1.1% |
| all four terms (x1) | LST+Consolidate | 515,232 | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead  (3 seeds) | 519,470 | +0.8% |
