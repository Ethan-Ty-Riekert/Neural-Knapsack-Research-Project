# on_rho075_tight: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 15 m_T * weighted tardiness + 115.4 m_U * weighted late jobs + 27.4 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | WLST+Consolidate | 29,502 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 30,044 | +1.8% |
| J + linear tardiness (x0.25) | WLST+Consolidate | 36,901 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 37,491 | +1.6% |
| J + linear tardiness (x0.5) | WLST+Consolidate | 44,301 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 44,938 | +1.4% |
| J + linear tardiness (x1) | WLST+Consolidate | 59,100 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 59,832 | +1.2% |
| J + linear tardiness (x2) | WLST+Consolidate | 88,697 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 89,620 | +1.0% |
| J + linear tardiness (x4) | COVERT+Consolidate | 147,619 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 148,827 | +0.8% |
| J + late jobs (x0.25) | WLST+Consolidate | 36,881 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 37,264 | +1.0% |
| J + late jobs (x0.5) | WLST+Consolidate | 44,259 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 44,483 | +0.5% |
| J + late jobs (x1) | COVERT+Consolidate | 58,930 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 58,922 | -0.0% |
| J + late jobs (x2) | COVERT+Consolidate | 86,367 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 87,136 | +0.9% |
| J + late jobs (x4) | COVERT+Consolidate | 141,242 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 143,513 | +1.6% |
| J + energy (x0.25) | WLST+Consolidate | 36,865 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 37,400 | +1.5% |
| J + energy (x0.5) | WLST+Consolidate | 44,228 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 44,756 | +1.2% |
| J + energy (x1) | WLST+Consolidate | 58,955 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 59,468 | +0.9% |
| J + energy (x2) | WLST+Consolidate | 88,408 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 88,892 | +0.5% |
| J + energy (x4) | WLST+Consolidate | 147,313 | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm star (5 seeds) | 147,733 | +0.3% |
| J + late jobs + energy (x1) | WLST+Consolidate | 88,470 | PPO Opt2 priority score +Consolidate windowed WLST expert +EDF imitati (3 seeds) | 88,346 | -0.1% |
| all four terms (x1) | COVERT+Consolidate | 117,579 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 117,993 | +0.4% |
