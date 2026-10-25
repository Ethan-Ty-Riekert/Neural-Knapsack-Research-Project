# on_rho050: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 14.7 m_T * weighted tardiness + 127.2 m_U * weighted late jobs + 4.9 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | WLST+Consolidate | 4,085 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 4,087 | +0.0% |
| J + linear tardiness (x0.25) | WLST+Consolidate | 5,106 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 5,108 | +0.0% |
| J + linear tardiness (x0.5) | WLST+Consolidate | 6,127 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 6,129 | +0.0% |
| J + linear tardiness (x1) | WLST+Consolidate | 8,169 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 8,171 | +0.0% |
| J + linear tardiness (x2) | WLST+Consolidate | 12,252 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 12,256 | +0.0% |
| J + linear tardiness (x4) | WLST+Consolidate | 20,419 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 20,425 | +0.0% |
| J + late jobs (x0.25) | WLST+Consolidate | 5,107 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 5,108 | +0.0% |
| J + late jobs (x0.5) | WLST+Consolidate | 6,128 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 6,129 | +0.0% |
| J + late jobs (x1) | WLST+Consolidate | 8,171 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 8,172 | +0.0% |
| J + late jobs (x2) | WLST+Consolidate | 12,257 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 12,258 | +0.0% |
| J + late jobs (x4) | WLST+Consolidate | 20,428 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 20,429 | +0.0% |
| J + energy (x0.25) | LPT+Consolidate | 5,089 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 5,097 | +0.2% |
| J + energy (x0.5) | LPT+Consolidate | 6,087 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 6,107 | +0.3% |
| J + energy (x1) | LPT+Consolidate | 8,084 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 8,125 | +0.5% |
| J + energy (x2) | LPT+Consolidate | 12,077 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 12,162 | +0.7% |
| J + energy (x4) | LPT+Consolidate | 20,064 | PPO Opt2 priority score +Consolidate windowed MDC expert +EDF imitatio (3 seeds) | 20,235 | +0.8% |
| J + late jobs + energy (x1) | LPT+Consolidate | 12,175 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 12,215 | +0.3% |
| all four terms (x1) | LPT+Consolidate | 16,266 | PPO Opt2 priority score +Consolidate windowed COVERT expert +EDF imita (3 seeds) | 16,300 | +0.2% |
