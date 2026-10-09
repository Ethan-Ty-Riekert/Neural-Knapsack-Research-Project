# off_tf05_w1: best heuristic vs best RL on composite objectives (50 test instances, lower is better)

J_comp = J + 16.7 m_T * weighted tardiness + 215 m_U * weighted late jobs + 75.7 m_E * active machine-ticks.

| objective | best heuristic | J_comp | best RL | J_comp | RL vs heuristic |
|---|---|---|---|---|---|
| J only | LST+FirstFit | 13,098 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 13,098 | +0.0% |
| J + linear tardiness (x0.25) | LST+FirstFit | 16,376 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 16,376 | +0.0% |
| J + linear tardiness (x0.5) | LST+FirstFit | 19,654 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 19,654 | +0.0% |
| J + linear tardiness (x1) | LST+FirstFit | 26,209 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 26,209 | +0.0% |
| J + linear tardiness (x2) | LST+FirstFit | 39,320 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 39,320 | +0.0% |
| J + linear tardiness (x4) | LST+FirstFit | 65,542 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 65,542 | +0.0% |
| J + late jobs (x0.25) | LST+FirstFit | 16,379 | A2C Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-awa (3 seeds) | 16,826 | +2.7% |
| J + late jobs (x0.5) | LST+FirstFit | 19,660 | A2C Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-awa (3 seeds) | 20,067 | +2.1% |
| J + late jobs (x1) | LST+FirstFit | 26,222 | A2C Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-awa (3 seeds) | 26,549 | +1.2% |
| J + late jobs (x2) | LST+FirstFit | 39,345 | A2C Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-awa (3 seeds) | 39,514 | +0.4% |
| J + late jobs (x4) | COVERT+BestFit | 59,077 | A2C Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-awa (3 seeds) | 65,443 | +10.8% |
| J + energy (x0.25) | LST+Consolidate | 16,231 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 16,278 | +0.3% |
| J + energy (x0.5) | LST+Consolidate | 19,365 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 19,459 | +0.5% |
| J + energy (x1) | LST+Consolidate | 25,631 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 25,819 | +0.7% |
| J + energy (x2) | LST+Consolidate | 38,164 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 38,446 | +0.7% |
| J + energy (x4) | LST+Consolidate | 63,230 | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival- (3 seeds) | 63,637 | +0.6% |
| J + late jobs + energy (x1) | LST+Consolidate | 38,755 | A2C Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-awa (3 seeds) | 39,619 | +2.2% |
| all four terms (x1) | LST+Consolidate | 51,866 | A2C Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-awa (3 seeds) | 52,931 | +2.1% |
