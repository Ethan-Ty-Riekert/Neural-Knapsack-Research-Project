# Optuna study on_rho095_o1cmlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | COMPLETE | 68172 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 8 | COMPLETE | 68172 | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 2 | COMPLETE | 127328 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 3 | COMPLETE | 143296 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 0 | COMPLETE | 189648 | default | default | default | default | default | default | default | default |
| 4 | COMPLETE | 189648 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 5 | COMPLETE | 189648 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 10 | COMPLETE | 189648 | 8.67e-05 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0237 |
| 11 | PRUNED | pruned at 75000 steps (100738 on 5) | 0.000982 | 2048 | 1024 | 3 | 0.999 | 0.98 | 0.3 | 0.029 |
| 9 | PRUNED | pruned at 75000 steps (131319 on 5) | 0.000388 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.1 | 0.00644 |
| 6 | PRUNED | pruned at 75000 steps (132796 on 5) | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 7 | PRUNED | pruned at 75000 steps (187107 on 5) | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
