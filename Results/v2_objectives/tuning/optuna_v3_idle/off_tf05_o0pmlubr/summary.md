# Optuna study off_tf05_o0pmlubr (protocol v3_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | COMPLETE | 32535 | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
| 6 | COMPLETE | 32600 | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 3 | COMPLETE | 32749 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 5 | COMPLETE | 42700 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 4 | COMPLETE | 54935 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 2 | COMPLETE | 107707 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 1 | COMPLETE | 264478 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 0 | COMPLETE | 1091091 | default | default | default | default | default | default | default | default |
| 9 | PRUNED | pruned at 75000 steps (43327 on 5) | 0.000388 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.1 | 0.00644 |
| 8 | PRUNED | pruned at 150000 steps (44812 on 5) | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 11 | PRUNED | pruned at 75000 steps (49200 on 5) | 8.2e-05 | 16384 | 64 | 5 | 0.99 | 0.95 | 0.1 | 0.000101 |
| 10 | PRUNED | pruned at 75000 steps (1263277 on 5) | 3.22e-05 | 2048 | 64 | 5 | 0.99 | 0.9 | 0.2 | 0.000222 |
