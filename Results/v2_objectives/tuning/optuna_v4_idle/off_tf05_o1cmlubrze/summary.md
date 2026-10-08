# Optuna study off_tf05_o1cmlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | COMPLETE | 37169 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 0 | COMPLETE | 37175 | default | default | default | default | default | default | default | default |
| 3 | COMPLETE | 37243 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 1 | COMPLETE | 37324 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 4 | COMPLETE | 37324 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 5 | COMPLETE | 37324 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 7 | COMPLETE | 37324 | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
| 8 | COMPLETE | 37324 | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 10 | COMPLETE | 37324 | 9.28e-05 | 2048 | 256 | 10 | 0.995 | 0.9 | 0.2 | 0.00124 |
| 6 | PRUNED | pruned at 75000 steps (35054 on 5) | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 9 | PRUNED | pruned at 75000 steps (36947 on 5) | 0.000388 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.1 | 0.00644 |
| 11 | PRUNED | pruned at 75000 steps (49973 on 5) | 0.00097 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.2 | 0.000101 |
