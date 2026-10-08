# Optuna study off_tf05_o4fmlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | COMPLETE | 28220 | 8.23e-05 | 2048 | 64 | 3 | 0.99 | 0.98 | 0.2 | 0.000118 |
| 10 | COMPLETE | 29157 | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 0 | COMPLETE | 29569 | default | default | default | default | default | default | default | default |
| 2 | COMPLETE | 30966 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 5 | COMPLETE | 31489 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 9 | COMPLETE | 31576 | 0.000895 | 16384 | 1024 | 3 | 0.99 | 0.98 | 0.3 | 0.00713 |
| 1 | COMPLETE | 37394 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 3 | COMPLETE | 44552 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 4 | COMPLETE | 51558 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 6 | PRUNED | pruned at 150000 steps (43294 on 5) | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 8 | PRUNED | pruned at 75000 steps (62328 on 5) | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
| 7 | PRUNED | pruned at 75000 steps (74206 on 5) | 0.000129 | 2048 | 1024 | 5 | 0.999 | 0.95 | 0.1 | 0.00963 |
