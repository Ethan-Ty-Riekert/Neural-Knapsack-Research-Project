# Optuna study off_tf05_o3wmlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 26578 | default | default | default | default | default | default | default | default |
| 2 | COMPLETE | 27972 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 5 | COMPLETE | 29597 | 0.000895 | 16384 | 1024 | 3 | 0.99 | 0.98 | 0.3 | 0.00713 |
| 1 | COMPLETE | 30126 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 4 | COMPLETE | 31181 | 0.000129 | 2048 | 1024 | 5 | 0.999 | 0.95 | 0.1 | 0.00963 |
| 3 | COMPLETE | 31777 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 9 | PRUNED | pruned at 225000 steps (31598 on 5) | 8.02e-05 | 16384 | 64 | 5 | 0.995 | 0.98 | 0.1 | 0.00525 |
| 7 | PRUNED | pruned at 150000 steps (35619 on 5) | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 8 | PRUNED | pruned at 75000 steps (44723 on 5) | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 11 | PRUNED | pruned at 75000 steps (48189 on 5) | 7.79e-05 | 8192 | 256 | 3 | 0.99 | 0.98 | 0.2 | 0.000996 |
| 10 | PRUNED | pruned at 75000 steps (48500 on 5) | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
| 6 | PRUNED | pruned at 75000 steps (62683 on 5) | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
