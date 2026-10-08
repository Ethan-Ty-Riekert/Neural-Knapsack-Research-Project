# Optuna study on_rho095_o3mlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | COMPLETE | 48997 | 8.02e-05 | 16384 | 64 | 5 | 0.995 | 0.98 | 0.1 | 0.00525 |
| 1 | COMPLETE | 53812 | default | default | default | default | default | default | default | default |
| 3 | COMPLETE | 60848 | 0.000895 | 16384 | 1024 | 3 | 0.99 | 0.98 | 0.3 | 0.00713 |
| 12 | COMPLETE | 71944 | 3.19e-05 | 16384 | 64 | 5 | 0.99 | 0.98 | 0.2 | 0.000118 |
| 11 | COMPLETE | 94431 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 2 | COMPLETE | 442367 | 0.000129 | 2048 | 1024 | 5 | 0.999 | 0.95 | 0.1 | 0.00963 |
| 5 | COMPLETE | 787306 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 9 | PRUNED | pruned at 150000 steps (85488 on 5) | 7.73e-05 | 8192 | 1024 | 5 | 0.995 | 0.95 | 0.2 | 0.0178 |
| 8 | PRUNED | pruned at 150000 steps (152039 on 5) | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 10 | PRUNED | pruned at 150000 steps (156378 on 5) | 0.000267 | 16384 | 1024 | 10 | 0.995 | 0.9 | 0.3 | 0.00216 |
| 6 | PRUNED | pruned at 75000 steps (165878 on 5) | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 7 | PRUNED | pruned at 75000 steps (182743 on 5) | 0.000128 | 8192 | 64 | 10 | 0.999 | 0.95 | 0.1 | 0.000732 |
