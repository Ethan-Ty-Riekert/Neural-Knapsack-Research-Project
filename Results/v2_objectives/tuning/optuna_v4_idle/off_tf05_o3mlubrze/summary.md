# Optuna study off_tf05_o3mlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 27091 | default | default | default | default | default | default | default | default |
| 9 | COMPLETE | 27885 | 0.000388 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.1 | 0.00644 |
| 8 | COMPLETE | 27903 | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 2 | COMPLETE | 28561 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 1 | COMPLETE | 33045 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 5 | COMPLETE | 33336 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 3 | COMPLETE | 39110 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 4 | COMPLETE | 62704 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 6 | PRUNED | pruned at 225000 steps (36946 on 5) | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 10 | PRUNED | pruned at 75000 steps (40066 on 5) | 7.79e-05 | 2048 | 64 | 3 | 0.99 | 0.98 | 0.2 | 0.0233 |
| 11 | PRUNED | pruned at 75000 steps (43247 on 5) | 0.0001 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.2 | 0.000101 |
| 7 | PRUNED | pruned at 75000 steps (63522 on 5) | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
