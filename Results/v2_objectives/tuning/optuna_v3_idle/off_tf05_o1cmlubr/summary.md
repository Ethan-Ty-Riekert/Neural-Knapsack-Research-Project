# Optuna study off_tf05_o1cmlubr (protocol v3_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 37324 | default | default | default | default | default | default | default | default |
| 1 | COMPLETE | 37324 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 3 | COMPLETE | 37324 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 4 | COMPLETE | 37324 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 7 | COMPLETE | 37324 | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
| 8 | COMPLETE | 37324 | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 9 | COMPLETE | 37324 | 0.000388 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.1 | 0.00644 |
| 10 | COMPLETE | 37324 | 7.79e-05 | 2048 | 64 | 3 | 0.99 | 0.98 | 0.2 | 0.0233 |
| 2 | COMPLETE | 39433 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 6 | COMPLETE | 39433 | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 5 | PRUNED | pruned at 75000 steps (36649 on 5) | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 11 | PRUNED | pruned at 75000 steps (36666 on 5) | 9.39e-05 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0288 |
