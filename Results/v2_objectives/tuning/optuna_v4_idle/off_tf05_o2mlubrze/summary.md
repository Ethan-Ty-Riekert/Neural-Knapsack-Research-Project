# Optuna study off_tf05_o2mlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | COMPLETE | 28566 | 0.000842 | 2048 | 256 | 3 | 0.999 | 0.98 | 0.2 | 0.0233 |
| 11 | COMPLETE | 29978 | 0.000988 | 2048 | 256 | 3 | 0.999 | 0.98 | 0.2 | 0.0279 |
| 8 | COMPLETE | 33109 | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 1 | COMPLETE | 34826 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 2 | COMPLETE | 35796 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 5 | COMPLETE | 36127 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 3 | COMPLETE | 36221 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 4 | COMPLETE | 51325 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 0 | COMPLETE | 68250 | default | default | default | default | default | default | default | default |
| 9 | PRUNED | pruned at 75000 steps (49228 on 5) | 0.000388 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.1 | 0.00644 |
| 6 | PRUNED | pruned at 75000 steps (52956 on 5) | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
| 7 | PRUNED | pruned at 75000 steps (61186 on 5) | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
