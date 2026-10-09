# Optuna study on_rho095_o1cmlubr (protocol v3_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | COMPLETE | 30857 | 0.000754 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.1 | 0.000779 |
| 11 | COMPLETE | 46770 | 8.87e-05 | 4096 | 256 | 3 | 0.999 | 0.98 | 0.2 | 0.000101 |
| 0 | COMPLETE | 68172 | default | default | default | default | default | default | default | default |
| 9 | COMPLETE | 68172 | 0.000388 | 4096 | 256 | 5 | 0.999 | 0.98 | 0.1 | 0.00644 |
| 1 | COMPLETE | 171392 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 5 | COMPLETE | 171392 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 3 | COMPLETE | 178410 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 2 | COMPLETE | 180674 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 4 | COMPLETE | 189648 | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 10 | PRUNED | pruned at 75000 steps (27277 on 5) | 0.000842 | 2048 | 256 | 3 | 0.999 | 0.98 | 0.2 | 0.0233 |
| 7 | PRUNED | pruned at 75000 steps (166138 on 5) | 3.21e-05 | 4096 | 64 | 5 | 0.99 | 0.9 | 0.1 | 0.000177 |
| 6 | PRUNED | pruned at 75000 steps (166941 on 5) | 0.000226 | 16384 | 256 | 5 | 0.999 | 0.95 | 0.1 | 0.00318 |
