# Optuna study on_rho095_o3wmlubrze (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | COMPLETE | 53572 | 0.000895 | 16384 | 1024 | 3 | 0.99 | 0.98 | 0.3 | 0.00713 |
| 4 | COMPLETE | 67697 | 8.02e-05 | 16384 | 64 | 5 | 0.995 | 0.98 | 0.1 | 0.00525 |
| 10 | COMPLETE | 75533 | 0.000219 | 16384 | 64 | 5 | 0.995 | 0.9 | 0.3 | 0.000413 |
| 11 | COMPLETE | 88788 | 0.000967 | 16384 | 256 | 3 | 0.99 | 0.98 | 0.3 | 0.000118 |
| 8 | COMPLETE | 102250 | 7.73e-05 | 8192 | 1024 | 5 | 0.995 | 0.95 | 0.2 | 0.0178 |
| 3 | COMPLETE | 105426 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 0 | COMPLETE | 114428 | default | default | default | default | default | default | default | default |
| 5 | COMPLETE | 115300 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 7 | COMPLETE | 160510 | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 1 | COMPLETE | 368500 | 0.000129 | 2048 | 1024 | 5 | 0.999 | 0.95 | 0.1 | 0.00963 |
| 9 | PRUNED | pruned at 75000 steps (104581 on 5) | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
| 6 | PRUNED | pruned at 150000 steps (179879 on 5) | 0.000128 | 8192 | 64 | 10 | 0.999 | 0.95 | 0.1 | 0.000732 |
