# Optuna study on_rho095_o0pmlubrze (protocol v4_idle)

7 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | batch-size | n-epochs | gamma | gae-lambda | clip-range | ent-coef |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 35669 | default | default | default | default | default | default | default | default |
| 8 | COMPLETE | 165219 | 0.000494 | 4096 | 256 | 10 | 0.995 | 0.9 | 0.1 | 0.00121 |
| 4 | COMPLETE | 285471 | 0.000129 | 2048 | 1024 | 5 | 0.999 | 0.95 | 0.1 | 0.00963 |
| 2 | COMPLETE | 990657 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 7 | COMPLETE | 990657 | 0.000206 | 2048 | 1024 | 3 | 0.999 | 0.95 | 0.3 | 0.0266 |
| 9 | PRUNED | pruned at 75000 steps (320981 on 5) | 0.000346 | 8192 | 1024 | 10 | 0.995 | 0.9 | 0.1 | 0.00422 |
| 10 | PRUNED | pruned at 75000 steps (381159 on 5) | 4.87e-05 | 8192 | 1024 | 5 | 0.99 | 0.95 | 0.2 | 0.00519 |
