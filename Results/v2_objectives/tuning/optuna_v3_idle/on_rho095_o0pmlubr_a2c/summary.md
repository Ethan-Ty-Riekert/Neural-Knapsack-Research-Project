# Optuna study on_rho095_o0pmlubr_a2c (protocol v3_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 10 | COMPLETE | 52717 | 0.00248 | 320 | 0.99 | 0.95 | 0.00279 |
| 11 | COMPLETE | 52717 | 0.00291 | 320 | 0.99 | 0.95 | 0.00361 |
| 5 | COMPLETE | 73552 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 0 | COMPLETE | 84410 | default | default | default | default | default |
| 4 | COMPLETE | 96772 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 2 | COMPLETE | 131399 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 9 | COMPLETE | 137880 | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 7 | COMPLETE | 155429 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 1 | COMPLETE | 413159 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 3 | COMPLETE | 3722546528 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 8 | PRUNED | pruned at 75000 steps (385088 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 6 | PRUNED | pruned at 75000 steps (152126967 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
