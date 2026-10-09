# Optuna study on_rho095_o3mlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 90350 | default | default | default | default | default |
| 4 | COMPLETE | 132866 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 6 | COMPLETE | 138225 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 1 | COMPLETE | 161869 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 7 | COMPLETE | 203405 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 3 | COMPLETE | 224811 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 2 | COMPLETE | 511499 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 11 | PRUNED | pruned at 75000 steps (92241 on 5) | 0.000277 | 80 | 0.995 | 1 | 0.029 |
| 10 | PRUNED | pruned at 75000 steps (100610 on 5) | 0.00245 | 320 | 0.99 | 0.9 | 0.00279 |
| 8 | PRUNED | pruned at 75000 steps (119912 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 9 | PRUNED | pruned at 75000 steps (182637 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 5 | PRUNED | pruned at 75000 steps (280917 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
