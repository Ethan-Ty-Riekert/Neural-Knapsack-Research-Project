# Optuna study on_rho095_o1cmlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 43668 | default | default | default | default | default |
| 4 | COMPLETE | 43668 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 2 | COMPLETE | 52717 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 3 | COMPLETE | 143296 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 1 | COMPLETE | 189648 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 5 | PRUNED | pruned at 225000 steps (28159 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 8 | PRUNED | pruned at 75000 steps (45660 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 6 | PRUNED | pruned at 225000 steps (100738 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 10 | PRUNED | pruned at 225000 steps (132796 on 5) | 0.00245 | 320 | 0.99 | 0.9 | 0.00279 |
| 11 | PRUNED | pruned at 225000 steps (132796 on 5) | 0.000277 | 80 | 0.995 | 1 | 0.029 |
| 9 | PRUNED | pruned at 75000 steps (166941 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 7 | PRUNED | pruned at 75000 steps (234755 on 5) | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
