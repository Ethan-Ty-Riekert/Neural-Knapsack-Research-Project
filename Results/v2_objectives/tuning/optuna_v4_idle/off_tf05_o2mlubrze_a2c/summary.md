# Optuna study off_tf05_o2mlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 4 | COMPLETE | 27038 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 7 | COMPLETE | 27273 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 3 | COMPLETE | 33700 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 0 | COMPLETE | 34631 | default | default | default | default | default |
| 1 | COMPLETE | 39058 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 2 | COMPLETE | 47341 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 10 | COMPLETE | 110860 | 0.00245 | 80 | 0.99 | 1 | 0.00254 |
| 11 | PRUNED | pruned at 150000 steps (33547 on 5) | 0.00225 | 80 | 0.995 | 0.95 | 0.000136 |
| 9 | PRUNED | pruned at 75000 steps (37652 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 6 | PRUNED | pruned at 75000 steps (42130 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 5 | PRUNED | pruned at 75000 steps (84888 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 8 | PRUNED | pruned at 75000 steps (108143 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
