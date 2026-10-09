# Optuna study off_tf05_o1cmlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 37324 | default | default | default | default | default |
| 3 | COMPLETE | 37324 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 4 | COMPLETE | 37324 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 5 | COMPLETE | 37324 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 6 | COMPLETE | 37324 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 7 | COMPLETE | 37324 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 8 | COMPLETE | 37324 | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 2 | COMPLETE | 39433 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 1 | COMPLETE | 72159 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 10 | COMPLETE | 188828 | 0.00245 | 320 | 0.99 | 0.9 | 0.00279 |
| 11 | PRUNED | pruned at 75000 steps (36666 on 5) | 0.000277 | 20 | 0.995 | 1 | 0.029 |
| 9 | PRUNED | pruned at 75000 steps (232135 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
