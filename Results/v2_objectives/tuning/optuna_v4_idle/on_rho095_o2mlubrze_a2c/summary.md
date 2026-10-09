# Optuna study on_rho095_o2mlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 6 | COMPLETE | 124199 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 0 | COMPLETE | 133730 | default | default | default | default | default |
| 11 | COMPLETE | 186223 | 0.000267 | 20 | 0.99 | 0.9 | 0.029 |
| 3 | COMPLETE | 186533 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 4 | COMPLETE | 189078 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 1 | COMPLETE | 254308 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 7 | COMPLETE | 274538 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 2 | COMPLETE | 385504 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 10 | PRUNED | pruned at 75000 steps (206232 on 5) | 0.000107 | 320 | 0.99 | 0.9 | 0.00191 |
| 8 | PRUNED | pruned at 75000 steps (232401 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 9 | PRUNED | pruned at 75000 steps (238850 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 5 | PRUNED | pruned at 75000 steps (250514 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
