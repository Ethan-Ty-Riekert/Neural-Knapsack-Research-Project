# Optuna study on_rho095_o1cnm_a2c

20 trials x 100000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 1 | COMPLETE | 41763 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 2 | COMPLETE | 44416 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 4 | COMPLETE | 44416 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 16 | COMPLETE | 143296 | 0.0002 | 80 | 0.995 | 1 | 0.00595 |
| 0 | COMPLETE | 189648 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 3 | COMPLETE | 189648 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 11 | COMPLETE | 189648 | 0.000316 | 80 | 0.995 | 1 | 0.0022 |
| 17 | COMPLETE | 189648 | 0.000245 | 80 | 0.995 | 1 | 0.00385 |
| 5 | COMPLETE | 297671 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 9 | PRUNED | pruned at 25000 steps (45503 on 5) | 0.000102 | 320 | 0.99 | 0.9 | 0.0229 |
| 19 | PRUNED | pruned at 25000 steps (45660 on 5) | 0.00104 | 320 | 0.99 | 0.9 | 0.00122 |
| 7 | PRUNED | pruned at 25000 steps (118705 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 8 | PRUNED | pruned at 50000 steps (132796 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 14 | PRUNED | pruned at 25000 steps (132796 on 5) | 0.00119 | 320 | 0.99 | 1 | 0.00107 |
| 10 | PRUNED | pruned at 25000 steps (166138 on 5) | 0.00245 | 80 | 0.999 | 1 | 0.00279 |
| 12 | PRUNED | pruned at 25000 steps (166138 on 5) | 0.000352 | 20 | 0.999 | 1 | 0.00191 |
| 13 | PRUNED | pruned at 25000 steps (166138 on 5) | 0.000376 | 80 | 0.995 | 1 | 0.00528 |
| 6 | PRUNED | pruned at 25000 steps (166941 on 5) | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 15 | PRUNED | pruned at 25000 steps (166941 on 5) | 0.000478 | 20 | 0.999 | 1 | 0.0292 |
| 18 | PRUNED | pruned at 25000 steps (166941 on 5) | 0.000537 | 20 | 0.999 | 1 | 0.0121 |
