# Optuna study on_rho095_o0pnm_a2c

20 trials x 100000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 10 | COMPLETE | 55437 | 0.00253 | 80 | 0.999 | 0.95 | 0.00265 |
| 6 | COMPLETE | 59514 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 2 | COMPLETE | 63618 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 4 | COMPLETE | 66060 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 13 | COMPLETE | 74393 | 0.00163 | 80 | 0.999 | 0.95 | 0.000477 |
| 11 | COMPLETE | 105089 | 0.00292 | 80 | 0.999 | 0.95 | 0.00175 |
| 0 | COMPLETE | 172458 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 1 | COMPLETE | 207172 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 3 | COMPLETE | 468463 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 12 | PRUNED | pruned at 25000 steps (79857 on 5) | 0.00258 | 80 | 0.999 | 0.95 | 0.003 |
| 18 | PRUNED | pruned at 25000 steps (170080 on 5) | 0.000413 | 80 | 0.999 | 1 | 0.000246 |
| 15 | PRUNED | pruned at 25000 steps (193286 on 5) | 0.00151 | 80 | 0.999 | 0.95 | 0.000489 |
| 14 | PRUNED | pruned at 25000 steps (232902 on 5) | 0.00156 | 80 | 0.999 | 0.95 | 0.00363 |
| 16 | PRUNED | pruned at 25000 steps (258908 on 5) | 0.00208 | 80 | 0.99 | 0.95 | 0.00303 |
| 17 | PRUNED | pruned at 25000 steps (299109 on 5) | 0.00122 | 320 | 0.999 | 0.95 | 0.00107 |
| 9 | PRUNED | pruned at 25000 steps (339283 on 5) | 0.000102 | 320 | 0.99 | 0.9 | 0.0229 |
| 7 | PRUNED | pruned at 25000 steps (352927 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 5 | PRUNED | pruned at 75000 steps (353935 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 19 | PRUNED | pruned at 25000 steps (410019 on 5) | 0.00103 | 80 | 0.99 | 0.9 | 0.000632 |
| 8 | PRUNED | pruned at 25000 steps (466538 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
