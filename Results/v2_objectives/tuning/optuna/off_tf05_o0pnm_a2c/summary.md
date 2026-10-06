# Optuna study off_tf05_o0pnm_a2c

20 trials x 100000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 15 | COMPLETE | 113300 | 0.000329 | 320 | 0.999 | 0.95 | 0.0297 |
| 14 | COMPLETE | 158248 | 0.000328 | 320 | 0.999 | 0.95 | 0.0214 |
| 4 | COMPLETE | 161193 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 19 | COMPLETE | 182507 | 0.000422 | 320 | 0.99 | 0.95 | 0.0273 |
| 1 | COMPLETE | 205088 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 2 | COMPLETE | 211429 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 11 | COMPLETE | 218526 | 0.00112 | 80 | 0.999 | 1 | 0.00364 |
| 0 | COMPLETE | 218557 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 13 | COMPLETE | 219877 | 0.00113 | 80 | 0.999 | 1 | 0.00358 |
| 5 | COMPLETE | 220865 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 3 | COMPLETE | 222297 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 12 | PRUNED | pruned at 25000 steps (205714 on 5) | 0.000333 | 80 | 0.999 | 1 | 0.000287 |
| 6 | PRUNED | pruned at 25000 steps (209482 on 5) | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 10 | PRUNED | pruned at 25000 steps (209482 on 5) | 0.00248 | 20 | 0.999 | 0.95 | 0.00259 |
| 7 | PRUNED | pruned at 25000 steps (213146 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 8 | PRUNED | pruned at 25000 steps (215323 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 17 | PRUNED | pruned at 25000 steps (225376 on 5) | 0.000294 | 320 | 0.999 | 0.95 | 0.0159 |
| 18 | PRUNED | pruned at 25000 steps (226011 on 5) | 0.000222 | 320 | 0.999 | 0.95 | 0.00533 |
| 16 | PRUNED | pruned at 25000 steps (233555 on 5) | 0.000305 | 320 | 0.99 | 0.95 | 0.0275 |
| 9 | PRUNED | pruned at 25000 steps (263054 on 5) | 0.000102 | 320 | 0.99 | 0.9 | 0.0229 |
