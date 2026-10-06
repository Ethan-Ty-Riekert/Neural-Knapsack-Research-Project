# Optuna study off_tf05_o4nfm_a2c

20 trials x 100000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 3 | COMPLETE | 67917 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 12 | COMPLETE | 79823 | 0.000304 | 80 | 0.999 | 1 | 0.000108 |
| 11 | COMPLETE | 88611 | 0.00037 | 80 | 0.999 | 1 | 0.00361 |
| 18 | COMPLETE | 90037 | 0.000222 | 80 | 0.995 | 1 | 0.000547 |
| 13 | COMPLETE | 91394 | 0.000284 | 80 | 0.995 | 1 | 0.00013 |
| 1 | COMPLETE | 92494 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 6 | COMPLETE | 112484 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 4 | COMPLETE | 132116 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 10 | COMPLETE | 138032 | 0.00245 | 80 | 0.995 | 1 | 0.00254 |
| 0 | COMPLETE | 139245 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 2 | COMPLETE | 230848 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 16 | PRUNED | pruned at 50000 steps (129683 on 5) | 0.000356 | 80 | 0.995 | 1 | 0.000231 |
| 5 | PRUNED | pruned at 50000 steps (161763 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 7 | PRUNED | pruned at 50000 steps (168275 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 9 | PRUNED | pruned at 50000 steps (171532 on 5) | 0.000102 | 320 | 0.99 | 0.9 | 0.0229 |
| 15 | PRUNED | pruned at 25000 steps (183572 on 5) | 0.000256 | 80 | 0.999 | 1 | 0.000296 |
| 19 | PRUNED | pruned at 25000 steps (190527 on 5) | 0.000396 | 80 | 0.99 | 0.9 | 0.00021 |
| 17 | PRUNED | pruned at 25000 steps (202179 on 5) | 0.00107 | 320 | 0.999 | 1 | 0.000105 |
| 14 | PRUNED | pruned at 25000 steps (217050 on 5) | 0.00115 | 80 | 0.99 | 1 | 0.000113 |
| 8 | PRUNED | pruned at 25000 steps (238610 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
