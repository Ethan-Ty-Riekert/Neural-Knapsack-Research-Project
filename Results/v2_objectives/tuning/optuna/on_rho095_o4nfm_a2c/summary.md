# Optuna study on_rho095_o4nfm_a2c

20 trials x 100000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 3 | COMPLETE | 331241 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 0 | COMPLETE | 332950 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 2 | COMPLETE | 340269 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 4 | COMPLETE | 343921 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 1 | COMPLETE | 348297 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 6 | COMPLETE | 464713 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 18 | COMPLETE | 497686 | 0.000222 | 20 | 0.995 | 1 | 0.000107 |
| 13 | PRUNED | pruned at 25000 steps (173960 on 5) | 0.00113 | 80 | 0.995 | 1 | 0.00367 |
| 9 | PRUNED | pruned at 25000 steps (179441 on 5) | 0.000102 | 320 | 0.99 | 0.9 | 0.0229 |
| 19 | PRUNED | pruned at 25000 steps (216034 on 5) | 0.0005 | 320 | 0.99 | 0.9 | 0.000235 |
| 5 | PRUNED | pruned at 25000 steps (239423 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 7 | PRUNED | pruned at 25000 steps (242876 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 12 | PRUNED | pruned at 25000 steps (258406 on 5) | 0.000333 | 20 | 0.995 | 1 | 0.000118 |
| 14 | PRUNED | pruned at 25000 steps (286659 on 5) | 0.000306 | 320 | 0.995 | 0.95 | 0.000326 |
| 11 | PRUNED | pruned at 25000 steps (293537 on 5) | 0.00037 | 80 | 0.995 | 0.95 | 0.00361 |
| 17 | PRUNED | pruned at 25000 steps (302995 on 5) | 0.00212 | 80 | 0.995 | 0.95 | 0.00845 |
| 8 | PRUNED | pruned at 25000 steps (324994 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 16 | PRUNED | pruned at 25000 steps (346580 on 5) | 0.000713 | 80 | 0.99 | 0.95 | 0.00143 |
| 15 | PRUNED | pruned at 25000 steps (348891 on 5) | 0.00128 | 20 | 0.995 | 1 | 0.0294 |
| 10 | PRUNED | pruned at 25000 steps (362332 on 5) | 0.00245 | 80 | 0.995 | 1 | 0.00254 |
