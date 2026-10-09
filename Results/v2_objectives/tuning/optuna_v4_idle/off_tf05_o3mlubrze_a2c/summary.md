# Optuna study off_tf05_o3mlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 2 | COMPLETE | 31539 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 5 | COMPLETE | 37132 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 0 | COMPLETE | 38480 | default | default | default | default | default |
| 3 | COMPLETE | 42200 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 1 | COMPLETE | 46185 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 4 | COMPLETE | 47367 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 7 | PRUNED | pruned at 225000 steps (47284 on 5) | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 10 | PRUNED | pruned at 225000 steps (67497 on 5) | 0.00245 | 80 | 0.99 | 1 | 0.00281 |
| 9 | PRUNED | pruned at 75000 steps (77015 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 6 | PRUNED | pruned at 75000 steps (88289 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 8 | PRUNED | pruned at 75000 steps (104082 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 11 | PRUNED | pruned at 75000 steps (118708 on 5) | 0.00112 | 320 | 0.999 | 0.95 | 0.029 |
