# Optuna study off_tf05_o0pmlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 3 | COMPLETE | 28213 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 0 | COMPLETE | 29438 | default | default | default | default | default |
| 10 | COMPLETE | 29676 | 0.000257 | 320 | 0.99 | 1 | 0.00191 |
| 4 | COMPLETE | 37376 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 1 | COMPLETE | 39369 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 6 | COMPLETE | 71806 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 2 | COMPLETE | 275780 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 8 | PRUNED | pruned at 150000 steps (56765 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 9 | PRUNED | pruned at 150000 steps (57201 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 7 | PRUNED | pruned at 75000 steps (209482 on 5) | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 11 | PRUNED | pruned at 75000 steps (209482 on 5) | 0.00276 | 20 | 0.995 | 1 | 0.029 |
| 5 | PRUNED | pruned at 75000 steps (263701 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
