# Optuna study off_tf05_o3wmlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 1 | COMPLETE | 28530 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 5 | COMPLETE | 29021 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 2 | COMPLETE | 29230 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 7 | COMPLETE | 32934 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 4 | COMPLETE | 39369 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 0 | COMPLETE | 41579 | default | default | default | default | default |
| 3 | COMPLETE | 216729 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 10 | PRUNED | pruned at 225000 steps (31660 on 5) | 0.00245 | 320 | 0.99 | 0.95 | 0.00281 |
| 11 | PRUNED | pruned at 150000 steps (42266 on 5) | 0.00118 | 20 | 0.995 | 0.95 | 0.029 |
| 8 | PRUNED | pruned at 75000 steps (64690 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 9 | PRUNED | pruned at 75000 steps (144486 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 6 | PRUNED | pruned at 75000 steps (166721 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
