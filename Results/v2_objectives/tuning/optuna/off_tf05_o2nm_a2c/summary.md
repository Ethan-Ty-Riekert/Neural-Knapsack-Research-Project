# Optuna study off_tf05_o2nm_a2c

20 trials x 100000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 1 | COMPLETE | 59482 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 15 | COMPLETE | 69670 | 0.000478 | 20 | 0.999 | 1 | 0.0292 |
| 2 | COMPLETE | 77706 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 10 | COMPLETE | 78629 | 0.00245 | 80 | 0.999 | 1 | 0.00279 |
| 14 | COMPLETE | 94052 | 0.00119 | 320 | 0.99 | 1 | 0.00107 |
| 5 | COMPLETE | 103331 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 6 | COMPLETE | 109900 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 3 | COMPLETE | 129220 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 0 | COMPLETE | 157614 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 4 | COMPLETE | 170350 | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 19 | PRUNED | pruned at 25000 steps (149266 on 5) | 0.00104 | 320 | 0.999 | 0.9 | 0.0189 |
| 13 | PRUNED | pruned at 25000 steps (150860 on 5) | 0.000376 | 80 | 0.995 | 1 | 0.00528 |
| 12 | PRUNED | pruned at 25000 steps (155950 on 5) | 0.000352 | 20 | 0.999 | 1 | 0.00191 |
| 11 | PRUNED | pruned at 25000 steps (174553 on 5) | 0.000316 | 80 | 0.995 | 1 | 0.0022 |
| 18 | PRUNED | pruned at 25000 steps (178161 on 5) | 0.000537 | 20 | 0.999 | 1 | 0.00523 |
| 17 | PRUNED | pruned at 25000 steps (186494 on 5) | 0.000245 | 80 | 0.999 | 1 | 0.0117 |
| 9 | PRUNED | pruned at 25000 steps (187706 on 5) | 0.000102 | 320 | 0.99 | 0.9 | 0.0229 |
| 16 | PRUNED | pruned at 25000 steps (192183 on 5) | 0.0002 | 80 | 0.999 | 1 | 0.0258 |
| 8 | PRUNED | pruned at 25000 steps (196118 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 7 | PRUNED | pruned at 25000 steps (203753 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
