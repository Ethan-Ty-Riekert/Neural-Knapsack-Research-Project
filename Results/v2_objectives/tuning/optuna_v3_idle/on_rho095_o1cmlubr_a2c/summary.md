# Optuna study on_rho095_o1cmlubr_a2c (protocol v3_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 3 | COMPLETE | 30857 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 4 | COMPLETE | 40378 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 1 | COMPLETE | 41763 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 6 | COMPLETE | 44416 | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
| 2 | COMPLETE | 189648 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 0 | COMPLETE | 303956 | default | default | default | default | default |
| 7 | PRUNED | pruned at 75000 steps (38023 on 5) | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 11 | PRUNED | pruned at 75000 steps (45660 on 5) | 0.00035 | 80 | 0.995 | 1 | 0.00175 |
| 9 | PRUNED | pruned at 75000 steps (100738 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 8 | PRUNED | pruned at 75000 steps (166941 on 5) | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 5 | PRUNED | pruned at 75000 steps (192491 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 10 | PRUNED | pruned at 75000 steps (192491 on 5) | 0.000257 | 320 | 0.99 | 1 | 0.00191 |
