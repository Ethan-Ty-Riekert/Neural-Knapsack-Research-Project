# Optuna study on_rho095_o3wmlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 7 | COMPLETE | 73434 | 0.00163 | 80 | 0.999 | 0.95 | 0.000502 |
| 8 | COMPLETE | 109227 | 0.000151 | 320 | 0.999 | 0.9 | 0.000171 |
| 11 | COMPLETE | 130038 | 0.000108 | 320 | 0.999 | 0.9 | 0.000115 |
| 1 | COMPLETE | 141960 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 2 | COMPLETE | 143962 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 4 | COMPLETE | 187879 | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 3 | COMPLETE | 224365 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 0 | COMPLETE | 286518 | default | default | default | default | default |
| 10 | PRUNED | pruned at 75000 steps (190536 on 5) | 0.00253 | 80 | 0.99 | 0.95 | 0.00265 |
| 9 | PRUNED | pruned at 75000 steps (229798 on 5) | 0.000709 | 20 | 0.995 | 0.95 | 0.0113 |
| 5 | PRUNED | pruned at 225000 steps (268010 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 6 | PRUNED | pruned at 75000 steps (272575 on 5) | 0.000173 | 20 | 0.99 | 0.9 | 0.000819 |
