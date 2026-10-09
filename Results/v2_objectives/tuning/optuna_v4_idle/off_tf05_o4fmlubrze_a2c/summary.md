# Optuna study off_tf05_o4fmlubrze_a2c (protocol v4_idle)

12 trials x 300000 steps (TPE seeded per worker, MedianPruner); validation seeds 600000-600019.

| trial | state | validation J | learning-rate | rollout-size | gamma | gae-lambda | ent-coef |
|---|---|---|---|---|---|---|---|
| 4 | COMPLETE | 29492 | 0.00048 | 20 | 0.995 | 1 | 0.00135 |
| 3 | COMPLETE | 30514 | 0.000413 | 20 | 0.999 | 1 | 0.00109 |
| 2 | COMPLETE | 30869 | 0.000604 | 80 | 0.999 | 1 | 0.00954 |
| 0 | COMPLETE | 31644 | default | default | default | default | default |
| 1 | COMPLETE | 38788 | 0.000647 | 20 | 0.995 | 0.95 | 0.00915 |
| 9 | PRUNED | pruned at 150000 steps (29731 on 5) | 0.000966 | 20 | 0.999 | 0.95 | 0.000329 |
| 6 | PRUNED | pruned at 150000 steps (31204 on 5) | 0.000691 | 80 | 0.995 | 1 | 0.000141 |
| 5 | PRUNED | pruned at 75000 steps (41866 on 5) | 0.00103 | 80 | 0.99 | 1 | 0.025 |
| 8 | PRUNED | pruned at 75000 steps (46142 on 5) | 0.000613 | 20 | 0.99 | 0.9 | 0.00902 |
| 11 | PRUNED | pruned at 75000 steps (64913 on 5) | 0.00284 | 320 | 0.995 | 0.95 | 0.00011 |
| 10 | PRUNED | pruned at 75000 steps (82041 on 5) | 0.000142 | 80 | 0.99 | 0.9 | 0.00165 |
| 7 | PRUNED | pruned at 75000 steps (86184 on 5) | 0.00029 | 320 | 0.999 | 0.9 | 0.0236 |
