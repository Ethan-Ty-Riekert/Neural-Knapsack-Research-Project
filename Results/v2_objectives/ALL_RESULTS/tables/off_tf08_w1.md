# off_tf08_w1: all methods (J = sum w_j T_j^2, lower is better)

| rank | method | family | J | on-time rate | weighted tardiness | max tardiness | mean wait | active machine-ticks | seeds | instances |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ATC+FirstFit | Heuristic | 210328 +/- 14586 | 0.243 +/- 0.039 | 3400 +/- 163 | 96.7 +/- 2.4 | 49.50 +/- 0.00 | 171 +/- 10 | 1 | 50 |
| 2 | ATC+BestFit | Heuristic | 210328 +/- 14586 | 0.243 +/- 0.039 | 3400 +/- 163 | 96.7 +/- 2.4 | 49.50 +/- 0.00 | 172 +/- 10 | 1 | 50 |
| 3 | ATC+Consolidate | Heuristic | 210328 +/- 14586 | 0.243 +/- 0.039 | 3400 +/- 163 | 96.7 +/- 2.4 | 49.50 +/- 0.00 | 166 +/- 7 | 1 | 50 |

Spread (+/-): std across seeds for multi-seed RL rows, otherwise std across instances (instance heterogeneity; the figures use the standard error instead).
