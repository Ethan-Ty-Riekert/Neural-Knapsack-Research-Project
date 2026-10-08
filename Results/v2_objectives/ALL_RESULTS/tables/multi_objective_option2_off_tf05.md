# Option 2 PPO, off_tf05: effect of the training objective (50 test instances, mean over seeds)

Reference WLST+FirstFit: J 26,670, late jobs 61.7, active machine-ticks 173.8. lambda_late = 146, lambda_energy = 153.

| training objective | J (sq. tardiness) | late jobs | on-time rate | max tardiness | weighted tardiness | active machine-ticks | energy (SPECpower) |
|---|---|---|---|---|---|---|---|
| J only (3 seeds) | 42,171 | 56.8 | 0.432 | 51.2 | 1,742 | 186.8 | 156.1 |
| J + late jobs (3 seeds) | 32,521 | 52.3 | 0.477 | 41.8 | 1,512 | 174.6 | 147.9 |
| J + energy (3 seeds) | 30,094 | 58.0 | 0.420 | 40.1 | 1,665 | 174.4 | 147.8 |
| J + late jobs + energy (3 seeds) | 34,264 | 53.6 | 0.464 | 41.9 | 1,610 | 175.4 | 148.4 |
