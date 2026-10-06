| preset | best heuristic (J) | best RL (J) | RL vs best heuristic | PSO (J) | CP-SAT (J) |
|---|---|---|---|---|---|
| off_c_50 | LST+FirstFit (98) | - | - | - | 234 |
| off_tf05_w1 | LST+FirstFit (13098) | A2C Opt1 rule selection +Consolidate non-delay [tuned] (13318, 3 seeds) | +1.7% | - | 16469 |
| off_tf08_w1 | ATC+FirstFit (210328) | - | - | - | - |
| off_tf02 | EDF+FirstFit (0) | - | - | - | - |
| off_tf05 | LST+FirstFit (39536) | PPO Opt3 ATC-prior score windowed non-delay [tuned] (31291, 3 seeds) | -20.9% | - | 35681 |
| off_tf08 | LST+FirstFit (390493) | - | - | - | - |
| on_rho050 | LST+Consolidate (4087) | - | - | - | - |
| on_rho075 | LST+Consolidate (6051) | - | - | - | - |
| on_rho075_tight | LST+Consolidate (29975) | - | - | - | - |
| on_rho095 | EDF+Consolidate (21986) | A2C Opt0 full action space pointer non-delay [tuned] (44321, 3 seeds) | +101.6% | - | - |
| on_rho110 | EDF+Consolidate (114142) | - | - | - | - |
