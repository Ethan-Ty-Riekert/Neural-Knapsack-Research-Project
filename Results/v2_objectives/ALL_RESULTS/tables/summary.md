| preset | best heuristic (J) | best RL (J) | RL vs best heuristic | PSO (J) | CP-SAT (J) |
|---|---|---|---|---|---|
| off_c_15 | LST+FirstFit (192) | - | - | - | 192 |
| off_tf02 | LST+FirstFit (0) | - | - | - | - |
| off_tf05 | LST+FirstFit (37457) | RL Opt1 rule selection (38885, 1 seed) | +3.8% | 92729 | - |
| off_tf08 | LST+FirstFit (383691) | RL Opt3 ATC-prior score (381318, 1 seed) | -0.6% | - | - |
| on_rho050 | ATC+Consolidate (3721) | - | - | - | - |
| on_rho075 | LST+Consolidate (7983) | RL Opt1 rule selection +Consolidate (12608, 1 seed) | +57.9% | - | - |
| on_rho075_tight | LST+Consolidate (32786) | - | - | - | - |
| on_rho095 | EDF+Consolidate (21923) | RL Opt1 rule selection (144248, 3 seeds) | +558.0% | 108604 | - |
| on_rho110 | EDF+Consolidate (99781) | RL Opt1 rule selection +Consolidate (373310, 3 seeds) | +274.1% | 422596 | - |
