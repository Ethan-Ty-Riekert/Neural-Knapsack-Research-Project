| preset | best heuristic (J) | best RL (J) | RL vs best heuristic | PSO (J) | CP-SAT (J) |
|---|---|---|---|---|---|
| off_c_50 | LST+FirstFit (98) | - | - | - | 234 |
| off_tf02_w1 | WLST+FirstFit (0) | - | - | - | - |
| off_tf05_w1 | LST+FirstFit (13098) | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival-aware critic lateness shaping [tuned] (13098, 3 seeds) | +0.0% | - | 16469 |
| off_tf08_w1 | WLST+FirstFit (129805) | - | - | - | - |
| off_tf02 | EDF+FirstFit (0) | - | - | - | - |
| off_tf05 | WLST+FirstFit (26670) | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle [tuned] (28061, 3 seeds) | +5.2% | - | 35681 |
| off_tf08 | WLST+FirstFit (257155) | - | - | - | - |
| on_rho050 | WLST+Consolidate (4085) | - | - | - | - |
| on_rho075 | WLST+Consolidate (5990) | - | - | - | - |
| on_rho075_tight | WLST+Consolidate (29502) | - | - | - | - |
| on_rho095 | EDF+Consolidate (21986) | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget [tuned] (26515, 3 seeds) | +20.6% | - | - |
| on_rho110 | MDC+Consolidate (106463) | - | - | - | - |
