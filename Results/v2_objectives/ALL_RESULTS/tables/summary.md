| preset | best heuristic (J) | best RL (J) | RL vs best heuristic | PSO (J) | CP-SAT (J) |
|---|---|---|---|---|---|
| off_c_50 | LST+FirstFit (98) | - | - | - | 234 |
| off_tf02_w1 | WLST+FirstFit (0) | - | - | - | - |
| off_tf05_w1 | LST+FirstFit (13098) | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival-aware critic lateness shaping [tuned] (13098, 3 seeds) | +0.0% | - | 16469 |
| off_tf08_w1 | WLST+FirstFit (129805) | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert (129805, 3 seeds) | +0.0% | - | - |
| off_tf02 | EDF+FirstFit (0) | - | - | - | - |
| off_tf05 | WLST+FirstFit (26670) | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert (27452, 4 seeds) | +2.9% | - | 35681 |
| off_tf08 | WLST+FirstFit (257155) | PPO Opt2 priority score windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert (266327, 3 seeds) | +3.6% | - | - |
| on_rho050 | WLST+Consolidate (4085) | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert (4090, 3 seeds) | +0.1% | - | - |
| on_rho075 | WLST+Consolidate (5990) | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert (6092, 2 seeds) | +1.7% | - | - |
| on_rho075_tight | WLST+Consolidate (29502) | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle WLST expert (30056, 8 seeds) | +1.9% | - | - |
| on_rho095 | EDF+Consolidate (21986) | PPO Opt2 priority score +Consolidate +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 1M-step budget (21827, 1 seed) | -0.7% | - | - |
| on_rho110 | MDC+Consolidate (106463) | PPO Opt2 priority score +Consolidate windowed +EDF imitation warm start look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle MDC expert (115590, 9 seeds) | +8.6% | - | - |
