| preset | best heuristic (J) | best RL (J) | RL vs best heuristic | PSO (J) | CP-SAT (J) |
|---|---|---|---|---|---|
| off_c_50 | LST+FirstFit (98) | - | - | - | 234 |
| off_tf02_w1 | WLST+FirstFit (0) | - | - | - | - |
| off_tf05_w1 | LST+FirstFit (13098) | PPO Opt1 rule selection +Consolidate look-ahead fixed scaling arrival-aware critic lateness shaping [tuned] (13098, 3 seeds) | +0.0% | - | 16469 |
| off_tf08_w1 | WLST+FirstFit (129805) | - | - | - | - |
| off_tf02 | EDF+FirstFit (0) | - | - | - | - |
| off_tf05 | WLST+FirstFit (26670) | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle [tuned] (28061, 3 seeds) | +5.2% | - | 35681 |
| off_tf08 | WLST+FirstFit (257155) | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle +late-count objective +energy objective (309725, 3 seeds) | +20.4% | - | - |
| on_rho050 | WLST+Consolidate (4085) | - | - | - | - |
| on_rho075 | WLST+Consolidate (5990) | PPO Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle [tuned] (10550, 3 seeds) | +76.1% | - | - |
| on_rho075_tight | WLST+Consolidate (29502) | - | - | - | - |
| on_rho095 | EDF+Consolidate (21986) | PPO Opt2 priority score look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle 2M-step budget [tuned] (24921, 2 seeds) | +13.3% | - | - |
| on_rho110 | MDC+Consolidate (106463) | PPO Opt3 ATC-prior score windowed look-ahead fixed scaling arrival-aware critic lateness shaping reward scaling event-driven idle [tuned] (190523, 3 seeds) | +79.0% | - | - |
