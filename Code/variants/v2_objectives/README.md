# Variant v2: selectable objectives + difficulty

**Status:** planned. The formal definitions come first and need user review; implementation follows.
**Design record:** `Future/research/2026-09-28-objective-redesign-discussion.md`

## Idea

1. **Select objectives**: any of
   - weighted tardiness ΣwT
   - weighted late-job count ΣwU
   - energy: active machine-ticks by default, or a SPECpower curve
   - dropped jobs (always included)
2. **Select difficulty**: load factor (arrival rate), deadline tightness (tardiness factor TF and due-date
   range RDD), weight range, size distribution, and later DAG depth.
3. **Compare every method** (heuristics, PSO, CP-SAT, RL) on that configuration, through `run.py`.

## Decided so far

- The reward is exactly the selected objectives. No +3/+50 bonuses, activation, hotspot, flat idle or
  invalid-action terms.
- A dropped job costs ρ_j, charged at its latest start (H − P_j). ρ_j is set above the largest late-cost
  that job could incur, so dropping never beats finishing late. Potential-based shaping and a slack
  observation feature help with credit assignment. **OPEN:** the user is still weighing this.
- Energy: under the linear power model, minimising energy is the same as minimising active machine-ticks.
  A SPECpower curve is optional.
- Hotspot term removed. Overload slowdown comes later, as a difficulty setting.
- Each objective stays in its physical units, with λ as an explicit exchange rate between objectives. One
  global constant scales the whole reward (which provably doesn't change the optimal policy).
- Idle stays a legal action, with no flat penalty.
