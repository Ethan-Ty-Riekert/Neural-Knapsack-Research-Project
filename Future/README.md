# Future Work

Placeholder for the more complex, realistic extensions to this project, once the
current fixed-size, static, single-stage scheduling environment (`Code/`) is
working reliably. Nothing here is implemented yet -- this folder exists so future
work has an obvious home and doesn't get bolted onto the existing simple
environment.

Candidate directions, roughly in order of how directly they build on the current
codebase (see `NotesForAI/` for the fuller research background):

- **Variable-size job sets without padding.** `Code/gym_scheduling_wrapper.py`
  currently handles a variable number of jobs per curriculum stage by padding
  observations/actions out to a fixed `max_jobs` capacity and masking the unused
  slots. That's a practical stopgap, not a generalisable solution -- it still
  caps the maximum problem size and wastes capacity on small instances. A
  set/sequence-based policy (pointer network, transformer encoder over job
  tokens, or a graph neural network over jobs+machines) would handle arbitrary
  job counts natively and should generalise to unseen problem sizes at
  evaluation time, matching "Option A" in the research notes.
- **Dynamic workloads.** VM/job arrivals and departures over time, rather than
  all jobs being known upfront (the current static, offline case).
- **Energy-aware and SLA-aware rewards.** Proper energy/power proxies and
  SLA-violation modelling beyond the current machine-activation and tardiness
  penalties.
- **Multi-objective evaluation.** Explicit Pareto-style comparison across
  energy, tardiness, and utilisation rather than a single weighted scalar
  reward.
- **Comparison against non-RL baselines** beyond the current EDF/SPT/LST
  heuristics (e.g. evolutionary methods, exact MILP solutions on small
  instances) for a stronger empirical baseline.
