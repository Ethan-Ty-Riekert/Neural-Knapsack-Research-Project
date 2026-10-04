# Neural-Knapsack Research Project

This repository contains the codebase, experiments, and documentation for my Curtin University  
Third-Year Research Project (2026) on neural combinatorial optimisation applied to  
bin packing and cloud resource allocation.

## Project Overview

Cloud datacentres must allocate virtual machines (VMs) to physical hosts under multi-dimensional
resource constraints such as CPU, RAM, and storage. This problem is commonly modelled as
Vector Bin Packing (VBP) or Multidimensional Bin Packing (MDBP), both of which are NP-hard.

Recent literature suggests that reinforcement learning (RL) and neural combinatorial optimisation
(NCO) may outperform classical heuristics for large-scale, dynamic cloud workloads.  
This project investigates whether RL-based policies can learn efficient packing strategies that 
improve a set of metrics like minimize active servers, energy usage, reduce SLA violations ...

## Current Stage of Development

**Updated 2026-09-18 (S2W9) -- this section previously described the project's very
first (2026-07/08) static-only implementation and had not been revised since, despite
dynamic arrivals having been designed, implemented, and evaluated. See
`PROGRESS.md` and `Future/research/training-log.md` for the full run-by-run record.**

The repository implements two related problem variants, both with a genuine time
dimension (deadlines, tardiness, a per-tick decision clock -- `Code/env/
scheduling_env.py`):

- **Offline case**: all jobs (VM requests) are known in advance, either as one fixed
  instance or resampled fresh per episode. This is the more heavily validated case --
  an exact CP-SAT baseline (`Code/baselines/exact_solver.py`) proves the true optimum
  on the real deployed 100-job instance (tardiness=8.0), and RL policies using an
  action-space-reduced design (see below) reach within a few units of it.
- **Online case** (`Code/env/online_scheduling_env.py`, `arrival_process.py`): jobs
  arrive dynamically via a Poisson process, optionally with heavy-tailed (log-normal)
  job sizes matching published Google/Azure cluster-trace characteristics, so the
  policy must decide under genuine uncertainty about future arrivals. A retrospective
  CP-SAT oracle (`solve_retrospective()`) gives this case a hindsight lower-bound
  reference point once an episode's realized arrival order is known.

Both cases support randomized per-job weights (`job_weight_range`) feeding into a
weighted-tardiness objective, and three action-space designs
(`Code/training/train_action_space_variant.py`, Options 1-3) that shrink the raw
`jobs x machines` action space -- identified as PPO's actual bottleneck after seven
other fixes (reward tuning, Lagrangian constraints, architecture changes) failed to
move it. Energy usage and SLA/QoS modelling are not yet implemented -- see Planned
Extensions.

## Repository Structure

Reorganised 2026-09-28 (S2W11) by **problem variant** and **method family** -- see
`Future/research/2026-09-28-objective-redesign-discussion.md` section 11. Old module paths
(`Code.env.*`, `Code.baselines.*`, `Code.policies.*`, `Code.training.*`, `Code.evaluation.*`)
still work as compatibility shims, so older commands and checkpoints keep loading.

```
Neural-Knapsack-Research-Project/
|
|-- run.py                     # ONE launcher: variant -> preset -> method (see below)
|-- experiments/               # saved run configurations (YAML) for run.py --experiment
|
|-- Code/
|   |-- core/                  # the shared problem: SchedulingEnv, OnlineSchedulingEnv,
|   |                          #   instance generators (env_config, arrival_process),
|   |                          #   base gym wrappers + action masking
|   |-- variants/              # one package per problem/reward definition, each with
|   |   |-- v1_legacy_reward/  #   PRESETS = exactly reproducible evaluation protocols
|   |   `-- v2_objectives/     #   selectable objectives + difficulty (planned)
|   |-- methods/               # every solution method, one package per family
|   |   |-- heuristics/        #   priority rule x placement rule registry (EDF, ATC, ...)
|   |   |-- metaheuristic/     #   PSO
|   |   |-- exact/             #   CP-SAT
|   |   `-- rl/                #   policies/, action_spaces/ (Options 1-4), training/, evaluation/
|   `-- utils/                 # paths (artefact locations), results_log, plotting, diagnostics
|
|-- Results/                   # curated results, one folder per variant (see Results/README.md)
|   |-- v1_legacy_reward/      #   everything up to 2026-09-28 + runs/ from run.py
|   `-- v2_objectives/
|-- tests/                     # regression scripts: python -m tests.<name>
|-- Future/research/           # training-log.md + dated design/investigation docs + references.bib
|-- PROGRESS.md                # narrative story of the project
`-- rl_training/               # generated, gitignored (or NK_ARTIFACTS_DIR/<machine>/, see below)
```

### Running things

```
python run.py                  # interactive menu
python run.py --list           # variants, presets, methods
python run.py --variant v1_legacy_reward --preset off_c_15 --method EDF
python run.py --variant v1_legacy_reward --preset off_c_small --method cpsat
python run.py --variant v1_legacy_reward --preset on_r_50 --method rl-eval:3
python run.py --experiment experiments/pso_vs_edf_off_c_15.yaml

# v2 RL: train (parallel envs, seeded; Option 1 may also choose the placement rule), then compare
python run.py --variant v2_objectives --preset on_rho095 --method rl-train:1 --n-envs 4 --seed 0 \
    --rule-placements FirstFit,Consolidate --timesteps 750000 --checkpoint-tag v2_on_rho095_o1c_s0
python run.py --variant v2_objectives --preset on_rho095 \
    --method "rl-eval:1:v2_on_rho095_o1c_s0,EDF+Consolidate,RandomRule+FirstFitConsolidate"
python Results/v2_objectives/ALL_RESULTS/scripts/build_folder.py   # rebuild all v2 tables/figures
```

Trained RL models save a sidecar `<model>.json` (action menu, ATC feature, window, seed, gamma), so
`rl-eval:<option>:<tag>` rebuilds the matching environment from the tag alone. Tag convention:
`v2_<preset>_o<option>[c]_s<seed>` (`c` = FirstFit+Consolidate menu).

Heuristic / PSO / CP-SAT runs are saved to `Results/<variant>/runs/<timestamp>_<preset>_<method>/`
(`run.json` with git commit, machine, full config and metrics, plus `per_instance.csv`). Every run
reports **jobs scheduled and dropped next to tardiness**, because tardiness alone hides dropped jobs.
RL methods delegate to the training/evaluation scripts with the preset's exact flags.

Individual modules can still be run directly as modules from the repo root, e.g.
`python -m Code.methods.rl.training.train_action_space_variant --option 1` -- not as script files,
which breaks the `Code.*` absolute imports.

### Two machines: artefact storage

Set `NK_ARTIFACTS_DIR` to a cloud-synced folder and (optionally) `NK_MACHINE` to a machine name;
generated artefacts then go to `NK_ARTIFACTS_DIR/<machine>/` instead of `rl_training/`. Each machine
writes only its own subfolder (sync tools corrupt `optuna.db` and fork appended CSVs otherwise).
Unset = the original `rl_training/` behaviour.

## Components

### Bin
Represents a physical machine with:
- A capacity vector  
- Remaining resource tracking  
- A list of items placed into the bin  

### BasicBinPackingEnv
A minimal RL-compatible environment that supports:
- State extraction (dictionary and vector forms)
- Action masking for feasible placements
- Dynamic bin creation
- Step-by-step item placement and reward feedback

### Testing and Visualisation
Includes:
- A simple first-fit baseline policy
- 3D visualisation of bin capacity and item placement for interpretability

## Hyperparameter Optimization

The project includes comprehensive hyperparameter optimization using Optuna to address common RL challenges:

### Current Challenges
- **Policy collapse to idling**: historically the agent learned to always idle instead
  of scheduling jobs. Largely traced to a since-fixed environment bug (capacity never
  properly reset between episodes) plus missing exploration/reward-normalisation in
  the hand-rolled A2C -- see `Future/research/training-log.md` and
  `Future/research/2026-08-09-pointer-network-action-head.md` for the current
  understanding and what's still open.
- **Large action space**: ~1000 discrete actions (jobs × machines) makes exploration difficult
- **PPO gradient clipping**: Can trap policies in suboptimal regions for large action spaces
- **Reward imbalance**: Idling penalty must be carefully balanced against other penalties

### Solution
Use Optuna to automatically find optimal hyperparameters:

**Quick Start** (run from the repo root):
```bash
# 1. Run hyperparameter optimization (50-100 trials recommended)
python -m Code.training.optuna_tune --algo ppo --trials 50

# 2. Train with optimized parameters
python -m Code.training.train_optimized --algo ppo
```

**What Gets Optimized:**
- Network architecture (layer sizes, depth, activation functions)
- Learning rates and batch sizes
- Entropy coefficient (critical for exploration)
- Reward penalties (λ₁, λ₂, λ₃, idle_penalty, invalid_penalty)
- PPO-specific: clip_range, n_epochs, GAE parameters
- A2C-specific: n_steps, value coefficient

**Results:**
Optimization results are saved to `rl_training/optuna_results/`:
- Best hyperparameters (JSON)
- All trials history (CSV)
- Interactive visualizations (HTML plots)
- Parameter importance analysis

**Documentation:**
See `docs/OPTUNA_GUIDE.md` for detailed instructions, troubleshooting, and advanced usage.

## Planned Extensions

**Updated 2026-09-18 (S2W9)** -- several items below are now implemented (struck
through); genuinely remaining future work follows.

- ~~Dynamic workloads with VM arrivals~~ -- implemented (`online_scheduling_env.py`,
  Poisson arrivals, optional heavy-tailed job sizes)
- VM/job *departures* and lifetimes -- not yet modelled; jobs currently run to
  completion once started, with no early termination or resource release mid-run
  beyond normal completion
- Energy-aware reward functions -- not yet implemented
- SLA/QoS modelling -- not yet implemented (tardiness/deadlines are modelled; explicit
  SLA-tier or QoS-class distinctions are not)
- ~~Multi-objective optimisation~~ -- implemented and explored (Pareto-front search
  over the reward's lambda weights; see `PROGRESS.md` Phase 12's Pareto arc)
- ~~RL agents (PPO, Actor-Critic)~~ -- implemented (`MaskablePPO` via `sb3_contrib`, a
  hand-rolled masked A2C); DQN not attempted (action-masking + the continuous-ish
  priority-scoring designs in Options 2/3 made policy-gradient methods the more
  natural fit given this project's action-space designs)
- ~~Comparison against classical heuristics~~ -- implemented, expanded well beyond
  First/Best Fit to a 9-heuristic suite (EDF, SPT, LST, FCFS, LPT, WSPT, ATC, Tetris,
  each paired with a placement rule) plus an exact CP-SAT baseline
  (`Code/baselines/`)
- Evaluation on real (not just synthetic) workload traces -- not yet done; the
  log-normal job-size distribution is grounded in published trace statistics (Reiss et
  al. 2012; Cortez et al. 2017) but no real trace file has been replayed directly
- A deployed-scale hyperparameter search for the action-space-reduced (Options 1-3)
  designs -- not yet run; current best numbers use un-tuned `MaskablePPO` defaults

## Academic Context

This project is supervised by Elham Mardaneh with co-supervision from Tony Mathew.

The work draws on literature in:
- Cloud resource allocation  
- Vector and multidimensional bin packing  
- Reinforcement learning  
- Neural combinatorial optimisation  

Additional notes, summaries, and mathematical formulations are maintained separately.
