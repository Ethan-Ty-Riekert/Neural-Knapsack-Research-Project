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

The repository currently implements the simplest static version of the bin-packing problem:

- All items (VM requests) are known in advance.
- Items arrive sequentially in a fixed order.
- Bins (physical machines) have fixed multi-dimensional capacities.
- The agent or heuristic chooses to place an item into an existing bin or open a new one.
- No time dimension, no arrivals or departures, and no energy or SLA modelling at this stage.

This forms the foundation for later extensions into dynamic cloud resource allocation.

## Repository Structure
Neural-Knapsack-Research-Project/
│
├── Code/
│   ├── scheduling_env.py       # Core resource-constrained scheduling environment
│   ├── gym_scheduling_wrapper.py # Gymnasium wrapper (flattened actions, obs, masking)
│   ├── env_config.py           # Centralised, seeded environment configuration
│   ├── train_rl_agent.py       # Curriculum training entry point (PPO / A2C)
│   ├── train_optimized.py      # Training with Optuna-optimized hyperparameters
│   ├── optuna_tune.py          # Hyperparameter optimization using Optuna
│   ├── eval_rl_agent.py        # Evaluation of a trained agent against heuristics
│   ├── plotting_utils.py       # Shared live + saved plotting for training/evaluation
│   ├── OPTUNA_GUIDE.md         # Comprehensive guide for hyperparameter tuning
│   ├── Policies/                # PPO (sb3_contrib) and hand-rolled maskable A2C
│   └── Testing/
│       └── test_env.py         # Test script and visualisation
│
├── rl_training/
│   ├── models/                 # Saved model checkpoints + env config snapshots
│   ├── logs/                   # TensorBoard logs
│   ├── optuna_results/         # Optuna optimization results and visualizations
│   └── plots/                  # Live-plotting output: training/<run>/ and eval/<run>/
│       └── ...                 # PNG snapshots + CSV reward logs per run (see plotting_utils.py)
│
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies



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
- **Policy collapse to idling**: Agent learns to always idle instead of scheduling jobs
- **Large action space**: ~1000 discrete actions (jobs × machines) makes exploration difficult
- **PPO gradient clipping**: Can trap policies in suboptimal regions for large action spaces
- **Reward imbalance**: Idling penalty must be carefully balanced against other penalties

### Solution
Use Optuna to automatically find optimal hyperparameters:

**Quick Start:**
```bash
cd Code

# 1. Run hyperparameter optimization (50-100 trials recommended)
python optuna_tune.py --algo ppo --trials 50

# 2. Train with optimized parameters
python train_optimized.py --algo ppo
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
See `Code/OPTUNA_GUIDE.md` for detailed instructions, troubleshooting, and advanced usage.

## Planned Extensions

Future work will expand the environment toward realistic cloud resource allocation, including:

- Dynamic workloads with VM arrivals and departures
- VM lifetimes and time-based events
- Energy-aware reward functions
- SLA/QoS modelling
- Multi-objective optimisation
- RL agents (PPO, DQN, Actor–Critic)
- Comparison against classical heuristics (First Fit, Best Fit, Vector Best Fit)
- Evaluation on synthetic and real workload traces

## Academic Context

This project is supervised by Elham Mardaneh with co-supervision from Tony Mathew.

The work draws on literature in:
- Cloud resource allocation  
- Vector and multidimensional bin packing  
- Reinforcement learning  
- Neural combinatorial optimisation  

Additional notes, summaries, and mathematical formulations are maintained separately.
