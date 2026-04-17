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
│   ├── binpacking.py          # Core environment and Bin class
│   ├── pycache/           # Python cache files
│   └── Testing/
│       ├── test_env.py        # Test script and visualisation
│       └── pycache/       # Python cache files
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
