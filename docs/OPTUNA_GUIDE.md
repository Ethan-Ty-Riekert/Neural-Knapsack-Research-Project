# Hyperparameter Optimization Guide

This guide explains how to use Optuna to find optimal hyperparameters for your reinforcement learning agents.

## Problem Context

Your current implementation faces several challenges:

1. **Policy collapse to idling**: The agent learns to always idle (reward stuck at -30.5)
2. **Large action space**: ~1000 discrete actions (jobs × machines) makes exploration difficult
3. **PPO gradient clipping**: Can trap the policy in suboptimal regions early in training
4. **Reward imbalance**: Idling penalty too low relative to other penalties

## Solution: Hyperparameter Optimization

Optuna automatically searches for the best combination of hyperparameters by:
- Testing different network architectures
- Tuning learning rates and exploration coefficients
- Balancing reward penalties
- Optimizing batch sizes and training parameters

## Quick Start

### 1. Run Hyperparameter Optimization

For PPO (recommended):
```bash
cd Neural-Knapsack-Research-Project/Code
python -m Code.training.optuna_tune --algo ppo --trials 50
```

For A2C:
```bash
python -m Code.training.optuna_tune --algo a2c --trials 50
```

**Options:**
- `--trials N`: Number of hyperparameter combinations to try (default: 50)
- `--jobs N`: Number of parallel trials (default: 1, use higher for faster search)
- `--storage URL`: Database for persistent storage (default: sqlite:///rl_training/optuna.db)

**Example with parallel trials:**
```bash
python -m Code.training.optuna_tune --algo ppo --trials 100 --jobs 4
```

### 2. Monitor Progress

The optimization will print progress updates:
```
Trial 0 finished with value: -15.2
Trial 1 finished with value: 8.5
Trial 2 finished with value: -30.5  (agent stuck on idle)
...
Best trial: 7
Best reward: 25.3
```

### 3. View Results

After optimization completes, results are saved to `rl_training/optuna_results/`:

- `ppo_best_params.json`: Best hyperparameters found
- `ppo_trials.csv`: All trials with their parameters and rewards
- `ppo_optimization_history.html`: Interactive plot of optimization progress
- `ppo_param_importances.html`: Which parameters matter most

**Open visualizations:**
```bash
# Windows
start rl_training/optuna_results/ppo_optimization_history.html

# Linux/Mac
open rl_training/optuna_results/ppo_optimization_history.html
```

### 4. Train with Optimized Parameters

Once optimization is complete, train a full model:

```bash
python -m Code.training.train_optimized --algo ppo
```

This will:
- Load the best hyperparameters from Optuna
- Train using curriculum learning (20 → 40 → 60 → 100 jobs)
- Save the final model to `rl_training/models/ppo_scheduling_optimized.zip`

**For single-stage training (no curriculum):**
```bash
python -m Code.training.train_optimized --algo ppo --no-curriculum
```

## What Gets Optimized

### Network Architecture
- **layer_size**: 128, 256, or 512 neurons per layer
- **n_layers**: 2 or 3 hidden layers
- **activation**: Tanh or ReLU

### PPO Hyperparameters
- **learning_rate**: 1e-5 to 1e-3 (log scale)
- **n_steps**: 512, 1024, 2048, or 4096 (rollout length)
- **batch_size**: 64, 128, 256, or 512
- **n_epochs**: 3 to 10 (gradient updates per rollout)
- **gamma**: 0.95 to 0.999 (discount factor)
- **gae_lambda**: 0.9 to 0.99 (advantage estimation)
- **ent_coef**: 0.001 to 0.1 (exploration bonus) - **CRITICAL**
- **clip_range**: 0.1 to 0.3
- **vf_coef**: 0.25 to 1.0 (value loss weight)
- **max_grad_norm**: 0.3 to 1.0

### Reward Penalties
- **lambda_1**: 0.5 to 2.0 (machine activation penalty)
- **lambda_2**: 0.5 to 2.0 (tardiness penalty)
- **lambda_3**: 0.5 to 2.0 (hotspot penalty)
- **idle_penalty**: 0.5 to 3.0 - **CRITICAL**
- **invalid_penalty**: 3.0 to 10.0

## Understanding Results

### Good Trial
```
Trial 15: mean_reward=35.2, idle_ratio=0.15
```
- High positive reward
- Low idle ratio (agent schedules jobs instead of idling)

### Bad Trial (Idle Collapse)
```
Trial 8: mean_reward=-28.5, idle_ratio=0.98
```
- Negative reward near -30.5
- High idle ratio (agent always idles)
- This trial gets penalized with -100 bonus penalty

### Identifying Important Parameters

Check `ppo_param_importances.html` to see which parameters have the biggest impact on performance.

**Expected important parameters:**
1. **ent_coef** (entropy coefficient): Controls exploration
2. **idle_penalty**: Must be high enough to discourage idling
3. **learning_rate**: Affects convergence speed
4. **n_epochs**: Too many can cause overfitting to single rollouts

## Tips for Success

### 1. Start with More Trials
- 50 trials is a good start, but 100-200 gives better results
- Use `--jobs 4` or higher if you have multiple CPU cores

### 2. Resume Interrupted Optimization
Optuna saves progress to a database. If optimization stops, just run the same command again:
```bash
python -m Code.training.optuna_tune --algo ppo --trials 100
```
It will continue from where it left off.

### 3. Analyze Failed Trials
Look at `ppo_trials.csv` to identify patterns:
```python
import pandas as pd
df = pd.read_csv('rl_training/optuna_results/ppo_trials.csv')

# Find trials that didn't collapse to idle
good_trials = df[df['user_attrs_idle_ratio'] < 0.5]
print(good_trials[['params_ent_coef', 'params_idle_penalty', 'value']])
```

### 4. Manual Tuning After Optuna
Use Optuna results as a starting point, then manually fine-tune:
- If agent still idles too much: increase `ent_coef` and `idle_penalty`
- If training is unstable: reduce `learning_rate` and `n_epochs`
- If agent is too random: reduce `ent_coef`

### 5. Curriculum Learning
The optimized hyperparameters are tuned on small problems (20 jobs).
For full-scale training (100 jobs), curriculum learning is recommended:
```bash
python -m Code.training.train_optimized --algo ppo  # Uses curriculum by default
```

## Troubleshooting

### All Trials Collapse to Idle
If every trial gets stuck idling:
1. Increase the search range for `ent_coef` (higher values)
2. Increase the search range for `idle_penalty` (higher values)
3. Check that action masking is working correctly

### Optimization is Too Slow
Each trial trains for 30k timesteps. To speed up:
1. Use parallel jobs: `--jobs 4`
2. Reduce `eval_timesteps` in `optuna_tune.py` (line 133)
3. Use A2C instead of PPO (faster updates)

### Out of Memory
Reduce batch sizes in the search space:
- Edit `optuna_tune.py`, line 79
- Change `batch_size` options from `[64, 128, 256, 512]` to `[64, 128, 256]`

### Import Errors
Make sure all dependencies are installed:
```bash
pip install optuna plotly kaleido
```

## Advanced Usage

### Custom Search Space
Edit `optuna_tune.py` to customize the search space:

```python
# Example: Focus on entropy coefficient
ent_coef = trial.suggest_float("ent_coef", 0.05, 0.2, log=True)

# Example: Test specific architectures
layer_size = trial.suggest_categorical("layer_size", [256, 512, 1024])
```

### Multi-Objective Optimization
Optimize for multiple objectives (e.g., reward AND training time):

```python
# In objective_ppo function, return multiple values:
return mean_reward, -training_time  # Minimize training time

# When creating study:
study = optuna.create_study(
    directions=["maximize", "maximize"]  # Both objectives
)
```

### Pruning Bad Trials Early
Optuna automatically prunes trials that are performing poorly (using MedianPruner).
This saves time by stopping bad trials early.

## Expected Results

After optimization, you should see:
- **Idle ratio**: < 0.3 (agent schedules jobs instead of idling)
- **Mean reward**: > 10 (positive reward from job completion)
- **Consistent behavior**: Similar performance across evaluation episodes

If the best trial still shows idle collapse, you may need to:
1. Revise the reward structure in `scheduling_env.py`
2. Consider action space decomposition (see your research notes)
3. Try alternative algorithms (e.g., DQN, SAC)

## Next Steps

1. **Run optimization**: `python -m Code.training.optuna_tune --algo ppo --trials 50`
2. **Analyze results**: Check visualizations and best parameters
3. **Train full model**: `python -m Code.training.train_optimized --algo ppo`
4. **Evaluate**: Use `eval_rl_agent.py` to compare against heuristics
5. **Iterate**: If results aren't good enough, run more trials or adjust search space

Good luck with your research project!
