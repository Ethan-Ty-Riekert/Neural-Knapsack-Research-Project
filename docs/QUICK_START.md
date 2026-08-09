# Quick Start Guide - Hyperparameter Optimization

## Current Problem

Your RL agent is stuck learning to always idle, resulting in:
- Training reward converged to **-30.5** (61 timesteps × 0.5 idle penalty)
- Policy places 100% probability on the idle action
- No job scheduling happening

## Root Causes

1. **Idle penalty too low** (0.5) vs other penalties (1.0-5.0)
2. **Entropy coefficient too low** (0.01) - insufficient exploration
3. **PPO gradient clipping** trapping policy in local optima
4. **Network capacity** may be insufficient (also had typo: 2356 → 256)

## Solution - 3 Step Process

### Step 1: Install Optuna
```bash
cd Neural-Knapsack-Research-Project
pip install -r requirements.txt
```

### Step 2: Run Hyperparameter Optimization
```bash

python -m Code.training.optuna_tune --algo ppo --trials 50
```

**What this does:**
- Tests 50 different combinations of hyperparameters
- Each trial trains for 30k timesteps on a small problem (20 jobs)
- Detects and penalizes "idle-only" behavior
- Saves best parameters to `rl_training/optuna_results/ppo_best_params.json`
- Takes ~2-4 hours depending on hardware

**Monitor progress:**
- Watch for trials with positive rewards (good!)
- Trials with -30.5 reward are being penalized (idle collapse)
- Check `idle_ratio` - should be < 0.3 for good trials

### Step 3: Train with Optimized Parameters
```bash
python -m Code.training.train_optimized --algo ppo
```

**What this does:**
- Loads best parameters from Step 2
- Trains full model with curriculum learning
- Uses larger network if Optuna found it beneficial
- Saves to `rl_training/models/ppo_scheduling_optimized.zip`
- Takes ~1-2 hours

## Expected Improvements

**Before optimization:**
- Reward: -30.5 (always idle)
- Jobs scheduled: 0%
- Idle ratio: 100%

**After optimization (expected):**
- Reward: 10-50+ (positive!)
- Jobs scheduled: 70-100%
- Idle ratio: 10-30%

## Key Parameters Being Optimized

**Most Critical:**
1. **ent_coef** (entropy coefficient): 0.001 → 0.1
   - Higher = more exploration, prevents idle collapse

2. **idle_penalty**: 0.5 → 3.0
   - Must be high enough to discourage idling

3. **learning_rate**: 1e-5 → 1e-3
   - Affects how quickly policy updates

4. **n_epochs**: 3 → 10
   - Fewer epochs reduces overfitting to single rollouts

**Also Important:**
- Network size (128, 256, 512 neurons)
- Batch sizes
- Reward penalties (λ₁, λ₂, λ₃)

## Troubleshooting

### All trials still idle
If every trial gets -30.5:
```python
# Edit optuna_tune.py, line 90
ent_coef = trial.suggest_float("ent_coef", 0.05, 0.2, log=True)  # Higher range

# Line 108
idle_penalty = trial.suggest_float("idle_penalty", 1.0, 5.0)  # Higher range
```

### Optimization too slow
```bash
# Use parallel trials (requires multiple cores)
python -m Code.training.optuna_tune --algo ppo --trials 50 --jobs 4
```

### Want to continue interrupted optimization
Just run the same command again - Optuna resumes from database:
```bash
python -m Code.training.optuna_tune --algo ppo --trials 100  # Adds 50 more to existing 50
```

## Analyzing Results

### View best parameters
```bash
cat rl_training/optuna_results/ppo_best_params.json
```

### View all trials
```python
import pandas as pd
df = pd.read_csv('rl_training/optuna_results/ppo_trials.csv')

# Show top 5 trials
print(df.nlargest(5, 'value')[['number', 'value', 'params_ent_coef', 'params_idle_penalty']])

# Show trials that didn't collapse
good = df[df['user_attrs_idle_ratio'] < 0.5]
print(f"Found {len(good)} trials that didn't collapse to idle")
```

### Interactive visualizations
Open in browser:
- `rl_training/optuna_results/ppo_optimization_history.html`
- `rl_training/optuna_results/ppo_param_importances.html`

## Alternative: Try A2C

A2C doesn't have PPO's gradient clipping issues:
```bash
# Optimize A2C
python -m Code.training.optuna_tune --algo a2c --trials 50

# Train with optimized A2C
python -m Code.training.train_optimized --algo a2c
```

## Full Documentation

See `OPTUNA_GUIDE.md` for:
- Detailed explanations
- Advanced usage
- Custom search spaces
- Multi-objective optimization
- Theory behind each parameter

## Next Steps After Optimization

1. **Evaluate performance:**
   ```bash
   python -m Code.evaluation.eval_rl_agent --model rl_training/models/ppo_scheduling_optimized.zip
   ```

2. **Compare to heuristics:**
   - First Fit
   - Best Fit
   - Your optimized RL agent

3. **Analyze learned policy:**
   - Does it schedule jobs efficiently?
   - When does it choose to idle (if ever)?
   - How does it handle deadlines?

4. **Iterate if needed:**
   - Run more Optuna trials (100-200 total)
   - Adjust reward structure if still not learning
   - Consider action space decomposition (see your research notes)

Good luck! The optimization should significantly improve your results.
