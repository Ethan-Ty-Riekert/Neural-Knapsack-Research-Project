# Can Reinforcement Learning Outperform Traditional Methods for Cloud Resource Allocation?

Curtin University, NPSC3000 third-year research project (2026).
Student: Ty Riekert. Supervisor: Elham Mardaneh. Co-supervisor: Tony Mathew.

This repository contains the code, experiment protocol and results behind the project's research
paper. **Reviewers: the paper's results are reproduced from the tagged release cited in the paper**
(see [Citing this work](#citing-this-work)); the results themselves are in
[`Results/v2_objectives/ALL_RESULTS/`](Results/v2_objectives/ALL_RESULTS/).

## Overview

Cloud data centres place jobs (virtual machines, containers) on physical machines with limited
multi-dimensional capacity (CPU, memory, ...). Each job has a duration, a resource demand, a deadline
and an importance weight, and the scheduler must decide **which job to start, on which machine, and
when** -- including whether to deliberately leave capacity free for work that has not arrived yet.

The project asks whether a reinforcement learning (RL) agent can learn scheduling policies that beat
the dispatching rules used in practice, measured by **weighted squared tardiness**

$$J = \sum_j w_j\, T_j^2, \qquad T_j = \max(0,\ C_j - d_j),$$

which penalises a few very late important jobs more than many slightly late ones. Two settings are
studied:

- **Offline**: all jobs are known at the start (100 jobs, 10 machines, 4 resources).
- **Online**: jobs arrive over time (Poisson arrivals, heavy-tailed sizes) and the scheduler does not
  know the future; studied at increasing load.

## Formulation

The problem is modelled as a Markov decision process (MDP). The state at time $t$ holds the
remaining capacity $R_{m,r,t}$ of every machine and resource, each machine's activation $y_m$, and a
feature row $(p_j, s_j, d_j, A_j, w_j, m_j)$ for every job (duration, start time, deadline, resource
demand, weight, machine), which makes the state Markov. The reward is exactly the objective: at
every step the agent is charged the lateness accrued in that step, so the episode return is $-J$ up
to a constant scale. The full definitions (states, actions, transition dynamics, reward and their
proofs) are in the paper's Methodology section; the code is in `Code/core/`
(`scheduling_env.py`, `online_scheduling_env.py`, `objectives.py`, `obs_layout.py`).

## Methods compared

| Family | Methods | Code |
|---|---|---|
| Dispatching heuristics | 38 rules: 9 priority rules (EDF, SPT, LST, FCFS, LPT, WSPT, ATC, WMDD, COVERT) x 4 placement rules (First/Best/Worst-Fit, energy-aware Consolidate), Tetris, random | `Code/methods/heuristics/` |
| Exact solver | CP-SAT (Google OR-Tools), 60 s per instance (offline) | `Code/methods/exact/` |
| Metaheuristic | Particle swarm optimisation | `Code/methods/metaheuristic/` |
| Reinforcement learning | PPO and A2C (Maskable PPO, Stable-Baselines3) on six action-space designs | `Code/methods/rl/` |

RL action-space designs (the paper's Options 0-4):

| Design | The agent chooses | Network |
|---|---|---|
| Option 0 | a (job, machine) pair, or idle | pointer network |
| Option 1 | a dispatching rule (priority x placement) per decision | MLP |
| Option 2 | the next job to start (First-Fit placement) | priority pointer network |
| Option 3 | as Option 2, with the ATC index as an engineered per-job feature | priority pointer network |
| Option 3w | as Option 3, over a 20-job deadline-ordered window | windowed pointer network |
| Option 4 | job and machine as two sub-actions (action branching) | branching network |

Final protocol (v3), identical for every design: free idling (the agent may leave capacity unused);
a capacity look-ahead over the longest job duration; fixed feature scaling; a critic that also sees a
summary of future arrivals (an input-dependent baseline, which leaves the policy gradient unbiased);
potential-based reward shaping on projected lateness (which leaves the optimal policy unchanged).
Hyperparameters are tuned with Optuna (TPE sampler, median pruning, 12 trials of 300k steps per design,
algorithm and setting, with the library defaults as the first trial), selected on validation instances
only. Each configuration is then trained with 3 seeds and evaluated on 50 held-out test instances.

| Instance set | Seeds | Used for |
|---|---|---|
| Training | below 500000 | RL training (a fresh instance every episode) |
| Validation | 600000-600019 | hyperparameter selection and any design decision |
| Test | 500000-500049 | the reported results only |

## Results

All tables, figures and the per-run data behind the paper are in
[`Results/v2_objectives/ALL_RESULTS/`](Results/v2_objectives/ALL_RESULTS/) (start with its README and
`tables/summary.md`). Each number there traces back to a run folder in `Results/v2_objectives/runs/`
that records the git commit, machine, full configuration and per-instance metrics.

<!-- RESULTS SUMMARY: filled in from tables/summary.md when the final (v3) runs are complete. -->

## Reproducing the results

Requirements: Python 3.13, CPU only (no GPU needed).

```
git clone https://github.com/Ethan-Ty-Riekert/Neural-Knapsack-Research-Project.git
cd Neural-Knapsack-Research-Project
git checkout <tag cited in the paper>
pip install -r requirements.txt
python -m tests.test_v2_variants          # protocol and correctness checks
```

Baselines on one setting (offline, medium deadlines):

```
python run.py --variant v2_objectives --preset off_tf05 --method heuristics
python run.py --variant v2_objectives --preset off_tf05 --method cpsat --time-limit 60 --cpsat-workers 2
```

RL: hyperparameter studies, final training runs and test evaluation are generated by one module and
run through a CPU-load-aware job queue (`tools/campaign/`):

```
python -m Code.methods.rl.training.tune_optuna_v2 jobs --protocol v3_idle \
    --enqueue-to rl_training/campaign/queue.txt > rl_training/campaign/queue.txt
python tools/campaign/queue_runner.py      # trains; finished studies queue their final runs
python tools/campaign/auto_eval.py         # scores every final model on the 50 test instances
python run.py --variant v2_objectives --preset off_tf05 --method rl-eval:3:<model tag>
python Results/v2_objectives/ALL_RESULTS/scripts/build_folder.py   # rebuild every table and figure
```

The full campaign takes roughly two days on a 16-core desktop CPU. Trained models are written to
`rl_training/` (not versioned).

## Repository structure

```
run.py                     one launcher: variant -> preset -> method
Code/core/                 environments, instance generators, objective/reward, observation layout
Code/variants/             reproducible problem definitions and evaluation presets
Code/methods/              heuristics, exact (CP-SAT), metaheuristic (PSO), rl (policies, action spaces,
                           training, evaluation)
Results/                   results per problem variant; v2_objectives/ALL_RESULTS = the paper's results
tests/                     regression and correctness checks (python -m tests.<name>)
tools/campaign/            job queue, automatic evaluation and CPU watchdog for long campaigns
Future/research/           experiment log, dated design documents, bibliography
docs/                      development guide (docs/DEVELOPMENT.md) and earlier project documents
```

`Results/v1_legacy_reward/` holds the project's earlier results under a previous reward definition,
kept for reference; the paper uses `Results/v2_objectives/`.

## Citing this work

Please cite the tagged release named in the paper (a tag fixes the exact code and results; the branch
does not), for example:

> T. Riekert, *Can reinforcement learning outperform traditional methods for cloud resource allocation?*,
> NPSC3000 research project, Curtin University, 2026. Code and results:
> https://github.com/Ethan-Ty-Riekert/Neural-Knapsack-Research-Project (release `paper-2026-10`).

## Generative AI statement

Generative AI (Claude, Anthropic, used through the Claude Code assistant) was used throughout this
project in the following ways:

1. **Coding assistant**: implementing the environments, RL policies and action-space designs, the
   training, tuning and evaluation pipelines, the baselines and the diagnostic tooling in this
   repository, under the author's direction. All design decisions and their literature grounding were
   directed or reviewed by the author; each is recorded with its justification in
   `Future/research/training-log.md`.
2. **Running experiments**: setting up, running and monitoring the training and evaluation campaigns
   (`tools/campaign/`), and checking finished runs for failures.
3. **Literature**: assisting literature search and citation checking for the background and the design
   decisions (`Future/research/references.bib`).
4. **Diagnosis**: helping investigate unexpected experimental results, at the author's request.
5. **Writing**: drafting and structuring documentation (including this README and the experiment log)
   and parts of the written report from the author's notes, data and experiment log, with the author
   reviewing and editing the text.

No result, table or figure was produced without the underlying code being run and its output
inspected; generative AI was not used as a substitute for running experiments or verifying their
results. AI-written code is checked by automated tests (`tests/`), including tests of the mathematical
properties the methods rely on. The author takes full responsibility for the content of this
repository and the paper.

## Acknowledgements

Supervised by Elham Mardaneh, with co-supervision from Tony Mathew, Curtin University.
