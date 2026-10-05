# Training-campaign tools

Long-running batches of `run.py` jobs (training, heuristic sweeps, CP-SAT, PSO), automatic test-set
evaluation of every finished model, and CPU protection. Scripts live here; runtime state lives in
`rl_training/campaign/` (gitignored, like all generated artefacts).

| Script | Role |
|---|---|
| `queue_runner.py` | Starts jobs from `rl_training/campaign/queue.txt` while CPU stays at or below `target_cpu` (`runner_settings.json`, re-read live). Jobs run at Below Normal priority. Adopts jobs left running by an earlier runner. |
| `auto_eval.py` | Evaluates every finished model on its preset's 50 test instances (`run.py rl-eval:<option>:<tag>`). Tuning trials (`_hp<k>`) are never evaluated on test. |
| `cpu_watchdog.py` | Pauses the newest training job if the 3-minute average CPU exceeds 95%, resumes it below 80%. |
| `common.py` | Shared paths, the tag pattern and status logging. |

Start all three from this folder (in the background, Below Normal priority):

```
python queue_runner.py
python auto_eval.py
python cpu_watchdog.py
```

Queue lines are `<tag> <run.py args>`, with `--train-args ...` (verbatim training-script flags) always
last. Tags follow `v2_<preset>_o<option><mods>[_a2c][_hp<k>|_tuned]_s<seed>`. Modifiers: `c` Consolidate
menu, `a` ATC feature, `p` pointer network (Option 0), `w` windowed, `t` per-tick, `n` work-conserving,
`f` placement repair, `m` full-MDP (Markov) observation. The paper's tables (`Results/v2_objectives/
ALL_RESULTS/`) include only `m` runs.

Why the CPU limits: on 2026-10-05 a sustained 100% CPU load hard-reset the desktop
(`Future/research/training-log.md`). Keep `target_cpu` at 90 or below.
