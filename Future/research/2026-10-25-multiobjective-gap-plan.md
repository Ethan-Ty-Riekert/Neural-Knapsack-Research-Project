# Plan: multi-objective training and evaluation where no composite-objective model exists (2026-10-25, S2W14)

**Status: PLAN ONLY, nothing queued. Needs the user's go-ahead on the decisions in section 5 (the methodology rule: announce, justify, then run).**

## 1. What exists (checked against the results archive and the submitted report)
- Main series: models trained on J = sum_j w_j T_j^2 alone (lambda_T = lambda_U = lambda_E = 0). Report: "The main objective of this project is J alone".
- Multi-objective experiments (report, "Multi-objective experiments"): Option 2/3 PPO agents trained on composite objectives J_lambda = J + lambda_T sum w T + lambda_U sum w U + lambda_E E, with lambda set per setting so that each added term equals J on the best heuristic's schedule, scaled by 0.25, 0.5, 1, 2, 4 (tag suffix `_lam`); 300,000 steps offline (default hyperparameters), 1,000,000 steps online, 3 seeds.
- In the archive (model groups by training objective):

| setting | + late | + energy | + linear | late + energy | all three | total groups |
|---|---|---|---|---|---|---|
| off_tf05 | 9 | 6 | 9 | 2 | 5 | 31 |
| off_tf05_w1 | 1 | - | 1 | - | - | 2 |
| off_tf08 | 1 | - | 1 | 1 | 1 | 4 |
| on_rho095 | 2 | 2 | 4 | 1 | 1 | 10 |
| **no composite model at all** | | | | | | on_rho050, on_rho075, on_rho075_tight, on_rho110, off_tf02, off_tf02_w1, off_tf08_w1 |

- Reference weights (lambda_T, lambda_U, lambda_E) exist in `Code/variants/v2_objectives/__init__.py` for only three settings (off_tf05, on_rho095, off_tf05_w1). The other settings need their own (the code comment warns offline values must not be reused online: J per machine-tick differs about 9x).
- Report finding to keep in view: at moderate weights the agents that won composite objectives were trained on J alone; only at larger weights did composite-trained agents take over. So J-only controls are essential.

## 2. Question this plan answers
For the seven empty settings: does training on a composite objective give a learned scheduler that is better on that composite than (a) the best heuristic and (b) the same architecture trained on J alone? Exploratory; no result is claimed before it exists.

## 3. Proposed protocol (uniform across all seven settings)
1. **Reference weights per setting**, computed (not hand-copied) from the archive as the code defines them: lambda_term = J / term on the schedule of the setting's best-J heuristic (mean over the 50 test instances). Logged in the training log; added to `REFERENCE_LAMBDAS`.
2. **Model:** windowed Option 2 (window 20), PPO from scratch with the composite reward, Consolidate placement online / FirstFit offline, the same recipe as the existing windowed from-scratch arm (1,000,000 steps; flags as in `queue.txt` scratch lines) plus `--objectives` and `--lambda-*` set from step 1.
3. **Objectives (3):** J + late jobs; J + energy; J + all three terms. **Weight multiplier (2):** x1 and x4 (the report's larger weights are where composite training mattered). **Seeds:** 3.
   = 3 x 2 x 3 = 18 runs per setting, 126 runs for seven settings, about 28.7 minutes each (measured on the existing from-scratch runs); about 8 to 10 hours at 6 to 8 jobs in parallel under the 80% CPU cap.
4. **Controls (already trained, no new compute):** windowed J-only from-scratch models (seeds 60-62 at the five settings queued 2026-10-25) and the J-only clones, all scored on each composite J_lambda from their stored metrics. Composite scores of every heuristic come from stored metrics too (J_lambda is linear in the recorded per-instance metrics).
5. **Optional cheap arm:** a clone of the best heuristic for each composite (about 5 minutes each) so the explorer has a learned model that follows the best rule for energy or late jobs in every setting.
6. **Evaluation:** score on the OLD 50 test instances for all arms; then the 50 fresh instances for the arms that lead. Paired per-instance bootstrap of seed-averaged differences against the best heuristic and against the J-only control on the composite. Pre-registered wording stays: "beats" = interval entirely below zero with 5 seeds per arm; 3-seed results are indicative; exploratory, all arms reported, no multiplicity correction. Any arm that looks ahead gets 5 seeds before it is called a win.

## 4. Schedule (freeze is 2026-10-26 evening)
- 10-25 afternoon: compute reference weights, log the announcement, queue the 126 runs.
- 10-25 night to 10-26 morning: training; `auto_eval` scores them.
- 10-26 morning: archive rebuild, controls scored, tables.
- 10-26 afternoon: fresh-instance checks for leading arms, findings doc, explorer rebuild.
- Cut-off rule: whatever has not finished by 10-26 15:00 is left out and listed as not run.

## 5. Decisions needed from the user
1. Full grid (126 runs) or reduced (x1 only: 63 runs, about 5 hours)?
2. From-scratch composite training as in the report (recommended), or also the cheap "copy the best rule for that objective" arm?
3. Which settings first if time runs short (suggest: on_rho075, on_rho110, off_tf02, off_tf08_w1, then the rest)?
4. 1M steps offline (uniform, recommended) or the report's 300k offline?

## 6. Risks
From-scratch windowed models varied widely between seeds and were weaker than the best rules in the earlier tests, so most arms may simply lose; that is a result, not a failure. The energy term needs the squared-tardiness term on (code constraint, satisfied since J is always included). If the CPU cap is lowered the 126 runs will not fit before the freeze.
