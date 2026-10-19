# Final-week plan: pre-presentation experiments and showcase app

**Date:** 2026-10-20 (S2W14). **Presentation:** about 2026-10-28, a one-hour open room where visitors walk between
stalls (so the demo must work for someone arriving at any time). **Status:** draft for user approval; the autonomous
loop starts only after approval. Decision rule from the user: announce each new design in the training log, then proceed.

## Starting point (from the training log, 2026-10-09 to 10-11)

- Offline: best RL (Option 2 PPO, tuned) is within +5% of WLST on J and beats heuristics on weighted
  tardiness / linear-tardiness composites.
- Online (the weak spot): RL is behind the best heuristic on every preset (on_rho095 +13% at 2M steps; on_rho075
  +76%; on_rho110 +79%). Leads: more steps helps Option 2; J + linear tardiness (k) seed 0 ties EDF+Consolidate
  (22,000 vs 21,986), one seed only.
- Compute queues are empty since 2026-10-11 (CPU runner idle; GPU runner waiting; held queue of 189 default-HP runs
  stays held).

## Goal ranking

1. **Online RL beats the best heuristic** (EDF+Consolidate on on_rho095, then rho075 / rho110) with >= 3 seeds, no
   checkpoint cherry-picking, free idling kept (user rules).
2. **Metric-combination coverage** so the demo can pick any metric weighting and show the best model.
3. **Rigour**: a daily critic pass (rubric below) so every claim is cited, derived, or labelled "observed only".

## Workstream A: online improvement (ordered; each step announced before running)

| # | Hypothesis | Grounding | Test | Stop rule |
|---|---|---|---|---|
| A0 | Do more steps keep helping? | Power-law learning curves (Kaplan et al. 2020, arXiv 2001.08361) | Fit J(steps) per option to 300k / 1M / 2M results, extrapolate with intervals; run only options whose fitted curve can cross the heuristic | Curve plateau above heuristic -> drop budget as a lever |
| A1 | J + linear tardiness (k) objective helps online | Own lead (seed 0 ties) | Seeds 1-4 at 1M, GPU | Mean J vs EDF+Consolidate over 5 seeds |
| A2 | Residual policy over a heuristic prior | Silver et al. 2018 (arXiv 1812.06298); Johannink et al. 2019 (arXiv 1812.03201) | Option 3 ATC-prior already exists; add EDF+Consolidate prior variant | Compare to Opt 2 at equal budget |
| A3 | Behaviour-cloning warm start from the best heuristic, then PPO | Ross et al. 2011 (DAgger, arXiv 1011.0686); Hester et al. 2018 (DQfD, arXiv 1704.03732) | Pretrain the policy to imitate EDF+Consolidate, fine-tune with PPO | Beats scratch at equal total steps |
| A4 | Larger batches / more envs on the GPU | PPO batch-size studies (Andrychowicz et al. 2021, arXiv 2006.05990) | 1M steps, n_envs 16 | Throughput and J |
| A5 | Multi-agent decomposition (MAPPO / QMIX) | Yu et al. 2022 (arXiv 2103.01955); Rashid et al. 2018 (arXiv 1803.11485) | Only if A1-A3 fail by day 4; needs new code | Time-boxed |

Literature citations above are from memory and must be verified (arXiv ID, venue) before they go into any report or
`references.bib`.

## Workstream B: metric-combination coverage

- Exact re-scoring: the composite objective is linear in per-instance metrics, so any weight vector can be scored on
  every already-evaluated model with no retraining (`multi_objective_report.py`). The demo uses this.
- Trained anchors: J + {late jobs, linear tardiness, energy} already exist offline and online. Add anchors for
  makespan and p95 tardiness, one preset each, 3 seeds, so the demo can show a trained specialist next to the
  re-scored generalist.

## Workstream C: critic

Once a day a separate reviewer agent scores, 1-5 with evidence: (1) mathematical grounding of each design, (2) literature
support, (3) statistical soundness (seeds, intervals, protocol uniformity), (4) optimisation headroom (what the
learning curve and CP-SAT / heuristic bounds say is still achievable), (5) implementation efficiency (hot paths).
Output goes to the training log; findings that raise a flaw are fixed before more runs depend on them.

## Workstream D: showcase app

Streamlit app (`app/`), reading `Results/v2_objectives/ALL_RESULTS/data/all_results.csv` and per-instance CSVs.
Tab 1: slides (PDF to images carousel, 3 slides). Tab 2: pick preset, tick metrics, set weights, see ranked models
with the best RL vs best heuristic and a Pareto plot. Tab 3: one-click "best model for this objective". Ships a
frozen data bundle so the laptop needs only `pip install streamlit pandas matplotlib`. Planning detail with the user
later; a skeleton is built on days 2-3 and rebuilt from new results automatically.

## Loop protocol (every 60-90 minutes)

1. Check CPU (keep 80-90%, never 100%), BELOW_NORMAL priority, runner and watchdog alive; restart if dead.
2. Evaluate finished runs, append a training-log entry (Config / Stats / Observation / Conclusion), rebuild
   ALL_RESULTS, confirm auto_push is running.
3. Queue the next menu items (GPU for online, CPU for offline), announce any new design in the log first.
4. Phone notification on milestones (an RL win, a failure needing a decision).

## Calendar

| Day | Date | Focus |
|---|---|---|
| 1 | Oct 20 | Plan approved; learning-curve fits (A0); queue A1 on GPU; CPU: B anchors |
| 2-3 | Oct 21-22 | A2 and A3 implementation + runs; critic pass 1; app skeleton |
| 4 | Oct 23 | Decide A5 yes/no; compare A-results; critic pass 2 |
| 5-6 | Oct 24-25 | Best online design at 3-5 seeds; widen to rho075 / rho110; app finished |
| 7 | Oct 26 | Results freeze at evening; final tables, highlights, data bundle |
| 8 | Oct 27 | Rehearse the demo on the laptop; slides; no new training |

## Protocol tiers (needs user decision)

New designs cannot be tuned with the full Optuna protocol (20 trials x 100k) in a week. Proposal: label every result with a
tier: **Tier 1** = full tuned protocol (existing results); **Tier 2** = default hyperparameters, same budget and seeds
for every compared option, announced as exploratory. A new design that wins at Tier 2 is a lead, not a ranking against
tuned Tier 1 models.
