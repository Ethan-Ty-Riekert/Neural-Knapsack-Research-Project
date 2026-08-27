# Multi-objective Optuna: a genuine reward/tardiness Pareto front

**Date:** 2026-08-28 (S2W6)

## 1. Motivation

Three independent lines of evidence converged on the same problem tonight
and across this project's history: this environment's reward function
rewards throughput (the `+3` per-placement, `+50` completion bonuses) more
than it penalizes tardiness (`lambda_2 * T_j/H`) at the weights tried so
far.

- Phase 6 (2026-08-17): Optuna's tardiness-focused search, tuned via a
  single scalar composite score (`mean_reward - TARDINESS_PENALTY_WEIGHT *
  mean_tardiness_norm`), still produced worse tardiness at deployment scale
  than the reward-tuned search.
- Phase 7 (2026-08-17): grounded this in Skalse et al. (2022)'s formal
  definition of reward hacking, and in Roijers et al. (2013)'s critique
  that a single fixed scalarization weight can only ever reach one point on
  a multi-objective Pareto front -- already cited in
  `2026-08-21-rcpo-constrained-tardiness.md` Section 1 as motivation for
  RCPO, but never actually acted on for the *tuning* side of the pipeline
  until now.
- Tonight (Stage B, PSO): an entirely different, gradient-free search
  method optimizing raw reward directly found the same exploit
  independently -- reward 15% above EDF, tardiness 20x EDF's.

Every tuning run this project has ever done (`objective_a2c`'s `"reward"`
and `"tardiness"` modes) is a single scalarization, exactly the practice
Roijers et al. critique. This experiment implements the thing that
critique actually recommends: search for the whole non-dominated front,
not one hand-weighted point on it.

## 2. Implementation

`Code/training/optuna_tune.py::objective_a2c`, new `optimize_for="pareto"`
mode: returns the raw pair `(mean_reward, mean_tardiness_norm)` instead of
a scalarized score. `run_optimization()` creates the study with
`directions=["maximize", "minimize"]` instead of a single `direction=`
string -- Optuna auto-selects its NSGA-II-based multi-objective sampler
(Deb, Pratap, Agarwal, & Meyarivan, 2002) once multiple directions are
given, since `TPESampler` (used for every prior single-objective study
here) doesn't support multiple objectives. Results are read from
`study.best_trials` (the Pareto front itself -- `study.best_trial`/
`study.best_value` don't exist for a multi-objective study, there is no
single "best") and saved to `a2c_pointer_pareto_pareto_front.json`.

**Smoke-tested with 3 trials** before committing to a real run: correctly
identified exactly 1 of 3 trials as non-dominated (the other two were each
beaten on *both* objectives simultaneously by trial 0) -- confirms the
Pareto-front logic is doing genuine dominance comparison, not just sorting
by one axis.

## 3. Search budget

40 trials, `n_jobs=1` (sequential), same `make_tuning_env` small-scale
tuning environment (20 jobs, 5 machines, horizon=30) every prior Optuna
study in this project has used -- ~40-45s/trial observed, so ~30 minutes
total. Not independently re-tuned for this run; reused directly from the
existing single-objective studies' budget for comparability.

## 4. Results

_Filled in after the 40-trial run completes._

<!-- PARETO_RESULTS_PLACEHOLDER -->

## References

1. Roijers, D. M., Vamplew, P., Whiteson, S., & Dazeley, R. (2013). "A
   Survey of Multi-Objective Sequential Decision-Making." *Journal of
   Artificial Intelligence Research*, 48, 67-113. -- the core critique this
   experiment directly acts on.
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A fast and
   elitist multiobjective genetic algorithm: NSGA-II." *IEEE Transactions
   on Evolutionary Computation*, 6(2), 182-197. -- the algorithm Optuna
   uses internally once a multi-objective study is created.
3. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). "Optuna:
   A Next-generation Hyperparameter Optimization Framework." *KDD 2019*. --
   the tuning framework itself.
4. Skalse, J., Howe, N., Krasheninnikov, D., & Krueger, D. (2022). "Defining
   and Characterizing Reward Hacking." arXiv:2209.13085 -- already cited in
   this project's Phase 7 literature review, the formal grounding for why
   this misalignment counts as reward hacking rather than ordinary
   underfitting.
