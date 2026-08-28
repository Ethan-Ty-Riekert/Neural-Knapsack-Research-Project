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

**The Pareto front collapsed to a single point** (`trial 20: mean_reward=
107.65, mean_tardiness_norm=0.0`) -- every one of the other 39 trials was
dominated by it. This is not a search failure: 17 of 40 trials (42.5%)
reached exactly zero mean tardiness at this tuning scale, and the single
best reward among all 40 trials (107.65) belongs to one of those
zero-tardiness trials -- the best *non*-zero-tardiness trial only reached
106.10 reward, strictly less. **There is no trade-off to characterize
here**: nothing had to be given up on reward to also reach zero tardiness,
so a single point legitimately dominates the entire 40-trial population.

### Honest reading

This is a genuine, informative null result, not a wasted run. It means the
reward/tardiness misalignment this project has repeatedly found (Phase 6,
PSO tonight) **is not present, or is far weaker, at this small tuning
scale** (20 jobs, 5 machines, horizon=30) -- the same scale every prior
Optuna study in this project has searched at. The misalignment that
matters shows up at deployment scale (100 jobs, horizon=100), where the
`+3`/`+50` throughput bonuses accumulate over many more placements per
episode relative to the `O(1)`-per-job tardiness penalty. This directly
corroborates Eimer, Lindauer, & Raileanu (2023)'s HPO tuning/deployment
mismatch warning -- already cited in this project's Phase 7 literature
review for a different symptom (tardiness didn't transfer from tuning
scale to deployment scale) -- now showing the same root cause applies to
the *shape of the Pareto front itself*, not just to which point on it gets
picked.

## 5. Conclusion / next step

Multi-objective tuning at small scale can't reveal the trade-off that
actually matters, because the trade-off barely exists there. The natural
follow-up -- already on this project's survey list independently, now
doubly motivated -- is rerunning this exact `optimize_for="pareto"` mode
against a tuning environment closer to deployment scale.

## 6. Follow-up: rerun at a larger tuning scale (same night)

Added `tuning_num_jobs`/`tuning_num_machines`/`tuning_horizon` parameters
to `objective_a2c`/`run_optimization`/the CLI (default `20, 5, 30`,
identical to every prior study) so this could actually be attempted instead
of only flagged. **Found and fixed a real bug while smoke-testing this at
`num_jobs=60`:** `make_tuning_env()` hardcoded `max_jobs=30` regardless of
`num_jobs` -- every study before tonight happened to use `num_jobs=20 <=
30`, so `num_jobs > max_jobs` was never exercised, and it crashed deep
inside `PointerActorCritic`'s observation-splitting logic with a cryptic
"index N is out of bounds" rather than a clear error at the actual
boundary. Fixed to `max(30, num_jobs)`.

A 2-trial smoke test at `num_jobs=60, num_machines=8, horizon=60`
immediately showed what the small-scale study couldn't: **both trials were
non-dominated** -- trial 1 (reward=154.85, tardiness_norm=7.05) and trial 0
(reward=109.24, tardiness_norm=1.37) each beat the other on one axis. A
real trade-off, visible at n=2, where the small-scale study needed 40
trials to confirm there wasn't one. A full 40-trial run at this scale is
in progress; results below once it completes.

**Results (40 trials, num_jobs=60, num_machines=8, horizon=60):**

**10 of 40 trials are non-dominated** -- a genuine front this time, not a
single point:

| trial | reward | tardiness_norm | late jobs | `lambda_2` |
|---|---|---|---|---|
| 9 | 195.39 | 7.15 | 17.00 | 4.068 |
| 33 | 158.81 | 6.68 | 20.00 | 1.130 |
| 27 | 158.51 | 3.98 | 12.00 | 1.292 |
| 23 | 143.19 | 3.82 | 12.00 | 0.732 |
| 18 | 134.39 | 3.73 | 13.00 | 3.889 |
| 16 | 122.15 | 1.68 | 8.00 | 8.618 |
| 12 | 103.75 | 0.75 | 4.00 | 4.850 |
| 13 | 100.21 | 0.33 | 3.00 | 9.097 |
| 7 | 55.02 | 0.08 | 1.00 | 16.130 |
| 38 | -117.05 | 0.00 | 0.00 | 6.658 |

### Honest reading

**Confirms the hypothesis cleanly.** At this scale, reward ranges from
-117.05 (zero tardiness) up to 195.39 (7.15 tardiness_norm, 17 late jobs)
across the front -- a steep, real trade-off, the opposite of the small-scale
study's single dominant point. Reaching zero tardiness here costs enough
reward to go *negative*, something that never happened, or needed to
happen, at the small tuning scale.

**`lambda_2` correlates with position on the front, but noisily, not
monotonically** -- the lowest-tardiness trials (7, 13, 38) do have among the
highest `lambda_2` values (16.13, 9.10, 6.66) in the front, but trial 33
(tardiness_norm=6.68, near the high-reward end) has a *lower* `lambda_2`
(1.13) than trial 23 (tardiness_norm=3.82, `lambda_2`=0.73) -- meaning other
sampled hyperparameters (architecture size, learning rate, entropy
coefficient) also materially shape where a trial lands, not `lambda_2`
alone. This is exactly the information a single scalarized search (the
"tardiness" mode, or any fixed weight) would never surface: which
combinations of hyperparameters, not just which `lambda_2`, produce a good
trade-off.

**No single "best" trial exists here, by design** -- that is the entire
point of a Pareto front over a scalarized score. Any of these 10 trials is
a legitimate choice depending on how much reward one is willing to trade
for tardiness; picking one for a future full deployment-scale run is a
downstream decision this experiment deliberately leaves open rather than
making for you.

## 6b. Closing the loop: the actual full deployment scale (same night)

Reran once more at `num_jobs=100, num_machines=10, horizon=100` -- the
actual deployed instance's exact dimensions, not a proxy -- with a smaller
budget (15 trials, ~2 min/trial at this scale) given the hour.

**5 of 15 trials non-dominated**, reward ranging from -89.94 (zero
tardiness) to 257.05 (tardiness_norm=19.37):

| trial | reward | tardiness_norm | `lambda_2` |
|---|---|---|---|
| 7 | 257.05 | 19.37 | 1.434 |
| 13 | 247.09 | 13.81 | 0.708 |
| 12 | 170.88 | 2.47 | 10.981 |
| 11 | 110.99 | 1.71 | 5.685 |
| 14 | -89.94 | 0.00 | 0.568 |

Confirms the 60-job finding robustly at the actual production scale: a
real, wide front exists here too, not just at the intermediate proxy scale.

**Important caveat before reading these numbers against any deployed
checkpoint:** every trial here trains for only `eval_timesteps=30_000`
(the same fixed budget every Optuna study in this project uses, deliberately
held constant -- see objective_a2c's `tuning_num_jobs` docstring), vastly
less than the ~500,000 timesteps a full curriculum run gives the actual
deployed checkpoints. These trials are all substantially undertrained
relative to Phase 8/RCPO's checkpoints -- the *existence and shape* of the
trade-off is the finding, not these specific reward/tardiness values, which
should not be compared directly to `eval_results.csv`'s fully-trained
numbers.

**One concrete cross-reference that does hold up:** the historical
reward-tuned `lambda_2=3.8529` (`a2c_pointer_best_params.json`, used to
warm-start every RCPO run) falls *between* trial 11 (`lambda_2=5.685`) and
trial 7 (`lambda_2=1.434`) on this front -- neither at the extreme-reward
nor extreme-tardiness end. Consistent with it being found by a
reward-maximizing search that happened to land in the middle of the
trade-off, not deliberately at either extreme.

## 7. Conclusion

The small-scale study's null result and this scaled-up study's rich,
10-point front are two halves of the same finding: **the reward/tardiness
misalignment this project keeps rediscovering (Phase 6, PSO, and
implicitly every RCPO experiment) is not a fixed property of the reward
function in isolation -- it is scale-dependent**, emerging as instance size
grows and vanishing at small scale. This matches the throughput-vs-tardiness
mechanism already suspected (the `+3`/`+50` bonuses accumulate over more
placements per episode as `num_jobs` grows, while the tardiness penalty
stays `O(1)` per job) but had not, until tonight, been directly
demonstrated by showing the trade-off appear and disappear as scale
changes alone. Recommended next step for a future session: rerun this
`optimize_for="pareto"` mode at full deployment scale (100 jobs,
horizon=100) to find the front that would actually inform a production
`lambda_2` choice, rather than this run's intermediate scale.

## 8. Validating a front point under full training (same night)

Section 6b's front is only informative about *relative* trade-off shape --
every trial there used the shared 30,000-timestep tuning budget, far short
of the ~500,000 timesteps a real curriculum run gives a deployed
checkpoint (Section 6b's caveat). The natural next step is to actually
train one of the front's points to full convergence and see whether the
trade-off it suggested survives, using the exact same pipeline
(`train_optimized.py`) every other checkpoint in this project was produced
by -- not a special-cased one-off script.

**Point chosen: trial 12** (`reward=170.88, tardiness_norm=2.47,
lambda_2=10.981`) -- the front's "knee": a large tardiness improvement over
the high-reward end (13.81-19.37) for a moderate reward cost, without
trial 14's reward collapse at the zero-tardiness end. Saved as
`rl_training/optuna_results/a2c_pointer_paretoknee_best_params.json` (all
14 fields from the trial, verbatim) so it loads through
`train_optimized.py`'s existing `--params-tag` mechanism exactly like the
project's other alternative-hyperparameter runs (e.g. `--params-tag
tardiness`) -- no new loading code needed.

Trained full curriculum, `--use-potential-shaping` (carrying forward Phase
8's shaping win, matching every other checkpoint this project has trained
since), tagged `a2c_pointer_s4-200000_shaped_paretoknee_S2W6`.

**Results:** _filled in after training and evaluation complete._

<!-- PARETOKNEE_RESULTS_PLACEHOLDER -->

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
