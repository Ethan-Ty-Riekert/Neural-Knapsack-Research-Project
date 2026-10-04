# Machine-Utilisation Graph Revision: Per-Machine Lines and the "Top 95%" Envelope

**Date:** 2026-09-13 (S2W8)
**Files:** `Code/utils/plotting_utils.py` (`plot_machine_utilisation`),
`Code/evaluation/eval_rl_agent.py` (`plot_results`)
**Status:** Complete. Statistical method chosen and implemented, ahead of the
final offline-case training/evaluation runs this method's output will be used
in.

---

## 1. Motivation

The previous machine-utilisation figure (`plot_results()`'s old "1. Mean
utilisation curve" block) collapsed utilisation across *both* the 50
evaluation runs *and* the 10 machines into a single curve per model, with a
`fill_between(mean - std, mean + std)` band. This throws away all
machine-to-machine structure -- exactly the information the supervisor asked
to see restored, per this project's revision request: one graph per model,
one colored line per machine, plus a min/average/near-max envelope.

Per this project's CLAUDE.md rule (ground every design decision in a citation
or derivation, don't just declare a statistic "correct"), the exact
definition of "top 95%, excluding outliers" needed to be pinned down and
justified before implementation, not guessed.

## 2. Candidates considered

**(a) Plain sample percentile** -- `np.percentile(data, 95)`. NumPy's default
interpolation is the Type-7 estimator surveyed in Hyndman & Fan (1996) [1],
one of nine linear estimators they compare; Type 7 is the one most stats
packages (R's default, NumPy, Excel) use. It is an order-statistic
interpolation, not literally "the max with outliers removed" -- with only 10
machines, the 95th percentile falls between the 9th and 10th order statistic,
i.e. within one slot of the true per-timestep max.

**(b) Tukey (1977) IQR fence** [2] -- define the upper fence as
`Q3 + 1.5*IQR`, then take the max value at or below that fence (the classic
boxplot upper whisker). This is a literal implementation of "the max,
excluding outliers": an outlier is *defined* as anything past the fence,
then you take the max of what remains.

Both share the same small-sample caveat: with `num_machines=10`
(`Code/env_config.py:8`), any order-statistic-based estimate (whether a
percentile or a quartile-derived fence) is computed from only 10 points per
timestep, so none of these methods is a strong outlier-rejection procedure at
this sample size -- they are all mild trims, not robust ones.

## 3. Aggregation axis: across machines only, not pooled with runs

A separate question from (a) vs (b): should the per-timestep sample set be
just the 10 machines (after averaging each machine's series across the 50
eval runs first), or the pooled 10 machines x 50 runs = 500 values?

Pooling doesn't help here for the RL model's own curve: `eval_rl_agent.py`'s
default (non-`--randomized-eval`) evaluation mode replays the *same fixed
instance* 50 times, and `MaskablePPO.predict(..., deterministic=True)` makes
the policy deterministic on a fixed instance -- so all 50 runs produce
numerically identical utilisation traces. Pooling would silently duplicate
the same 10 machine-values 50x, adding zero information while making the
sample look artificially large. (This would matter more under
`--randomized-eval`, or for a heuristic with genuine run-to-run randomness,
e.g. `Random` -- not the common case this graph targets.) Averaging each
machine's series across runs first is also the natural choice because it's
the exact same `(timesteps, n_machines)` array the per-machine line plot
needs anyway -- every element of the figure (lines, envelope, average) then
comes from one consistent array.

## 4. Decision

Confirmed with the user: **plain 95th percentile (candidate (a)), computed
across machines only** (10 samples/timestep, machines' series pre-averaged
across the 50 eval runs) -- matching the supervisor's literal "95%" framing,
and consistent with the per-machine data already being plotted.

Implemented in `Code/utils/plotting_utils.py::plot_machine_utilisation()`:
- `per_machine = util.mean(axis=0)` -- shape `(timesteps, n_machines)`, runs
  averaged out first.
- One line per machine: `per_machine[:, m]` for each `m`.
- `avg_line = per_machine.mean(axis=1)`, `min_line = per_machine.min(axis=1)`,
  `top95_line = np.percentile(per_machine, 95, axis=1)`.
- `fill_between(min_line, top95_line)` shaded band.

**Caveat carried into the figure/code (not silently dropped):** with only 10
machines, the 95th-percentile line sits within one order statistic of the
true per-timestep max -- readers of the resulting paper figure should not
read "95th percentile" as an aggressive outlier trim at this sample size. If
a future revision increases `num_machines` substantially, this caveat
weakens and the percentile becomes a more meaningful trim relative to the
max; the Tukey fence (candidate (b)) remains a documented alternative if the
supervisor later prefers a literal outlier-exclusion definition over the
percentile framing.

## 5. References

[1] Hyndman, R. J., & Fan, Y. (1996). *Sample Quantiles in Statistical
Packages*. The American Statistician, 50(4), 361-365.

[2] Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
(Boxplot / IQR outlier fences, Chapter 2.)
