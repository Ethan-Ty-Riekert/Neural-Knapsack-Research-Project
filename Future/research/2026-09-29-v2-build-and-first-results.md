# v2 Objective Program: Build Record and First Results

**Date:** 2026-09-29 (S2W11)
**Status:** Build steps 1–5 are implemented and tested. First baseline results (heuristics, PSO, CP-SAT) are in. The
RL trainings under v2 were started but **killed by the system (low memory) before finishing**. They produced no
RL results yet (section 4).
**Design:** `2026-09-28-v2-objective-formal-definition.md` (the objective and proofs) and
`2026-09-28-objective-redesign-discussion.md` (the decisions).

## CORRECTION (added the same day, after checking which jobs were dropped)

Finding 1 below ("every heuristic is far from the best schedule, mostly through dropped jobs") is **mostly an
artefact of the drop surcharge B = H**:
- On `off_c_15`, every job LST and EDF dropped has a deadline of 100–109, i.e. at or after the horizon H = 100.
  The v1 deadline range is 10–109, so these jobs are due after the window closes.
- Each drop was charged about 100, against about 0.4 of lateness per scheduled job.
- Without the surcharge, LST's J is 38.2, as good as CP-SAT's 40.4.

On `off_tf05` the dropped jobs are due at about ticks 73–80, so they'd genuinely be about 20–30 ticks late, but
100 of each drop's roughly 123 charge is still B. Drop counts and lateness (reported separately in every table)
remain valid. The **J rankings in this document hold only under B = H**. The user has decided to replace
drops with an extended horizon (true lateness, no B); see the decision record, section 4a. All v2 tables will be
re-run under that mode.

**Update 2026-09-30 (extended horizon, the new default):** on `off_c_15`, CP-SAT gives J = 38.20 (12 of 15
proven optimal) and **LST equals it on every instance**. So LST is optimal for total lateness on the standard
offline benchmark. The earlier "gap" is entirely gone. See the training log for 2026-09-30.

---

## 1. What was built

| Step | Commit | What | Verified by |
|---|---|---|---|
| 1 | `2c51753` | `reward_mode="objective"`: dense weighted tardiness, drop charge at the latest start (B = H), optional late count, global scale c, drop-risk shaping | `tests/test_objective_reward.py`: the exact component totals, shaping telescoping and drop dominance hold on 12 random episodes, offline and online |
| 2 | `32971ac` | CP-SAT solves the same J (optional jobs, K_j drop cost); `run.py` comparison tables | the CP-SAT optimum equals the replayed J (75.0 = 75.0, with 5 drops) |
| 3 | `3fe1b12` | Energy: `linear` (active machine-ticks) or `specpower_ml110g5`, charged exactly at placement; `Consolidate` placement rule (7 new heuristics) | energy charged = energy recomputed from the final capacity grid, both power models |
| 4 | `f2d61e7` | Difficulty presets: `off_tf02/05/08` (Potts–Van Wassenhove TF/RDD, adapted), `on_rho050/075/095/110`, `on_rho075_tight` | `tests/test_difficulty.py`: realised load within 0.03 of target; deadlines tighten monotonically |
| 5 | `4a863bd` | RL under v2: `--reward-mode objective`, `--difficulty`, shaping γ = PPO γ; `run.py` v2 `rl-train` / `rl-eval` | end-to-end smoke tests: train → save → load → evaluate, offline (Option 1) and online (Option 3) |
| perf | `4a863bd` | Online observation vectorised; ATC mean_p hoisted out of the per-slot loop | `tests/test_online_obs_vectorised.py`: bit-identical observations. Online ATC episode 2.78 s → 0.96 s; Option 3 online 29.1 s → 5.2 s |

All 16 regression scripts pass (`python -m tests.<name>`).

## 2. Results: offline, lateness + drops (J, lower is better)

The objective is J = Σ_finished w_j T_j + Σ_dropped w_j (max(0, H − d_j) + H), with 15 held-out instances (seeds
500000–500014) per preset. CP-SAT is capped at 60 s per instance with 8 workers. Its J is therefore an
**upper bound on the optimum** where the status was FEASIBLE rather than OPTIMAL, although it is still the best
schedule found. Code for each table: `off_c_15` ran on `32971ac` (its header says `f2d61e7-dirty`, which was
stamped when the run finished, a bug since fixed in `0188dd2`); the `off_tf*` tables ran on `0188dd2`.

| Preset | CP-SAT J (drops) | Best heuristic J (drops) | EDF J (drops) | PSO J (drops) |
|---|---|---|---|---|
| `off_c_15` (original deadlines, weights 1) | **40.4** (0.00) | LST 238.2 (2.00) | 330.3 (2.80) | 952.4 (0.13) |
| `off_tf02` (loose, weights 1–5) | **0.5** (0.00) | LST 406.7 (1.60) | 840.0 (3.20) | not run |
| `off_tf05` (medium) | **1306.7** (0.07) | ATC 2243.6 (5.67) | 3116.2 (3.20) | not run |
| `off_tf08` (tight) | **6770.9** (0.40) | ATC 8400.2 (5.67) | 10622.5 (3.20) | not run |

Full tables are in `Results/v2_objectives/comparisons/`.

**Findings (observed, 15 instances each):**
1. **Every heuristic is far from the best known schedule under the real objective, and most of the gap is
   dropped jobs.** With loose deadlines CP-SAT reaches J ≈ 0.5 and never drops a job. The best heuristic drops
   1.6 jobs per instance (J = 407).
2. **Drops are structural, not a deadline effect.** 100 jobs, a 100-tick horizon and one start per tick make
   the offline problem nearly saturated. Rules that ignore each job's latest start (H − P_j) leave long jobs
   until they can no longer fit. EDF drops 3.2 jobs per instance at *every* tightness level. A
   latest-start-aware rule, or a learned policy, should be able to remove most drops. Not yet tested.
3. **The ranking of heuristics depends on deadline tightness.** LST is best with loose deadlines (few late jobs,
   few drops), and ATC is best with medium and tight deadlines on weighted instances. It's a small-scale example
   of the "regime map" research question.
4. **PSO with a reward fitness is now honest but weak.** Under v2 it drops almost nothing (0.13) but has
   lateness 932.7, so J = 952 is worse than LST. A 450-evaluation budget over a 110-dimensional priority encoding
   doesn't find good schedules. It no longer "wins" by gaming the reward, as it did under v1.
5. **Placement rules don't change lateness offline**, since the one-start-per-tick clock binds, not capacity.
   They do change energy: Consolidate saves about 5–10 active machine-ticks, and WorstFit roughly triples them.

## 3. Results: lateness + energy (λ_E = 1, J includes energy)

| Preset | Power model | Best J | Plain rule | Consolidate | WorstFit (spread) |
|---|---|---|---|---|---|
| `off_tf05` | linear | ATC+Consolidate 2395.9 | ATC 2401.1 | (best) | EDF+WorstFit 3589.7 vs EDF 3277.8 |
| `off_tf05` | SPECpower | ATC+Consolidate 2373.4 | ATC 2377.0 | (best) | EDF+WorstFit 3466.0 vs EDF 3253.4 |
| `on_rho075` | linear | LST+Consolidate 10748.9 | LST 10996.5 | (best) | EDF+WorstFit 13429.1 vs EDF 11327.1 |
| `on_rho075` | SPECpower | LST+Consolidate 10657.1 | LST 10900.0 | (best) | EDF+WorstFit 13311.9 vs EDF 11229.0 |

**Finding (online):** Consolidate improves *every* measure, not just energy. LST+Consolidate against LST:
drops 31.7 vs 32.4, weighted lateness 383 vs 398, active machine-ticks 929 vs 943. Packing jobs onto
already-busy machines leaves whole machines free for large arrivals. Spreading jobs (WorstFit) is worst on
every measure. Offline, where capacity doesn't bind, Consolidate saves energy only. The SPECpower and linear
models rank the methods the same way here.

**Online drops:** about 32 of about 719 jobs per instance are dropped by every heuristic. How many of these are
"infeasible on arrival" (they arrive after their own latest start, so no policy could have done better) is
reported per run in `Results/v2_objectives/runs/*/per_instance.csv` (`infeasible_on_arrival`). It isn't
summarised here yet.

## 4. Not completed (honest status)

- **RL under v2: no results.** Three 300k-step PPO trainings (Option 1 offline, Option 3 offline, Option 1
  online at `on_rho075`) were running in parallel with the CP-SAT sweep. The laptop appears to have slept for
  part of the run: each process used only 600–770 CPU seconds over about 80 minutes. Claude Code then stopped
  all of them for low system memory, at about 37k, 10k and 16k steps. Stable-Baselines3 saves only at the
  end, so no checkpoint exists. **To do:** rerun **one training at a time**, not in parallel with CP-SAT.
  Expected times alone on this laptop: Option 1 offline about 25 min, Option 1 online about 45 min, and
  Option 3 offline longer.
- **Online difficulty sweep (job A):** stopped by the same memory event after the three offline presets.
  `on_rho050/075/095/110` and `on_rho075_tight` still need their heuristic tables (about 3 min each, run
  alone). `on_rho075` has lateness + energy tables from job E.
- **PSO on the difficulty presets:** not run (about 30–45 min per preset).

## 4a. Short v2 RL runs (smoke-scale, NOT final; 2026-09-29, 3 in parallel, about 10 min each)

| Run | Steps | J | Dropped | Weighted lateness | Note |
|---|---|---|---|---|---|
| Option 1 offline (`off_c_15`) | 120k | 82.7 | 0.07 | 76.0 | Avoids drops by accepting 2× LST's lateness: the B = H trade-off |
| Option 3 offline (`off_c_15`) | 50k | 233.1 | 0.07 | 226.5 | Same pattern, undertrained |
| Option 1 online (`on_rho075`) | 60k | 10053.3 | 32.4 | 398.0 | Identical to LST in every metric: collapsed onto one rule, as in v1 |

These are valid only under B = H and are superseded once the extended-horizon mode exists. They're checkpoints
`action_space_option{1,3}_ppo_v2_off_c_short.zip` and `action_space_option1_ppo_v2_on_rho075_short.zip` in
`rl_training/models/` on this laptop (gitignored).

## 5. Other findings recorded during the build

- **The v1 online protocol's load label is wrong.** The canonical `on_r_50` / `on_r_20` protocol (rate 9,
  log-normal sizes) was labelled "ρ ≈ 0.75", but its measured load is about 0.87 per resource (0.92 on the
  busiest). The log-normal generator matches E[P] but not E[P·a]. v1 results are unaffected, but the label in
  any write-up needs correcting. The v2 presets use the measured load.
- **Job sizes are 1–9, not 1–10.** `rng.integers(1, 10)` excludes 10, and the project map has been corrected.
  It's consistent with ρ = 9 × 5 × 5 / 300 = 0.75 for *uniform* sizes, which is where the old label came from.

## References

1. Potts, C.N. & Van Wassenhove, L.N. (1985). A branch and bound algorithm for the total weighted tardiness problem. *Operations Research* 33(2):363–377. (TF/RDD scheme; adapted here)
2. Kleinrock, L. (1975). *Queueing Systems, Vol. 1*. Wiley. (load factor)
3. Beloglazov, A., Abawajy, J. & Buyya, R. (2012). Energy-aware resource allocation heuristics for efficient management of data centers for Cloud computing. *FGCS* 28(5):755–768. (power-aware placement, the basis of Consolidate)
4. Beloglazov, A. & Buyya, R. (2012). *CCPE* 24(13):1397–1420. (SPECpower HP ML110 G5 power table)
5. Fan, X., Weber, W.-D. & Barroso, L.A. (2007). Power provisioning for a warehouse-sized computer. *ISCA '07*. (linear power model)
