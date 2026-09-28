# Decision Record: Objective / Reward Redesign Discussion

**Date:** 2026-09-28 (S2W11)
**Type:** Design discussion + decisions (no code changed by this document)
**Status:** Decisions recorded below; items marked **OPEN** are still being weighed by the user.
**Supersedes nothing** — every result produced under the legacy and dense rewards stays valid *as a result
under that reward* and is preserved (see §11).

---

## 1. What triggered this

The PSO baseline study (`2026-08-28-pso-metaheuristic-baseline.md`) reported, on 15 held-out instances
(seeds 500000–500014):

| | reward | tardiness | late jobs |
|---|---|---|---|
| PSO (reward fitness) | 329.68 | 1012.40 | 40.93 |
| EDF | 286.00 | 50.33 | 14.20 |

User question: *how can PSO have the better reward while being worse on everything else?*

**Re-check (2026-09-28, current code), and a correction made during this same session.**
- *First re-check (wrong horizon).* This used `generate_env_config`'s default horizon of 110, but the
  protocol uses **H = 100** (see `results_data.py` `PROTOCOLS["off_c_15"]`). It gave reward 346.34,
  tardiness 50.87 and 100/100 jobs scheduled. From that I concluded the gap was the hotspot bug and that EDF
  never drops jobs. **Both conclusions were wrong and are retracted.**
- *Correct re-check (H = 100).* EDF gives reward **286.00**, tardiness **50.33**, late **14.20**. That is an
  exact reproduction of August, so the reward code for this path hasn't changed since. EDF schedules only
  **97.2 / 100 jobs on average (worst case 95)**, so it **drops about 3 jobs per instance** and never earns
  the +50 completion bonus.

**Explanation of the anomaly.** Take a schedule that fits all 100 jobs in, even very late ones. Compared with
EDF it earns +50 (completion) and about 3 × 2.8 ≈ +8.4 (per-job bonus), and pays only about −(1012 − 50)/100
≈ −9.6 in extra tardiness. That nets roughly +49, against the observed gap of +43.7. So PSO most likely wins
by **finishing every job** and ignoring lateness. That is exactly the "flat bonuses dominate the bounded
tardiness penalty" defect. (Confirmation run of PSO's jobs-scheduled count: see §1a.)

Two consequences:
- EDF's low tardiness is **partly bought by dropping jobs**. The tardiness metric doesn't count dropped jobs,
  so EDF's 50.33 and PSO's 1012 aren't measured on the same set of jobs. Neither number is a fair comparison
  on its own; jobs scheduled must be reported alongside tardiness.
- The horizon itself changes the conclusion. At H = 110 EDF fits every job in and earns the bonus. Any
  comparison must pin the protocol's exact dimensions, and the launcher presets do this (§11).

### 1a. Confirmation run (2026-09-28, current code, H = 100, PSO swarm 15 × 30 iterations, seed 0)

| Instance | PSO reward | PSO tard | PSO sched | EDF reward | EDF tard | EDF sched |
|---|---|---|---|---|---|---|
| 500000 | 337.65 | 935 | **100** | 286.00 | **0** | **97** |
| 500001 | 335.66 | 1134 | **100** | 277.42 | 58 | **95** |

Confirmed. PSO wins on reward by scheduling every job and ignoring lateness. On instance 500000, EDF has
zero tardiness because it **drops the 3 jobs that would have been late** rather than finishing them late.
Neither behaviour is what the project wants. The legacy reward favours the first, and the legacy tardiness
metric makes the second look perfect.

## 2. Audit of the current reward (both modes)

| Term | What it measures | Problem |
|---|---|---|
| −λ₂·w_j·T_j/H | weighted tardiness | smallest term: 1 tardiness unit ≈ 0.009 reward at H=110 |
| +3/job, +50 all done (legacy) | throughput | constant for any full schedule → cannot rank full schedules |
| −λ₁ per machine activation | machine *ever* used | one-off per episode, −10 once all machines used; not energy |
| −λ₃·Δθ hotspot | rise in global peak utilisation | sums to ≤ 1/episode; silent once any machine hits 100%; penalises packing (anti-energy); no modelled consequence |
| −0.5 per idle | shaping against idle collapse | changes the optimum, not derived (violates CLAUDE.md rigour rule) |
| −5 invalid | invalid action | unnecessary under masking |

Exchange rates implied: 1 idle step ≈ 55 tardiness units; 1 activation ≈ 110 tardiness units.

Other findings from the code audit:
- Unscheduled jobs are invisible to the tardiness / late-jobs / P95 metrics (tardiness stays 0); they only
  appear in `jobs_scheduled`.
- `Code/evaluation/eval_rl_agent.py::make_env` always builds the legacy reward with λ=1, so evaluation
  `total_reward` is legacy regardless of the training reward mode.
- Seven separate env factories exist (train_optimized, train_action_space_variant ×2, train_a2c,
  optuna_tune, train_rl_agent, eval_rl_agent) — no single place to thread a reward config.
- CP-SAT (`exact_solver.py`) minimises Σ w_j·T_j with all jobs forced scheduled — a different objective
  from the RL reward.

## 3. Tardiness must dominate

**User position:** agrees tardiness must be made more important.
**Decision:** the reward contains only terms that are part of the chosen objective. Bonuses (+3/+50),
activation λ₁, hotspot λ₃, idle penalty and invalid penalty are removed from the new objective mode
(kept untouched in legacy/dense modes for reproducibility).

## 4. Dropped / unfinished jobs

**User position:** thought dense tardiness already handled this (it compounds over time). Wants every job
finished before H, accepts that may be impossible. Future jobs will be DAGs (a job cannot start until its
parents finish), so the design must extend to precedence.

**Analysis:** dense tardiness charges an unscheduled job each tick past its deadline, but only until the
episode ends at H. So dropping job j costs ≈ (H+1−d_j)/H — almost nothing for late-deadline jobs
(d_j = 105 → ≈ 0.05, cheaper than one idle step). The metric doesn't count it at all.

**How industry does it:** work is not silently dropped. Borg (Verma et al. 2015) keeps jobs pending,
admits by priority/quota and preempts lower-priority work; Decima (Mao et al. 2019) runs every DAG job to
completion and minimises job completion time. The formal model for "may reject at a cost" is scheduling
with rejection (Bartal et al. 2000).

**Design:**
- Each job has a **latest start** LS_j = H − P_j: by the existing feasibility rule (t + P_j ≤ H) it can
  never be started after that tick. So the moment a job becomes a certain drop is known exactly — no
  judgement needed. (DAG: LS_j = H − longest remaining chain through j, including descendants.)
- At tick LS_j + 1, if j has not started, charge a one-off **drop penalty ρ_j**, and stop j's per-tick
  tardiness charge (no double counting).
- ρ_j is set strictly above the largest tardiness cost j could incur if scheduled at all (finishing at H),
  so **dropping a job never beats finishing it late** — provable, and it makes completion lexicographically
  first.
- Late jobs (finished after deadline) and dropped jobs (never finished) are reported separately.

**Worked example.** H = 110, job A: P = 8, d = 60 → LS = 102.

| Tick | Job A | Charge |
|---|---|---|
| 0–59 | waiting, not late | none |
| 60–102 | waiting, late | normal dense tardiness each tick |
| 103 | can no longer finish → certain drop | −ρ_A once |

**User question — isn't end-of-episode-only penalising a problem?** Yes: a terminal-only penalty is
sparse (the credit-assignment problem dense tardiness was built to fix). Charging at LS_j instead gives
the same total, but earlier and tied to one job.

**User question — the penalty lands at a tick caused by many earlier decisions; how does the agent know
the cause, and won't it distort the current decision?**
1. *It does not distort the current decision.* At tick 103 the penalty happens whatever the agent does,
   so it adds equally to every action's return and cancels out of the advantage
   A(s,a) = Q(s,a) − V(s) (and out of argmax_a Q(s,a)). It adds variance, not bias.
2. *The value function moves blame back.* The observation contains time, P_A and d_A, so "A has 1 tick
   of slack left" is visible. V(s) learns to predict the upcoming −ρ_A; at tick 101, "start A" then has
   much higher advantage than "start something else". Same mechanism already carries dense tardiness.
3. *Made explicit, not left to learning:*
   - potential-based shaping Φ(s) on each job's slack-to-latest-start (Ng, Harada & Russell 1999 —
     provably leaves the optimal policy unchanged), so each decision that eats into a job's slack pays a
     share of the penalty immediately;
   - an explicit per-job "slack to latest start" observation feature
     (`Code/env/gym_scheduling_wrapper.py::_get_obs` currently has time/duration/deadline only).

**Status: OPEN** — user is still weighing this explanation; revisit before the formal definitions (§12)
are finalised.

## 5. Activation penalty → energy

**User position:** remove or rework it; energy / active servers stays as an objective in the
multi-objective setting.

**Derivation:** under the linear server power model P(u) = P_idle + (P_max − P_idle)·u
(Fan, Weber & Barroso 2007), total energy over an episode is

E = P_idle · (active machine-ticks) + (P_max − P_idle) · (total work).

Total work is fixed once every job is scheduled, so minimising energy ⇔ minimising **active
machine-ticks** (a machine is active at tick t if it runs ≥ 1 job). This replaces λ₁, derived rather than
invented.

**Decision:** energy objective with a **selectable power model** — linear (active machine-ticks) as
default; a piecewise-linear SPECpower curve (10% load steps) from a real modern server as an option.

## 6. Hotspot term / utilisation–efficiency

**User position:** the hotspot term was meant to (a) push toward energy efficiency and (b) keep space free
for important arriving jobs. Accepts it is probably wrong. Idea: find what utilisation real data centres
consider most efficient and shape a utilisation term to match — maybe a target like 50%, or a more complex
curve with several local peaks (e.g. 10%, 32%, 90%).

**What the literature says:**
- Real servers mostly run at 10–50% utilisation; idle draws ~50% of peak power; for older servers
  efficiency (work per watt) rises all the way to 100% (Barroso & Hölzle 2007).
- For modern energy-proportional servers, SPECpower data show peak efficiency at an *intermediate* load
  (Wong, ISCA 2016), often ~70–80% (Ruan & Chen 2015 — exact level **unverified**).
- The curve is smooth and single-peaked, not multi-peaked.

**Problems with the current term:** see §2 — and crucially goals (a) and (b) conflict: energy wants
packing, headroom wants spare room. That tension is the core of VM-consolidation research (Beloglazov &
Buyya 2012: consolidate, but keep hosts under an overload threshold).

**Options considered:**
- **A.** Remove it. Energy via the power-curve objective (§5); headroom learned through tardiness in the
  online case (heavily-weighted arrivals with nowhere to go are penalised directly).
- **B.** Per-machine per-tick utilisation cost f(u): B1 f = power curve (= A + energy); B2 overload
  threshold penalty (e.g. u > 80%) — arbitrary without a modelled consequence of overload.
- **C.** Model overload: jobs on a machine above a threshold run slower (interference). High utilisation
  then *causes* lateness and tardiness penalises it with no invented term.
- **D.** Explicit reserve for high-priority work (Borg priority bands / preemption).

**Decision:** **A now** (with the power-curve energy objective of §5 — this is what made B1 appealing to
the user); **C later** as an optional realism/difficulty setting. B2 and D recorded as considered.
No hand-shaped utilisation target: the efficiency peak emerges from the chosen power curve.

## 7. Idling

**User position:** originally added because sometimes no valid move exists; wants to keep idling as a real
decision — the agent may learn to wait because important jobs may arrive and need space.

**Analysis:** deliberate waiting is "inserted idle time" (Kanet & Sridharan 2000), valuable precisely with
dynamic arrivals and due dates. Under dense tardiness, idling already costs the per-tick charge of every
overdue job, so a flat idle penalty is unnecessary and biases the optimum.

**Decision:** idle stays a legal action (forced when nothing is feasible — mask already does this); flat
idle penalty removed from the new objective. If idle collapse returns, use potential-based shaping
(policy-invariant) rather than a flat penalty.

## 8. The offline one-start-per-tick clock

**Decision (user):** intentional. A tick is a decision point, not a fixed wall-clock duration. Offline is
meant to be simple. Keep. (CP-SAT already models it via `AddAllDifferent(start)`.)

## 9. Tardiness measures

| Measure | Meaning | As per-tick reward? |
|---|---|---|
| ΣwT (total weighted tardiness) | how late, in total | yes, exactly (existing dense charge) |
| ΣwU (weighted late count) | SLA violation rate — closest to industry SLAs | yes, −w_j once at tick d_j if unfinished |
| T_max | worst case / fairness | no — report only |
| P95 / P99 tardiness | tail latency (Dean & Barroso 2013) | no — report only |

**Recommendation:** ΣwT and ΣwU selectable as objectives; all four always reported, plus dropped-job
count. **Status:** recommendation made; user has not objected.

## 9a. Reported metrics vs. objectives

**User question:** do we need more metrics? The user cited Nagabushnam, Choi & Kim (2025), a survey of
51 fog task-scheduling papers. It evaluates algorithms on **energy consumption, computational latency,
task completion time and quality of service**.

**Analysis:** before this, the project reported tardiness and late jobs only, which is a slice of QoS, and
dropped jobs were invisible. **Reported metrics** (measured for every method, always) are kept separate
from **objectives** (what a reward optimises, chosen per experiment). The reported set can be broad
because it doesn't change training. Mapping onto this model:

| Survey criterion | Metric(s) (`Code/core/metrics.py`) | Note |
|---|---|---|
| QoS / SLA | completion rate, dropped, on-time rate, late jobs, tardiness (total, weighted, max, P95), tardiness with dropped-job lower bound | on-time rate counts a dropped job as not on time |
| Latency | mean and P95 waiting time (start − arrival) | no network in this model, so latency = queueing delay; fog transmission delay needs a topology (out of scope) |
| Task completion time | mean flow time (completion − arrival), mean slowdown (flow / duration, DeepRM's objective), makespan | |
| Energy | active machine-ticks, mean utilisation of active machines | equals energy up to constants under the linear power model (section 5) |

**Decision (user):** add these metrics. They're computed from the final environment state, so they don't
depend on the reward. Every method reports them through `run.py`. A hand-checked regression test is in
`tests/test_schedule_metrics.py`.

**First reading (off_c_15, legacy reward):**
- EDF: tardiness with dropped-job lower bound 55.4, on-time rate 0.83, 162.4 active machine-ticks.
- PSO (seed 500000 only): 935, 0.63 and 155.

Even when dropped jobs are charged, EDF is far better on QoS. PSO only wins on completion rate, and on
energy by a small margin.

## 10. Scale / normalisation

**First proposal (dropped):** normalise each component by its value under a reference heuristic (EDF).
**User objection:** it feels wrong. **The user is right**, for three reasons:
it bakes a heuristic into the objective; it divides by zero whenever EDF has zero tardiness (5 of 15
held-out instances); and when training over many instances it silently down-weights hard instances.

**Decision (accepted):** separate two concerns.
1. *Exchange rates between objectives:* keep each component in physical units (weighted late job-ticks;
   active machine-ticks or kWh). λ then reads as "1 late job-tick costs as much as λ machine-ticks" —
   the same way operators price SLA penalties against electricity.
2. *Numerical scale for learning:* divide the whole reward by one global constant c > 0 (e.g. num_jobs).
   Proof: scaling every reward by c scales every return by c, so the optimal policy is unchanged.

Ideal/nadir normalisation (Hayes et al. 2022) is used only for plotting Pareto fronts, never in the reward.

## 11. The "objective program" vision, difficulty, preservation, restructure

**User vision:** a program where you (1) select objectives to minimise/maximise — tardiness only, or
tardiness + energy, or more; (2) select problem difficulty — arrival rate, deadline tightness, etc.;
(3) compare every method on that configuration, to see what each algorithm is best at. Heuristics likely
won't handle multi-objective out of the box, but may be tuned to consider other objectives.

**Design:**
- *Objectives:* registry of components (per-tick charge, drop charge, eval metric); weighted sum with
  explicit λ; constrained RL (existing RCPO / PPO-Lagrangian) and Pareto-front reporting (existing Optuna
  multi-objective); weight-conditioned policies (Abels et al. 2019) as a stretch goal.
- *Difficulty knobs:* load factor ρ = arrival rate × E[size × duration] / capacity; deadline tightness via
  tardiness factor TF and due-date range RDD (Potts & Van Wassenhove 1985 — which paper introduced them is
  **partly unverified**); weight range; size distribution (uniform / lognormal); later DAG depth/fan-out
  (Alibaba 2018 trace, Tian, Zheng & Wang 2019).
- *Heuristics for energy:* the registry already composes priority rule × placement rule; add a
  consolidating placement (prefer already-active, most-loaded machines) → EDF+Consolidate,
  ATC+Consolidate, etc.

**Preservation (user):** every result matters regardless of how the problem was designed. Before any
redesign, snapshot everything: commit the untracked `Results/`, tag
`pre-objective-redesign-2026-09-28`, frozen branch `legacy-reward-v1`; redesign on `objective-redesign`.
`rl_training/` outputs on the D: machine are gitignored and must be backed up separately.

**Restructure (user):** organise by environment/problem variant so each variant's decisions, code and
results sit together; reorganise `Code/` as well (not only docs/results), plus a launcher program for
presentation that chooses which code to run. Plan: `Code/core/` (shared env, objectives, difficulty,
factory, metrics), `Code/variants/{v1_legacy_reward,v2_objectives}/`, `Code/methods/{heuristics,
metaheuristic,exact,rl}/`, `Results/{v1_legacy_reward,v2_objectives}/…`, `run.py` + `experiments/*.yaml`;
old import paths kept as re-export shims so existing checkpoints and commands still work.

**Two-machine artefact storage (decided 2026-09-28).** The user works on two machines: this laptop and the
machine with `D:\University\Year3\...`, which holds all the gitignored `rl_training/` outputs to date.
- *Git* holds code, docs, and the curated result tables and figures. *A cloud-synced folder* (OneDrive or
  Google Drive, not yet chosen) holds models, logs, `results_by_setting` and Optuna databases.
  Git LFS was rejected because the free quota is 1 GB. DVC was considered but deferred as too much overhead
  for now.
- `Code/utils/paths.py` reads an `NK_ARTIFACTS_DIR` environment variable per machine and falls back to
  `REPO_ROOT/rl_training`, so existing commands keep working.
- Each machine writes only to `artifacts/<machine-name>/…`. Sync tools corrupt SQLite (`optuna.db`) and
  fork appended files (`eval_results.csv`, `env_config.npz`) when both machines write them. Either machine
  can read everything.
- Every run writes a `run.json` manifest: git commit, machine, date, full config, final metrics.
- A merge script combines both machines' eval CSVs into the git `Results/`.
- **Timing:** the user is back at the D: machine in about one week (around 2026-10-05). Only then can the D:
  outputs be copied into the synced folder. Until that copy is made, they have no backup.
- **OPEN:** which sync service, and what each machine is called.

## 12. Order of work

1. This record.
2. Backup / tag.
3. Restructure + launcher, no behaviour change; verify v1 presets reproduce EDF 346.34 / 50.87.
4. Formal objective definitions and proofs (for user review; §4 is OPEN).
5. Implement the objective program.
6. Later: comprehensive scientific review of all results (separate plan).

## References

Verified (bibliographic details checked online 2026-09-28) unless marked.

- Barroso, L.A. & Hölzle, U. (2007). The Case for Energy-Proportional Computing. *IEEE Computer* 40(12):33–37. doi:10.1109/MC.2007.443
- Fan, X., Weber, W.-D. & Barroso, L.A. (2007). Power provisioning for a warehouse-sized computer. *ISCA '07*, 13–23. doi:10.1145/1250662.1250665
- SPECpower_ssj2008 methodology. https://www.spec.org/power_ssj2008/
- Wong, D. (2016). Peak Efficiency Aware Scheduling for Highly Energy Proportional Servers. *ISCA '16*, 481–492.
- Ruan, X. & Chen, H. (2015). Performance-to-Power Ratio Aware VM Allocation in Energy-Efficient Clouds. *IEEE CLUSTER 2015*. (peak-efficiency level **unverified**)
- Beloglazov, A. & Buyya, R. (2012). Optimal online deterministic algorithms and adaptive heuristics for energy and performance efficient dynamic consolidation of VMs in Cloud data centers. *CCPE* 24(13):1397–1420. doi:10.1002/cpe.1867 (ESV metric attribution **unverified**)
- Beloglazov, A., Abawajy, J. & Buyya, R. (2012). Energy-aware resource allocation heuristics for efficient management of data centers for Cloud computing. *FGCS* 28(5):755–768. doi:10.1016/j.future.2011.04.017
- Verma, A. et al. (2015). Large-scale cluster management at Google with Borg. *EuroSys '15*. doi:10.1145/2741948.2741964
- Tirmazi, M. et al. (2020). Borg: the Next Generation. *EuroSys '20*. doi:10.1145/3342195.3387517
- Cortez, E. et al. (2017). Resource Central. *SOSP '17*. doi:10.1145/3132747.3132772
- Hadary, O. et al. (2020). Protean: VM Allocation Service at Scale. *OSDI '20*.
- Tian, H., Zheng, Y. & Wang, W. (2019). Characterizing and Synthesizing Task Dependencies of Data-Parallel Jobs in Alibaba Cloud. *SoCC '19*. doi:10.1145/3357223.3362710
- Dean, J. & Barroso, L.A. (2013). The Tail at Scale. *CACM* 56(2):74–80. doi:10.1145/2408776.2408794
- Mao, H. et al. (2016). Resource Management with Deep Reinforcement Learning (DeepRM). *HotNets '16*. doi:10.1145/3005745.3005750
- Mao, H. et al. (2019). Learning Scheduling Algorithms for Data Processing Clusters (Decima). *SIGCOMM '19*. doi:10.1145/3341302.3342080
- Grandl, R. et al. (2014). Multi-resource packing for cluster schedulers (Tetris). *SIGCOMM '14*. doi:10.1145/2619239.2626334
- Grandl, R. et al. (2016). GRAPHENE: Packing and Dependency-Aware Scheduling for Data-Parallel Clusters. *OSDI '16*.
- Hayes, C.F. et al. (2022). A practical guide to multi-objective reinforcement learning and planning. *JAAMAS* 36(1):26. doi:10.1007/s10458-022-09552-y
- Abels, A. et al. (2019). Dynamic Weights in Multi-Objective Deep Reinforcement Learning. *ICML 2019*, PMLR 97:11–20.
- Ng, A.Y., Harada, D. & Russell, S. (1999). Policy Invariance Under Reward Transformations. *ICML '99*, 278–287.
- Bartal, Y. et al. (2000). Multiprocessor scheduling with rejection. *SIAM J. Discrete Math.* 13(1):64–78. (DOI **unverified**)
- Kanet, J.J. & Sridharan, V. (2000). Scheduling with Inserted Idle Time: Problem Taxonomy and Literature Review. *Operations Research* 48(1):99–110. doi:10.1287/opre.48.1.99.12447
- Potts, C.N. & Van Wassenhove, L.N. (1985). A branch and bound algorithm for the total weighted tardiness problem. *Operations Research* 33(2):363–377. doi:10.1287/opre.33.2.363 (origin of TF/RDD **partly unverified**)
- Vepsalainen, A.P.J. & Morton, T.E. (1987). Priority rules for job shops with weighted tardiness costs (ATC). *Management Science* 33(8):1035–1047. doi:10.1287/mnsc.33.8.1035
- Pinedo, M.L. (2022). *Scheduling: Theory, Algorithms, and Systems*, 6th ed. Springer. doi:10.1007/978-3-031-05921-6
