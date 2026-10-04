# Reflection Brief: Redesigning the Project for the Online (Dynamic-Arrival) Case

> **For the reflection agent:** Part A is the factual record of the redesign, taken from the
> project's research log (`Future/research/training-log.md`) and design docs
> (`Future/research/2026-09-16-online-arrival-mdp-design.md`, `2026-09-17-*.md`,
> `2026-09-18-*.md`, `2026-09-21-*.md`). Part B is an example reflection written from it.
> Treat every number in Part A as the ground truth, and don't add results that aren't listed
> here. Where Part B describes feelings or motivations, those are placeholders the student
> should confirm or rewrite in their own voice.

---

## Part A — Information collection

### A1. Project context (one paragraph)
Curtin Year 3 research project (NPSC3001). Cloud resource allocation is modelled as a
resource-constrained scheduling / vector bin-packing problem and framed as an MDP. The
question is whether RL (PPO, A2C) can beat classical heuristics on tardiness, jobs
scheduled and reward. The project was built **offline first**: every job is known at t=0,
with 100 jobs, 10 machines and horizon 100. This made it possible to validate the
formulation against an exact CP-SAT solver (proven optimal tardiness = 8.00). The
**online** case, where jobs arrive live without foreknowledge, is the more realistic version
of cloud RA and was attempted second, once the offline formulation was trusted.

### A2. Why redesign for online at all
- Real cloud jobs arrive in real time, so the offline case is a simplification.
- Offline gave a tractable, provable baseline to validate the MDP first. Online was always
  the end goal.
- Ordering rationale: validate the easy case, then extend. Don't debug two unknowns at once.

### A3. Redesign decisions (each presented as options and chosen by the student, not assumed)
| Decision | Choice | Justification |
|---|---|---|
| Arrival process | Poisson, rate ν jobs/tick | Standard queueing model (Kleinrock 1975); memoryless, one parameter; matches DeepRM (Mao et al. 2016) |
| Deadlines | Relative to arrival: d_j = τ_j + Δ_j | An absolute deadline would let late-arriving jobs be born already overdue |
| Decision clock | Only `idle` advances time; multiple placements allowed per tick (Option 2 of 3 considered) | Keeping "every action = 1 tick" imposes an artificial ν < 1 throughput cap stricter than actual capacity |
| Horizon | Finite window [0, H) | Infinite-horizon average-reward objective needs a whole new training objective (flagged as future work) |
| Load level | ρ = ν/12 via Little's Law (L = λW); target ρ ≈ 0.7 initially | Standard utilisation criterion, derived from 10 machines, capacity 30 and E[p]=E[a]=5 |
| Code structure | **Subclass, don't modify**: `OnlineSchedulingEnv(SchedulingEnv)`, `OnlineGymSchedulingEnv` | Offline training/eval must keep working untouched |
| Fixed-size episodes | "Phantom padding": always `max_jobs` records; unarrived slots get arrival = H+1 | Keeps `num_jobs` constant across episodes as the base env requires |
| Baselines | FCFS now uses real arrival time; new ATC rule (Vepsalainen & Morton 1987) computed only over revealed jobs | Baselines must be causally restricted to what is known at decision time |
| Oracle | Retrospective ("hindsight") CP-SAT with a new `start ≥ arrival` constraint | CP-SAT can't run live, but once an episode is realised it gives a hindsight bound |

### A4. Bugs the redesign exposed (all caught by tests or suspicious results, not by inspection)
1. **Unreachable sentinel was reachable.** Time increments to H+1 before the done check, so
   every padding job got revealed on the terminal step. Fixed by gating reveals on t ≤ H.
2. **Valid placements counted as invalid actions.** The offline wrapper used "did time
   change?" as a proxy for "was the action valid?". Online, valid placements deliberately
   don't advance time, so correct behaviour would have been counted as invalid and would
   truncate episodes, silently and with no exception. Fixed by checking whether the job
   actually left `remaining_jobs`. *Most consequential find.*
3. **Observation leak of future jobs.** With phantom padding, not-yet-arrived jobs' real
   attributes appeared in the observation, which breaks causality. Anticipated in planning and
   fixed by gating on `revealed_jobs`.
4. **CP-SAT oracle said INFEASIBLE.** The offline `AddAllDifferent(start)` constraint (one
   job per tick) was reused, but online allows many jobs per tick. Forcing 200+ jobs into 100
   distinct start times is a pigeonhole contradiction. Caught because INFEASIBLE is a proof,
   and EDF had scheduled the same instance fine.
5. Infrastructure bugs: concurrent runs at different arrival rates shared checkpoint paths,
   and the A2C eval template env read a shared config file written by another run.

### A5. First results showed the online problem was too easy
- First campaign (uniform job sizes, ρ ≈ 0.5 / 0.8 / 0.95, PPO and A2C, 300k steps each):
  EDF/LST/ATC scored **0.00 tardiness at every load**, even ρ ≈ 0.95.
- PPO: 21.73 (ρ 0.8), 392.93 (ρ 0.95). A2C: 176.00, 1031.00.
- **Ranking reversal:** A2C was the better learner in every offline result and was clearly
  worse online. The lesson was not to carry offline conclusions over unchecked.
- The student diagnosed that just pushing ρ ≥ 1 to force lateness "essentially just makes it
  offline": raw volume tests throughput, not decisions under uncertainty.
- **Second redesign: heavy-tailed job sizes.** Log-normal durations and demands,
  *mean-matched* to the uniform case so ρ stays valid and only the distribution's shape
  changes. Grounded in Google (Reiss et al. 2012) and Azure (Cortez et al. 2017) traces:
  "mostly mice, occasionally elephants". Uniform max duration was 10; log-normal max was 40.
- Heuristics only differentiate at **ρ ≈ 0.75–1.0**. At ρ ≈ 0.25 all 9 heuristics gave
  identical results, because the same 11 jobs were structurally unschedulable.

### A6. Action-space redesign carried into online
The offline breakthrough was shrinking PPO's action space from ~1000 (job × machine) to 8.
In **Option 1**, RL picks which of 7 classical rules to apply each tick. That reached 11.00
against a proven optimum of 8.00 offline. Online results at ρ ≈ 0.75, log-normal,
dense + weighted reward, 50-instance eval:

| Method | Weighted tardiness |
|---|---|
| ATC (best heuristic) | 648.16 |
| WSPT+BestFit | 709.42 |
| **Option 1, 300k steps (best online RL)** | **730.30** (20-instance sample; genuinely mixed SPT + LST rule use) |
| Option 1, 900k steps (×3 runs) | 788–798 (**digit-for-digit identical to SPT**) |
| SPT | 798.46 |
| EDF | 800.90 |
| Option 3, windowed (15), EDF-ordered | 952.74 |
| Option 3, windowed, FIFO-ordered | 1190.48 |

- Earlier single-instance check (unweighted): hindsight CP-SAT best bound 105, ATC 120,
  Option 1 163 (identical to SPT).
- RL beats most heuristics under load but **has not beaten ATC online**.
- Option 4 (learned placement via action branching) was worse than fixed FirstFit at matched
  budget. This is evidence that action-space size, not skipping placement, drove the gains.

### A7. The headline open problem: "more online training makes it worse"
- Offline, more training always helped. Online, 300k → 900k made **both** Option 1 and
  Option 3 worse. This was confirmed 3 times independently.
- Diagnostics built to investigate:
  - Rule choice over 9,332 decisions: **SPT 89.2%, idle 10.8%, every other rule 0%**.
    WSPT and ATC, the only weight-aware rules, were never chosen.
  - SPT use doesn't vary with congestion (queue length 0–97). The hypothesis that SPT was
    chosen *adaptively* under load was **refuted**.
  - Policy confidence: top-1 minus top-2 probability = **exactly 1.0 on all 5,591
    decisions**, so the actor is fully saturated. The critic is not collapsed (V(s) std 1.44).
    This is an actor-specific collapse.
  - This corrected the student's own earlier claim ("argmax locks in before entropy decays"),
    which had confused aggregate action-histogram entropy with per-state entropy.
- Fixes tried: `ent_coef = 0.01` kept training entropy healthy but still collapsed to SPT.
  Mechanism: the entropy-bonus gradient vanishes as probabilities saturate. `ent_coef = 0.1`
  was launched, but **no result is recorded yet**.
- Partial theory: SPT minimises mean flow time (Smith 1956), and by Little's Law it therefore
  minimises queue length. That makes it a coherent local optimum under heavy load, but it
  ignores weights and deadlines. WSPT/ATC are the weight-aware refinement it never reaches.

### A8. Honest framing and limitations
- Option 1 is a **selection hyper-heuristic** (Burke et al. 2013). It can only learn *when*
  to use existing rules, never invent new ones, so its ceiling is set by its rule menu.
- Online results are mostly single-seed at a 300k "first-pass" budget. The online case has
  no Optuna tuning of its own and reuses offline hyperparameters, and there is no online
  curriculum.
- The hotspot-penalty and ATC-as-feature ablations were scoped but not run.
- The report's own words: the online case "remains incredibly complex and this project only
  touches on it."

### A9. Future-work directions already identified
- Resolve the online collapse before extending anything else to it. Candidates: KL / trust
  region, logit-level regularisation, a large entropy bonus applied early only, and
  stochastic evaluation.
- GP-evolved rule pool + RL selection (Chen et al. 2024), tried offline first.
- An infinite-horizon average-reward formulation, online-specific HPO, a curriculum, and
  multi-seed rigour.

### A10. Suggested reflection angles
1. What surprised you: the "easy" online problem, the A2C/PPO reversal, and more training
   hurting.
2. Process: subclass-don't-modify, test-driven bug catching, and treating INFEASIBLE as a
   clue.
3. Being wrong productively: two of your own hypotheses were refuted by your own
   diagnostics (congestion-adaptivity, the entropy framing).
4. What you'd do differently: design the load/difficulty regime (heavy tails) *before* the
   first training campaign, run multi-seed from the start, and budget for online-specific
   tuning.
5. Transferable lesson: offline conclusions don't transfer automatically. Re-validate every
   assumption when the problem changes.

---

## Part B — Example reflection (~750 words)

*(Structured loosely on Gibbs' cycle: description → feelings → evaluation → analysis →
conclusion → action plan. Personal feelings are placeholders to adjust.)*

**Redesigning for the online case**

By the time I turned to the online case, I had a working offline formulation that I trusted.
It had been validated against an exact CP-SAT solver, and an RL hyper-heuristic came within
three tardiness units of the proven optimum. The online case was always the real goal,
because cloud jobs don't arrive all at once. I went into it expecting mainly an engineering
extension: add an arrival process, reveal jobs over time, and retrain. It turned out to be a
redesign of what the problem even *is*.

The first round of decisions went well, I think, because I treated each one as a choice to
justify rather than a default to accept. I chose Poisson arrivals because they are the
standard, one-parameter queueing model. I made deadlines relative to arrival, because an
absolute deadline would let a job arrive already late. I used Little's Law to set arrival
rates from actual machine capacity instead of picking numbers by feel. The decision I'm
proudest of is the decision clock. Keeping the offline rule that every action costs a tick
would have silently capped throughput below what the machines could handle, so only the idle
action advances time. I also kept the offline code untouched by subclassing it. That paid
off immediately, because I could always compare the online behaviour against a known-good
reference.

That same decision-clock change is where the redesign bit back. Several assumptions in the
offline code were true only because time always advanced. The wrapper used "did time change?"
to detect invalid actions, so online it would have punished the agent for exactly the
behaviour I designed it to have. It would have done this silently, truncating episodes
without ever raising an error. My hindsight CP-SAT oracle reused a one-job-per-tick
constraint and reported the online problem as provably infeasible. Neither bug would have
been obvious from reading the code. I caught them because I had written tests for the
mechanism, and because I took an "INFEASIBLE" result seriously as evidence instead of
dismissing it. The lesson I take from this is that when you change a core rule of a system,
every place that quietly depended on the old rule becomes suspect.

The first training campaign humbled me in a different way. EDF, LST and ATC scored zero
tardiness at every load level, even at 95% utilisation. My online problem was too easy to
tell good scheduling from bad. My first instinct was to push the load higher, but I realised
that overloading the system just turns it back into a throughput problem, which is
effectively the offline case again. What makes online genuinely hard is uncertainty about
what's coming. So I redesigned the workload using heavy-tailed, log-normal job sizes,
grounded in published Google and Azure cluster traces. I mean-matched them so load stayed
comparable and only the shape of the distribution changed. Only then did the heuristics start
to differ from each other. Looking back, I should have designed the difficulty regime
*before* running the first campaign, not after.

The results since then are mixed, and I've tried to report them honestly. My best online RL
policy beats most heuristics under heavy load (730 weighted tardiness), but it doesn't beat
ATC (648). More striking is that training longer consistently made things worse. Three
independent 900k-step runs collapsed onto a policy numerically identical to SPT. A2C, the
better algorithm offline, was the worse one online. I built diagnostics to understand the
collapse, and they overturned two of my own explanations. SPT use didn't depend on
congestion, so it wasn't an adaptive strategy. My earlier claim about entropy had confused
two different entropy measures. What the data actually showed was a fully saturated actor
with a healthy critic, and an entropy bonus whose gradient vanishes at exactly the point it's
needed. I don't yet know the root cause. I did find a plausible reason SPT specifically is
attractive: it minimises queue length, but it ignores job weights entirely.

*[Feelings placeholder: e.g. frustration that more compute made things worse, then
satisfaction at turning a mystery into a precisely characterised phenomenon.]*

Evaluating the redesign as a whole, I think the formulation work was strong and well
justified. The experimental design lagged behind it. I ran single seeds at a first-pass
budget, reused offline hyperparameters, and had no online curriculum. The biggest thing I'd
change is to stop assuming offline conclusions transfer. A2C's superiority, "more training
helps" and the adequacy of my load settings each held offline, and each failed online.

If I continued, I would fix the online training collapse before adding anything else, for
example with trust-region or logit-level regularisation, or a strong early entropy bonus. I
would then add multi-seed runs and online-specific tuning. Only after that would I extend the
hyper-heuristic with machine-generated rules. More broadly, this redesign taught me that
changing a problem's assumptions means re-checking conclusions as well as code.
