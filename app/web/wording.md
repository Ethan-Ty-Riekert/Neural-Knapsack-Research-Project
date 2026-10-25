// =====================================================================================================================
// WORDING FILE for the Scheduler Race demo. Every sentence the visitor reads is in this file. Edit it in any text editor,
// then run:   python tools/analysis/make_demo_artifacts.py     (and republish app/web/dist/variant_a.html)
//
// HOW TO EDIT
//  * Lines starting with // are comments and are ignored.
//  * "## NAME" starts a list of "key: value" lines (one line per value). Change the text after the colon, never the key.
//  * "### topic-id" starts one explanation card. Under it: "kicker:", "title:", "related:" (ids of cards to link to),
//    then sections that start with "--- Section heading" (add "(open)" to show the section expanded). Write the section
//    text under the heading; a blank line starts a new paragraph.
//  * Formatting you can use in any text:  **bold**   *italic*   `code`   a_{j} (subscript)   x^{2} (superscript)
//  * {{name}} drops in a formula or a table that the program builds. You can move it, repeat it or delete it. Available:
//      {{tard}} T_j     {{J}} cost J     {{late}} late jobs     {{wait}} mean wait     {{flow}} mean flow time
//      {{energy}} energy     {{rho}} load factor     {{dueOff}} offline deadlines     {{dueOn}} online deadlines
//      {{score}} score S     {{pct}} bar percentage     {{copy}} copying loss     {{ppo}} PPO objective
//      {{reward}} reward and shaping     {{policy}} Option 2 network     {{policy_atc}} Option 3 network
//      {{opt1}} Option 1 action     {{opt0}} Option 0 action
//      Racer card only: {{racer_table}} {{racer_sum}} {{racer_what}} {{racer_math}} {{racer_deeper}}
//      Objective card only: {{weights_table}}
//  * Text in {curly braces like this} inside "## TEMPLATES" is filled in by the program (for example {name}); keep it.
//  * Keep rule ids (EDF, MDC ...), option numbers and topic ids exactly as they are; they are how the program finds things.
// =====================================================================================================================

## UI
heading: Scheduler Race
legend.rule: Classic rule
legend.copied: AI scheduler that copies a rule
legend.own: AI scheduler with its own model
step1: The situation
step2: 2 · What matters to you
step3: 3 · The race (100% = best result found)
start: Start the race
idle: Choose your situation and what matters, then press Start the race.
nothing: Nothing to compare for this combination - choose something that matters.
winner: Winner: {title}
kind.rule: classic rule
kind.ai: AI scheduler
bar_tooltip: Click for the exact score and the maths
row_tooltip: Click for the maths
winner_tooltip: Click to see how to read this
back: ← Back
close: Close
related: Related
row.arrival: Jobs arrive
row.machines: Machines
row.deadlines: Deadlines
row.jobs: Jobs
hint.tight_only_normal: tight only tested at Normal

## LABELS
// the words on the buttons (the left part, before the colon, is the code's name for each choice)
arrival.online: Over time
arrival.offline: Known in advance
busy.Quiet: Quiet
busy.Normal: Normal
busy.Busy: Busy
busy.Overloaded: Overloaded
busyhint.Quiet: about half used
busyhint.Normal: about 3/4 used
busyhint.Busy: nearly full
busyhint.Overloaded: more work than capacity
deadline.Normal: Normal
deadline.Tight: Tight
deadlines.Relaxed: Relaxed
deadlines.Medium: Medium
deadlines.Tight: Tight
jobs.equal: All equal
jobs.unequal: Some matter more
level.0: Ignore
level.2: Some
level.5: A lot
// the short names on the "Related" buttons at the bottom of a card
chip.setup: The setup
chip.arrival: Arrival
chip.machines: Busyness ρ
chip.deadlines: Deadlines
chip.importance_jobs: Job weights
chip.objective: The objective
chip.f_delay: Delay cost J
chip.f_late: Late jobs
chip.f_wait: Wait
chip.f_flow: Flow time
chip.f_energy: Energy
chip.bar: The bar
chip.winner: The winner
chip.kind_rule: Classic rules
chip.kind_copied: Copying AI
chip.kind_own: Own-model AI
objective.table.thing: Thing
objective.table.setting: Setting
objective.table.lambda: λ
objterm.lin: linear lateness
objterm.late: late jobs
objterm.energy: energy
metric.objective_J: Cost J (squared lateness)
metric.late_jobs: Late jobs
metric.mean_wait: Mean wait
metric.mean_flow_time: Mean flow time
metric.active_machine_ticks: Energy (machine-ticks)
example.1: Never be late
example.2: Customers hate waiting
example.3: Energy Usage
example.4: A bit of everything

## FACTORS
// the five things a visitor can care about: label shown on the row, and the small hint beside it
late.label: Avoid late jobs
late.hint: how many miss their deadline
delay.label: Avoid long delays
delay.hint: a very late job hurts extra
wait.label: Short waiting
wait.hint: before a job starts
flow.label: Finish each job quickly
flow.hint: after it arrives
energy.label: Save energy
energy.hint: fewer machines switched on

## NOTES
// the sentence under the winner line
noai: We did not train an AI scheduler for this exact setting, so four classic rules race instead.
noimprove: No AI scheduler beat the classical results, the normal result found in this project
noise: The best AI scheduler is only {pct}% ahead of the best classic rule, which is within the run-to-run noise of this project (gaps under about 5% were not significant).

## RULES
// name = what the racer is called; plain = one sentence; cite = where the rule comes from. Formulas are built by the program.
EDF.name: Earliest deadline first
EDF.plain: Starts the job whose deadline comes first.
EDF.cite: Classical (Pinedo, Scheduling: Theory, Algorithms, and Systems).
EDFL.name: Earliest deadline first (longest first on ties)
EDFL.plain: Earliest deadline first; when deadlines tie, the longer job goes first.
EDFL.cite: Added in this project (2026-10-21) so ties are decided by a job feature a network can read.
WLST.name: Weighted least slack
WLST.plain: Weighted least slack: slack divided by weight while a job can still be on time, multiplied by weight once it is late.
WLST.cite: Constructed for this project (not a published rule).
LST.name: Least slack first
LST.plain: Starts the job with the least slack: time left before it must start to finish on time.
LST.cite: Classical (least slack time).
WSPT.name: Weighted shortest job first
WSPT.plain: Weighted shortest job first: important and short jobs first.
WSPT.cite: Smith (1956), Naval Research Logistics Quarterly 3.
SPT.name: Shortest job first
SPT.plain: Starts the shortest job first.
SPT.cite: Classical (shortest processing time).
LPT.name: Longest job first
LPT.plain: Starts the longest job first.
LPT.cite: Graham (1969), Bounds on multiprocessing timing anomalies.
WMDD.name: Weighted modified due date
WMDD.plain: Weighted modified due date: time to the deadline (but at least the job's own length), divided by its weight.
WMDD.cite: Kanet & Li (2004), Journal of Scheduling 7(4); designed for linear tardiness, applied unchanged to the squared cost.
WEDF.name: Weighted earliest deadline first
WEDF.plain: Weighted earliest deadline first: time to the deadline, weight-scaled the same way as WLST.
WEDF.cite: Constructed for this project (not a published rule).
MDC.name: Marginal delay cost (MDC)
MDC.plain: Marginal delay cost: starts the job whose one-tick delay adds the most squared lateness per unit of its own length.
MDC.cite: Derived in this project from the cost J (2026-10-07), using Smith's ratio logic (1956). A heuristic for parallel machines, not a guarantee.
COVERT.name: COVERT (urgency versus cost)
COVERT.plain: COVERT (cost over time): the weighted-shortest-job index, scaled by how close the job is to being late.
COVERT.cite: Carroll (1965), MIT PhD thesis; compared with ATC in Vepsalainen & Morton (1987). Simplified lead-time estimate (flagged in the code).
ATC.name: Apparent tardiness cost (ATC)
ATC.plain: Apparent Tardiness Cost: the weighted-shortest-job index, switched on as a job runs out of slack.
ATC.cite: Vepsalainen & Morton (1987), Management Science 33(8). k = 2 is the literature default, not re-tuned here.
FCFS.name: First come, first served
FCFS.plain: First come, first served.
FCFS.cite: Classical.
RandomRule.name: Random order
RandomRule.plain: Picks a waiting job at random (a sanity-check baseline).
RandomRule.cite: Baseline.

## PLACEMENTS
// how a started job is given a machine
FirstFit.short: first machine that fits
FirstFit.plain: First fit: the first machine (lowest index) where the job fits.
FirstFit.cite: Johnson (1973), Near-optimal bin packing algorithms.
BestFit.short: tightest-fitting machine
BestFit.plain: Best fit: the machine that is left with the least spare capacity.
BestFit.cite: Johnson (1973).
Consolidate.short: packs jobs onto few machines
Consolidate.plain: Consolidate: the machine whose use switches on the fewest new machine-ticks over the job's run (ties by best fit), so work is packed onto few machines and the rest stay off.
Consolidate.cite: Per-job version of the 'allocate where power increases least' idea of Beloglazov, Abawajy & Buyya (2012), Future Generation Computer Systems.

## OPTIONS
// the kinds of AI model. short = the words shown under "AI scheduler" on a bar; name and plain = the card
0.short: picks job and machine directly
0.name: Option 0, full action space
0.plain: The network picks a (job, machine) pair, or idle, directly: max_jobs × M + 1 choices.
1.short: picks which rule to apply
1.name: Option 1, rule selection
1.plain: Instead of directly placing jobs, it selects a rule to do so
2.short: scores each waiting job
2.name: Option 2, priority learning
2.plain: The network gives every waiting job a score (within a window) and the best-scoring job is started. A fixed placement rule then picks its machine.
3.short: scores each waiting job
3.name: Option 3, priority learning with an ATC hint
3.plain: As Option 2, along with the ATC urgency index.
4.short: picks a job-machine pair
4.name: Option 4, job × machine branching
4.plain: The network picks a job and then a machine in two steps.

## TEMPLATES
// sentences the program fills in. Words in {braces} are replaced; keep them.
ai.title: AI scheduler
detail.rule: classic rule; {placement}
detail.rule.noplacement: classic rule
detail.ai: {what} | {algo}{budget}
detail.copies: copies {rule}
detail.copies.generic: copies a classic rule
detail.atc_hint: {what} using an urgency hint
budget.1M: (1M steps)
budget.2M: (2M steps)
racer.title.rule: {rule} + {placement}
racer.title.ai: {title}: {detail}
racer.kicker.rule: Classic rule
racer.kicker.ai: AI scheduler
racer.results: Mean over {inst} test instances ({runs}). Highlighted rows are the things you weighted.
racer.runs.many: average of {n} independent training runs; ± is the standard deviation across runs
racer.runs.one: ± is the standard deviation across the test instances
racer.table.measured: Measured
racer.table.value: Value
racer.table.bestrule: Best rule
racer.table.ratio: Ratio
racer.table.weight: λ
racer.table.contrib: λ × ratio
racer.sum: Score *S* = {terms} = **{score}**. Best score in this scenario *S*^{*} = {best}, so the gap is {gap}% and the bar shows 100 × exp(−4 × {gapfrac}) = **{pct}%**.
racer.rule.what: **{name}**: {plain}\n\nMachine choice: {placement}
racer.rule.math: A job is chosen from those that can start now by this key; *t* is the current time.
racer.rule.deeper: Source: {cite}\n\nPlacement: {pcite}\n\nA rule is deterministic, so its numbers have no training-run spread. Its table values are means over the test instances.
racer.ai.what: **{optname}.** {optplain}
racer.ai.window: It looks at the {win} most urgent (earliest-deadline) jobs that can start now, plus a count of the backlog beyond them, and a fixed placement rule ({plc}) picks the machine.
racer.ai.allslots: It looks at every job slot, and a fixed placement rule ({plc}) picks the machine.
racer.ai.copied: Trained by copying **{rule}** (behaviour cloning).
racer.ai.copied.generic: Trained by copying **a classic rule** (behaviour cloning).
racer.ai.scratch: Trained with {algo} from scratch (no rule copied).
racer.ai.budget: Budget: {steps} environment steps.
racer.ai.obj.j: Training objective: the squared-lateness cost J only.
racer.ai.obj.multi: Training objective: the squared-lateness cost J plus {terms}, at fixed reference weights.
racer.ai.obj.copy: Its objective only trains its value estimate and any fine-tuning; the choices themselves are copied from the rule.
racer.ai.math.other: See Code/methods/rl/action_spaces in the project for this option's action space.
racer.ai.deeper.copy: Copying: expert labels come from the copied rule on the states the network itself visits (DAgger; Ross, Gordon & Bagnell 2011), then the value head is warmed up with the policy frozen.
racer.ai.deeper.scratch: Training objective: the shaped reward below, which keeps the optimal policy of the objective the model was trained on (Ng, Harada & Russell 1999).
racer.ai.deeper.tail: Here *c* is the number of jobs in the instance, γ the discount, and Φ the lateness potential. {algonote} Each AI result is the mean over its training runs; one training run per seed is evaluated once on every test instance.
racer.ai.deeper.scorer: Options 2 and 3 score each job from its own features only (shared weights), so the network cannot compare two jobs directly; the surrounding context only affects the idle option and the critic.
racer.ai.algonote.A2C: A2C (Mnih et al. 2016) replaces PPO's clipped objective with a plain advantage actor-critic update.
racer.ai.algonote.PPO: PPO: Schulman et al. 2017.
factor.insore: This row sets λ for the metric `{metric}`: Ignore 0, Some 2, A lot 5. Its contribution to a method's score is λ × (method's value ÷ best rule's value).
factor.section.plain: In plain words
factor.section.formulation: Formulation
factor.section.insore: In the score
// short notes printed beside or under the formulas
math.smallest: (smallest first)
math.largest: (largest first)
math.largest_k: (largest first, k = 2)
math.mdc_tie: (largest first; jobs with ΔC = 0 go by least slack)
math.wlst_if: if σ ≥ 0,
math.wlst_else: otherwise
math.wlst_with: with σ =
math.wedf_scaled: weight-scaled
math.slack_range: slack range
math.energy_cond: machine m runs at least one job at tick t
math.unfinished: unfinished
math.visited: states visited
math.policy.shared: (shared weights, embedding size 128)
math.policy.over: over the jobs that can start now, plus an idle option
math.policy.note: The idle option's score comes from the mean job embedding, the mean machine embedding and the time. Jobs that cannot start are masked out.
math.opt1.machine: machine chosen by π
math.opt1.note: The network scores each (rule, placement) pair and picks one with a softmax; the chosen rule then selects the job.
math.opt0.z: z is the score of job j on machine m, computed from their embeddings
math.opt0.note: A pointer-style network scores every job-machine pair. The action space has max_jobs × M + 1 choices, which is why learning from scratch is hard here.
math.reward.note: Ĉ_{j} is the earliest time job j can still finish (fixed once it starts). Φ is potential-based, so it adds no new optimum.

### setup
kicker: The situation
title: The scheduling problem and the experimental setup
related: arrival, machines, deadlines, objective
--- In plain words (open)
Users send jobs to a server that require machines to process their request. Each job have values like deadline, resource demand (CPU, memory, ...), and processing duration. A scheduler must decide which waiting job to start on which machine whilst fitting a set of constraints e.g a machine at 100% capacity can't take in another job.
--- Formulation (open)
Job *j* has arrival a_{j}, processing duration p_{j}, deadline d_{j}, weight w_{j} and resource demand u_{jr}. There are *M* = 10 machines with capacity C_{r} per resource *r*. A job started at time s_{j} on machine *m* occupies u_{jr} of resource *r* on machine *m* for ticks [s_{j}, s_{j} + p_{j} − 1] and may never exceed C_{r}. It completes at C_{j} = s_{j} + p_{j}, and is late by

{{tard}}

The objective is a weighted sum of up to four terms (linear lateness, squared lateness, late jobs, energy), each with a switch λ:

{{objfull}}

Which terms are switched on depends on the model. Most models here use only the squared-lateness term (λ_{S} = 1, the others 0), which gives the cost

{{J}}

Models labelled '+tardiness obj', '+late-job obj' or '+energy obj' were trained with the extra terms switched on as well. Jobs are never dropped here, so the dropped-job term of the full objective is always zero.

Idling is always allowed (it can pay to leave a machine empty).
--- Experimental setup
Each scenario has 50 held-out test instances (generated job sets, random seeds 500000 and up). Every method is run once on every instance and the metrics are averaged over the 50. Learned models are trained separately (3 to 10 training runs per kind) and their average is shown. A second set of 50 fresh instances (seeds 700000 and up) is used only for the confirmation results.

Given **all jobs are known** in advance, we give the environment 100 time units with 100 jobs whose values are drawn uniformly to find its best schedule. In the case of **online job arrivals**, the same setup is used with the addition of Poisson arrivals over the same 100 time units at differing job arrival loads ρ.

### arrival
kicker: The situation
title: How jobs reach you: online or known in advance
related: setup, machines, deadlines
--- In plain words (open)
**Over time (online):** a job is only revealed when it arrives, so the scheduler cannot plan for jobs it has not seen. **Known in advance (offline):** every job is known at time 0 and the scheduler can see them all.
--- Formulation
Online, arrival times a_{j} come from a Poisson process with rate λ and a decision at time *t* may use only jobs with a_{j} ≤ *t*. Offline, a_{j} = 0 for all jobs. Offline instances have 100 jobs; online instances run over a horizon of 100 ticks with a preferred horizon *H*, and jobs may finish after it (the cost still counts their lateness).

### machines
kicker: The situation
title: How busy the machines are (load factor ρ)
related: setup, arrival, deadlines
--- In plain words (open)
Quiet: the machines are about half used on average. Normal: about three quarters. Busy: nearly full. Overloaded: more work arrives than the machines can ever finish, so a queue builds up.
--- Formulation (open)
ρ (rho) says how busy the machines are. λ (lambda) is how many jobs arrive per tick (the rate of the Poisson arrivals). They are linked: the more jobs arrive per tick, the busier the machines. Standard queueing utilisation (Kleinrock 1975), for resource *r*, with *M* machines of capacity C_{r}, where *E*[p u_{r}] is the average work one job brings on that resource:

{{rho}}

We choose the busyness we want to test, ρ = 0.50, 0.75, 0.95 or 1.10, and then set λ to the arrival rate that gives exactly that ρ. Because ρ grows in proportion to λ, this is a single scaling: λ = ρ ÷ (the ρ you would get at one job per tick), where that reference value is measured on a large sample from the project's own job generator (lognormal job sizes). ρ < 1: the machines can keep up on average. ρ ≥ 1: more work arrives than they can finish, so a queue builds up.

### deadlines
kicker: The situation
title: How tight the deadlines are
related: setup, importance_jobs, machines
--- In plain words (open)
Tight deadlines leave jobs little spare time, so more of them end up late and the choice of schedule matters more.
--- Formulation (open)
**Offline** (tardiness factor TF and due-date range RDD, Potts & Van Wassenhove 1985), with *L* a lower bound on the makespan:

{{dueOff}}

Relaxed, Medium, Tight use TF = 0.2, 0.5, 0.8 with RDD = 0.6; deadlines are never earlier than a job's own length.

**Online**:

{{dueOn}}

Tight deadlines use a slack range of 5 to 25 ticks and exist only at Normal load.

### importance_jobs
kicker: The situation
title: Are all jobs equally important?
related: objective, setup
--- In plain words (open)
If all jobs are equal, lateness of any job costs the same. If some matter more, being late on an important job costs more.
--- Formulation (open)
Weights w_{j} enter the cost through J = Σ w_{j} T_{j}^{2}. The 'All equal' option sets w_{j} = 1; 'Some matter more' draws integer weights from 1 to 5. Tested offline only.

{{J}}

### objective
kicker: What matters to you
title: The objective and your choices
related: f_delay, f_late, f_wait, f_flow, f_energy, bar
--- In plain words (open)
Every option provided to rate is a measured quantity of a finished schedule. The ratings decide the weights of how much each metric mattters and how the models will be scored against. Lower overall score is better.
--- Formulation (open)
With metric values x_{m} of a method, the best rule's value x_{m,ref} on that metric, and your weights λ_{m} (Ignore = 0, Some = 2, A lot = 5):

{{score}}

The race will then compares *S* with the best score *S*^{*} found among all rules and learned models to make a relative score:

{{pct}}

With the current selected weights:

{{weights_table}}
--- How it relates to what the AI was trained for (open)
The learned schedulers are trained by reinforcement learning to minimise an objective built from the same kinds of terms as your choices:

{{objfull}}

Most of the models raced here were trained with only the squared-lateness term switched on, which is the cost

{{J}}

A smaller group, labelled '+tardiness obj', '+late-job obj' and '+energy obj' in the results, were trained with the extra terms (linear lateness, late jobs, energy) switched on as well, at fixed reference weights. Models that copy a rule are trained by imitation, so the objective only trains their value estimate and any fine-tuning.

Your own weights cannot be trained into a model in advance, because there are endless combinations of them. They are applied afterwards to the recorded measurements of each finished schedule. So no model was trained for your exact mix: a model trained only for lateness can look good on delay and poor on energy. Each racer's card says which objective it was trained on.

### f_delay
kicker: What matters to you
title: Avoid long delays
related: objective, bar
metric: objective_J
--- In plain words (open)
Counts how late jobs are, with big delays hurting much more than small ones (squared) and important jobs counting more.
--- Formulation (open)
{{tard}}

{{J}}

This is the cost that most of the learned models were trained to minimise. Models labelled '+tardiness obj', '+late-job obj' or '+energy obj' were trained on this cost plus the extra terms.

### f_late
kicker: What matters to you
title: Avoid late jobs
related: objective, bar
metric: late_jobs
--- In plain words (open)
How many jobs finish after their deadline, however late.
--- Formulation (open)
{{late}}

A job counts as late if T_{j} > 0. Late jobs are never refused: they are counted. This row counts every late job once; the late-job term used when training some models is weighted, each late job counting w_{j} times.

### f_wait
kicker: What matters to you
title: Short waiting before a job starts
related: objective, bar
metric: mean_wait
--- In plain words (open)
How long a job sits in the queue before it starts, on average.
--- Formulation (open)
{{wait}}

s_{j} is the start time and a_{j} the arrival. This is a score, not a limit: no rule guarantees a maximum wait.

### f_flow
kicker: What matters to you
title: Finish each job quickly after it arrives
related: objective, bar
metric: mean_flow_time
--- In plain words (open)
How long a job takes from arrival to finish, on average (waiting plus running).
--- Formulation (open)
{{flow}}

C_{j} is the completion time.

### f_energy
kicker: What matters to you
title: Save energy
related: objective, bar
metric: active_machine_ticks
--- In plain words (open)
How many machines are switched on, and for how long. Packing jobs onto fewer machines saves energy.
--- Formulation (open)
{{energy}}

Counted in active machine-ticks (a machine that runs at least one job at a tick counts as one). A linear power model is assumed; a measured SPECpower profile exists in the archive but is not used in this tool.

### bar
kicker: The race
title: How the bar is computed
related: objective, winner
--- In plain words (open)
Each bar shows how close a method gets to the best result found for your situation and your priorities. 100% is the best of all rules and learned models in this scenario (not only the five racers). The scale is stretched so that small differences are visible: 5% worse than the best shows as about 82%, 25% worse as about 37%.
--- Formulation (open)
{{pct}}

where *S* is the method's score (see the objective) and *S*^{*} the lowest score in the scenario. If the best score is not positive the gap is divided by (worst − best) instead.
--- Deeper
The exponential scale is a display choice, not a statistical statement. A bar at 0% only means the score is more than about 2.2 times the best.

### winner
kicker: The race
title: Reading the winner and the 'within noise' note
related: bar, setup
--- In plain words (open)
The winner is the racer with the lowest score. When a learned model is within about 5% of the best rule, the project treats the two as equivalent: differences that small were not statistically significant in the earlier tests.
--- Deeper
These results are exploratory. 'Beats' was defined in advance as a paired interval over instances lying entirely below zero with at least 5 training runs; none of the learned models met it on the fresh instances. Training-run variance is not included in any interval. See the slides for the fresh-instance results.

### kind_rule
kicker: Racers
title: Classic rules (grey)
related: kind_copied, kind_own
--- In plain words (open)
A classic rule ranks the waiting jobs by a simple formula and starts the best-ranked one; a placement rule then picks the machine. They need no training and are fast. Click a rule's bar to see its formula.

### kind_copied
kicker: Racers
title: AI scheduler that copies a rule (blue)
related: kind_own, kind_rule
--- In plain words (open)
A neural network was shown what a classic rule chooses in thousands of situations and trained to make the same choice. It is a learning check: can the network represent the rule? It is not expected to beat the rule it copies.
--- Formulation (open)
{{copy}}

### kind_own
kicker: Racers
title: AI scheduler with its own model (orange)
related: kind_copied, kind_rule
--- In plain words (open)
A neural network trained by reinforcement learning alone, from random weights, with no rule to copy. It improves its choices from the reward it receives, which is built from its training objective. Results varied widely between training runs.
--- Formulation (open)
{{ppo}}

{{reward}}

### racer
// the card opened by clicking a bar. The titles come from ## TEMPLATES; the content is built from the racer's data.
related: objective, bar
--- Exact performance (open)
{{racer_table}}

{{racer_sum}}
--- What this is (open)
{{racer_what}}
--- The equations (open)
{{racer_math}}
--- Deeper
{{racer_deeper}}
