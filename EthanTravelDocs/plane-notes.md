# Plane notes -- read offline, no internet needed

Companion to `report/main.tex`, built to stand in for asking me questions directly while
you have no internet and no chat access. Three jobs: (1) a **glossary** (Section 0) for
every non-obvious term or method the report and supervisor document use; (2) project-
specific reference material (Sections 1-8) -- answers to the questions your draft left open,
a plain-prose project summary, citation notes, a "what belongs where" writing guide, and a
LaTeX troubleshooting cheat-sheet; and (3) an extensive **knowledge dump** (Section 9) --
design-decision FAQs, a codebase map, a cheat-sheet of every experiment tried, the reward
function fully explained, anticipated marker/supervisor questions with suggested answers,
and how the pipeline works end-to-end. Not part of the report itself. If you have a
question and aren't sure where to look, start at Section 9 -- it's the closest thing to
actually asking me.

---

## 0. Concepts & terminology, explained

Organised by where they show up in the report. Ctrl+F for a term if you're looking for
something specific.

### Reinforcement learning basics

- **MDP (Markov Decision Process)** -- the standard mathematical framework for a
  decision-making problem where an agent repeatedly observes a *state*, picks an *action*,
  and gets a *reward*, and the next state depends only on the current state and action (not
  the full history -- that's the "Markov" property). Formally a tuple $(S, A, P, R, \gamma)$:
  state space, action space, transition probabilities, reward function, discount factor.
  Every RL problem, including this one, starts by defining these five things.
- **Policy ($\pi$)** -- the agent's strategy: a (possibly probabilistic) rule for picking an
  action given a state. "Training" an RL agent means adjusting the policy's parameters to get
  more reward over time.
- **Value function ($V$) / action-value function ($Q$)** -- $V(s)$ estimates "how good is it
  to be in state $s$, if I keep following my current policy from here?" (in expected future
  reward). $Q(s,a)$ is the same idea but for a specific state-action pair. These are what the
  "critic" in actor-critic methods learns to estimate.
- **Bellman equation** -- the recursive relationship that says a state's value equals the
  immediate reward plus the (discounted) value of whatever state comes next. It's the
  mathematical backbone of almost all RL, but solving it *exactly* requires enumerating every
  state and action -- infeasible here, which is exactly why neural-network function
  approximation (i.e. "deep" RL) is needed instead of classical dynamic programming.
- **Discount factor ($\gamma$)** -- how much future reward is worth relative to immediate
  reward, between 0 (only care about right now) and 1 (future reward counts fully). This
  project uses $\gamma = 1$ (undiscounted) because the objective already sums costs evenly
  across the whole planning horizon, with no reason to prefer an early win over a late one.
- **Policy gradient** -- a family of RL methods that directly compute the gradient of
  expected reward with respect to the policy's parameters, and do gradient ascent. Contrast
  with *value-based* methods (like Q-learning) that instead learn a value function first and
  derive a policy from it.
- **Actor-critic** -- a policy-gradient method that pairs a policy ("actor," decides what to
  do) with a learned value function ("critic," judges how good that decision turned out to
  be), using the critic's judgement to reduce the noisiness (variance) of the actor's
  learning signal. PPO and A2C, this project's two algorithms, are both actor-critic methods.
- **Entropy / entropy regularisation** -- entropy here measures how "spread out" (vs.
  deterministic) the policy's action probabilities are. Adding an entropy bonus to the
  training objective encourages the policy to keep exploring instead of prematurely
  committing to one action. Relevant to the online case's "more training hurts" mystery: it
  was tested as a fix and *didn't* solve the problem (see report Results, online case).

### The two algorithms used

- **PPO (Proximal Policy Optimisation)** -- the main algorithm used in this project. Its key
  trick: when updating the policy, it "clips" how far the new policy is allowed to move away
  from the old one in a single update. This prevents one bad, overly-aggressive update from
  wrecking training, which is the classic failure mode of naive policy-gradient methods. It's
  popular because it's simple to implement and usually stable.
- **A2C (Advantage Actor-Critic)** -- the second algorithm. "Advantage" refers to using
  $A(s,a) = Q(s,a) - V(s)$ (how much better this action was than average) instead of the raw
  value as the learning signal, which reduces variance. A2C runs multiple copies of the
  environment in parallel and updates synchronously (all at once), as opposed to A3C which
  updates asynchronously. This project's A2C is hand-rolled (custom-written) because the RL
  library used (Stable-Baselines3) doesn't ship a maskable A2C implementation (see "action
  masking" below).
- **RCPO (Reward Constrained Policy Optimisation)** -- a way to add a *constraint* (e.g. "keep
  tardiness below some target") to an actor-critic method, using a Lagrange multiplier that
  automatically adjusts how hard the constraint is enforced during training. Sounds elegant,
  and initially looked like the project's best result -- but turned out to be exploitable
  (the agent found a cheap way to satisfy the constraint -- abandoning jobs -- instead of the
  intended way of scheduling them well). This is a real, general risk with constrained RL:
  the constraint has to be specified *exactly* right, or the optimiser will find the easiest
  way to satisfy it, not the intended one.
- **Reward hacking** -- the general name for what happened to RCPO above: an agent finds a
  way to score well on the *literal* reward/constraint that doesn't match the *intended*
  behaviour. Worth knowing as a term because it's a recognised, citable phenomenon in the RL
  literature, not just "a bug."
- **Potential-based reward shaping** -- a mathematically safe way to add extra guidance to a
  reward function (making learning easier/faster) *without* changing which policy is
  actually optimal at the end. Proven by Ng, Harada & Russell (1999) -- cited as `ng1999shaping`.
  Used in this project's A2C-shaping result.
- **Action masking** -- restricting which actions the policy is even allowed to sample at a
  given state (e.g. don't let it "select" a job/machine placement that would violate
  capacity). Used throughout instead of just penalising invalid actions after the fact,
  because it's more sample-efficient (no wasted training signal on actions that could never
  be valid).

### Scheduling terms

- **Tardiness** -- how late a job finishes relative to its deadline: $T_j = \max(0,\ \text{finish
  time} - d_j)$. Zero if on time or early. The main metric this whole project optimises for.
- **Weighted tardiness** -- tardiness multiplied by the job's importance weight $w_j$ before
  summing, so being late on an important job counts for more than being late on a trivial
  one.
- **Makespan** -- the total time to finish *all* jobs (i.e. when the last job completes).
  Different objective from tardiness -- minimising makespan doesn't necessarily minimise
  tardiness and vice versa.
- **EDF (Earliest Deadline First)** -- a classical dispatching rule: always schedule whichever
  waiting job has the closest deadline next.
- **SPT (Shortest Processing Time)** -- always schedule the shortest job next. Provably
  optimal for minimising *average completion time* (Smith 1956), but ignores deadlines and
  weights entirely -- which is exactly why the online case's RL policy collapsing onto SPT is
  a bad outcome despite SPT being a "real" strategy, not a random one.
- **LST (Least Slack Time)** -- schedule whichever job has the least slack (deadline minus
  remaining processing time minus now) -- i.e., whichever job is closest to "must start now or
  miss its deadline." Usually this project's strongest heuristic baseline.
- **WSPT (Weighted Shortest Processing Time)**, **FCFS (First Come First Served)**, **LPT
  (Longest Processing Time)** -- other classical rules, all used as baselines; the names
  describe exactly what they do.
- **ATC (Apparent Tardiness Cost)** -- a more sophisticated composite rule (see the formula
  and full walkthrough in the report's Online Case section / `mathformulation.tex`) that
  combines a job's value-per-time-unit with an urgency term that ramps up as its deadline
  approaches. Used both as a baseline and as an engineered input feature for Option 3.
- **First-Fit / Best-Fit / Worst-Fit** -- classical *placement* rules (which machine to put a
  job on, as opposed to dispatching rules which decide *which job* goes next): First-Fit picks
  the first machine with enough room; Best-Fit picks the machine that fits tightest; Worst-Fit
  picks the machine with the most room left.

### This project's specific designs

- **Hyper-heuristic** -- a heuristic whose job is to choose *between* other heuristics
  (rather than solve the problem directly itself). Split into two kinds (Burke et al. 2013):
  a **selection** hyper-heuristic picks among a fixed pool of existing heuristics; a
  **generation** hyper-heuristic invents new ones (e.g. via genetic programming). Option 1 is
  a selection hyper-heuristic -- this is the precise, honest way to describe the project's
  best result, and the important caveat is that a selection hyper-heuristic, by construction,
  can never do better than the best combination of its underlying rules allows.
- **Action-space reduction (Options 1-4)** -- the project's core methodological pivot.
  Instead of asking the RL agent to directly choose a (job, machine) pair out of ~1000
  combinations (too large for PPO to learn effectively), each option restructures what the
  agent's action *means*: Option 1 picks a dispatching rule (8 choices); Options 2/3 pick a
  job directly, with placement fixed to First-Fit (~100 choices); Option 4 restores a learned
  placement decision but as two smaller sequential choices instead of one large joint one
  (~100+10 choices, added rather than multiplied).
- **Action branching** -- the general technique Option 4 uses: instead of one action space
  that's the *product* of two decisions (e.g. 100 jobs times 10 machines = 1000), split it
  into two independent, smaller decisions made in the same step (100 + 10 = 110). Much
  cheaper, at the cost of the two decisions not being able to directly condition on each
  other within a single forward pass.

### Evaluation methodology

- **CP-SAT / exact solver** -- a constraint-programming solver (via Google OR-Tools) that can
  find a *provably optimal* solution for small/tractable problem instances, used as a ground-
  truth floor to measure everything else against. Doesn't scale to large instances because
  this scheduling problem is NP-hard (Lenstra, Rinnooy Kan & Brucker, 1977) -- meaning no
  known algorithm can solve arbitrarily large instances to proven optimality in reasonable
  time. Where CP-SAT numbers appear in this report, they're for the one specific
  computationally-tractable instance, not a general bound.
- **NP-hard** -- a formal complexity classification meaning (very roughly) that no known
  algorithm can solve every instance of the problem quickly (in time that scales
  "reasonably" as the problem grows) -- so exact methods only stay practical at small scale,
  which is exactly why approximate methods (heuristics, metaheuristics, RL) matter at all for
  problems like this one.
- **Held-out instance / 50-instance protocol** -- evaluating a trained policy on problem
  instances it never saw during training (different random seeds), to check whether it
  actually generalised or just memorised the one thing it was trained on. This project's
  standard protocol averages over 50 such instances.
- **Seeds / 3-seed verification** -- a "seed" controls the random number generator, so
  training the "same" model with 3 different seeds gives 3 independently-trained models. If
  their results agree closely, a finding is more trustworthy than a single run; if they
  disagree a lot (like RCPO's later 3-seed check), the original single-run result wasn't
  reliable. This is standard practice for taking any ML result seriously, not just something
  this project does for extra rigor.
- **mean $\pm$ std** -- results reported as "mean $\pm$ standard deviation" across multiple
  instances/seeds. A large std relative to the mean is itself informative -- it means the
  result is unreliable/high-variance, not just "the average was X."

### Online-case specific

- **Poisson process** -- the standard mathematical model for events (here, job arrivals)
  that happen independently at a constant average rate. Fully described by one number, the
  rate $\nu$. Chosen over alternatives because it's the standard, well-understood choice and
  introduces no artificial structure beyond that one rate.
- **Little's Law** -- a queueing-theory result, $L = \lambda W$: the average number of items
  in a system equals the arrival rate times the average time each item spends in the system.
  Used in this project to derive a safe arrival-rate range from the environment's machine
  capacity (see the report's Online Case section for the full derivation).
- **$\rho$ (rho) / utilisation** -- the fraction of total system capacity being used on
  average, $\rho = \nu / 12$ in this project's specific configuration. $\rho < 1$ means arrivals
  are (on average) within the system's capacity to handle them; $\rho \geq 1$ means the system
  is structurally overloaded no matter how good the scheduling is.

---

## 1. The current state of the project, in plain English

**Offline case.** Early on, PPO could not learn deadline-aware scheduling no matter what
was tried -- reward tuning, a constrained (Lagrangian) variant, a hyperparameter search, a
fancier network architecture. Seven different fixes, all landed in the same bad place
(tardiness ~1300, same as just ignoring deadlines). A2C did better with a shaping trick, but
only on the one instance it was trained on -- it fell apart on new random instances. A
constrained version of A2C (RCPO) briefly looked like the best result ever, until a closer
look showed it was cheating (abandoning half the jobs instead of scheduling them), and even
after that was fixed, it wasn't reliable across different random seeds.

The thing that actually worked: **shrinking the action space**. Instead of asking PPO to
directly pick "job X on machine Y" out of ~1000 combinations, "Option 1" asks it to pick
which of 7 already-existing scheduling rules (EDF, SPT, LST, etc.) to use *right now*. That
dropped the choice down to 8 options, and PPO went from useless to nearly optimal: 11.00
tardiness vs. a mathematically-proven best-possible of 8.00. That's the project's headline
result.

**The honest caveat, and why it matters for how you write this up**: Option 1 works by
learning *when to use which existing rule*, not by inventing new scheduling behaviour. If a
marker or your supervisors ask "so did the AI invent a better way to schedule jobs?", the
honest answer is no -- it learned to be a smart switchboard between rules humans already
know about. That's still a real, useful, citable result (it's a recognised category in the
literature -- a "selection hyper-heuristic"), just a more modest claim than "RL beats
humans at scheduling."

**Online case.** Jobs arriving live, not known in advance. RL beats most heuristics here
too under heavy load, but there's a weird, still-unsolved problem: training it for *longer*
makes it *worse*, every single time it's been tried (3 independent runs, 2 different
designs). It seems to collapse onto "just clear the queue as fast as possible" (the SPT
rule), which is a real strategy but ignores which jobs actually matter (their weights). Why
this happens isn't fully understood yet -- flagged as the biggest open question in the
report.

---

## 2. Answers to your own draft's open questions

**"How do I mention the reasons for why we choose to model such a case or scenario as
opposed to another way?"** -- You did this well already in Methodology's Problem
Formulation section (the job weight comes from "some outsider" -- I filled that in as
SLA-tier/business-priority, which is standard framing). For the offline-vs-online choice
specifically: offline is the tractable starting point that lets you validate the MDP
formulation and get a provable CP-SAT comparison; online is the more realistic version of
the actual cloud RA problem (jobs really do arrive live), attempted second once the offline
formulation was validated. That's the honest ordering and it's fine to say so plainly.

**Hotspot "???" questions** -- resolved in the Online Case section of
`mathformulation.tex` now (Section on Hotspot Redefinition). Short version: there are three
different reasons you *could* want a hotspot penalty (real-time contention risk, training
stability, long-run fragility), they're not the same thing, and the project hasn't picked
one definitively -- that's stated as an open design question rather than hidden. Separately,
a real bug was found and fixed in how utilisation was being computed (it was silently
understating it the whole project). And there's a genuine tension: the hotspot penalty as
currently written works *against* the standard definition of energy efficiency in the
literature (spreading load vs. consolidating it). All of this is now written up properly in
the supervisor document.

**Discussion -- "What should I talk about?"** -- I drafted full Discussion content in
`main.tex` already, but if you want to add your own voice/reflection, good angles to expand
on: (a) what surprised you most (probably: that shrinking the action space mattered more
than every reward/algorithm change combined), (b) what you'd do differently starting over
(probably: check action-space size against comparable literature earlier), (c) your own
take on the hyper-heuristic-vs-novel-scheduling distinction -- do you think it matters for
how "impressive" the result is, or is "beats every heuristic but one" enough on its own?

---

## 3. Citations I resolved for you

- Your draft cited `6d37e1a0bc084f698b5172146b3f68bc` (a Zotero/Mendeley-style hash key) for
  the "RL-based adaptive order dispatching... outperform... heuristic" quote next to the ATC
  discussion. I matched this to **`min2022atctuning`**, already in `references.bib` (Min &
  Kim 2022, IEEE Access, DRL-tuned ATC parameter) -- the description lines up closely. Worth
  a 30-second double-check against wherever you originally got that hash key, in case it was
  actually a different paper.
- `wang2005reinforcement` and `chen2013flexible` -- both found and added to `references.bib`
  with full details (Wang & Usher 2005, Engineering Applications of AI; Chen & Matis 2013,
  Int'l Journal of Production Economics). Both are legitimate, on-topic papers.
- `alkhalifa2025comparative` (used once in Background, heuristics-vs-metaheuristics
  classification) -- I could **not** verify this one via search. Added as a placeholder bib
  entry flagged "unverified" so it doesn't silently disappear, but you should find your
  original source for this and fix the entry before submitting -- it's currently a stub, not
  a real citation.
- The `hu2021attend2pack` / `DBLP:...` duplicate I flagged in an earlier session was already
  fixed on disk by the time I got back to it (good -- someone/something resolved it while I
  wasn't looking).

---

## 4. GenAI statement -- why I wrote it the way I did

I drafted a factual, specific statement (in the report itself) rather than a vague one,
because a generic "AI was used to assist" line undersells how much Claude was actually
involved (coding, literature search, diagnosing the RCPO reward-hacking finding, drafting
this report from your notes) and a marker who reads `training-log.md` would immediately
notice a mismatch. I'd rather you look slightly more AI-assisted and completely honest than
the reverse. Edit it if anything in there doesn't match how you'd describe your own
involvement -- it's your declaration, not mine.

---

## 5. Git / housekeeping note for when you're back

Your current branch (`autonomous-overnight-2026-08-28`) is **43 commits behind
`origin/main`** -- a lot of separate work has landed on main that this branch never picked
up. I pushed everything from this session (research log catch-up, report, online-case
section, this artefacts folder) to the *current* branch, as you asked, without trying to
merge or rebase onto main myself -- that's a real decision (could have conflicts) better
made by you than guessed at while you're about to be offline. Worth reconciling the two
branches when you're back, before doing much more work on either.

---

## 6. Report word count

Body text (Abstract through Future Work, excluding figures/tables/captions/references/the
Data Management Plan and GenAI statement, which the marking guide explicitly excludes) is
currently **~4,700 words**, inside the 3,000-5,000 range -- there's a little headroom if you
want to expand Discussion with your own reflections (see section 2 above), but not a lot, so
keep additions tight.

---

## 7. If you want to edit/extend a section yourself, here's what belongs where

Since you can't ask me "does this fit in Results or Discussion" while offline, here's the
dividing line each section is already following, so you can keep new writing consistent
with it:

- **Background** -- literature and general motivation only (why cloud RA matters, why RL is a
  reasonable thing to try). No project-specific numbers belong here.
- **Methodology** -- what you built and why, in general terms (the MDP, the algorithms, the
  Options 1-4 designs, the evaluation protocol). No *results* (numbers/outcomes) belong here
  -- if you're about to write a tardiness value, it belongs in Results instead.
- **Results** -- numbers and what they show, method by method, with minimal interpretation.
  If you catch yourself writing "this means..." or "this is important because...", that
  sentence probably belongs in Discussion instead.
- **Discussion** -- interpretation, relevance, and what you personally take away from the
  results. This is the section most worth adding your own voice to (see section 2's
  suggestions above) -- unlike Results, it's supposed to sound like you, not just report
  numbers.
- **Conclusion** -- a few sentences, no new information. If you're introducing something that
  hasn't appeared earlier in the report, it belongs earlier, not here.
- **Future Work** -- ideas for what's next, not things you actually did. If it's something you
  tried and have a result for, it belongs in Results, not here.

**Rule of thumb if you're unsure:** Methodology = what/how, Results = what happened,
Discussion = what it means.

---

## 8. LaTeX quick reference (for when you're stuck and can't search)

- **Add a citation:** `\cite{keyname}` -- the key must already exist in
  `Future/research/references.bib` (browse that file directly, it's plain text, to find a
  key or check one exists) or the bibliography step will silently skip it and the reference
  will show as `[?]` in the compiled PDF.
- **Add a new reference:** open `references.bib`, copy an existing entry's format (e.g.
  `@article{...}` with `author`, `title`, `journal`, `year` fields), give it a new unique key
  (this project's convention: `<firstauthor><year><shortslug>`, e.g. `smith2020example`),
  and cite it with `\cite{smith2020example}`.
- **Add a table row:** inside a `\begin{tabular}{...}...\end{tabular}` block, each row is
  `col1 & col2 & col3 \\` (double backslash ends the row, single `&` separates columns --
  the number of `&`s must match the number of columns declared in `{lccc}` etc. after
  `\begin{tabular}`).
- **Bold / italic:** `\textbf{bold text}`, `\textit{italic text}` or `\emph{emphasised text}`.
- **Common compile errors:**
  - *"Undefined control sequence"* -- usually a typo in a command name, or a missing
    `\usepackage{...}` for a command that needs one.
  - *"Missing $ inserted"* -- you used a math symbol (like `_`, `^`, or a Greek letter name)
    outside of `$...$` (inline math) or `\[...\]` (display math). Wrap it in `$...$`.
  - *"Runaway argument"* / *"Paragraph ended before ... was complete"* -- almost always a
    missing closing `}` somewhere above the error line -- count braces working backwards from
    where LaTeX complains.
  - Citations showing as `[?]` in the PDF -- the bibliography needs a second compile pass
    after the first one generates the `.bbl` file; in Overleaf this happens automatically,
    but if compiling locally you may need to run the bibliography step (`bibtex`) then
    compile again.
- **If something won't compile and you can't figure out why:** comment out the specific
  paragraph/section you just edited (wrap it in `\begin{comment}...\end{comment}` if the
  `comment` package is available, or just delete it temporarily) to confirm the rest of the
  document still compiles, then add your edit back in smaller pieces to isolate exactly which
  line broke it.

---

## 9. Anything else you might want to ask -- an extensive knowledge dump

Everything below is the kind of thing you'd normally just ask me mid-conversation. Organised
so you can jump straight to the category you need.

### 9.1 Design-decision FAQ ("why did we do it this way, not that way")

- **Why PPO and A2C, not DQN/SAC/TD3?** The action space is discrete (choose a job/machine
  pair, or later a rule/priority score), which immediately rules out SAC and TD3 (both
  require continuous actions). DQN works with discrete actions but is a value-based method,
  generally considered less stable/scalable than policy-gradient actor-critic methods for
  problems with structure like this one (see the glossary's "value-based vs policy-based"
  distinction). PPO and A2C were chosen for being well-established, well-supported by
  available libraries, and directly comparable to each other as two different actor-critic
  training styles (clipped-trust-region vs. synchronous-parallel).
- **Why these seven heuristics as baselines (EDF/SPT/LST/FCFS/LPT/WSPT/ATC)?** They're the
  standard, literature-grounded dispatching rules for tardiness-focused scheduling (see the
  glossary's Scheduling Terms section for what each does) -- using well-known named rules
  rather than inventing ad hoc baselines makes every comparison citable and reproducible by
  someone else.
- **Why 100 jobs / 10 machines / horizon 100 as the "deployed" scale?** This is the scale
  large enough to be a meaningful test (not trivially small) while still keeping training
  times and CP-SAT solve times practical on this project's hardware (CPU-only, 16 cores, no
  GPU). Smaller "curriculum" stages (15/30/60 jobs) were used during training to help the
  agent learn incrementally before being evaluated at full scale.
- **Why both a fixed instance AND a randomized-instance distribution?** A fixed instance lets
  you get a clean CP-SAT proven-optimal comparison (CP-SAT only stays tractable at small/
  single-instance scale) and isolates "did the agent learn scheduling skill at all" from "did
  it also generalise." The randomized-instance results then separately test generalisation --
  a model can do very well on the one instance it trained on and fall apart on a new one (this
  is exactly what happened to the early A2C result), so testing both is what actually catches
  that.
- **Why $\gamma = 1$ (undiscounted return)?** The underlying optimisation objective
  (Equation 2 in the report) sums costs uniformly across the whole planning horizon with no
  built-in preference for early vs. late improvement -- so there's no principled reason to
  discount future reward relative to immediate reward here. Discounting is mainly useful for
  infinite-horizon problems (to keep the sum finite) or when you genuinely care more about
  near-term outcomes, neither of which applies to this fixed-horizon offline case.
- **Why a weighted-sum objective ($\lambda_1, \lambda_2, \lambda_3$), not a lexicographic
  one?** A lexicographic objective (fully solve for machine count first, only then break ties
  on tardiness, etc.) would mean the agent never trades off a small increase in one objective
  for a large improvement in another -- unrealistic for a real operator, who genuinely does
  care about relative magnitudes, not just priority order. The mathematical formulation
  (`mathformulation.tex`, Section 1) documents the lexicographic alternative too, for
  completeness, but the weighted-sum version is what was actually implemented and trained
  against.
- **Why action masking instead of just penalising invalid actions?** Penalising after the
  fact still wastes a training step sampling an action that could never have been valid --
  masking removes invalid actions from the probability distribution entirely, so every
  sampled action is at least feasible, which is strictly more sample-efficient. It's also
  directly supported by this project's RL library (Stable-Baselines3's `MaskablePPO`), so
  there was no real reason not to use it.
- **Why "wrap the environment, don't modify it" for every Option (1-4)?** Each Option changes
  what the *action* means, not the underlying scheduling mechanics (feasibility, capacity,
  reward computation). Implementing each as a Gym wrapper around the same unmodified
  environment means the offline and online cases, and every heuristic baseline, stay
  automatically correct and comparable for every Option, instead of needing the environment's
  core logic re-verified separately for each one.
- **Why CP-SAT / OR-Tools specifically as the exact solver?** It's a well-supported,
  actively-maintained constraint-programming solver with native support for interval
  variables and cumulative-resource constraints -- exactly the primitives a resource-
  constrained scheduling problem needs -- and it's free/open-source, so anyone can reproduce
  the exact-solver comparisons without a commercial license.
- **Why 50 held-out instances specifically?** A compromise between statistical stability
  (more instances = a more reliable average and a meaningful standard deviation) and
  practical evaluation time (each instance requires running every method being compared, and
  this project compares against many heuristics plus the trained agent every time). 50 was
  found to give a stable enough mean/std to be trustworthy without evaluation becoming the
  bottleneck.
- **Why offline before online?** The offline case is the tractable starting point: it lets
  the MDP formulation, the reward function, and the evaluation protocol all be validated
  against a provable CP-SAT floor before adding the extra complexity of dynamic arrivals,
  an unbounded-feeling horizon, and a stochastic arrival process on top. Online was always the
  more realistic target (real cloud jobs do arrive live), attempted second by design, not
  because it was an afterthought.
- **Why not a graph neural network (GNN) or multi-agent approach from the start?** Both are
  named as Future Work, not oversights -- they're reasonable next steps once the current
  action-space-reduction findings are well-established, but neither was necessary to test the
  project's core early hypotheses (does RL beat heuristics at all; is action-space size the
  bottleneck), and both add real implementation/tuning complexity that would have made
  diagnosing the actual bottleneck (action-space size) harder to isolate cleanly.

### 9.2 Codebase map -- what's in each folder

(As of this session; a few of these may have grown further by the time you're reading this,
since research kept going in the background even while this report was being written.)

- **`Code/env/`** -- the environment itself. `scheduling_env.py` is the core offline
  environment (state/action/reward/transition mechanics); `online_scheduling_env.py` and
  `arrival_process.py` add dynamic arrivals for the online case. `gym_scheduling_wrapper.py`
  / `online_gym_wrapper.py` adapt these to the standard Gym interface RL libraries expect.
  `rule_selection_gym_wrapper.py` (Option 1), `priority_only_gym_wrapper.py` (Options 2/3),
  `windowed_priority_gym_wrapper.py` (Option 3's windowed variant), and
  `action_branching_gym_wrapper.py` (Option 4) are the action-space-reduction wrappers --
  each one changes what an "action" means without touching the underlying environment.
  `env_config.py` generates job/machine configurations (including the `seed` parameter that
  controls instance randomization).
- **`Code/policies/`** -- the neural network architectures. `a2c_policy.py` is the hand-rolled
  masked A2C implementation (with RCPO support). `pointer_policy.py` /
  `priority_pointer_policy.py` / `windowed_priority_pointer_policy.py` are pointer-network-
  style architectures (shared job/machine encoders) for different action-space designs, each
  with a matching `*_ppo_policy.py` file wiring it into Stable-Baselines3's `MaskablePPO`.
  `action_branching_policy.py` / `action_branching_ppo_policy.py` implement Option 4.
  `ppo_lagrangian.py` extends the Lagrangian-constrained approach (RCPO's mechanism) to PPO.
- **`Code/training/`** -- entry points for actually training a model. `train_optimized.py` is
  the main offline/curriculum trainer (PPO via Stable-Baselines3, with Optuna hyperparameter
  search support). `train_action_space_variant.py` is the separate, simpler trainer used for
  Options 1-4 (deliberately not wired into the curriculum/Optuna machinery, for fast first-
  pass comparisons). `train_a2c.py` trains the hand-rolled A2C. `train_rl_agent.py` is an
  older, simpler variant kept for reference/comparison, not the current main path.
- **`Code/baselines/`** -- classical, non-learned methods. `priority_rules.py` (which job
  next: EDF/SPT/LST/etc., including the ATC formula) and `placement_rules.py` (which
  machine: First-Fit/Best-Fit/Worst-Fit) are combined via `registry.py` into named
  heuristics like `"EDF+FirstFit"`. `exact_solver.py` is the CP-SAT exact solver. `pso.py`
  is a particle-swarm-optimisation metaheuristic baseline.
- **`Code/evaluation/`** -- scripts that load a trained model and measure its performance.
  `eval_rl_agent.py` is the main offline evaluator (implements the 50-held-out-instance
  protocol). `eval_action_space_variant.py` is the equivalent for Options 1-4.
  `diagnose_policy_confidence.py` and `diagnose_rule_choice_congestion.py` are targeted, one-
  off diagnostic scripts built to investigate the online "more training hurts" mystery
  specifically -- not part of the standard evaluation pipeline.
- **`Code/utils/`** -- shared infrastructure. `results_log.py` auto-appends every evaluation
  run's results to `rl_training/results/eval_results.csv` (the source of this project's
  results CSV artefact). `training_diagnostics.py` provides the TensorBoard logging
  callbacks used to investigate training dynamics (entropy, per-rule action frequency, etc.).
  `paths.py` centralises where output files go. `plotting_utils.py` generates the matplotlib
  plots the evaluation scripts produce.
- **`Future/research/`** -- the project's documentation, not code. `training-log.md` is the
  master chronological experiment log (append-only, newest entries first -- the single most
  authoritative source for "what actually happened and when"). Dated files
  (`YYYY-MM-DD-<topic>.md`) are deep-dive write-ups on one specific investigation.
  `references.bib` is the shared bibliography.

### 9.3 Every major experiment thread, one-line summaries

A cheat-sheet of everything tried on the offline case, roughly in chronological order, so you
don't have to reconstruct this from `training-log.md`'s hundreds of entries:

| Thread | What it was | Outcome |
|---|---|---|
| PPO, raw action space, reward tuning | Adjust $\lambda_2$ (tardiness weight) | Failed -- stuck at ~1300 tardiness regardless of weight |
| PPO-Lagrangian | Constrained variant, adaptive tardiness weight | Collapsed to zero jobs scheduled (multiplier ran away) |
| PPO, tardiness-tuned hyperparameter search | Optuna search targeting tardiness directly | Helped at small tuning scale, didn't transfer to full scale |
| PPO, pointer-network architecture | Shared job/machine encoders instead of per-index weights | No better than flat MLP -- ruled out architecture as the cause |
| PPO, dense per-tick reward redesign | Reward every tick instead of only on completion | Better-behaved training dynamics, but no better final tardiness |
| A2C, potential-based shaping | Provably policy-invariant reward shaping | Best early-project result on the fixed instance (27-31 tardiness) |
| A2C, randomized-instance training | Train directly on shuffled instances | Much worse than fixed-instance training + held-out eval |
| RCPO (constrained A2C) | Adaptive Lagrangian tardiness constraint | "Best-ever" result was reward hacking (job abandonment); fix didn't survive 3 seeds |
| **Option 1** -- rule selection | Pick which of 7 heuristics to apply | **Best result overall: 11.00 tardiness vs. proven optimum 8.00** |
| Option 2 -- raw-feature priority scoring | Pick a job directly from raw features | Works, but far behind Option 1 (525.00 at first-pass scale) |
| Option 3 -- ATC-primed priority scoring | Like Option 2, plus the ATC score as a feature | Better than Option 2, still behind Option 1 (152.00) |
| Option 4 -- action-branching | Restore learned machine placement | Worse than Options 1/3 at matched budget; unresolved training instability |

Online case (separate thread, all built on the Options 1-4 action-space designs):

| Thread | Outcome |
|---|---|
| Option 1, online, 300k timesteps | Best online result found so far (weighted tardiness 730.30, genuinely mixed rule behaviour) |
| Option 1, online, 900k timesteps (x3 independent runs) | Consistently worse (~790-800), collapses toward SPT -- unresolved |
| Option 3, online, 300k vs 900k | Same "more training hurts" pattern, independent confirmation with a different design |
| Entropy regularisation as a fix | Tested directly, ruled out -- entropy stayed healthy but the collapse still happened |

### 9.4 The reward function, fully explained

The per-step reward (Equation in `mathformulation.tex`, Section on the MDP reward function)
has four parts:
1. **Machine activation penalty** ($-\lambda_1$ per newly-activated machine) -- discourages
   spinning up more machines than necessary.
2. **Tardiness penalty** ($-\lambda_2 \sum w_j T_j$ for jobs completing this step) -- the main
   term, penalising weighted lateness.
3. **Hotspot penalty** ($-\lambda_3 \Delta\theta$, the increase in peak utilisation) --
   discourages concentrating load. **Known issue:** this term had a real bug (used a mutable
   capacity reference instead of a fixed baseline) that silently understated utilisation for
   this project's entire history until it was found and fixed in the last week of work --
   see the report's Discussion and the Online Case section's Hotspot Redefinition
   subsection. **Known open tension:** this term structurally discourages the kind of load
   consolidation that the energy-efficiency literature says actually saves power -- i.e. the
   hotspot term and a "proper" energy-efficiency term would currently pull in opposite
   directions if both existed. Not resolved, named as future work.
4. **Invalid-action penalty** ($-\lambda_4$, or handled via masking so this rarely fires in
   practice) -- discourages/prevents infeasible placements.
Plus a terminal penalty at the end of an episode for any job still unscheduled. All
$\lambda$s were fixed by convention (not tuned via HPO) for most of this project's history,
since HPO on the relative weighting between objectives was explicitly deferred as a decision
needing more design input, not something to search blindly.

### 9.5 Questions a marker or supervisor might realistically ask, and how to answer them

These come from an independent, critical self-review the project generated
(`report.md` in the repo root, not part of the report itself -- worth reading in full if you
want the unfiltered version).

- **"Is this really 'RL learned to schedule,' or did it just learn to pick a heuristic?"** --
  Be direct: Option 1, the best result, is a *selection hyper-heuristic* -- it picks among 7
  existing rules, it does not invent new scheduling behaviour. That's explicitly stated in
  the report's Results and Discussion. Options 2/3 are the designs that *could* demonstrate
  genuinely novel learned behaviour, and they're currently weaker -- say this plainly if
  asked, don't let it be "discovered" by the marker instead.
- **"How do you know your 50-instance result isn't just luck?"** -- Point to the 3-seed
  verification practice used for the project's major claims (and the RCPO example where a
  3-seed check specifically *overturned* an earlier single-run result) -- this project has
  direct, in-report evidence that it takes this risk seriously, not just a claim that it does.
- **"Why didn't you finish the online case?"** -- Because it surfaced a genuine, reproducible,
  currently-unexplained instability (more training makes it worse) rather than because it
  was deprioritised or ignored. This is presented as an honest open research finding in the
  report, not a gap glossed over -- say so if asked.
- **"How do you know PPO actually failed, and it's not just a bug in your implementation?"**
  -- Two things to point to: (1) the same underlying environment, action-masking, and reward
  computation is shared by A2C, which *did* learn well under the right configuration --
  ruling out a broadly broken environment; (2) CP-SAT independently confirmed the fixed
  instance's true optimum (8.00) and that all 100 jobs *could* be scheduled on time, ruling
  out "the instance itself was impossible" as an excuse for PPO's ~1300 result.
- **"What's your test coverage / how do you know the code is correct?"** -- Honest answer:
  regression tests exist for specific historical bugs (run as plain scripts, e.g. `python -m
  tests.test_bugfixes`, not a formal `pytest` suite), and this project has caught and fixed
  roughly ten real bugs over its history purely through this kind of after-the-fact,
  hypothesis-driven debugging (not a proactive CI/test-suite process) -- named as a real
  structural gap in Future Work if you want to volunteer it, rather than claim more rigor
  here than actually exists.
- **"Did you tune your hyperparameters properly?"** -- Mostly no, and say so: Options 1-4
  (the project's best results) are currently running on un-tuned library defaults, not a
  searched configuration -- named explicitly as next-step work. Where HPO *was* done
  (earlier PPO/A2C variants), it sometimes didn't transfer from the scale it was tuned at to
  full deployment scale (a documented, citable phenomenon -- Eimer et al. 2023), which is
  itself a real finding, not just an excuse.

### 9.6 How the pipeline works end-to-end, if you need to explain or re-run it

1. **Configure**: `Code/env/env_config.py::generate_env_config(seed=...)` produces a job/
   machine instance (or, online, `arrival_process.py` produces a Poisson arrival stream).
2. **Train**: `Code/training/train_optimized.py` (or `train_action_space_variant.py` for
   Options 1-4) builds the (possibly wrapped) Gym environment, attaches a policy from
   `Code/policies/`, and runs Stable-Baselines3's training loop, optionally logging
   diagnostics via `Code/utils/training_diagnostics.py`. Produces a saved model checkpoint
   under `rl_training/models/` (gitignored -- these are regenerable, not committed).
3. **Evaluate**: `Code/evaluation/eval_rl_agent.py` (or `eval_action_space_variant.py`) loads
   a checkpoint, runs it against the 50-held-out-instance (or 50-arrival-sequence) protocol,
   and compares it against every heuristic in `Code/baselines/registry.py` and, where
   tractable, `exact_solver.py`.
4. **Log**: every evaluation run auto-appends its results to
   `rl_training/results/eval_results.csv` via `Code/utils/results_log.py`, and (by
   convention, not automatically) a corresponding dated entry is added to
   `Future/research/training-log.md` describing the run, its config, and what was concluded.
5. **Plot**: `Code/utils/plotting_utils.py` generates the matplotlib figures used in earlier
   evaluation runs (training curves, per-heuristic comparison bars, machine-utilisation
   plots) -- note Options 1-4's own trainer/evaluator do *not* currently call this, which is
   why this session had to generate this report's figures from the CSV data directly instead
   of reusing existing plot files.
