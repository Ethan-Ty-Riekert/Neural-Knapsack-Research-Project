# Plane notes -- read offline, no internet needed

Companion to `report/main.tex`. This is scratch/reference material for *you*, not part of
the report -- answers to the specific questions in your draft, a plain-prose summary of
where the project actually stands (so you don't have to re-parse `training-log.md` cold),
and a few things worth knowing before you touch git again.

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
