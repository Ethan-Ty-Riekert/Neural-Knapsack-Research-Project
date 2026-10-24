# Presentation plan: skeletons, scripts, Q&A, logistics (2026-10-23, S2W14)

**Format (from the user):** end-of-year open room, ~1 hour, visitors walk between stalls, mirrored from the laptop, "3 slide thing" plus an
interactive Streamlit explorer (`app/app.py`). Presentation ~2026-10-28. Nobody sits through a whole talk: people arrive at random times,
stay 1-5 minutes, and decide in about 10 seconds whether to stop. So the design goal is **a screen that explains itself, a hook you can say
in 30 seconds, and a demo a stranger can play with**.

All wording follows the findings doc ("For the presentation" section of `2026-10-21-final-week-findings.md`): "no significant difference",
never "RL beats/matches the heuristics"; label OLD test set vs FRESH set; results are exploratory.

## 1. The one-sentence story (pick this first)

> We trained neural networks to schedule cloud jobs and found that, honestly, **they learn to copy good rules of thumb but did not beat the
> best one** - and we built a tool where you choose what you care about and see which method wins.

This is a defensible, memorable negative-plus-useful result. It is stronger in a room of academics than an overclaimed "AI wins".

## 2. Three skeletons (choose one; recommendation: B)

### A. Question-driven (closest to the existing 4 slides)
1. **Hook:** "Can reinforcement learning beat the best scheduling heuristic?" (`01_headline.png`)
2. **Evidence:** fresh-instance bar chart at load 0.95 (`02_fresh_on_rho095.png`)
3. **Across scenarios:** clones land within 0-8% of their heuristic; PPO fine-tuning adds nothing (`03_scenarios.png`)
4. **Caveats:** what this does and does not show (`04_caveats.png`)
5. **Demo tab:** "pick what you care about"
Best for: people who stay 3-5 minutes. Risk: reads as a poster of caveats.

### B. Demo-first, slides as backup (recommended)
Screen is the explorer, already on a sensible default (on_rho095, quick objective). Slides are the second tab.
1. **Landing view:** explorer with a one-line banner: "Pick what matters to you - see which scheduler wins." Visitor plays immediately.
2. **Tab "Slides":** the same four slides, for the 'tell me the research story' visitor.
3. **Three scripted 'try this' prompts** printed on a card or shown at top of the explorer (section 4).
Best for: a roaming crowd; interaction hooks people faster than text. Risk: you must keep the default state clean after each visitor.

### C. One-screen poster
Single slide: left = problem (jobs, machines, tardiness cost J), centre = fresh-instance chart, right = three bullets + QR/repo. Explorer
on a second window. Best for: very short stops; weakest for depth.

**Recommendation:** B, with slides 01-04 as the second tab and the 30-second hook (section 3) as your opener. If a mentor prefers a
conventional talk slot, use A in that slot.

## 3. Scripts

### 3a. 30-second hook (stall opener, for people who slow down)
"This is about scheduling jobs on cloud machines so fewer of them finish late. We trained neural networks with reinforcement learning to do it
and compared them with classic rules like earliest-deadline-first. The honest result: on 50 fresh test instances no learned policy was
significantly better than the best rule. What we *did* build is this tool - tell it what you care about, like lateness, energy or number of
machines, and it ranks the methods for that goal. Want to try it?"

### 3b. Two-minute version (people who stay)
1. **Problem (20 s):** "Jobs arrive over time with deadlines and costs. We assign them to machines. The score J is the weighted squared
   lateness - lower is better."
2. **What we tried (30 s):** "Classic rules (EDF, MDC, COVERT, ATC...) with different placement strategies; neural policies trained by PPO from
   scratch; and policies that first copy a rule (behaviour cloning) and are then fine-tuned."
3. **Result (40 s):** "At load 0.95 the best rule is MDC with consolidation. No learned policy was significantly better. Cloned policies
   mostly land within a few percent of the rule they copied; PPO fine-tuning didn't help; PPO from scratch varied widely between seeds."
4. **So what (30 s):** "Which method is 'best' depends on the metric. That is why we built the explorer: change the weights and the winner changes."

### 3c. Five-minute deep dive (supervisors / technical visitors)
Add, in this order: (i) the old-vs-fresh test-set point ("the best heuristic was picked on the old set, so it looks better there; we
re-checked on 50 fresh instances"); (ii) the exact clone idea and why it is a baseline sanity check, not an improvement; (iii) the
windowed policy (20 earliest-deadline schedulable jobs) and the compact encoding (~6.6x faster, identical outputs, test-verified);
(iv) clairvoyant headroom (3 instances, up to ~-70%, not an online policy); (v) open question: windowed from scratch trains worse than
all-slot and bigger windows are worse - unexplained, untested; (vi) future work: v3 multi-metric objective, arrival forecasting is *untested*.

### 3d. Demo walkthrough ("try this" prompts)
1. **"Minimise lateness"** at on_rho095: rule MDC+Consolidate is best; learned clones sit next to the rule they copy.
2. **"Save energy / fewer machines"** (change weights): the ranking reshuffles; point at the trade-off scatter ("lower-left is better").
3. **Switch scenario to on_rho110 (overload) or off_tf05 (offline):** ranking differs again; "scores are only comparable within one scenario".
4. Show the **median-RL-vs-best-heuristic banner**: "a typical learned model is X% from the best rule; the best-of-many pick looks better than it
   is."
Say out loud: "Data here is the old test set; the fresh-set numbers are on slides 2-3."

## 4. Q&A prep (honest answers; check numbers against the findings doc before the day)

| Question | Answer |
|---|---|
| "Papers say RL beats heuristics. Why not you?" | Literature is mixed; many wins are against weak or mismatched heuristics. Ours were scored against 5-7 tuned rules with several placements, on fresh instances. Not claiming RL cannot win, only that we did not find it here. |
| "Did you tune PPO enough?" | Fine-tuning lr (1e-5) was chosen on validation instances; from-scratch arms used defaults and were not tuned. That is a stated limitation. |
| "How many seeds?" | 3-5 for most arms, 10 for the windowed headline arms at on_rho095. Seed variance is in no confidence interval. |
| "Isn't cloning cheating?" | It is a control: it shows the network *can* represent the rule. It is not claimed as an RL win. |
| "Why windowed?" | Only ~10 of ~1,235 job slots are schedulable; windowed is ~40x faster; the all-slot model was retired by decision on 2026-10-21. |
| "Is it significant?" | 'Beats' = paired interval entirely below 0 with 5 seeds/arm; nothing met it on fresh instances. 3-seed verdicts are indicative. Exploratory, no multiplicity correction. |
| "What would you do next?" | Explain the from-scratch gap (untested hypotheses), v3 weighted multi-metric objective, and test whether a forecast of arrivals captures the oracle headroom. |
| "Does the oracle headroom show RL could win?" | Not by itself: 3 instances, clairvoyant, not an online policy. |

## 5. Stall logistics checklist
- [ ] Laptop: `git pull; pip install streamlit==1.66.0 pandas matplotlib numpy; streamlit run app/app.py` - test on the laptop **before** 2026-10-27.
- [ ] Offline fallback: slide PNGs in `app/slides/` + a screen recording of the explorer (in case wifi/Streamlit fails).
- [ ] Browser zoom/fullscreen so text is readable from 2 m; hide bookmarks bar; dark mode off for projector.
- [ ] Default state: scenario `on_rho095`, quick objective = lateness. Reload the page between long visitors.
- [ ] A printed A4 card: one-sentence story, QR to repo, three "try this" prompts.
- [ ] Power adaptor, mouse, water; laptop sleep disabled (`keep_awake.py` on this desktop does not cover the laptop).
- [ ] Rehearse 3a three times aloud, 3b once with a timer; ask one person to play with the explorer without help and note where they get stuck.

## 6. Calendar to the presentation
- **10-23 to 10-25:** coverage-grid clones finish (111 jobs), archive rebuild, check explorer labels; choose skeleton; draft slides edits.
- **10-26 (evening, freeze):** `make_slides.py`, `build_folder.py`, explorer check, final log entry "Final state before the presentation".
- **10-27:** laptop dry run + rehearsal. No new training.
- **10-28:** present.

## 7. Open decisions for the user
1. Skeleton A, B or C (recommended B).
2. Do you want a "where RL wins and where it doesn't" slide (RL vs textbook rules / mismatched heuristics, e.g. offline composite wins)? It answers the most likely audience question.
3. Slide design: keep the plain matplotlib look or restyle for the projector (larger fonts, less text)?
4. Will you present alone or with a partner at the stall (changes the script split)?

## 8. Decisions (2026-10-24)
User chose skeletons **A + B** (explorer is the landing tab, the four slides are the second tab, in A's order) and scripts **3b** (two-minute
pitch) and **3d** (demo walkthrough). Implemented in `app/app.py`: tab order Explore then Slides, a headline "Pick what matters to you - see
which scheduler wins", and three one-click "Try this" buttons matching 3d (lateness at load 0.95, energy at 0.95, lateness under overload
1.10). Checked with Streamlit's AppTest (buttons switch scenario and weights without error). Still open: items 2-4 of section 7.

**Revision 2026-10-25 (user sketch):** the "Try this" buttons were replaced by a simpler race UI: (1) choose the environment (online
load light/moderate/heavy/overload, tight deadlines at moderate load, or offline deadline factor), (2) choose what matters (default examples
Energy / Tardiness / Tardiness + late jobs + energy, or own weights in an expander), (3) press **Start the race**: the top 2 rules and top 2
learned models (different models) fill horizontal bars to their percentage on an absolute scale (100% = best score of the scenario among all
rules and learned models; percent = best/score, min-max when scores are not all positive). Winner line, a noise note when the best
learned model is within 5% of the best rule, full ranking in an expander. Logic: `race_entries` in `app/scoring.py` (tested). Demo script 3d
now reads: pick Tardiness at Heavy -> Start; press Energy -> Start; switch to Overload -> Start.

**Scale (2026-10-25):** bars are non-linear: percent = 100*exp(-4*relative gap to the best), so 5% worse = ~82%, 25% worse = ~37%, 2x worse = ~2%; a display scale chosen so near-ties are visible, not a statistical statement (`RACE_STEEPNESS` in `scoring.py`).

## 9. Assessment fit (Industry Showcase, 5 criteria x 20%; added 2026-10-25)

The unit brief: a digital poster (a few PowerPoint slides, interactive welcome), a brief synopsis, attendance at the showcase (marks are given
only there), audience = academic staff, industry representatives and peers from other science disciplines; reference to professional and
technical skill development across the year; relevance to other fields. Deck revised accordingly (6 slides, plain language, jargon only in footers).

| Criterion (20% each) | How the deck / stall covers it | Still yours to do |
|---|---|---|
| Layout and visual appeal | Consistent navy/blue theme, cards, one chart per slide, big stat callouts, little text; interactive race tab | Check fonts on the laptop; keep slide text short |
| Synopsis and results | Slide 3 (what we did), slide 4 (result with plain-language chart), synopsis below, limits stated | Rehearse reading the chart in 15 s |
| Oral overview | Script 3b (two minutes) with the plain-language opener; pace, eye contact, ask the visitor a question ("what would you weight?") | Practice aloud; respond to comments rather than reciting |
| Personal and professional value | Slide 6 "Skills developed" | **Add your own career aspiration** (placeholder on slide 6 and in the line below) and one personal example (e.g. a mistake you corrected) |
| Diverse audience | Slide 2 (other fields, by analogy; only the cloud case was tested), jargon-free slides, two talk tracks below | Prepare one-sentence answers for non-technical and expert questions |

**Two talk tracks.** Non-technical (manager/investor): "Work arrives with deadlines; something must decide what runs first. Simple rules do
this well. We asked whether a self-taught AI scheduler could do better, and found it learned to match good rules but did not beat the best one;
here is a tool to see which wins for your priorities." Technical: add PPO vs behaviour cloning, fresh vs old test sets, paired intervals over
instances, 10 seeds, the unexplained from-scratch gap.

**Synopsis (assessment part b; ~150 words, edit freely).** Many real systems must decide which job runs next and on which machine, from cloud
data centres to hospitals and logistics. We studied this scheduling problem for cloud computing, where lateness and energy use trade off. We
compared classic rules (e.g. earliest-deadline-first) with neural-network schedulers trained by reinforcement learning, including ones first
taught by copying a rule. On 50 fresh test cases no learned scheduler was significantly better than the best classic rule; copies of a rule
performed close to it, extra training did not help, and learning from scratch varied widely between runs. The results are exploratory (few
training runs per setting). We built an interactive tool where a visitor picks the load, deadlines and what matters to them, and sees the best
rules race the best learned models. Skills developed: reinforcement learning, careful experiment design, and honest reporting of negative results.

**Revision 2026-10-25 (plain-language app):** the user found the race UI too sparse and technical. The app now asks: how do jobs reach you,
how busy are the machines, tight deadlines (normal busyness only), offline deadline tightness, equal vs unequal job importance (offline), and
five importance sliders (late jobs, long delays, waiting, finishing quickly, energy) with four examples. Racers get plain names ("Modified due
date (MDC)", "AI scheduler (copied)"); an expander states what was and was not tested (machines count, job sizes and hard waiting/lateness
limits were NOT tested; late jobs are scored, never refused). Settings without a trained AI scheduler (offline relaxed) race four rules.

**Revision 2026-10-25 (AI kinds, corrected):** the race shows 2 classic rules + 1 AI scheduler that copied a rule + 2 different AI schedulers that did NOT copy (trial and error); "copied then trained" is not raced. Missing non-copying models at on_rho050, on_rho075_tight, off_tf08_w1, off_tf02(_w1) are being trained (log 2026-10-25). `race_entries(ai_mix=AI_MIX)` in `app/scoring.py`.
