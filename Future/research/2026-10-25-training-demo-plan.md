# Plan: interactive "watch it learn" demo (reward loop + training), 2026-10-25 (S2W14)

**Status 2026-10-25: BUILT (v1). Learners changed from the plan: the actor-critic uses generalised advantage estimation (lambda 0.9) with a 7-feature critic because a one-step critic did not learn; both learners use actor rate 0.01 and a 600-episode budget; see tests/test_training_core.js for the acceptance results (10/10 seeds within 10% of the best rule, 9-10 of 10 with a clearly negative time-left weight).**

## Decisions so far (user)
- Main message: **reward drives learning** (the state, action, reward loop; the cost falls as the learner improves).
- Jobs: **all known up front** (offline toy).
- Learner: **REINFORCE** (simplified) and **actor-critic**, with a switch to swap between them and compare.
- Controls: **Step, Train, Test** (plus Reset and a speed button). No energy slider.
- Wording: all text in one wording file the user can rewrite (same approach as `app/web/wording.md`).

## The toy world (honestly labelled as a toy)
- 2 machines, 8 jobs, all waiting at time 0. Each job: length p (1 to 5 ticks), weight w (1 to 3), deadline d (set so some jobs will be late whatever is chosen).
- A decision is made whenever a machine becomes free: choose one waiting job (no idling in the toy).
- Reward at each decision: minus the weighted squared lateness of the chosen job, -w (max(0, C - d))^2 / c, with C = start + p known at selection. The episode return is minus the cost J, the same objective as the research (J = sum of w T^2), in miniature.
- A new random set of jobs every training episode; Test uses 200 fresh sets the learner never trained on (mirrors the fresh-instance protocol).

## Learners (both tiny, written in plain JavaScript, no libraries)
- Policy: score of waiting job j = theta . (time left, length, importance) (3 weights), probabilities by softmax over waiting jobs.
- **REINFORCE:** after each episode, theta moves along (return - running-average baseline) x gradient of log-probability of the actions taken.
- **Actor-critic:** the same policy (actor) plus a small value estimate V(state) (critic, linear in a few summary features); updates every decision using the advantage r + V(next) - V(now).
- Reference lines on the learning curve: a random scheduler (average) and the best classic rule (earliest deadline first / least time left), computed on the same fresh sets.

## The screen (one page, white, soft-blue borders, like the race page)
- Left: the world: waiting queue chips (length, deadline, weight), two machine lanes that fill as the clock runs; late jobs orange.
- Bottom strip: the loop State -> Action -> Reward -> Next state with the live values of the current decision.
- Right: learning curve (cost per episode, with the two reference lines; the second learner's curve overlays in another colour when swapped) and the three learned weights as bars.
- Buttons: Step (one decision), Train (about 20 seconds, with a speed button), Test, Reset, Learner: REINFORCE | Actor-critic.
- Click any label for a card (state, action, reward, policy, REINFORCE, actor-critic, learning curve, the cost J), same card system as the race page.

## Build order
1. Split the card engine out of `explain.js` into a generic file shared by both pages (text parsing, slide-over card, related links, Back); formulas stay per page.
2. Toy world and the two learners in `training_core.js`, tested in Node before any UI: pass criteria below.
3. Page template, `training_wording.md`, cards, charts (drawn on canvas, no libraries).
4. Build script, tests, publish, link from slide 2 (`training_demo.html` next to the deck).

## Acceptance tests (run before anything is claimed on the page)
- For each learner, over 10 random seeds, after the training budget the mean cost on 200 fresh test sets must be clearly below the random scheduler and within about 10% of the best rule in at least 8 of 10 seeds. If not, tune the learning rates or budget; if it still fails, the wording must not say it matches the rule.
- Learned weights: the page may say "it rediscovered least-time-left-first" only if the time-left weight is the most negative (smaller time left = higher score) in most seeds. Otherwise the wording says only what is measured.
- Training must finish in 15 to 25 seconds on a laptop-class machine, and the page must fit one screen at 1280x720, 1366x768 and 1920x1080.

## Risks
Plain REINFORCE has high variance and may need several hundred episodes and a tuned baseline; a toy that fails to learn would undermine the message, hence the acceptance tests. The toy ignores multi-resource machines and idling, which is stated on the page.
