// =====================================================================================================================
// WORDING FILE for the "Watch it learn" training demo. Every sentence the visitor reads is in this file. Edit it in any text editor,
// then run:   python tools/analysis/make_training_demo.py     (and republish app/web/dist/training_demo.html)
// The format is the same as app/web/wording.md (see the header of that file for the full rules):
//  * "## NAME" starts a list of "key: value" lines (change the text after the colon, never the key).
//  * "### id" starts one explanation card: kicker, title, related (ids of cards to link to), then "--- Section heading [(open)]"
//    sections with paragraphs (a blank line starts a new paragraph).
//  * Markup:  **bold**   *italic*   `code`   a_{j} (subscript)   x^{2} (superscript)   {{name}} (a formula the program builds):
//      {{J}} cost   {{reward}} reward of a decision   {{score}} job score   {{policy}} softmax   {{ret}} return
//      {{reinforce}} REINFORCE update   {{advantage}} actor-critic advantage   {{critic}} value estimate
//  * Words in {curly braces} inside ## UI and ## TEMPLATES lines are filled in by the program (for example {job}); keep them.
//  * Keep ids and keys exactly as they are; they are how the program finds things.
// =====================================================================================================================

## UI
heading: Watch it learn
tagline: A toy scheduler with 2 machines and 8 jobs. Press Step to see one decision at a time, or Train to watch the reward improve it.
toy_note: A toy version of the real problem: 2 machines, no resource limits.
btn.step: Step
btn.train: Train
btn.pause: Pause
btn.test: Test
btn.reset: Reset
speed.1: ×1
speed.3: ×3
speed.10: ×10
learner.reinforce: REINFORCE
learner.ac: Actor-critic
learner.label: Learner
panel.world: The world
panel.queue: Waiting jobs
panel.machines: Machines
panel.loop: The loop
panel.curve: Cost while learning (lower is better)
panel.weights: What it learned
panel.test: Test on new jobs
loop.state: State
loop.action: Action
loop.reward: Reward
loop.next: Next state
loop.idle: Press Step to see the first decision.
job.label: Job {id}
job.detail: length {p} · due {d} · weight {w}
curve.random: Random order
curve.rule: Best classic rule ({name})
curve.empty: Press Train to start learning.
episodes: Episodes: {n}
weights.timeleft: Time left
weights.length: Length
weights.importance: Importance
weights.hint: A bar to the left means it prefers a small value; to the right, a large value.
test.random: Random order
test.rule: Best classic rule ({name})
test.learner: This learner
test.caption: Average cost on {n} new job sets (lower is better).
test.note: In this toy it can beat a single simple rule; the real 10-machine problem is much harder (see the card).
related: Related
back: ← Back
close: Close

## TEMPLATES
loop.state.text: Time {t} · machine {m} is free · {n} jobs waiting
loop.action.text: Start job {job} (picked with probability {pct}%)
loop.reward.late: {reward} · job {job} finishes {late} ticks late (weight {w})
loop.reward.ontime: 0 · job {job} finishes on time
loop.next.text: Time {t} · {n} jobs left
loop.next.done: Episode over · total cost {cost}
msg.episode: Episode {n} finished with cost {cost}. The learner has updated itself from its rewards.
msg.trained: Training finished after {n} episodes. Press Test to try it on new jobs.
msg.budget: This learner has used its {n} training episodes. Press Reset to start again.
msg.testing: Testing on {n} new job sets...

## LABELS
chip.world: The toy world
chip.state: State
chip.action: Action
chip.reward: Reward
chip.policy: The scoring rule
chip.reinforce: REINFORCE
chip.actor_critic: Actor-critic
chip.curve: Learning curve
chip.weights: What it learned
chip.test: The test

### world
kicker: The toy world
title: What this demo is, and what it is not
related: state, action, reward, curve
--- In plain words (open)
Eight jobs wait to be run on two machines. Each job has a length, a deadline and an importance. Every time a machine becomes free, the scheduler picks one waiting job to start. A job that finishes after its deadline costs more the later it is and the more important it is.
--- Formulation (open)
A job *j* has length p_{j}, deadline d_{j} and weight w_{j}. Started at time s_{j} it finishes at C_{j} = s_{j} + p_{j} and is late by T_{j} = max(0, C_{j} − d_{j}). The cost of an episode (one set of 8 jobs) is

{{J}}
--- How it relates to the research
This is the same kind of cost as in the main project, in miniature. The real problem has 10 machines, jobs that need CPU and memory, 100 jobs per instance and jobs that can arrive over time. The scheduler there is a neural network with many more weights, trained for millions of steps. Here the learner has three numbers, so you can watch every part of the loop.

### state
kicker: The loop
title: State: what the learner sees
related: action, reward, policy
--- In plain words (open)
The state is everything the learner knows when it must decide: the clock, which machine just became free, and which jobs are still waiting with their lengths, deadlines and importance.
--- Formulation
At each decision time *t* the state is the list of waiting jobs together with the time left for each: (d_{j} − t − p_{j}), which is how long the job could still wait and finish on time. A negative number means it is already going to be late.

### action
kicker: The loop
title: Action: which job to start
related: state, reward, policy
--- In plain words (open)
The action is the choice of one waiting job to start on the free machine. The learner does not choose the machine: the first machine to become free takes the job, as in the project's priority-learning models.
--- Formulation
The learner gives each waiting job a score (see the scoring rule) and picks a job at random in proportion to the scores, so that it keeps exploring while it learns.

### reward
kicker: The loop
title: Reward: what tells it how well it did
related: state, action, curve
--- In plain words (open)
After each decision the learner receives a reward: zero if the job will finish on time, and a negative number if it will finish late. The later and the more important the job, the more negative. The learner never sees a list of right answers, only these rewards.
--- Formulation (open)
{{reward}}

Adding up the rewards of an episode gives minus the cost J of that episode, so improving the total reward is the same as reducing the cost.
--- Deeper
The cost is divided by a constant (10) so that the numbers are easy to read. A job's reward is known as soon as it is started, because its finishing time is then fixed.

### policy
kicker: The learner
title: The scoring rule it learns
related: weights, reinforce, actor_critic
--- In plain words (open)
The learner scores each waiting job from three numbers: how much time is left before it is due, how long it is, and how important it is. Three weights say how much each number counts. The job with the highest score is the most likely to be picked.
--- Formulation (open)
With the three signals gathered in a list φ_{j} (time left, length, importance, each scaled to a similar size) and three weights θ:

{{score}}

{{policy}}

Learning means changing the three weights θ so that choices with better rewards become more likely.
--- Deeper
This is the same idea as Option 2 in the project, where a larger network scores each waiting job from its own features, but with one layer and three weights so that you can read it.

### reinforce
kicker: The learner
title: REINFORCE: learning from the whole episode
related: policy, actor_critic, curve
--- In plain words (open)
After each episode, REINFORCE looks at how much total reward came after each decision and compares it with the usual amount at that point. Decisions followed by better-than-usual rewards become more likely; decisions followed by worse-than-usual rewards become less likely.
--- Formulation (open)
With the return G_{t} (the rewards from decision t to the end) and a baseline b_{t} (a running average of that return):

{{ret}}

{{reinforce}}
--- Deeper
Plain REINFORCE is noisy because one episode is only one sample; the baseline reduces that noise. Source: Williams (1992).

### actor_critic
kicker: The learner
title: Actor-critic: a learner with a guess about the future
related: policy, reinforce, curve
--- In plain words (open)
Actor-critic adds a second small estimator, the critic, that guesses how well things will go from the current situation. The actor (the scoring rule) is then pushed by how much better or worse each decision turned out than the critic expected.
--- Formulation (open)
The critic estimates the value of a state from a few summary numbers (how many jobs wait, how many are already doomed to be late, how much slack is left). The advantage of a decision is how much better it went than the critic expected, summed over the rest of the episode with a discount factor:

{{critic}}

{{advantage}}
--- Deeper
The advantage is a generalised advantage estimate (Schulman et al. 2016) with a smoothing factor of 0.9. The critic is trained towards what actually happened.

### curve
kicker: The learner
title: Reading the learning curve
related: reward, test, weights
--- In plain words (open)
Each dot is the cost of one training episode on a fresh set of jobs, and the line is the average of the last 25 episodes. While it learns, the learner still picks jobs at random in proportion to their scores, so the cost is higher than when it picks its best guess. The grey lines show a random scheduler and the best of five classic rules, measured on new job sets.
--- Deeper
The two learners start from the same place (all three weights zero, so every job is equally likely). Switch the learner and train again to compare their curves. Each press of Reset starts both learners again.

### weights
kicker: The learner
title: What it learned
related: policy, curve
--- In plain words (open)
The three bars are the learned weights. A bar to the left means the learner prefers a small value of that signal; a bar to the right, a large one. After training, the time-left weight is strongly to the left in nearly every run: the learner has discovered that jobs with little time left should go first, which is the idea behind the classic rules.
--- Deeper
It also usually prefers short jobs (to the left) and important jobs (to the right). The exact numbers differ from run to run.

### test
kicker: The learner
title: The test on new jobs
related: curve, weights
--- In plain words (open)
Test uses 200 new sets of jobs that the learner has never trained on, and lets it pick its best-scoring job every time. It is compared with a random scheduler and with the best of five classic rules on the same sets.
--- Deeper
In this toy the learner can even beat a single simple rule, because it blends three signals. The real problem, with 10 machines and 100 jobs, is much harder: there the learned schedulers matched the best rules but did not beat them.

Using new job sets matters for the same reason as in the project: a learner can look good on the jobs it trained on and worse on new ones.
