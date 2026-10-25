/* test_training_core.js - acceptance tests for the toy world and the two learners (app/web/training_core.js).
   Run:  node tests/test_training_core.js            (prints a table and exits non-zero if a criterion fails)
   Criteria (docs: Future/research/2026-10-25-training-demo-plan.md): for each learner, over 10 seeds, after the training budget the greedy
   policy's mean cost on 200 fresh job sets is clearly below the random scheduler's, and within 10% of the best classic rule in at
   least 8 of 10 seeds. Pass an optional JSON of learner options as argv[2] to try other learning rates. */
const C = require("../app/web/training_core.js");
const BUDGET = Number(process.env.EPISODES || C.TRAIN_EPISODES), SEEDS = 10;
const opts = process.argv[2] ? JSON.parse(process.argv[2]) : {};
const ONLINE = process.env.ONLINE === "1";  // ONLINE=1: jobs arrive over time
const tests = C.testSets(200, 424242, ONLINE);
const random = C.meanCost(tests, C.randomChooser(C.makeRng(7)));
const rules = Object.fromEntries(Object.entries(C.RULES).map(([k, f]) => [k, C.meanCost(tests, f)]));
const bestName = Object.keys(rules).sort((a, b) => rules[a] - rules[b])[0], best = rules[bestName];
console.log(`random ${random.toFixed(1)} | ` + Object.entries(rules).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(" | ") + ` | best rule ${bestName}`);
let failed = false;
for (const kind of ["reinforce", "actor-critic"]) {
  const finals = [], thetas = [], timeLeft = [];
  for (let seed = 1; seed <= SEEDS; seed++) {
    const L = C.createLearner(kind, seed, opts), r = C.makeRng(1000 + seed);
    for (let e = 0; e < BUDGET; e++) L.train(C.makeInstance(r, ONLINE));
    timeLeft.push(L.theta[0]); finals.push(C.meanCost(tests, L.greedy)); thetas.push(L.theta.map(x => x.toFixed(2)).join(","));
  }
  const near = finals.filter(x => x <= best * 1.10).length, clearly = finals.every(x => x < random * 0.8);
  console.log(`${kind}: final cost per seed ${finals.map(x => x.toFixed(1)).join(" ")}\n   within 10% of best rule in ${near}/${SEEDS}; all clearly below random: ${clearly}\n   theta (time left, length, importance): ${thetas.join(" | ")}`);
  // the page may say the learner favours jobs with little time left only if the time-left weight is clearly negative in nearly every seed
  const THRESHOLD = ONLINE ? -2 : -5, negative = timeLeft.filter(x => x < THRESHOLD).length;  // online costs are smaller, so the weights are smaller
  console.log(`   time-left weight below ${THRESHOLD} in ${negative}/${SEEDS} seeds`);
  if (near < 8 || !clearly || negative < 9) failed = true;
}
console.log(failed ? "FAILED" : "PASSED");
process.exit(failed ? 1 : 0);
