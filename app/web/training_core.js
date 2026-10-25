/* training_core.js - the toy scheduling world and its two learners for the "watch it learn" demo. No DOM, no libraries: runs in the
   browser and in Node (tests/test_training_core.js).

   World: 2 machines, 8 jobs known at time 0. Job: length p (1-5), weight w (1-3), deadline d (p .. p+9). Whenever a machine becomes free
   the scheduler picks one waiting job (no idling). A job started at s finishes at C = s + p; its lateness is T = max(0, C - d) and its
   cost is w T^2, so an episode's cost is J = sum w T^2 (the research objective in miniature). The reward of a decision is minus that
   job's cost divided by COST_SCALE, so the episode return is -J / COST_SCALE.

   Policy: score of waiting job j = theta . (time left, length, importance) with time left = (d - t - p) / 10, length = p / 5,
   importance = w / 3; probabilities by softmax over the waiting jobs.
   Learners: "reinforce" (episode return minus a running per-step average) and "actor-critic" (per-decision advantage from a linear
   value estimate). */
(function (root) {
  const MACHINES = 2, JOBS = 8, COST_SCALE = 10, TRAIN_EPISODES = 600, ARRIVAL_SPAN = 6, ONLINE_SLACK = 7;  // budget of the demo: 600 episodes, about 20 s at 30 episodes per second

  function makeRng(seed) {  // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // online = true: jobs arrive over time (arrival a, spread over the first ARRIVAL_SPAN ticks; the deadline is counted from the arrival)
  function makeInstance(rand, online) {
    const jobs = [];
    for (let i = 0; i < JOBS; i++) {
      const p = 1 + Math.floor(rand() * 5), w = 1 + Math.floor(rand() * 3), a = online ? Math.floor(rand() * ARRIVAL_SPAN) : 0, d = a + p + Math.floor(rand() * (online ? ONLINE_SLACK : 10));
      jobs.push({ id: i, p, w, d, a });
    }
    if (online) jobs.sort((x, y) => x.a - y.a).forEach((j, i) => { j.id = i; });  // job numbers follow the arrival order
    return jobs;
  }

  const features = (job, t) => [(job.d - t - job.p) / 10, job.p / 5, job.w / 3];

  // One episode. choose(state) -> index into state.waiting (the jobs that have arrived and are not yet started). A machine with nothing
  // to do waits for the next arrival (an idle gap, recorded in idles). Returns the cost J, every decision, the machine blocks and idles.
  function runEpisode(jobs, choose) {
    const free = new Array(MACHINES).fill(0), left = jobs.map((_, i) => i), steps = [], blocks = [], idles = [];
    let cost = 0;
    while (left.length) {
      let m = 0;
      for (let k = 1; k < MACHINES; k++) if (free[k] < free[m]) m = k;
      let t = free[m];
      let waiting = left.filter(j => jobs[j].a <= t);
      if (!waiting.length) {  // nothing has arrived yet: this machine idles until the next arrival
        const next = Math.min(...left.map(j => jobs[j].a));
        idles.push({ machine: m, start: t, end: next }); free[m] = next; continue;
      }
      const state = { t, machine: m, waiting: waiting.slice(), jobs, free: free.slice(), arrived: jobs.filter(j => j.a <= t).length };
      const k = choose(state), j = waiting[k], job = jobs[j], C = t + job.p, T = Math.max(0, C - job.d), c = job.w * T * T;
      steps.push({ state, k, job: j, start: t, end: C, lateness: T, cost: c, reward: -c / COST_SCALE });
      blocks.push({ job: j, machine: m, start: t, end: C, late: T > 0 });
      cost += c; free[m] = C; left.splice(left.indexOf(j), 1);
    }
    return { cost, steps, blocks, idles, makespan: Math.max(...free) };
  }

  // classic rules, as choosers
  const argBest = (state, score) => { let best = 0, bs = Infinity; state.waiting.forEach((j, k) => { const s = score(state.jobs[j], state.t); if (s < bs) { bs = s; best = k; } }); return best; };
  const RULES = {
    EDF: s => argBest(s, j => j.d), LST: s => argBest(s, (j, t) => j.d - t - j.p), SPT: s => argBest(s, j => j.p),
    WSPT: s => argBest(s, j => -j.w / j.p), WEDF: s => argBest(s, (j, t) => (j.d - t) / j.w),
  };
  const randomChooser = rand => state => Math.floor(rand() * state.waiting.length);

  function softmax(z) {
    const m = Math.max(...z), e = z.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / s);
  }
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

  // summary features of a state for the critic (what a person would look at to guess how bad things will get)
  function stateFeatures(state) {
    const left = state.waiting.map(j => (state.jobs[j].d - state.t - state.jobs[j].p) / 10), n = state.waiting.length;
    const work = state.waiting.reduce((s, j) => s + state.jobs[j].p, 0);
    const doomed = state.waiting.reduce((s, j, k) => s + state.jobs[j].w * Math.pow(Math.max(0, -left[k]), 2), 0);  // already late if started now
    const spare = left.reduce((s, x) => s + Math.max(0, x), 0) / n;
    const other = Math.max(...state.free) - state.t;  // how far ahead the other machine is
    return [1, n / JOBS, doomed, spare, Math.min(...left), work / 24, other / 5];
  }

  function createLearner(kind, seed, opts) {
    const o = Object.assign({ actorRate: 0.01, criticRate: 0.02, lambda: 0.9 }, opts);
    const rand = makeRng(seed), theta = [0, 0, 0], critic = new Array(7).fill(0), base = new Array(JOBS).fill(0);
    let episodes = 0;
    const probs = state => softmax(state.waiting.map(j => dot(theta, features(state.jobs[j], state.t))));
    const learner = {
      kind, theta, critic, probs, rand,
      value: state => dot(critic, stateFeatures(state)),
      baseline: t => base[Math.min(t, JOBS - 1)],  // REINFORCE's running average of the reward from decision t onwards (for display)
      greedy: state => { const p = probs(state); let b = 0; p.forEach((x, k) => { if (x > p[b]) b = k; }); return b; },
      sample: state => { const p = probs(state); let u = rand(), k = 0; while (k < p.length - 1 && u > p[k]) { u -= p[k]; k++; } return k; },
      get episodes() { return episodes; },
      // one training episode on a fresh job set; returns the episode record
      train(jobs) { const ep = runEpisode(jobs, learner.sample); learner.update(ep); return ep; },
      // learn from a finished episode (the demo's Step mode plays an episode out decision by decision, then calls this)
      update(ep) {
        const jobs = ep.steps[0].state.jobs;
        const grads = ep.steps.map(st => {  // gradient of log pi(action): phi_a - sum_j pi_j phi_j
          const p = probs(st.state), phis = st.state.waiting.map(j => features(jobs[j], st.state.t));
          const mean = [0, 1, 2].map(i => phis.reduce((s, f, k) => s + p[k] * f[i], 0));
          return phis[st.k].map((x, i) => x - mean[i]);
        });
        if (kind === "reinforce") {
          let g = 0;
          for (let t = ep.steps.length - 1; t >= 0; t--) {  // return from step t, minus the running average return from step t
            g += ep.steps[t].reward;
            const adv = g - base[t];
            base[t] += 0.05 * (g - base[t]);
            for (let i = 0; i < 3; i++) theta[i] += o.actorRate * adv * grads[t][i];
          }
        } else {
          // actor-critic: the critic estimates how good each situation is; the advantage of a decision is how much better it went than
          // the critic expected (generalised advantage estimation over the episode); the critic is trained towards what happened
          const n = ep.steps.length, psis = ep.steps.map(st => stateFeatures(st.state)), vs = psis.map(ps => dot(critic, ps));
          let adv = 0;
          for (let t = n - 1; t >= 0; t--) {
            const delta = ep.steps[t].reward + (t + 1 < n ? vs[t + 1] : 0) - vs[t];
            adv = delta + o.lambda * adv;
            for (let i = 0; i < 7; i++) critic[i] += o.criticRate * adv * psis[t][i];
            for (let i = 0; i < 3; i++) theta[i] += o.actorRate * adv * grads[t][i];
          }
        }
        episodes++;
      },
    };
    return learner;
  }

  // mean cost of a chooser over a list of job sets
  const meanCost = (sets, choose) => sets.reduce((s, jobs) => s + runEpisode(jobs, choose).cost, 0) / sets.length;
  const testSets = (n, seed, online) => { const r = makeRng(seed); return Array.from({ length: n }, () => makeInstance(r, online)); };

  const core = { MACHINES, JOBS, COST_SCALE, TRAIN_EPISODES, makeRng, makeInstance, features, runEpisode, RULES, randomChooser, createLearner, meanCost, testSets, softmax, stateFeatures };
  if (typeof module !== "undefined" && module.exports) module.exports = core; else root.TrainingCore = core;
})(typeof window !== "undefined" ? window : globalThis);
