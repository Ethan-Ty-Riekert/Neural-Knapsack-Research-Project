/* training_ui.js - the "Watch it learn" page: the toy world, the State -> Action -> Reward -> Next state strip, Step / Train / Test, the
   learning curve and the learned weights. The world and the learners are in training_core.js; every sentence is in training_wording.md
   (exported as RACE_DATA.text); explanation cards use the shared engine (cards.js). */
(function () {
  const C = window.TrainingCore, D = window.RACE_DATA, TX = D.text, UI = TX.ui, TP = TX.templates, LB = TX.labels;
  const F = {};
  const K = Cards.create({ text: TX, formulas: F });
  const { el, fill, css } = TrainingShared;
  TrainingShared.formulas(F);  // formulas for the cards

  const build = (id) => K.topicCard(id, {});

  // ---- state ---------------------------------------------------------------------------------------------------------------
  const KINDS = { reinforce: "reinforce", ac: "actor-critic" }, HORIZON = 20, WINDOW = 25;
  const REF = TrainingShared.refs(false), testSets = REF.sets, refRandom = REF.random, refRule = REF.rule;
  let S, cfg, raf = 0, resets = 0;

  function reset() {
    cancelAnimationFrame(raf); raf = 0;
    const seed = 100 + resets++ * 7;
    S = { active: "reinforce", running: false, step: null, speed: 1, lastEp: null, jobRng: C.makeRng(seed + 999), tested: null,
          L: { reinforce: C.createLearner("reinforce", seed), ac: C.createLearner("actor-critic", seed + 1) }, hist: { reinforce: [], ac: [] } };
    drawAll(); setMsg(""); cfg.loop.querySelectorAll(".val").forEach(v => { v.textContent = ""; });
    cfg.loop.querySelector('[data-box="state"] .val').textContent = UI["loop.idle"];
    setButtons();
  }
  const learner = () => S.L[S.active];
  const setMsg = t => { cfg.msg.textContent = t; };

  // ---- the world --------------------------------------------------------------------------------------------------------------
  // jobs: the instance; waiting: indices still waiting; blocks: scheduled blocks; now: decision time; probs: {job index: probability}; pick: chosen job
  function drawWorld(jobs, waiting, blocks, now, probs, pick, newBlock) {
    cfg.queue.textContent = "";
    waiting.forEach(j => {
      const job = jobs[j], chip = el("div", "job" + (j === pick ? " pick" : ""));
      chip.append(el("b", null, fill(UI["job.label"], { id: j + 1 })), el("span", null, fill(UI["job.detail"], job)));
      if (probs && probs[j] != null) {
        const bar = el("div", "pbar"), fillEl = el("i"); fillEl.style.width = (probs[j] * 100).toFixed(0) + "%"; bar.append(fillEl);
        chip.append(bar, el("em", null, (probs[j] * 100).toFixed(0) + "%"));
      }
      cfg.queue.append(chip);
    });
    cfg.lanes.forEach((lane, m) => {
      lane.textContent = "";
      blocks.filter(b => b.machine === m).forEach(b => {
        const e = el("div", "blk" + (b.late ? " late" : "") + (b === newBlock ? " fresh" : ""), "J" + (b.job + 1));
        e.style.left = (b.start / HORIZON * 100) + "%"; e.style.width = (b.end - b.start) / HORIZON * 100 + "%"; lane.append(e);
      });
      if (now != null) { const line = el("div", "nowline"); line.style.left = (now / HORIZON * 100) + "%"; lane.append(line); }
    });
  }
  function drawFinal(ep) {  // a finished episode: every block, nothing waiting
    drawWorld(ep.steps[0].state.jobs, [], ep.blocks, null, null, null, null);
  }

  // ---- the loop strip ------------------------------------------------------------------------------------------------------------
  function loopText(box, text) { cfg.loop.querySelector(`[data-box="${box}"] .val`).textContent = text; }
  function showDecision(ep, k) {
    const st = ep.steps[k], jobs = st.state.jobs, L = learner(), p = L.probs(st.state), probs = {};
    st.state.waiting.forEach((j, i) => { probs[j] = p[i]; });
    const blocks = ep.blocks.slice(0, k + 1);
    drawWorld(jobs, st.state.waiting, blocks, st.start, probs, st.job, blocks[k]);
    loopText("state", fill(TP["loop.state.text"], { t: st.state.t, m: st.state.machine + 1, n: st.state.waiting.length }));
    loopText("action", fill(TP["loop.action.text"], { job: st.job + 1, pct: Math.round(p[st.k] * 100) }));
    loopText("reward", st.cost > 0 ? fill(TP["loop.reward.late"], { reward: (st.reward).toFixed(1), job: st.job + 1, late: st.lateness, w: jobs[st.job].w }) : fill(TP["loop.reward.ontime"], { job: st.job + 1 }));
    const last = k + 1 >= ep.steps.length;
    loopText("next", last ? fill(TP["loop.next.done"], { cost: ep.cost }) : fill(TP["loop.next.text"], { t: ep.steps[k + 1].state.t, n: ep.steps[k + 1].state.waiting.length }));
  }

  // ---- curve and weights ------------------------------------------------------------------------------------------------------
  function drawCurve() {
    TrainingShared.drawCurve(cfg.curve, { hist: S.hist, active: S.active, refs: REF, empty: UI["curve.empty"],
      labels: { random: UI["curve.random"], rule: fill(UI["curve.rule"], { name: refRule.name }) }, colours: { reinforce: css("--copied"), ac: css("--win") } });
  }
  function drawWeights() {
    cfg.weights.textContent = "";
    const th = learner().theta;
    [["weights.timeleft", 0], ["weights.length", 1], ["weights.importance", 2]].forEach(([key, i]) => {
      const row = el("div", "wrow"), bar = el("div", "wbar"), fillEl = el("i");
      const frac = Math.max(-1, Math.min(1, th[i] / 8));
      fillEl.style.left = frac >= 0 ? "50%" : (50 + frac * 50) + "%"; fillEl.style.width = Math.abs(frac) * 50 + "%";
      bar.append(fillEl, el("u"));
      row.append(el("span", "wl", UI[key]), bar, el("span", "wv", th[i].toFixed(1)));
      cfg.weights.append(row);
    });
  }
  function drawTest() {
    cfg.test.textContent = "";
    if (!S.tested) return;
    const rows = [[fill(UI["test.random"], {}), refRandom, "rulebar"], [fill(UI["test.rule"], { name: refRule.name }), refRule.cost, "rulebar"],
                  [UI["test.learner"] + " (" + (S.active === "reinforce" ? UI["learner.reinforce"] : UI["learner.ac"]) + ")", S.tested, "copied"]];
    const top = Math.max(...rows.map(r => r[1]));
    rows.forEach(([label, v, colour]) => {
      const row = el("div", "trow"), bar = el("div", "tbar"), f = el("i"); f.style.width = (v / top * 100) + "%"; f.style.background = `var(--${colour})`; bar.append(f);
      row.append(el("span", "tl", label), bar, el("b", null, v.toFixed(0)));
      cfg.test.append(row);
    });
    cfg.test.append(el("p", "tcap", fill(UI["test.caption"], { n: testSets.length })), el("p", "tcap", UI["test.note"]));
  }
  function drawAll() {
    drawCurve(); drawWeights(); drawTest();
    cfg.episodes.textContent = fill(UI.episodes, { n: learner().episodes });
    if (S.lastEp) drawFinal(S.lastEp); else drawWorld([], [], [], null, null, null, null);
  }

  // ---- controls ------------------------------------------------------------------------------------------------------------------
  function setButtons() {
    cfg.buttons.step.disabled = S.running; cfg.buttons.test.disabled = S.running;
    cfg.buttons.train.textContent = S.running ? UI["btn.pause"] : UI["btn.train"];
    cfg.learnerBox.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.kind === S.active)); b.disabled = S.running; });
    cfg.speedBox.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(+b.dataset.speed === S.speed)));
  }
  function record(ep) { S.hist[S.active].push(ep.cost); S.lastEp = ep; }

  function doStep() {
    const L = learner();
    if (L.episodes >= C.TRAIN_EPISODES && !S.step) { setMsg(fill(TP["msg.budget"], { n: C.TRAIN_EPISODES })); return; }
    if (!S.step) { S.step = { ep: C.runEpisode(C.makeInstance(S.jobRng), L.sample), k: 0 }; }
    const { ep } = S.step;
    showDecision(ep, S.step.k);
    S.step.k++;
    if (S.step.k >= ep.steps.length) {  // last decision of the episode: the learner updates from its rewards
      L.update(ep); record(ep); S.step = null; S.tested = null;
      cfg.episodes.textContent = fill(UI.episodes, { n: L.episodes }); drawCurve(); drawWeights(); drawTest();
      setMsg(fill(TP["msg.episode"], { n: L.episodes, cost: ep.cost }));
    } else setMsg("");
  }
  function doTrain() {
    if (S.running) { S.running = false; cancelAnimationFrame(raf); setButtons(); return; }
    const L = learner();
    if (L.episodes >= C.TRAIN_EPISODES) { setMsg(fill(TP["msg.budget"], { n: C.TRAIN_EPISODES })); return; }
    S.step = null; S.running = true; setButtons();
    let last = performance.now(), owed = 0;
    const tick = now => {
      if (!S.running) return;
      owed += Math.min(now - last, 100) / 1000 * 30 * S.speed; last = now;
      while (owed >= 1 && L.episodes < C.TRAIN_EPISODES) { owed -= 1; record(L.train(C.makeInstance(S.jobRng))); }
      S.tested = null; drawAll(); setMsg("");
      if (L.episodes >= C.TRAIN_EPISODES) { S.running = false; setButtons(); setMsg(fill(TP["msg.trained"], { n: L.episodes })); return; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }
  function doTest() {
    setMsg(fill(TP["msg.testing"], { n: testSets.length }));
    setTimeout(() => { S.tested = C.meanCost(testSets, learner().greedy); setMsg(""); drawTest(); }, 30);
  }

  // ---- start-up ------------------------------------------------------------------------------------------------------------------
  function init(c) {
    cfg = c;
    document.querySelectorAll("[data-ui]").forEach(e => { e.textContent = UI[e.dataset.ui]; });
    K.init(c.explain, () => ({}), build);
    // buttons
    const mk = (label, fn, cls) => { const b = el("button", cls || "go", label); b.type = "button"; b.onclick = fn; return b; };
    cfg.buttons = { step: mk(UI["btn.step"], doStep), train: mk(UI["btn.train"], doTrain), test: mk(UI["btn.test"], doTest) };
    cfg.controls.append(cfg.buttons.step, cfg.buttons.train, cfg.buttons.test, mk(UI["btn.reset"], reset, "ghost"));
    cfg.speedBox = el("div", "seg small");
    [1, 3, 10].forEach(s => { const b = el("button", null, UI["speed." + s]); b.type = "button"; b.dataset.speed = s; b.onclick = () => { S.speed = s; setButtons(); }; cfg.speedBox.append(b); });
    cfg.controls.append(cfg.speedBox);
    cfg.learnerBox = el("div", "seg");
    [["reinforce", "learner.reinforce"], ["ac", "learner.ac"]].forEach(([kind, key]) => {
      const b = el("button", null, UI[key]); b.type = "button"; b.dataset.kind = kind; b.setAttribute("role", "radio");
      b.onclick = () => { S.active = kind; S.step = null; S.lastEp = null; S.tested = null; setButtons(); drawAll(); setMsg(""); };
      cfg.learnerBox.append(b);
    });
    cfg.learner.append(cfg.learnerBox);
    document.addEventListener("click", e => {
      const t = e.target.closest("[data-explain]");
      if (t && !t.closest("#explain")) K.open(t.dataset.explain === "learner" ? (S.active === "reinforce" ? "reinforce" : "actor_critic") : t.dataset.explain, {});
    });
    window.addEventListener("resize", drawCurve);
    reset();
  }
  window.TrainingUI = { init };
})();
