/* training_ui.js - the "Watch it learn" page: the toy world, the State -> Action -> Reward -> Next state strip, Step / Train / Test, the
   learning curve and the learned weights. The world and the learners are in training_core.js; every sentence is in training_wording.md
   (exported as RACE_DATA.text); explanation cards use the shared engine (cards.js). */
(function () {
  const C = window.TrainingCore, D = window.RACE_DATA, TX = D.text, UI = TX.ui, TP = TX.templates, LB = TX.labels;
  const { I, N, O, T, R, sub, sup, frac, sum, M } = MathML;
  const F = {};
  const K = Cards.create({ text: TX, formulas: F });
  const fill = K.fill;

  // ---- formulas for the cards ---------------------------------------------------------------------------------------------
  const wj = sub(I("w"), I("j")), Tj = sub(I("T"), I("j")), phi = sub(I("φ"), I("j")), th = I("θ");
  const grad = R(I("∇"), I("log"), sub(I("π"), th), O("("), sub(I("a"), I("t")), O("|"), sub(I("s"), I("t")), O(")"));
  Object.assign(F, {
    J: M(I("J"), O("="), sum(I("j"), R(wj, sup(Tj, N(2))))),
    reward: M(sub(I("r"), I("t")), O("="), O("−"), frac(R(wj, sup(Tj, N(2))), N(10)), T("   (job j started at time t)")),
    score: M(sub(I("z"), I("j")), O("="), sup(th, O("⊤")), phi, O("="), sub(th, T("time left")), O("·"), T("time left"), O("+"), sub(th, T("length")), O("·"), T("length"), O("+"), sub(th, T("importance")), O("·"), T("importance")),
    policy: M(I("π"), O("("), I("j"), O("|"), I("s"), O(")"), O("="), frac(R(I("exp"), O("("), sub(I("z"), I("j")), O(")")), R(sum(I("k"), R(I("exp"), O("("), sub(I("z"), I("k")), O(")")))))),
    ret: M(sub(I("G"), I("t")), O("="), sum(R(I("k"), O("≥"), I("t")), sub(I("r"), I("k")))),
    reinforce: M(th, O("←"), th, O("+"), I("α"), O("("), sub(I("G"), I("t")), O("−"), sub(I("b"), I("t")), O(")"), grad) +
      M(I("∇"), I("log"), sub(I("π"), th), O("("), sub(I("a"), I("t")), O("|"), sub(I("s"), I("t")), O(")"), O("="), sub(phi, T("chosen")), O("−"), sum(I("j"), R(I("π"), O("("), I("j"), O(")"), phi))),
    critic: M(I("V"), O("("), I("s"), O(")"), O("="), sup(I("w"), O("⊤")), I("ψ"), O("("), I("s"), O(")")),
    advantage: M(sub(I("δ"), I("t")), O("="), sub(I("r"), I("t")), O("+"), I("V"), O("("), sub(I("s"), R(I("t"), O("+"), N(1))), O(")"), O("−"), I("V"), O("("), sub(I("s"), I("t")), O(")")) +
      M(sub(I("A"), I("t")), O("="), sum(I("k"), R(sup(N("0.9"), I("k")), sub(I("δ"), R(I("t"), O("+"), I("k"))))), O(","), T("   "), th, O("←"), th, O("+"), I("α"), sub(I("A"), I("t")), grad),
  });

  const build = (id) => K.topicCard(id, {});

  // ---- state ---------------------------------------------------------------------------------------------------------------
  const KINDS = { reinforce: "reinforce", ac: "actor-critic" }, HORIZON = 20, WINDOW = 25;
  const testSets = C.testSets(200, 424242);
  const refRandom = C.meanCost(testSets, C.randomChooser(C.makeRng(7)));
  const refRules = Object.entries(C.RULES).map(([k, f]) => [k, C.meanCost(testSets, f)]).sort((a, b) => a[1] - b[1]);
  const refRule = { name: refRules[0][0], cost: refRules[0][1] };
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
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
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
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function drawCurve() {
    const cv = cfg.curve, dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const padL = 38, padR = 10, padT = 8, padB = 22, maxY = 200, maxX = C.TRAIN_EPISODES;
    const X = e => padL + (w - padL - padR) * e / maxX, Y = c => padT + (h - padT - padB) * (1 - Math.min(c, maxY) / maxY);
    g.font = "12px " + css("--body"); g.fillStyle = css("--muted"); g.strokeStyle = css("--rule"); g.lineWidth = 1;
    [0, 50, 100, 150, 200].forEach(v => { g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(w - padR, Y(v)); g.stroke(); g.textAlign = "right"; g.fillText(v, padL - 5, Y(v) + 4); });
    [0, 200, 400, 600].forEach(v => { g.textAlign = "center"; g.fillText(v, X(v), h - 6); });
    const line = (cost, colour, dash, label) => {
      g.strokeStyle = colour; g.lineWidth = 2; g.setLineDash(dash); g.beginPath(); g.moveTo(padL, Y(cost)); g.lineTo(w - padR, Y(cost)); g.stroke(); g.setLineDash([]);
      g.fillStyle = colour; g.textAlign = "right"; g.fillText(label, w - padR - 2, Y(cost) - 4);
    };
    line(refRandom, css("--rulebar"), [6, 4], UI["curve.random"]);
    line(refRule.cost, css("--rulebar"), [], fill(UI["curve.rule"], { name: refRule.name }));
    Object.entries(S.hist).forEach(([kind, hist]) => {
      if (!hist.length) return;
      const colour = kind === "reinforce" ? css("--copied") : css("--win"), active = kind === S.active;
      g.globalAlpha = active ? 0.35 : 0.18; g.fillStyle = colour;
      hist.forEach((c, i) => { g.beginPath(); g.arc(X(i + 1), Y(c), 2, 0, 6.3); g.fill(); });
      g.globalAlpha = 1; g.strokeStyle = colour; g.lineWidth = active ? 3 : 2; g.beginPath();
      hist.forEach((c, i) => { const s = Math.max(0, i - WINDOW + 1), avg = hist.slice(s, i + 1).reduce((a, b) => a + b, 0) / (i + 1 - s); i ? g.lineTo(X(i + 1), Y(avg)) : g.moveTo(X(1), Y(avg)); });
      g.stroke();
    });
    if (!S.hist.reinforce.length && !S.hist.ac.length) { g.fillStyle = css("--muted"); g.textAlign = "center"; g.font = "15px " + css("--body"); g.fillText(UI["curve.empty"], w / 2, h / 2); }
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
