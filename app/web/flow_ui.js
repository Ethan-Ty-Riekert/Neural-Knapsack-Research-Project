/* flow_ui.js - the animated "Watch the loop" layouts (flow_a.html, flow_b.html). Layout: job queue on the left, two machines on top, the
   centre box (a centre module: the chain of the episode, or the decision ring) and the controls, and on the right the actor-critic
   network with the learning curve in its lower box. One decision is played as a short film: the state appears, a token travels to the
   network, the network scores the waiting jobs, the chosen action travels back, the job flies from the queue to a machine, and the reward
   pops up and joins the chain. Every sentence comes from training_wording.md; the world and learners from training_core.js.
   A centre module (FlowCentre) provides: css, create(root) -> { reset(), snap(ep), out, in, state(info), action(info), reward(info), next(info) }. */
(function () {
  const C = window.TrainingCore, D = window.RACE_DATA, TX = D.text, UI = TX.ui, TP = TX.templates;
  const { el, fill, css } = TrainingShared;
  const F = {};
  const K = Cards.create({ text: TX, formulas: F });
  TrainingShared.formulas(F);
  const ABORT = { abort: true }, UNIT = 0.5;  // UNIT: rem per tick in a machine box
  let S, cfg, centre, resets = 0;

  const online = () => S.mode === "online", learner = () => S.L[S.mode][S.active], hist = () => S.hist[S.mode];
  const refs = () => TrainingShared.refs(online());
  const setNarr = t => { cfg.narr.textContent = t; };
  const setMsg = t => { cfg.msg.textContent = t; };

  function newState() {
    cancel();
    const seed = 300 + resets++ * 11;
    S = { mode: S.mode || "offline", active: S.active || "reinforce", speed: S.speed || 1, runId: 0, playing: false, busy: false, ep: null, k: 0, shown: new Set(), blocks: [], last: null,
          L: { offline: { reinforce: C.createLearner("reinforce", seed), ac: C.createLearner("actor-critic", seed + 1) }, online: { reinforce: C.createLearner("reinforce", seed + 2), ac: C.createLearner("actor-critic", seed + 3) } },
          hist: { offline: { reinforce: [], ac: [] }, online: { reinforce: [], ac: [] } }, rng: { offline: C.makeRng(seed + 999), online: C.makeRng(seed + 1999) } };
  }
  function cancel() { if (S) S.runId++; if (cfg && cfg.fx) cfg.fx.textContent = ""; }
  const guard = id => { if (S.runId !== id) throw ABORT; };
  const sleep = async (ms, id) => { await new Promise(r => setTimeout(r, ms / S.speed)); guard(id); };

  // ---- flying tokens and pop-ups (an overlay above the page) ----------------------------------------------------------------------
  const point = x => { if (x.getBoundingClientRect) { const r = x.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; } return x; };
  async function fly(from, to, id, cls, text, ms) {
    const a = point(from), b = point(to), t = el("div", "token " + (cls || ""), text || "");
    cfg.fx.append(t);
    const at = p => `translate(${p.x}px, ${p.y}px) translate(-50%, -50%)`;
    const anim = t.animate([{ transform: at(a) + " scale(.5)", opacity: 0 }, { transform: at(a) + " scale(1)", opacity: 1, offset: 0.15 }, { transform: at(b) + " scale(1)", opacity: 1 }],
      { duration: ms / S.speed, easing: "ease-in-out", fill: "forwards" });
    try { await anim.finished; } catch (e) { /* cancelled */ }
    t.remove(); guard(id);
  }
  async function popup(anchor, text, cls, id, ms) {
    const p = point(anchor), t = el("div", "popup " + (cls || ""), text);
    t.style.left = p.x + "px"; t.style.top = p.y + "px"; cfg.fx.append(t);
    const anim = t.animate([{ transform: "translate(-50%, -50%) scale(.4)", opacity: 0 }, { transform: "translate(-50%, -120%) scale(1.1)", opacity: 1, offset: 0.3 }, { transform: "translate(-50%, -190%) scale(1)", opacity: 1 }],
      { duration: ms / S.speed, easing: "ease-out", fill: "forwards" });
    try { await anim.finished; } catch (e) { /* cancelled */ }
    guard(id); return t;
  }

  // ---- queue and machines ---------------------------------------------------------------------------------------------------------
  function qHeight(p) { return (1.15 + 0.42 * p) + "rem"; }
  function drawQueue(jobs, waiting, chosen, nextArrival) {
    cfg.queue.textContent = "";
    if (!waiting.length) cfg.queue.append(el("div", "qempty", UI["queue.empty"]));
    waiting.forEach(j => {
      const job = jobs[j], b = el("div", "qjob" + (j === chosen ? " pick" : "") + (S.shown.has(j) ? "" : " fresh"), "");
      b.dataset.job = j; b.style.height = qHeight(job.p);
      b.append(el("b", null, "J" + (j + 1)), el("span", null, "due " + job.d));
      cfg.queue.append(b); S.shown.add(j);
    });
    if (nextArrival != null) cfg.queue.append(el("div", "qnext", fill(UI["queue.next"], { t: nextArrival })));
  }
  function drawMachines(blocks, now, newBlock) {
    cfg.machines.forEach((box, m) => {
      box.textContent = "";
      blocks.filter(b => b.machine === m).forEach(b => {
        const e = el("div", "mblk" + (b.late ? " late" : "") + (b === newBlock ? " fresh" : ""), "J" + (b.job + 1));
        e.style.top = (b.start * UNIT) + "rem"; e.style.height = ((b.end - b.start) * UNIT) + "rem"; box.append(e);
      });
      if (now != null) { const l = el("div", "mnow"); l.style.top = (now * UNIT) + "rem"; box.append(l); }
    });
  }

  // ---- the network (SVG): inputs -> score -> softmax -> probabilities; critic branch below ------------------------------------------
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v)); if (text != null) e.textContent = text; return e; };
  const INPUTS = [["net.timeleft", 38], ["net.length", 86], ["net.importance", 134]], SCORE = { x: 140, y: 86 }, SOFT = { x: 208, y: 86 };
  function buildNet() {
    const g = cfg.net; g.textContent = "";
    const ac = S.active === "ac";
    g.append(svg("text", { x: 4, y: 12, class: "bracket" }, UI["net.actor"]));
    INPUTS.forEach(([key, y], i) => {
      g.append(svg("line", { id: "e" + i, x1: 54, y1: y, x2: SCORE.x - 16, y2: SCORE.y, class: "edge" }));
      g.append(svg("circle", { cx: 40, cy: y, r: 13, class: "node" }), svg("text", { x: 40, y: y + 4, class: "ntxt", "text-anchor": "middle" }, "x" + (i + 1)));
      g.append(svg("text", { x: 4, y: y + 26, class: "small" }, UI[key]));
      g.append(svg("text", { id: "w" + i, x: 88, y: y + (SCORE.y - y) * 0.45 - 5, class: "wtxt" }, ""));
    });
    g.append(svg("circle", { cx: SCORE.x, cy: SCORE.y, r: 16, class: "node score", id: "nscore" }), svg("text", { x: SCORE.x, y: SCORE.y + 4, class: "ntxt", "text-anchor": "middle" }, "z"));
    g.append(svg("text", { x: SCORE.x, y: SCORE.y + 32, class: "small", "text-anchor": "middle" }, UI["net.score"]));
    g.append(svg("line", { x1: SCORE.x + 16, y1: SCORE.y, x2: SOFT.x - 14, y2: SOFT.y, class: "edge thin" }));
    g.append(svg("rect", { x: SOFT.x - 14, y: SOFT.y - 14, width: 28, height: 28, rx: 6, class: "node soft", id: "nsoft" }), svg("text", { x: SOFT.x, y: SOFT.y + 4, class: "ntxt", "text-anchor": "middle" }, "π"));
    g.append(svg("text", { x: SOFT.x, y: SOFT.y + 32, class: "small", "text-anchor": "middle" }, UI["net.softmax"]));
    for (let r = 0; r < 8; r++) {  // probability bars for up to 8 waiting jobs
      g.append(svg("line", { x1: SOFT.x + 14, y1: SOFT.y, x2: 252, y2: 14 + r * 15, class: "edge faint", id: "o" + r }));
      g.append(svg("rect", { x: 254, y: 8 + r * 15, width: 0, height: 11, rx: 3, class: "pbar", id: "pb" + r }), svg("text", { x: 254, y: 17 + r * 15, class: "ptxt", id: "pt" + r }, ""));
    }
    // critic branch
    g.append(svg("text", { x: 4, y: 188, class: "bracket" }, ac ? UI["net.critic"] : UI["net.baseline"]));
    g.append(svg("rect", { x: 8, y: 198, width: 72, height: 30, rx: 6, class: "node", id: "csum" }), svg("text", { x: 44, y: 217, class: "ntxt", "text-anchor": "middle" }, ac ? "ψ(s)" : "t"));
    g.append(svg("line", { x1: 80, y1: 213, x2: 120, y2: 213, class: "edge thin", id: "cedge" }));
    g.append(svg("circle", { cx: 138, cy: 213, r: 17, class: "node value", id: "cval" }), svg("text", { x: 138, y: 217, class: "ntxt", "text-anchor": "middle", id: "cvaltxt" }, ac ? "V" : "b"));
    g.append(svg("text", { x: 164, y: 217, class: "small", id: "cnum" }, ""));
    g.append(svg("text", { x: 8, y: 244, class: "small" }, ac ? UI["net.value"] : ""));
    paintWeights();
  }
  function paintWeights() {
    const th = learner().theta;
    th.forEach((w, i) => {
      const e = cfg.net.querySelector("#e" + i); if (!e) return;
      e.style.strokeWidth = (1.5 + Math.min(7, Math.abs(w) * 0.9)).toFixed(1); e.style.stroke = w >= 0 ? css("--copied") : css("--scratch");
      cfg.net.querySelector("#w" + i).textContent = w.toFixed(1);
    });
  }
  function paintProbs(st, p, chosenK) {
    const rows = st ? st.state.waiting.length : 0;
    for (let r = 0; r < 8; r++) {
      const bar = cfg.net.querySelector("#pb" + r), txt = cfg.net.querySelector("#pt" + r), edge = cfg.net.querySelector("#o" + r);
      if (r < rows) {
        bar.setAttribute("width", Math.max(2, p[r] * 58)); bar.setAttribute("class", "pbar" + (r === chosenK ? " pick" : ""));
        txt.textContent = "J" + (st.state.waiting[r] + 1) + " " + Math.round(p[r] * 100) + "%"; txt.setAttribute("dx", 3); edge.setAttribute("class", "edge faint show");
      } else { bar.setAttribute("width", 0); txt.textContent = ""; edge.setAttribute("class", "edge faint"); }
    }
  }
  function paintCritic(st, t) {
    const L = learner(), ac = S.active === "ac";
    cfg.net.querySelector("#cnum").textContent = st ? (ac ? L.value(st.state) : L.baseline(t)).toFixed(1) : "";
  }
  async function pulse(x0, y0, x1, y1, id, ms) {
    const c = svg("circle", { r: 4, class: "pulse", cx: 0, cy: 0 }); cfg.net.append(c);
    const anim = c.animate([{ transform: `translate(${x0}px, ${y0}px)` }, { transform: `translate(${x1}px, ${y1}px)` }], { duration: ms / S.speed, easing: "ease-in-out", fill: "forwards" });
    try { await anim.finished; } catch (e) { /* cancelled */ }
    c.remove(); guard(id);
  }

  // ---- curve ------------------------------------------------------------------------------------------------------------------------
  function drawCurve() {
    const R = refs();
    TrainingShared.drawCurve(cfg.curve, { hist: hist(), active: S.active, refs: R, maxY: online() ? 100 : 200, ticks: online() ? [0, 25, 50, 75, 100] : [0, 50, 100, 150, 200],
      empty: UI["curve.empty"], labels: { random: UI["curve.random"], rule: fill(UI["curve.rule"], { name: R.rule.name }) }, colours: { reinforce: css("--copied"), ac: css("--win") } });
    cfg.episodes.textContent = fill(UI.episodes, { n: learner().episodes });
  }

  // ---- one decision as a short film --------------------------------------------------------------------------------------------------
  function startEpisode() {
    const L = learner(), jobs = C.makeInstance(S.rng[S.mode], online());
    S.ep = C.runEpisode(jobs, L.sample); S.k = 0; S.shown = new Set(); S.blocks = []; S.last = null;
    centre.reset(); drawQueue(jobs, [], null, null); drawMachines([], null, null); paintProbs(null); paintCritic(null);
  }
  async function playDecision(id) {
    centre.speed = S.speed;
    const ep = S.ep, k = S.k, st = ep.steps[k], jobs = st.state.jobs, L = learner(), p = L.probs(st.state), t = st.state.t, jobsLeft = ep.steps.length - k - 1;
    const chosen = st.job, pct = Math.round(p[st.k] * 100);
    // 1. state
    setNarr(UI["narr.state"]);
    const future = jobs.map((j, i) => i).filter(i => !st.state.waiting.includes(i) && !ep.steps.slice(0, k).some(s => s.job === i) && jobs[i].a > t).map(i => jobs[i].a);
    const nextArr = online() && future.length ? Math.min(...future) : null;
    drawQueue(jobs, st.state.waiting, null, nextArr);
    drawMachines(S.blocks, t, null);
    await centre.state({ k, n: ep.steps.length, t, m: st.state.machine + 1, waiting: st.state.waiting.length });
    await sleep(250, id); guard(id);
    // 2. state -> network
    setNarr(UI["narr.net"]);
    await fly(centre.out, cfg.net.querySelector("#e1"), id, "tk state", "s", 600);
    paintProbs(null); paintCritic(st, k);
    for (let i = 0; i < 3; i++) { const a = [54, INPUTS[i][1]]; pulse(a[0], a[1], SCORE.x - 16, SCORE.y, id, 450); }
    await sleep(500, id);
    await pulse(SCORE.x + 16, SCORE.y, SOFT.x - 14, SOFT.y, id, 350);
    paintProbs(st, p, st.k);
    await sleep(700, id);
    // 3. action back to the chain, the job flies to the machine
    setNarr(fill(UI["narr.action"], { job: chosen + 1, pct }));
    const bar = cfg.net.querySelector("#pb" + st.k);
    await fly(bar, centre.in, id, "tk action", "J" + (chosen + 1), 650);
    await centre.action({ job: chosen + 1, pct });
    drawQueue(jobs, st.state.waiting, chosen, nextArr);
    const qEl = cfg.queue.querySelector(`[data-job="${chosen}"]`);
    setNarr(fill(UI["narr.place"], { job: chosen + 1, m: st.state.machine + 1 }));
    const newBlock = { job: chosen, machine: st.state.machine, start: st.start, end: st.end, late: st.lateness > 0 };
    S.blocks.push(newBlock);
    const mBox = cfg.machines[st.state.machine];
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize), mb = mBox.getBoundingClientRect();
    await fly(qEl || cfg.queue, { x: mb.left + mb.width / 2, y: mb.top + ((st.start + st.end) / 2) * UNIT * rem }, id, "tk job", "J" + (chosen + 1), 650);
    drawQueue(jobs, st.state.waiting.filter(j => j !== chosen), null, nextArr); drawMachines(S.blocks, t, newBlock);
    await sleep(250, id);
    // 4. reward pops up, joins the chain
    const late = st.lateness > 0, rtxt = late ? st.reward.toFixed(1) : "0";
    setNarr(fill(UI["narr.reward"], { reward: rtxt, outcome: late ? fill(UI["narr.late"], { late: st.lateness }) : UI["narr.ontime"] }));
    const blkEl = mBox.querySelector(".mblk.fresh") || mBox;
    await popup(blkEl, rtxt, late ? "bad" : "good", id, 900);
    await centre.reward({ reward: rtxt, late, lateness: st.lateness, w: jobs[chosen].w });
    // 5. next state
    setNarr(UI["narr.next"]);
    const last = k + 1 >= ep.steps.length;
    await centre.next({ done: last, cost: ep.cost, t: last ? st.end : ep.steps[k + 1].state.t, n: last ? 0 : ep.steps[k + 1].state.waiting.length });
    S.k++;
    if (last) await finishEpisode(id);
  }
  async function finishEpisode(id) {
    const L = learner(), ep = S.ep;
    setNarr(fill(UI["narr.update"], { cost: ep.cost }));
    await fly(centre.in, cfg.net.querySelector("#e1"), id, "tk reward", "Σr", 700);
    L.update(ep); hist()[S.active].push(ep.cost); S.ep = null; S.tested = null;
    paintWeights(); drawCurve(); cfg.net.classList.add("flash"); setTimeout(() => cfg.net.classList.remove("flash"), 600);
    await sleep(500, id);
  }

  // ---- controls ---------------------------------------------------------------------------------------------------------------------
  const budgetLeft = () => learner().episodes < C.TRAIN_EPISODES;
  function setButtons() {
    cfg.b.play.textContent = S.playing ? UI["btn.pause"] : UI["btn.play"];
    cfg.b.step.disabled = S.playing || S.busy; cfg.b.train.disabled = S.playing || S.busy; cfg.b.test.disabled = S.playing || S.busy;
    cfg.modeBox.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.mode === S.mode)); b.disabled = S.playing || S.busy; });
    cfg.learnerBox.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.kind === S.active)); b.disabled = S.playing || S.busy; });
    cfg.speedBox.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(+b.dataset.speed === S.speed)));
    cfg.netTitle.textContent = S.active === "ac" ? UI["panel.network.ac"] : UI["panel.network.reinforce"];
  }
  async function run(one) {
    const id = ++S.runId; S.busy = true; setButtons();
    try {
      do {
        if (!S.ep) { if (!budgetLeft()) { setMsg(fill(TP["msg.budget"], { n: C.TRAIN_EPISODES })); break; } startEpisode(); }
        await playDecision(id);
      } while (!one && S.playing && budgetLeft());
    } catch (e) { if (e !== ABORT) throw e; }
    if (S.runId === id) { S.busy = false; S.playing = false; setButtons(); }
  }
  function doPlay() { if (S.playing) { S.playing = false; setButtons(); return; } S.playing = true; setButtons(); run(false); }
  function doStep() { if (!S.busy) run(true); }
  function doTrain() {  // no film: the learner trains fast, the curve and weights update, the chain shows the latest episode
    if (S.busy) return; const L = learner(); if (!budgetLeft()) { setMsg(fill(TP["msg.budget"], { n: C.TRAIN_EPISODES })); return; }
    const id = ++S.runId; S.busy = true; S.ep = null; setButtons(); let last = performance.now(), owed = 0;
    const tick = now => {
      if (S.runId !== id) return;
      owed += Math.min(now - last, 100) / 1000 * 30 * S.speed; last = now;
      let ep = null; while (owed >= 1 && budgetLeft()) { owed -= 1; ep = L.train(C.makeInstance(S.rng[S.mode], online())); hist()[S.active].push(ep.cost); }
      if (ep) { centre.snap(ep); drawQueue(ep.steps[0].state.jobs, [], null, null); drawMachines(ep.blocks, null, null); }
      paintWeights(); drawCurve(); setNarr("");
      if (!budgetLeft()) { S.busy = false; setButtons(); setMsg(fill(TP["msg.trained"], { n: L.episodes })); return; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  function doTest() {
    const R = refs(); setMsg(fill(TP["msg.testing"], { n: R.sets.length }));
    setTimeout(() => { const v = C.meanCost(R.sets, learner().greedy); setMsg(fill(UI["test.line"], { n: R.sets.length, random: R.random.toFixed(0), rule: R.rule.name, rulecost: R.rule.cost.toFixed(0), learner: v.toFixed(0) })); }, 30);
  }
  function resetAll() { newState(); redraw(); setMsg(""); setNarr(UI["narr.idle"]); setButtons(); }
  function redraw() {
    centre.reset(); drawQueue([], [], null, null); drawMachines([], null, null); buildNet(); paintProbs(null); paintCritic(null); drawCurve(); setButtons();
  }

  // ---- start-up ---------------------------------------------------------------------------------------------------------------------
  function init(c, centreModule) {
    cfg = c; S = {};
    const st = document.createElement("style"); st.textContent = centreModule.css; document.head.append(st);
    document.querySelectorAll("[data-ui]").forEach(e => { e.textContent = UI[e.dataset.ui]; });
    K.init(c.explain, () => ({}), id => K.topicCard(id, {}));
    centre = centreModule.create(c.centre);
    const mk = (label, fn, cls) => { const b = el("button", cls || "go", label); b.type = "button"; b.onclick = fn; return b; };
    cfg.b = { play: mk(UI["btn.play"], doPlay), step: mk(UI["btn.step"], doStep, "ghost"), train: mk(UI["btn.train"], doTrain, "ghost"), test: mk(UI["btn.test"], doTest, "ghost") };
    c.controls.append(cfg.b.play, cfg.b.step, cfg.b.train, cfg.b.test, mk(UI["btn.reset"], resetAll, "ghost"));
    cfg.speedBox = el("div", "seg small");
    [1, 3, 10].forEach(s => { const b = el("button", null, UI["speed." + s]); b.type = "button"; b.dataset.speed = s; b.onclick = () => { S.speed = s; setButtons(); }; cfg.speedBox.append(b); });
    c.controls.append(cfg.speedBox);
    cfg.learnerBox = el("div", "seg small");
    [["reinforce", "learner.reinforce"], ["ac", "learner.ac"]].forEach(([kind, key]) => {
      const b = el("button", null, UI[key]); b.type = "button"; b.dataset.kind = kind;
      b.onclick = () => { S.active = kind; S.ep = null; redraw(); setNarr(UI["narr.idle"]); setMsg(""); };
      cfg.learnerBox.append(b);
    });
    c.controls.append(cfg.learnerBox);
    cfg.modeBox = el("div", "seg small");
    ["offline", "online"].forEach(m => {
      const b = el("button", null, UI["mode." + m]); b.type = "button"; b.dataset.mode = m;
      b.onclick = () => { S.mode = m; S.ep = null; redraw(); setNarr(UI["narr.idle"]); setMsg(""); };
      cfg.modeBox.append(b);
    });
    c.modeSlot.append(cfg.modeBox);
    document.addEventListener("click", e => {
      const t = e.target.closest("[data-explain]");
      if (t && !t.closest("#explain")) K.open(t.dataset.explain === "learner" ? (S.active === "reinforce" ? "reinforce" : "actor_critic") : t.dataset.explain, {});
    });
    window.addEventListener("resize", drawCurve);
    S = { mode: "offline", active: "reinforce", speed: 1 };
    resetAll();
  }
  window.FlowUI = { init };
})();
