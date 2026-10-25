/* stage_ui.js - "Watch the loop", stage layout (stage.html). The layout is the loop: the environment (job queue, two machine lanes) on the
   left, the learner (actor network with its probability distribution as a vertical bar chart) on the right, and
   a small State, Action, Reward, Next state loop between them that lights up as the data travels. Under them a learning row: the reward
   ledger of the episode, and the learning curve with the weights.
   Play keeps stepping through episode after episode at the chosen speed (x30 is a fast "spam the Step button") until Stop.
   Every sentence comes from training_wording.md; the world and learners from training_core.js. */
(function () {
  const C = window.TrainingCore, D = window.RACE_DATA, TX = D.text, UI = TX.ui, TP = TX.templates;
  const { el, fill, css } = TrainingShared;
  const F = {};
  const K = Cards.create({ text: TX, formulas: F });
  TrainingShared.formulas(F);
  const ABORT = { abort: true }, HORIZON = 20, LAMBDA = 0.9, COLOURS = ["#2a6fd1", "#14866d", "#8793a1", "#e0702a", "#7a5bd1", "#c2a02b", "#4aa3b5", "#b05080"];
  let S, cfg;
  let resets = 0;

  const online = () => S.mode === "online", learner = () => S.L[S.mode][S.active], hist = () => S.hist[S.mode], isAC = () => S.active === "ac";
  const refs = () => TrainingShared.refs(online());
  const setNarr = t => { cfg.narr.textContent = t; };
  const setMsg = t => { cfg.msg.textContent = t; };
  const plain = (x, d = 1) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
  const num = (x, d = 1) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
  const q = (root, sel) => root.querySelector(sel);

  function newState() {
    cancel();
    const seed = 500 + resets++ * 13;
    S = { mode: S.mode || "offline", active: S.active || "reinforce", speed: S.speed || 1, runId: 0, playing: false, busy: false, ep: null, k: 0, shown: new Set(), blocks: [], rewards: [], deltas: [],
          L: { offline: { reinforce: C.createLearner("reinforce", seed), ac: C.createLearner("actor-critic", seed + 1) }, online: { reinforce: C.createLearner("reinforce", seed + 2), ac: C.createLearner("actor-critic", seed + 3) } },
          hist: { offline: { reinforce: [], ac: [] }, online: { reinforce: [], ac: [] } }, rng: { offline: C.makeRng(seed + 999), online: C.makeRng(seed + 1999) } };
  }
  function cancel() { if (S) S.runId++; if (cfg && cfg.fx) cfg.fx.textContent = ""; }
  const guard = id => { if (S.runId !== id) throw ABORT; };
  // wait for an animation, but never hang if the browser pauses animations (for example in a hidden tab): give up shortly after its duration
  const done = (anim, ms) => Promise.race([anim.finished.catch(() => {}), new Promise(r => setTimeout(r, ms + 150))]);
  const sleep = async (ms, id) => { await new Promise(r => setTimeout(r, ms / S.speed)); guard(id); };
  // at x1 the film is calm: one orb at a time travels slowly and the part it reaches is highlighted and held; from x3 up it moves faster
  const calm = () => S.speed === 1;
  const hold = (calmMs, fastMs, id) => sleep(calm() ? calmMs : fastMs, id);
  function spot(...els) { document.querySelectorAll(".spot").forEach(e => e.classList.remove("spot")); els.forEach(e => { const b = e && e.closest(".lbox, .panel"); if (b) b.classList.add("spot"); }); }

  // ---- flying tokens, pop-ups, pulses -----------------------------------------------------------------------------------------------
  const point = x => { if (x.getBoundingClientRect) { const r = x.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; } return x; };
  async function fly(from, to, id, cls, text, ms) {
    const a = point(from), b = point(to), t = el("div", "token " + (cls || ""), text || "");
    cfg.fx.append(t);
    const at = p => `translate(${p.x}px, ${p.y}px) translate(-50%, -50%)`;
    const anim = t.animate([{ transform: at(a) + " scale(.5)", opacity: 0 }, { transform: at(a) + " scale(1)", opacity: 1, offset: 0.15 }, { transform: at(b) + " scale(1)", opacity: 1 }],
      { duration: ms * (calm() ? 1.4 : 1) / S.speed, easing: "ease-in-out", fill: "forwards" });
    await done(anim, ms * (calm() ? 1.4 : 1) / S.speed);
    t.remove(); guard(id);
  }
  async function popup(anchor, text, cls, id, ms) {
    const p = point(anchor), t = el("div", "popup " + (cls || ""), text);
    t.style.left = p.x + "px"; t.style.top = p.y + "px"; cfg.fx.append(t);
    const anim = t.animate([{ transform: "translate(-50%, -50%) scale(.4)", opacity: 0 }, { transform: "translate(-50%, -120%) scale(1.1)", opacity: 1, offset: 0.3 }, { transform: "translate(-50%, -190%) scale(1)", opacity: 1 }],
      { duration: ms / S.speed, easing: "ease-out", fill: "forwards" });
    await done(anim, ms / S.speed);
    t.remove(); guard(id);
  }
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v)); if (text != null) e.textContent = text; return e; };
  async function pulse(root, x0, y0, x1, y1, id, ms) {
    const c = svg("circle", { r: 4, class: "pulse", cx: 0, cy: 0 }); root.append(c);
    const anim = c.animate([{ transform: `translate(${x0}px, ${y0}px)` }, { transform: `translate(${x1}px, ${y1}px)` }], { duration: ms / S.speed, easing: "ease-in-out", fill: "forwards" });
    await done(anim, ms / S.speed);
    c.remove(); guard(id);
  }
  // ---- the loop: four connected boxes (State, Action, Reward, Next state); one is lit at a time, together with the arrow into it ---------------
  const BOXES = ["state", "action", "reward", "next"], byId = id => document.getElementById(id);
  function buildLoop() {}
  function wire(name, on, bad) {
    BOXES.forEach(k => {
      const lit = on && k === name, b = byId("b-" + k); b.classList.toggle("on", lit); b.classList.toggle("bad", lit && !!bad); byId("a-" + k).classList.toggle("on", lit);
    });
  }
  const loopVal = (k, text) => { const e = byId("v-" + k); if (e) e.textContent = text; };
  // box 1 shows the state itself: the time and, for every waiting job, how much time it has left
  function loopState(t, st) {
    const box = byId("v-state"); box.textContent = ""; box.append(el("div", "lbig", fill(UI["loop.v.state"], { t, n: st.state.waiting.length })));
    st.state.waiting.forEach(j => { const r = el("div", "sjob"); r.style.borderLeftColor = jobColour(j); r.append(el("b", null, "J" + (j + 1)), el("span", null, UI["net.timeleft"] + " " + Math.max(0, st.state.jobs[j].d - t))); box.append(r); });
  }
  const loopCentre = a => { byId("lstep").textContent = a || ""; };

  // ---- environment: queue shelf, machine lanes -------------------------------------------------------------------------------------------
  const jobColour = j => COLOURS[j % COLOURS.length];
  function drawQueue(jobs, waiting, chosen, nextArrival) {
    cfg.queue.textContent = "";
    if (!waiting.length) cfg.queue.append(el("div", "qempty", UI["queue.empty"]));
    waiting.forEach(j => {
      const job = jobs[j], b = el("div", "qjob" + (j === chosen ? " pick" : "") + (S.shown.has(j) ? "" : " fresh"), "");
      b.dataset.job = j; b.style.borderLeft = "6px solid " + jobColour(j); b.style.minWidth = (2.4 + 0.3 * job.p) + "rem";
      b.append(el("b", null, "J" + (j + 1)), el("span", null, "due " + job.d));
      cfg.queue.append(b); S.shown.add(j);
    });
    if (nextArrival != null) cfg.queue.append(el("div", "qnext", fill(UI["queue.next"], { t: nextArrival })));
  }
  function drawMachines(blocks, now, newBlock) {
    cfg.machines.forEach((lane, m) => {
      lane.textContent = "";
      blocks.filter(b => b.machine === m).forEach(b => {
        const e = el("div", "mblk" + (b.late ? " late" : "") + (b === newBlock ? " fresh" : ""), "J" + (b.job + 1));
        e.style.background = b.late ? "" : jobColour(b.job); e.style.left = (b.start / HORIZON * 100) + "%"; e.style.width = ((b.end - b.start) / HORIZON * 100) + "%"; lane.append(e);
      });
      if (now != null) { const l = el("div", "mnow"); l.style.left = (now / HORIZON * 100) + "%"; lane.append(l); }
    });
  }

  // ---- learner: the actor diagram ----------------------------------------------------------------------------------------------------
  const IN = [["net.timeleft", 30], ["net.length", 75], ["net.importance", 120]], Z = { x: 112, y: 75 }, PI = { x: 182, y: 75 };
  function buildActor() {
    const g = cfg.actor; g.textContent = "";
    IN.forEach(([key, y], i) => {
      g.append(svg("line", { id: "ae" + i, x1: 40, y1: y, x2: Z.x - 15, y2: Z.y, class: "edge" }));
      g.append(svg("circle", { cx: 26, cy: y, r: 12, class: "node" }), svg("text", { x: 26, y: y + 4, class: "ntxt", "text-anchor": "middle" }, "x" + (i + 1)));
      g.append(svg("text", { x: 4, y: y + 24, class: "small" }, UI[key]), svg("text", { id: "aw" + i, x: 56, y: y + (Z.y - y) * 0.4 - 4, class: "wtxt" }, ""));
    });
    g.append(svg("circle", { cx: Z.x, cy: Z.y, r: 15, class: "node soft" }), svg("text", { x: Z.x, y: Z.y + 4, class: "ntxt", "text-anchor": "middle" }, "z"), svg("text", { x: Z.x, y: Z.y + 28, class: "small", "text-anchor": "middle" }, UI["net.score"]));
    g.append(svg("line", { x1: Z.x + 15, y1: Z.y, x2: PI.x - 14, y2: PI.y, class: "edge" }));
    g.append(svg("rect", { x: PI.x - 14, y: PI.y - 14, width: 28, height: 28, rx: 6, class: "node soft" }), svg("text", { x: PI.x, y: PI.y + 4, class: "ntxt", "text-anchor": "middle" }, "π"), svg("text", { x: PI.x, y: PI.y + 28, class: "small", "text-anchor": "middle" }, UI["net.softmax"]));
    paintWeights();
  }
  function paintWeights(before) {
    const th = learner().theta;
    th.forEach((w, i) => {
      const e = q(cfg.actor, "#ae" + i); if (!e) return;
      e.style.strokeWidth = (1.5 + Math.min(7, Math.abs(w) * 0.9)).toFixed(1); e.style.stroke = w >= 0 ? css("--copied") : css("--scratch");
      q(cfg.actor, "#aw" + i).textContent = before ? `${plain(before[i], 2)}→${plain(w, 2)}` : plain(w, 2);
    });
    cfg.wbars.textContent = "";
    [["weights.timeleft", 0], ["weights.length", 1], ["weights.importance", 2]].forEach(([key, i]) => {
      const d = before ? th[i] - before[i] : 0, row = el("div", "wrow" + (before && Math.abs(d) > 0.005 ? (d > 0 ? " up" : " down") : "")), bar = el("div", "wbar"), f = el("i");
      const frac = Math.max(-1, Math.min(1, th[i] / 8)); f.style.left = frac >= 0 ? "50%" : (50 + frac * 50) + "%"; f.style.width = Math.abs(frac) * 50 + "%";
      bar.append(f, el("u")); row.append(el("span", null, UI[key]), bar, el("span", "wv", plain(th[i], 2) + (before && Math.abs(d) > 0.005 ? (d > 0 ? " ▲" : " ▼") : "")));
      cfg.wbars.append(row);
    });
  }
  async function actorPulses(id) {
    for (let i = 0; i < 3; i++) pulse(cfg.actor, 40, IN[i][1], Z.x - 15, Z.y, id, 450);
    await sleep(500, id); await pulse(cfg.actor, Z.x + 15, Z.y, PI.x - 14, PI.y, id, 350);
  }

  // ---- learner: the probability distribution (vertical bars) ----------------------------------------------------------------------------
  function drawDist(st, p, chosenK) {
    cfg.dist.textContent = "";
    if (!st) { cfg.dist.append(el("div", "baempty", UI["dist.title"])); return; }
    const n = p.length;
    cfg.dist.append(el("div", "dtitle", UI["dist.title"]));
    const W = 170, H = 165, x0 = 4, x1 = W - 4, yb = 140, yt = 20, top = Math.max(0.4, Math.max(...p) * 1.15), slot = (x1 - x0) / n, bw = Math.min(26, slot * 0.7);
    const yOf = v => yb - (v / top) * (yb - yt), cx = k => x0 + slot * (k + 0.5), g = svg("svg", { id: "vbars", viewBox: `0 0 ${W} ${H}` });
    g.append(svg("line", { x1: x0, y1: yb, x2: x1, y2: yb, class: "vaxis" }), svg("line", { x1: x0, y1: yOf(1 / n), x2: x1, y2: yOf(1 / n), class: "vchance" }));
    const bars = [];
    p.forEach((pr, k) => {
      const j = st.state.waiting[k], r = svg("rect", { x: cx(k) - bw / 2, y: yb, width: bw, height: 0, rx: 3, fill: jobColour(j), class: "vbar" + (k === chosenK ? " pick" : "") });
      g.append(r, svg("text", { x: cx(k), y: yb + 14, "text-anchor": "middle", class: "vlbl" }, "J" + (j + 1)));
      const pc = svg("text", { x: cx(k), y: yb, "text-anchor": "middle", class: "vpct" + (k === chosenK ? " pick" : ""), opacity: 0 }, Math.round(pr * 100) + "%"); g.append(pc); bars.push([r, pc, pr, k]);
    });
    const line = svg("polyline", { points: p.map((pr, k) => `${cx(k)},${yOf(pr)}`).join(" "), class: "vline", opacity: 0 }), dots = p.map((pr, k) => svg("circle", { cx: cx(k), cy: yOf(pr), r: 3, class: "vdot", opacity: 0 }));
    g.append(line, ...dots); cfg.dist.append(g, el("div", "dlegend", "- - - " + UI["dist.uniform"]));
    setTimeout(() => {
      bars.forEach(([r, pc, pr]) => { r.setAttribute("y", yOf(pr)); r.setAttribute("height", yb - yOf(pr)); pc.setAttribute("y", yOf(pr) - 5); });
    }, 20);
    setTimeout(() => { bars.forEach(([, pc]) => pc.setAttribute("opacity", 1)); line.setAttribute("opacity", 1); dots.forEach(d => d.setAttribute("opacity", 1)); }, 520 / S.speed);
  }
  // ---- learner: the critic (actor-critic) or the baseline (REINFORCE) ----------------------------------------------------------------------
  function buildCritic() { cfg.critic.textContent = ""; }
  const paintCriticWeights = () => {};
  const paintCritic = () => { if (cfg.critic) cfg.critic.textContent = ""; };
  async function criticPulses(id) { await sleep(300, id); }
  async function criticCompare(st, k, ep, id) {  // actor-critic only: the critic's guess against what happened
    if (!isAC()) return;
    const L = learner(), v = L.value(st.state), next = k + 1 < ep.steps.length ? L.value(ep.steps[k + 1].state) : 0, delta = st.reward + next - v;
    cfg.critic.textContent = fill(UI["crit.got"], { r: plain(st.reward, 1), n: plain(next, 1), v: plain(v, 1) }) + " — " + fill(delta >= 0 ? UI["crit.better"] : UI["crit.worse"], { d: Math.abs(delta).toFixed(1) });
    cfg.critic.style.color = delta >= 0 ? css("--win") : css("--scratch");
    await sleep(900, id);
  }

  // ---- ledger ---------------------------------------------------------------------------------------------------------------
  function ledgerReset() {
    cfg.ledger.textContent = ""; S.rewards = [];
    const row = el("div", "lrow"); for (let i = 0; i < 8; i++) { const c = el("div", "lchip empty"); c.dataset.k = i; c.append(el("small", null, String(i + 1)), el("b", null, "·"), el("em", null, "")); row.append(c); }
    cfg.ledger.append(row, el("div", "lret", UI["ledger.empty"]));
  }
  function ledgerAdd(k, reward) {
    S.rewards[k] = reward;
    const c = q(cfg.ledger, `.lchip[data-k="${k}"]`); c.classList.remove("empty"); c.classList.add("got", reward < 0 ? "neg" : "zero"); q(c, "b").textContent = plain(reward, 1);
    q(cfg.ledger, ".lret").textContent = fill(UI["ledger.return"], { r: plain(S.rewards.reduce((s, x) => s + (x || 0), 0), 1) });
    return c;
  }
  function advantages(ep) {  // the credit each decision gets (how much better than expected), from the learner as it is before the update
    const L = learner(), r = ep.steps.map(s => s.reward), n = r.length, adv = new Array(n);
    if (isAC()) { const vs = ep.steps.map(s => L.value(s.state)); let a = 0; for (let t = n - 1; t >= 0; t--) { a = r[t] + (t + 1 < n ? vs[t + 1] : 0) - vs[t] + LAMBDA * a; adv[t] = a; } }
    else { let g = 0; for (let t = n - 1; t >= 0; t--) { g += r[t]; adv[t] = g - L.baseline(t); } }
    return adv;
  }
  function ledgerCredit(adv) { adv.forEach((a, t) => { const c = q(cfg.ledger, `.lchip[data-k="${t}"]`); if (c) { c.classList.add(a >= 0 ? "up" : "down"); q(c, "em").textContent = num(a, 1); } }); }

  // ---- curve ---------------------------------------------------------------------------------------------------------------------------------
  function drawCurve() {
    const R = refs();
    TrainingShared.drawCurve(cfg.curve, { hist: hist(), active: S.active, refs: R, maxY: online() ? 100 : 200, ticks: online() ? [0, 50, 100] : [0, 100, 200],
      empty: UI["curve.empty"], labels: { random: UI["curve.random"], rule: fill(UI["curve.rule"], { name: R.rule.name }) }, colours: { reinforce: css("--copied"), ac: css("--win") } });
  }

  // ---- one decision as a short film ------------------------------------------------------------------------------------------------------------
  function startEpisode() {
    const L = learner(), jobs = C.makeInstance(S.rng[S.mode], online());
    S.ep = C.runEpisode(jobs, L.sample); S.k = 0; S.shown = new Set(); S.blocks = [];
    ledgerReset(); drawQueue(jobs, [], null, null); drawMachines([], null, null); drawDist(null); paintCritic(null);
  }
    async function playDecision(id) {
    const ep = S.ep, k = S.k, st = ep.steps[k], jobs = st.state.jobs, L = learner(), p = L.probs(st.state), t = st.state.t, chosen = st.job, pct = Math.round(p[st.k] * 100), envBox = byId("env");
    // 1. State: the situation leaves the environment
    setNarr(UI["narr.state"]); loopCentre(fill(UI["loop.step"], { k: k + 1, n: ep.steps.length }));
    ["state", "reward", "next"].forEach(n => loopVal(n, ""));
    const future = jobs.map((j, i) => i).filter(i => !st.state.waiting.includes(i) && !ep.steps.slice(0, k).some(s => s.job === i) && jobs[i].a > t).map(i => jobs[i].a);
    const nextArr = online() && future.length ? Math.min(...future) : null;
    drawQueue(jobs, st.state.waiting, null, nextArr); drawMachines(S.blocks, t, null);
    wire("state", true); loopState(t, st); spot(envBox);
    await hold(800, 250, id);
    await fly(cfg.queue, q(cfg.actor, "#ae1"), id, "tk state", "s", 650);
    // 2. the actor scores the waiting jobs and turns the scores into probabilities; the critic estimates the situation
    setNarr(UI["narr.net"]); spot(cfg.actor);
    drawDist(null); paintCritic(st, k);
    await actorPulses(id);
    drawDist(st, p, st.k);
    await sleep(600, id);
    await hold(700, 0, id);
    // 3. Action: the job leaves the queue and starts on a machine
    wire("action", true); loopVal("action", fill(UI["loop.v.action"], { job: chosen + 1 })); setNarr(fill(UI["narr.action"], { job: chosen + 1, pct }));
    drawQueue(jobs, st.state.waiting, chosen, nextArr); spot(cfg.queue);
    const qEl = cfg.queue.querySelector(`[data-job="${chosen}"]`), from = q(cfg.dist, ".vbar.pick") || cfg.dist;
    await fly(from, qEl || cfg.queue, id, "tk action", "J" + (chosen + 1), 650);
    await hold(700, 0, id);
    setNarr(fill(UI["narr.place"], { job: chosen + 1, m: st.state.machine + 1 }));
    const newBlock = { job: chosen, machine: st.state.machine, start: st.start, end: st.end, late: st.lateness > 0 }; S.blocks.push(newBlock);
    const lane = cfg.machines[st.state.machine], lr = lane.getBoundingClientRect();
    await fly(qEl || cfg.queue, { x: lr.left + lr.width * ((st.start + st.end) / 2) / HORIZON, y: lr.top + lr.height / 2 }, id, "tk job", "J" + (chosen + 1), 650);
    drawQueue(jobs, st.state.waiting.filter(j => j !== chosen), null, nextArr); drawMachines(S.blocks, t, newBlock); spot(lane);
    await hold(700, 200, id);
    // 4. Reward: it appears, is written into the ledger, and (actor-critic) the critic compares it with its guess
    const late = st.lateness > 0, rtxt = late ? plain(st.reward, 1) : "0";
    wire("reward", true, late); loopVal("reward", rtxt); setNarr(fill(UI["narr.reward"], { reward: rtxt, outcome: late ? fill(UI["narr.late"], { late: st.lateness }) : UI["narr.ontime"] }));
    const blkEl = lane.querySelector(".mblk.fresh") || lane;
    await popup(blkEl, rtxt, late ? "bad" : "good", id, 800);
    const chip = ledgerAdd(k, st.reward); spot(cfg.ledger);
    await fly(blkEl, chip, id, late ? "reward" : "good", rtxt, 700);
    chip.classList.add("pulse"); setTimeout(() => chip.classList.remove("pulse"), 500);
    await hold(800, 0, id);
    if (isAC()) { setNarr(UI["narr.compare"]); spot(cfg.critic); await fly(chip, cfg.critic, id, "reward", "r", 650); await criticCompare(st, k, ep, id); await hold(600, 0, id); }
    // 5. Next state
    const nxt = ep.steps[k + 1]; wire("next", true); loopVal("next", nxt ? fill(UI["loop.v.next"], { t: nxt.state.t }) : "end"); setNarr(UI["narr.next"]); spot();
    await hold(500, 200, id);
    S.k++;
    if (S.k >= ep.steps.length) await finishEpisode(id);
  }
  async function finishEpisode(id) {
    const L = learner(), ep = S.ep, thetaBefore = L.theta.slice(), adv = advantages(ep);
    setNarr(fill(UI["narr.update"], { cost: ep.cost })); wire("reward", true); loopCentre(fill(UI["loop.done"], { cost: ep.cost }));
    ledgerCredit(adv); spot(cfg.ledger); await hold(1600, 900, id);
    const chips = [...cfg.ledger.querySelectorAll(".lchip.got")];
    chips.forEach(c => fly(c, cfg.actor, id, "reward", q(c, "em").textContent, 800).catch(() => {}));
    await sleep(900, id);
    L.update(ep); hist()[S.active].push(ep.cost); S.ep = null;
    setNarr(UI["narr.learn"]); spot(cfg.actor, cfg.wbars);
    paintWeights(thetaBefore); drawCurve();
    await hold(4200, 2600, id);
    spot(); paintWeights(); paintCritic(null); wire("none", false); loopCentre("");
  }

  // ---- controls ------------------------------------------------------------------------------------------------------------------------------
  const budgetLeft = () => learner().episodes < C.TRAIN_EPISODES;
  function setButtons() {
    cfg.b.play.textContent = S.playing ? UI["btn.stop"] : UI["btn.play"];
    ["step", "train", "test"].forEach(k => { cfg.b[k].disabled = S.playing || S.busy; });
    cfg.modeBox.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.mode === S.mode)); b.disabled = S.playing || S.busy; });
    cfg.learnerBox.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.kind === S.active)); b.disabled = S.playing || S.busy; });
    cfg.speedBox.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(+b.dataset.speed === S.speed)));
    cfg.agentTitle.textContent = isAC() ? UI["panel.network.ac"] : UI["panel.network.reinforce"];
    cfg.episodes.textContent = fill(UI.episodes, { n: learner().episodes });
  }
  async function run(one) {
    const id = ++S.runId; S.busy = true; setButtons();
    try {
      do {
        if (!S.ep) { if (!budgetLeft()) { setMsg(fill(TP["msg.budget"], { n: C.TRAIN_EPISODES })); break; } startEpisode(); }
        await playDecision(id);
        setButtons();
      } while (!one && S.playing && budgetLeft());
    } catch (e) { if (e !== ABORT) throw e; }
    if (S.runId === id) { S.busy = false; S.playing = false; setButtons(); }
  }
  function doPlay() { if (S.playing) { S.playing = false; setButtons(); return; } S.playing = true; setButtons(); run(false); }
  function doStep() { if (!S.busy) run(true); }
  function doTrain() {  // no film: fast training for when the learner should just get on with it
    if (S.busy) return; const L = learner(); if (!budgetLeft()) { setMsg(fill(TP["msg.budget"], { n: C.TRAIN_EPISODES })); return; }
    const id = ++S.runId; S.busy = true; S.ep = null; setButtons(); let last = performance.now(), owed = 0;
    const tick = now => {
      if (S.runId !== id) return;
      owed += Math.min(now - last, 100) / 1000 * 30; last = now;
      let ep = null; while (owed >= 1 && budgetLeft()) { owed -= 1; ep = L.train(C.makeInstance(S.rng[S.mode], online())); hist()[S.active].push(ep.cost); }
      if (ep) { drawQueue(ep.steps[0].state.jobs, [], null, null); drawMachines(ep.blocks, null, null); }
      paintWeights(); drawCurve(); setButtons(); setNarr("");
      if (!budgetLeft()) { S.busy = false; setButtons(); setMsg(fill(TP["msg.trained"], { n: L.episodes })); return; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  function doTest() {
    const R = refs(); setMsg(fill(TP["msg.testing"], { n: R.sets.length }));
    setTimeout(() => { const v = C.meanCost(R.sets, learner().greedy); setMsg(fill(UI["test.line"], { n: R.sets.length, random: R.random.toFixed(0), rule: R.rule.name, rulecost: R.rule.cost.toFixed(0), learner: v.toFixed(0) })); }, 30);
  }
  function redraw() {
    ledgerReset(); drawQueue([], [], null, null); drawMachines([], null, null); buildLoop(); buildActor(); buildCritic(); drawDist(null); paintCritic(null); drawCurve(); wire("none", false); loopCentre(""); setButtons();
  }
  function resetAll() { newState(); redraw(); setMsg(""); setNarr(UI["narr.idle"]); }

  function init(c) {
    cfg = c; S = {};
    document.querySelectorAll("[data-ui]").forEach(e => { e.textContent = UI[e.dataset.ui]; });
    K.init(c.explain, () => ({}), id => K.topicCard(id, {}));
    const mk = (label, fn, cls) => { const b = el("button", cls || "go", label); b.type = "button"; b.onclick = fn; return b; };
    cfg.b = { play: mk(UI["btn.play"], doPlay), step: mk(UI["btn.step"], doStep, "ghost"), train: mk(UI["btn.train"], doTrain, "ghost"), test: mk(UI["btn.test"], doTest, "ghost") };
    cfg.episodes = el("span", "sub");
    const seg = (cls, items, attr, onPick) => { const box = el("div", "seg " + cls); items.forEach(([val, label]) => { const b = el("button", null, label); b.type = "button"; b.dataset[attr] = val; b.onclick = () => onPick(val); box.append(b); }); return box; };
    cfg.speedBox = seg("small", [1, 3, 10, 30].map(s => [s, UI["speed." + s]]), "speed", s => { S.speed = +s; setButtons(); });
    cfg.learnerBox = seg("small", [["reinforce", UI["learner.reinforce"]], ["ac", UI["learner.ac"]]], "kind", kind => { S.active = kind; S.ep = null; redraw(); setNarr(UI["narr.idle"]); setMsg(""); });
    cfg.modeBox = seg("small", [["offline", UI["mode.offline"]], ["online", UI["mode.online"]]], "mode", m => { S.mode = m; S.ep = null; redraw(); setNarr(UI["narr.idle"]); setMsg(""); });
    c.controls.append(cfg.b.play, cfg.b.step, cfg.b.train, cfg.b.test, mk(UI["btn.reset"], resetAll, "ghost"), cfg.speedBox, cfg.learnerBox, cfg.episodes);
    c.modeSlot.append(cfg.modeBox);
    document.addEventListener("click", e => {
      const t = e.target.closest("[data-explain]");
      if (t && !t.closest("#explain")) K.open(t.dataset.explain === "learner" ? (S.active === "reinforce" ? "reinforce" : "actor_critic") : t.dataset.explain, {});
    });
    window.addEventListener("resize", drawCurve);
    S = { mode: "offline", active: "reinforce", speed: 1 };
    resetAll();
  }
  window.StageUI = { init };
})();
