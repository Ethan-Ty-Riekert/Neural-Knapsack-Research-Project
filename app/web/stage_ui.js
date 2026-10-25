/* stage_ui.js - "Watch the loop", stage layout (stage.html). The layout is the loop: the environment (job queue, two machine lanes) on the
   left, the learner (actor network with its probability distribution as bars or a wheel, and the critic or baseline) on the right, and
   three wires between them that light up as the state, the action and the reward travel. Under them a film strip of the episode's
   decisions and a learning row: the reward ledger, the same situation before and after learning, and the learning curve with the weights.
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
    S = { mode: S.mode || "offline", active: S.active || "reinforce", speed: S.speed || 1, view: S.view || "bars", runId: 0, playing: false, busy: false, ep: null, k: 0, shown: new Set(), blocks: [], rewards: [], deltas: [],
          L: { offline: { reinforce: C.createLearner("reinforce", seed), ac: C.createLearner("actor-critic", seed + 1) }, online: { reinforce: C.createLearner("reinforce", seed + 2), ac: C.createLearner("actor-critic", seed + 3) } },
          hist: { offline: { reinforce: [], ac: [] }, online: { reinforce: [], ac: [] } }, rng: { offline: C.makeRng(seed + 999), online: C.makeRng(seed + 1999) }, wheelRng: C.makeRng(seed + 7) };
  }
  function cancel() { if (S) S.runId++; if (cfg && cfg.fx) cfg.fx.textContent = ""; }
  const guard = id => { if (S.runId !== id) throw ABORT; };
  // wait for an animation, but never hang if the browser pauses animations (for example in a hidden tab): give up shortly after its duration
  const done = (anim, ms) => Promise.race([anim.finished.catch(() => {}), new Promise(r => setTimeout(r, ms + 150))]);
  const sleep = async (ms, id) => { await new Promise(r => setTimeout(r, ms / S.speed)); guard(id); };

  // ---- flying tokens, pop-ups, pulses -----------------------------------------------------------------------------------------------
  const point = x => { if (x.getBoundingClientRect) { const r = x.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; } return x; };
  async function fly(from, to, id, cls, text, ms) {
    const a = point(from), b = point(to), t = el("div", "token " + (cls || ""), text || "");
    cfg.fx.append(t);
    const at = p => `translate(${p.x}px, ${p.y}px) translate(-50%, -50%)`;
    const anim = t.animate([{ transform: at(a) + " scale(.5)", opacity: 0 }, { transform: at(a) + " scale(1)", opacity: 1, offset: 0.15 }, { transform: at(b) + " scale(1)", opacity: 1 }],
      { duration: ms / S.speed, easing: "ease-in-out", fill: "forwards" });
    await done(anim, ms / S.speed);
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
  function wire(name, on, bad) { Object.entries(cfg.wires).forEach(([k, e]) => { e.classList.toggle("on", on && k === name); e.classList.toggle("bad", !!bad); }); }

  // ---- environment: queue shelf, machine lanes -------------------------------------------------------------------------------------------
  function drawQueue(jobs, waiting, chosen, nextArrival) {
    cfg.queue.textContent = "";
    if (!waiting.length) cfg.queue.append(el("div", "qempty", UI["queue.empty"]));
    waiting.forEach(j => {
      const job = jobs[j], b = el("div", "qjob" + (j === chosen ? " pick" : "") + (S.shown.has(j) ? "" : " fresh"), "");
      b.dataset.job = j; b.style.minWidth = (2.6 + 0.55 * job.p) + "rem";
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
        e.style.left = (b.start / HORIZON * 100) + "%"; e.style.width = ((b.end - b.start) / HORIZON * 100) + "%"; lane.append(e);
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

  // ---- learner: the probability distribution (bars or wheel) ----------------------------------------------------------------------------
  function drawDist(st, p, chosenK) {
    cfg.dist.textContent = "";
    if (!st) { cfg.dist.append(el("div", "baempty", UI["dist.title"])); return; }
    const n = p.length;
    if (S.view === "wheel") {
      cfg.dist.append(el("div", "dtitle", UI["dist.title.wheel"]));
      const w = svg("svg", { id: "wheel", viewBox: "0 0 120 128" }), grp = svg("g", { id: "wg", style: "transform-origin: 60px 66px; transform-box: view-box" });
      let a0 = 0;
      p.forEach((pr, k) => {
        const a1 = a0 + pr * 2 * Math.PI, r = 54, cx = 60, cy = 66, big = pr > 0.5 ? 1 : 0;
        const x0 = cx + r * Math.sin(a0), y0 = cy - r * Math.cos(a0), x1 = cx + r * Math.sin(a1), y1 = cy - r * Math.cos(a1);
        grp.append(svg("path", { d: n === 1 ? `M${cx - r},${cy} a${r},${r} 0 1,0 ${2 * r},0 a${r},${r} 0 1,0 ${-2 * r},0` : `M${cx},${cy} L${x0},${y0} A${r},${r} 0 ${big} 1 ${x1},${y1} Z`, fill: COLOURS[st.state.waiting[k] % COLOURS.length], class: "slice" }));
        if (pr > 0.07) { const am = (a0 + a1) / 2; grp.append(svg("text", { x: cx + r * 0.62 * Math.sin(am), y: cy - r * 0.62 * Math.cos(am) + 3, class: "lbl", "text-anchor": "middle" }, "J" + (st.state.waiting[k] + 1) + " " + Math.round(pr * 100) + "%")); }
        a0 = a1;
      });
      w.append(grp, svg("path", { d: "M54,2 L66,2 L60,14 Z", class: "ptr" })); cfg.dist.append(w);
      S.mids = p.map((_, k) => { let s = 0; for (let i = 0; i < k; i++) s += p[i]; return { mid: (s + p[k] / 2) * 360, width: p[k] * 360 }; });
      return;
    }
    cfg.dist.append(el("div", "dtitle", UI["dist.title"]));
    p.forEach((pr, k) => {
      const row = el("div", "drow" + (k === chosenK ? " pick" : "")), bar = el("div", "dbar"), f = el("i"), mark = el("u");
      f.style.width = "0%"; mark.style.left = (100 / n) + "%"; bar.append(f, mark);
      row.append(el("span", null, "J" + (st.state.waiting[k] + 1)), bar, el("b", null, Math.round(pr * 100) + "%")); cfg.dist.append(row);
      setTimeout(() => { f.style.width = (pr * 100) + "%"; }, 20);
    });
    cfg.dist.append(el("div", "dlegend", "<s></s>".replace(/<s><\/s>/, "") + "| " + UI["dist.uniform"]));
  }
  async function spinWheel(chosenK, id) {
    const g = cfg.dist.querySelector("#wg"); if (!g || !S.mids) return;
    const m = S.mids[chosenK], jitter = (S.wheelRng() - 0.5) * m.width * 0.7, target = 360 * 3 - (m.mid + jitter);
    const anim = g.animate([{ transform: "rotate(0deg)" }, { transform: `rotate(${target}deg)` }], { duration: 1300 / S.speed, easing: "cubic-bezier(.15,.8,.2,1)", fill: "forwards" });
    await done(anim, 1300 / S.speed);
    guard(id);
  }

  // ---- learner: the critic (actor-critic) or the baseline (REINFORCE) ----------------------------------------------------------------------
  function buildCritic() {
    const box = cfg.critic; box.textContent = "";
    if (isAC()) {
      box.append(el("div", "dtitle", UI["net.critic"] + " · " + UI["net.summary"]));
      const grid = el("div", "cgrid"); for (let i = 0; i < 7; i++) { const c = el("div", "cchip"); c.dataset.i = i; c.append(el("span", null, UI["crit.f" + i]), el("b", null, "·"), el("em", null, "")); grid.append(c); }
      const out = el("div", "cout"), v = el("div", "cv"), dl = el("div", "cdelta");
      v.id = "cval"; v.append(el("span", null, "V(s)"), el("b", null, "")); out.append(v, el("div", null, UI["net.value"]), dl); dl.id = "cdelta"; box.append(grid, out);
    } else {
      box.append(el("div", "dtitle", UI["net.baseline"]));
      const bars = el("div", "bbars"); for (let t = 0; t < 8; t++) { const b = el("div", "bb"); b.dataset.t = t; b.append(el("b", null, ""), el("i"), el("span", null, fill(UI["base.step"], { k: t + 1 }))); bars.append(b); }
      box.append(bars);
    }
    paintCriticWeights();
  }
  function paintCriticWeights(before) {
    const L = learner();
    if (isAC()) {
      L.critic.forEach((w, i) => { const c = q(cfg.critic, `.cchip[data-i="${i}"]`); c.classList.toggle("pos", w >= 0.005); c.classList.toggle("neg", w <= -0.005); q(c, "em").textContent = before ? `×${plain(before[i], 2)}→${plain(w, 2)}` : `×${plain(w, 2)}`; });
    } else {
      const top = Math.max(2, ...Array.from({ length: 8 }, (_, t) => Math.abs(L.baseline(t))));
      for (let t = 0; t < 8; t++) { const b = q(cfg.critic, `.bb[data-t="${t}"]`), v = L.baseline(t); q(b, "i").style.height = (Math.abs(v) / top * 2.2) + "rem"; q(b, "b").textContent = plain(v, 1); }
    }
  }
  function paintCritic(st, k) {
    const L = learner();
    if (isAC()) {
      const f = st ? C.stateFeatures(st.state) : null;
      for (let i = 0; i < 7; i++) q(cfg.critic, `.cchip[data-i="${i}"] b`).textContent = f ? f[i].toFixed(2) : "·";
      q(cfg.critic, "#cval b").textContent = st ? plain(L.value(st.state), 1) : ""; q(cfg.critic, "#cdelta").textContent = "";
    } else for (let t = 0; t < 8; t++) q(cfg.critic, `.bb[data-t="${t}"]`).classList.toggle("cur", !!st && t === k);
  }
  async function criticPulses(id) {
    if (!isAC()) { await sleep(300, id); return; }
    const chips = [...cfg.critic.querySelectorAll(".cchip")]; chips.forEach(c => { c.style.transition = "background .25s"; c.style.background = "color-mix(in srgb, var(--accent) 14%, transparent)"; });
    await sleep(500, id); chips.forEach(c => { c.style.background = ""; });
  }
  async function criticCompare(st, k, ep, id) {
    if (!isAC()) return;
    const L = learner(), v = L.value(st.state), next = k + 1 < ep.steps.length ? L.value(ep.steps[k + 1].state) : 0, delta = st.reward + next - v;
    const d = q(cfg.critic, "#cdelta"); d.textContent = fill(UI["crit.got"], { r: plain(st.reward, 1), n: plain(next, 1), v: plain(v, 1) }) + " — " + fill(delta >= 0 ? UI["crit.better"] : UI["crit.worse"], { d: Math.abs(delta).toFixed(1) });
    d.style.color = delta >= 0 ? css("--win") : css("--scratch");
    const cv = q(cfg.critic, "#cval"); cv.style.borderColor = delta >= 0 ? css("--win") : css("--scratch"); cv.style.borderWidth = ".35rem";
    await sleep(800, id); cv.style.borderColor = ""; cv.style.borderWidth = "";
  }

  // ---- film strip, ledger ---------------------------------------------------------------------------------------------------------------
  function stripReset() {
    cfg.strip.textContent = "";
    for (let i = 0; i < 8; i++) { const s = el("div", "slot"); s.dataset.k = i; s.append(el("small", null, fill(UI["strip.slot"], { k: i + 1 })), el("b", null, "·"), el("span", "r", "")); cfg.strip.append(s); }
    const e = el("div", "slot end"); e.dataset.end = 1; e.append(el("small", null, "■"), el("b", null, "")); cfg.strip.append(e);
  }
  function stripCur(k) { cfg.strip.querySelectorAll(".slot").forEach(s => s.classList.toggle("cur", +s.dataset.k === k && !s.dataset.end)); }
  function stripFill(k, job, reward, late) {
    const s = q(cfg.strip, `.slot[data-k="${k}"]`); q(s, "b").textContent = "J" + (job + 1); const r = q(s, ".r"); r.textContent = late ? plain(reward, 1) : "0";
    s.classList.add(late ? "bad" : "good", "pop"); setTimeout(() => s.classList.remove("pop"), 500);
  }
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

  // ---- before / after: the same situation with the old and the new probabilities ----------------------------------------------------------------
  const probsWith = (theta, st) => C.softmax(st.state.waiting.map(j => theta.reduce((s, w, i) => s + w * C.features(st.state.jobs[j], st.state.t)[i], 0)));
  function baReset() { cfg.ba.textContent = ""; cfg.ba.append(el("div", "baempty", UI["ba.empty"])); }
  function baShow(ep, adv, thetaBefore, thetaAfter) {
    let best = -1, bk = -1; ep.steps.forEach((st, k) => { if (st.state.waiting.length > 1 && Math.abs(adv[k]) > best) { best = Math.abs(adv[k]); bk = k; } });
    cfg.ba.textContent = ""; if (bk < 0) { baReset(); return; }
    const st = ep.steps[bk], p0 = probsWith(thetaBefore, st), p1 = probsWith(thetaAfter, st), box = el("div", "babox");
    [[UI["ba.before"], p0], [UI["ba.after"], p1]].forEach(([title, p]) => {
      const col = el("div", "bacol"); col.append(el("div", "dtitle", title));
      p.forEach((pr, k) => {
        const row = el("div", "drow" + (k === st.k ? " pick" : "")), bar = el("div", "dbar"), f = el("i"), mark = el("u"); f.style.width = (pr * 100) + "%"; mark.style.left = (100 / p.length) + "%"; bar.append(f, mark);
        row.append(el("span", null, "J" + (st.state.waiting[k] + 1)), bar, el("b", null, Math.round(pr * 100) + "%")); col.append(row);
      });
      box.append(col);
    });
    const a0 = Math.round(p0[st.k] * 100), a1 = Math.round(p1[st.k] * 100);
    cfg.ba.append(box, el("div", "baline", fill(UI["ba.line"], { job: st.job + 1, credit: num(adv[bk], 1), p0: a0, p1: a1 }) + " " + (p1[st.k] >= p0[st.k] ? UI["ba.up"] : UI["ba.down"])));
  }

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
    stripReset(); ledgerReset(); drawQueue(jobs, [], null, null); drawMachines([], null, null); drawDist(null); paintCritic(null);
  }
  async function playDecision(id) {
    const ep = S.ep, k = S.k, st = ep.steps[k], jobs = st.state.jobs, L = learner(), p = L.probs(st.state), t = st.state.t, chosen = st.job, pct = Math.round(p[st.k] * 100);
    // 1. the state leaves the environment
    stripCur(k); setNarr(UI["narr.state"]);
    const future = jobs.map((j, i) => i).filter(i => !st.state.waiting.includes(i) && !ep.steps.slice(0, k).some(s => s.job === i) && jobs[i].a > t).map(i => jobs[i].a);
    const nextArr = online() && future.length ? Math.min(...future) : null;
    drawQueue(jobs, st.state.waiting, null, nextArr); drawMachines(S.blocks, t, null);
    wire("state", true); await sleep(250, id);
    await fly(cfg.queue, q(cfg.actor, "#ae1"), id, "tk state", "s", 650);
    // 2. the actor scores the waiting jobs and turns the scores into probabilities; the critic estimates the situation
    setNarr(UI["narr.net"]);
    drawDist(null); paintCritic(st, k);
    await actorPulses(id);
    drawDist(st, p, st.k);
    if (S.view === "wheel") await spinWheel(st.k, id); else await sleep(600, id);
    await criticPulses(id);
    // 3. the action goes back to the environment; the job moves from the queue onto a machine
    wire("action", true); setNarr(fill(UI["narr.action"], { job: chosen + 1, pct }));
    drawQueue(jobs, st.state.waiting, chosen, nextArr);
    const qEl = cfg.queue.querySelector(`[data-job="${chosen}"]`), from = S.view === "wheel" ? q(cfg.dist, "#wheel") : (q(cfg.dist, ".drow.pick") || cfg.dist);
    await fly(from, qEl || cfg.queue, id, "tk action", "J" + (chosen + 1), 650);
    setNarr(fill(UI["narr.place"], { job: chosen + 1, m: st.state.machine + 1 }));
    const newBlock = { job: chosen, machine: st.state.machine, start: st.start, end: st.end, late: st.lateness > 0 }; S.blocks.push(newBlock);
    const lane = cfg.machines[st.state.machine], lr = lane.getBoundingClientRect();
    await fly(qEl || cfg.queue, { x: lr.left + lr.width * ((st.start + st.end) / 2) / HORIZON, y: lr.top + lr.height / 2 }, id, "tk job", "J" + (chosen + 1), 650);
    drawQueue(jobs, st.state.waiting.filter(j => j !== chosen), null, nextArr); drawMachines(S.blocks, t, newBlock);
    await sleep(200, id);
    // 4. the reward comes back: pops up, travels along the reward wire, drops into the ledger and the strip
    const late = st.lateness > 0, rtxt = late ? plain(st.reward, 1) : "0";
    wire("reward", true, late); setNarr(fill(UI["narr.reward"], { reward: rtxt, outcome: late ? fill(UI["narr.late"], { late: st.lateness }) : UI["narr.ontime"] }));
    const blkEl = lane.querySelector(".mblk.fresh") || lane;
    await popup(blkEl, rtxt, late ? "bad" : "good", id, 800);
    const chip = ledgerAdd(k, st.reward); stripFill(k, chosen, st.reward, late);
    await fly(blkEl, chip, id, late ? "reward" : "good", rtxt, 700);
    chip.classList.add("pulse"); setTimeout(() => chip.classList.remove("pulse"), 500);
    if (isAC()) { setNarr(UI["narr.compare"]); await fly(chip, q(cfg.critic, "#cval"), id, "reward", "r", 650); await criticCompare(st, k, ep, id); }
    // 5. the next state
    wire("state", false); setNarr(UI["narr.next"]);
    S.k++;
    if (S.k >= ep.steps.length) await finishEpisode(id); else await sleep(200, id);
  }
  async function finishEpisode(id) {
    const L = learner(), ep = S.ep, thetaBefore = L.theta.slice(), criticBefore = isAC() ? L.critic.slice() : null, adv = advantages(ep);
    const endSlot = q(cfg.strip, ".slot.end"); endSlot.classList.add("cur"); q(endSlot, "b").textContent = fill(UI["strip.cost"], { cost: ep.cost });
    setNarr(fill(UI["narr.update"], { cost: ep.cost })); wire("reward", true);
    ledgerCredit(adv); await sleep(900, id);
    const chips = [...cfg.ledger.querySelectorAll(".lchip.got")];
    chips.forEach(c => fly(c, cfg.ba, id, "reward", q(c, "em").textContent, 800).catch(() => {}));
    await sleep(900, id);
    L.update(ep); hist()[S.active].push(ep.cost); S.ep = null;
    setNarr(UI["narr.learn"]);
    baShow(ep, adv, thetaBefore, L.theta.slice()); paintWeights(thetaBefore); paintCriticWeights(criticBefore); drawCurve();
    await sleep(2600, id);
    paintWeights(); paintCriticWeights(); paintCritic(null); wire("reward", false);
  }

  // ---- controls ------------------------------------------------------------------------------------------------------------------------------
  const budgetLeft = () => learner().episodes < C.TRAIN_EPISODES;
  function setButtons() {
    cfg.b.play.textContent = S.playing ? UI["btn.stop"] : UI["btn.play"];
    ["step", "train", "test"].forEach(k => { cfg.b[k].disabled = S.playing || S.busy; });
    cfg.modeBox.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.mode === S.mode)); b.disabled = S.playing || S.busy; });
    cfg.learnerBox.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.kind === S.active)); b.disabled = S.playing || S.busy; });
    cfg.speedBox.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(+b.dataset.speed === S.speed)));
    cfg.viewSeg.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.view === S.view)));
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
      paintWeights(); paintCriticWeights(); drawCurve(); setButtons(); setNarr("");
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
    stripReset(); ledgerReset(); baReset(); drawQueue([], [], null, null); drawMachines([], null, null); buildActor(); buildCritic(); drawDist(null); paintCritic(null); drawCurve(); wire("none", false); setButtons();
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
    cfg.viewSeg = seg("small", [["bars", UI["view.bars"]], ["wheel", UI["view.wheel"]]], "view", v => { S.view = v; drawDist(null); setButtons(); });
    c.controls.append(cfg.b.play, cfg.b.step, cfg.b.train, cfg.b.test, mk(UI["btn.reset"], resetAll, "ghost"), cfg.speedBox, cfg.learnerBox, cfg.episodes);
    c.modeSlot.append(cfg.modeBox); c.viewBox.append(cfg.viewSeg);
    document.addEventListener("click", e => {
      const t = e.target.closest("[data-explain]");
      if (t && !t.closest("#explain")) K.open(t.dataset.explain === "learner" ? (S.active === "reinforce" ? "reinforce" : "actor_critic") : t.dataset.explain, {});
    });
    window.addEventListener("resize", drawCurve);
    S = { mode: "offline", active: "reinforce", speed: 1, view: "bars" };
    resetAll();
  }
  window.StageUI = { init };
})();
