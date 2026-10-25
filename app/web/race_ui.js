/* race_ui.js - shared interaction for the browser demo variants: controls, state, race rendering and animation.
   A variant supplies the containers and CSS; RaceUI.init(cfg) wires everything. Uses RaceCore (race_core.js). */
(function () {
  const C = window.RaceCore, D = C.data;
  const state = { arrival: "online", busy: "Busy", tight: false, deadlines: "Medium", equal: false,
                  imp: Object.assign({ late: 0, delay: 0, wait: 0, flow: 0, energy: 0 }, D.examples["Never be late"]) };
  let cfg, running = 0;
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const levelName = v => (v === 0 ? "Ignore" : v <= 2 ? "Some" : "A lot");
  // second line of a racer: "[what it does] | [model]" (first and last part of the exported detail; capitalised)
  const shortDetail = d => { const p = d.split(" | "); const s = p.length > 2 ? p[0] + " | " + p[p.length - 1] : d; return s.charAt(0).toUpperCase() + s.slice(1); };

  function seg(options, value, onPick, disabled) {
    const box = el("div", "seg"); box.setAttribute("role", "radiogroup");
    options.forEach(o => {
      const b = el("button", null, o); b.type = "button"; b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(o === value)); if (disabled && disabled.includes(o)) b.disabled = true;
      b.onclick = () => onPick(o); box.appendChild(b);
    });
    return box;
  }
  function row(label, hint, control, topic) {  // topic: id of the explanation card opened by clicking the label
    const r = el("div", "row"), q = el("div", "q", label);
    if (hint) q.appendChild(el("span", "hint", hint));
    if (topic) { q.dataset.explain = topic; q.classList.add("explainable"); q.title = "Click for the maths"; q.appendChild(el("span", "info", " ⓘ")); }
    r.append(q, control); return r;
  }

  function renderControls() {
    const sit = cfg.situation; sit.textContent = "";
    sit.appendChild(row("Jobs arrive", null, seg(["Over time", "Known in advance"], state.arrival === "online" ? "Over time" : "Known in advance",
      o => { state.arrival = o === "Over time" ? "online" : "offline"; changed(); }), "arrival"));
    if (state.arrival === "online") {
      const hints = { Quiet: "about half used", Normal: "about 3/4 used", Busy: "nearly full", Overloaded: "more work than capacity" };
      sit.appendChild(row("Machines", hints[state.busy], seg(Object.keys(C.BUSY), state.busy, o => { state.busy = o; if (o !== "Normal") state.tight = false; changed(); }), "machines"));
      sit.appendChild(row("Deadlines", state.busy === "Normal" ? null : "tight only tested at Normal",
        seg(["Normal", "Tight"], state.tight ? "Tight" : "Normal", o => { state.tight = o === "Tight"; changed(); }, state.busy === "Normal" ? [] : ["Tight"]), "deadlines"));
    } else {
      sit.appendChild(row("Deadlines", null, seg(Object.keys(C.DEADLINES), state.deadlines, o => { state.deadlines = o; changed(); }), "deadlines"));
      sit.appendChild(row("Jobs", null, seg(["All equal", "Some matter more"], state.equal ? "All equal" : "Some matter more", o => { state.equal = o === "All equal"; changed(); }), "importance_jobs"));
    }
    cfg.examples.textContent = "";
    Object.keys(D.examples).forEach(name => {
      const b = el("button", "chip", name); b.type = "button";
      b.onclick = () => { Object.keys(state.imp).forEach(k => { state.imp[k] = D.examples[name][k] || 0; }); changed(); };
      cfg.examples.appendChild(b);
    });
    cfg.factors.textContent = "";
    D.factors.forEach(f => {
      cfg.factors.appendChild(row(f.question.split(" (")[0], (f.question.match(/\((.*)\)/) || [])[1] || null,
        seg(["Ignore", "Some", "A lot"], levelName(state.imp[f.key]), o => { state.imp[f.key] = C.LEVELS[o]; changed(); }), "f_" + f.key));
    });
  }

  // Lanes are built once per race and then moved: each lane is positioned by its current rank (leader on top) and slides
  // (CSS transition on transform) when another racer passes it. The starting order is random.
  let field = [], lastRes = null;
  function build(result) {
    lastRes = result;
    cfg.lanes.textContent = ""; cfg.lanes.style.setProperty("--n", result.racers.length);
    const order = result.racers.map((r, i) => ({ r, i, shuffle: Math.random() })).sort((a, b) => a.shuffle - b.shuffle);
    field = order.map((o, slot) => {
      const lane = el("div", "lane k-" + (o.r.family === "H" ? "rule" : o.r.kind));
      lane.innerHTML = cfg.laneHTML; lane.dataset.explain = "racer"; lane.dataset.idx = o.i; lane.title = "Click for the exact score and the maths";
      lane.querySelector(".t").textContent = o.r.title; lane.querySelector(".d").textContent = shortDetail(o.r.detail);
      lane.style.transform = "translateY(" + slot * 100 + "%)";
      cfg.lanes.appendChild(lane);
      return { r: o.r, lane, slot, p: 1.4 + Math.random() * 2.2 };  // p: this racer's own speed curve (higher = faster start)
    });
  }
  function progress(f, t) { return 1 - Math.pow(1 - t, f.p); }  // 0 at start, 1 at the finish for every racer
  function update(t) {
    const now = field.map(f => ({ f, w: f.r.percent * progress(f, t) }));
    now.forEach(x => { x.f.lane.querySelector(".pct").textContent = Math.round(x.w) + "%"; cfg.place(x.f.lane, x.w); });
    now.slice().sort((a, b) => b.w - a.w || a.f.slot - b.f.slot).forEach((x, rank) => { x.f.lane.style.transform = "translateY(" + rank * 100 + "%)"; });
  }
  function clearRace(msg) {
    field = []; cfg.lanes.textContent = ""; cfg.verdict.textContent = "";
    cfg.lanes.appendChild(el("div", "idle", msg));
  }

  function verdict(result) {
    cfg.verdict.textContent = "";
    if (!result.winner) { cfg.verdict.textContent = "Nothing to compare for this combination - choose something that matters."; return; }
    const w = result.winner, kind = w.family === "H" ? "classic rule" : "AI scheduler";
    cfg.verdict.append(el("b", null, "Winner: " + w.title), el("span", null, " - " + kind));
    if (result.note) cfg.verdict.appendChild(el("div", "note", result.note));
  }

  function current() { return C.run(C.presetFor(state), state.imp); }
  const IDLE = "Choose your situation and what matters, then press Start the race.";
  let finished = false;  // a finished race stays on screen until step 1 or 2 changes
  function changed() { running++; finished = false; lastRes = null; window.Explain.close(); renderControls(); clearRace(IDLE); }  // any change to steps 1 or 2 resets the race

  function finish(res) {
    const w = field.find(f => f.r === res.winner);
    if (w) w.lane.classList.add("win");
    verdict(res); finished = true;
  }
  function start() {
    finished = false;
    const res = current(), id = ++running;
    if (!res.racers.length) { clearRace("Nothing to compare for this combination - choose something that matters."); return; }
    build(res); cfg.verdict.textContent = "";
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { update(1); finish(res); return; }
    const t0 = performance.now(), dur = 4200;
    update(0);
    (function frame(now) {
      if (id !== running) return;
      const t = Math.min((now - t0) / dur, 1); update(t);
      if (t < 1) requestAnimationFrame(frame); else finish(res);
    })(t0);
  }

  // Boot (or re-boot after the page is updated in place): restore the choices and, if a race had finished, show its final result.
  function boot(data) {
    if (data && data.state) { Object.assign(state, data.state); }
    renderControls(); cfg.start.onclick = start;
    const res = current();
    if (data && data.finished && res.racers.length) { build(res); update(1); finish(res); } else clearRace(IDLE);
  }
  // Clicking anything marked data-explain opens its card; a racer card gets that racer's numbers, the others the current race context.
  function context(target) {
    const res = lastRes || current();
    if (target.dataset.explain === "racer") return { racer: res.racers[+target.dataset.idx], ctx: res.ctx };
    return res.ctx;
  }
  window.RaceUI = { init(c) {
    cfg = c;
    window.Explain.init(c.explain, () => context({ dataset: {} }));
    c.verdict.dataset.explain = "winner"; c.verdict.title = "Click to see how to read this";
    document.addEventListener("click", e => {
      const t = e.target.closest("[data-explain]");
      if (t && !t.closest(".seg") && !t.closest("#explain")) window.Explain.open(t.dataset.explain, context(t));
    });
    window.claude?.hot?.snapshot?.(() => ({ state: JSON.parse(JSON.stringify(state)), finished }));
    window.claude?.hot?.ready ? window.claude.hot.ready(boot) : boot(window.claude?.hot?.data ?? {});
  } };
})();
