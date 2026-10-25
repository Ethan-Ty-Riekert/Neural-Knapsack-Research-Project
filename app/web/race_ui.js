/* race_ui.js - interaction for the browser demo: controls, state, race rendering and animation.
   A page template supplies the containers and CSS; RaceUI.init(cfg) wires everything. Uses RaceCore (race_core.js) and Explain
   (explain.js). Every visible sentence comes from app/web/wording.md (RACE_DATA.text). */
(function () {
  const C = window.RaceCore, D = C.data, TX = D.text, UI = TX.ui, LB = TX.labels;
  const exampleNames = Object.keys(D.examples);
  const state = { arrival: "online", busy: "Busy", tight: false, deadlines: "Medium", equal: false,
                  imp: Object.assign({ late: 0, delay: 0, wait: 0, flow: 0, energy: 0 }, D.examples[exampleNames[0]]) };
  let cfg, running = 0;
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const fill = (tpl, v) => tpl.replace(/\{(\w+)\}/g, (m, k) => (v[k] != null ? v[k] : m));
  const levelKey = v => (v === 0 ? 0 : v <= 2 ? 2 : 5);
  // second line of a racer: "[what it does] | [model]" (first and last part of the exported detail; capitalised)
  const shortDetail = d => { const p = d.split(" | "); const s = p.length > 2 ? p[0] + " | " + p[p.length - 1] : d; return s.charAt(0).toUpperCase() + s.slice(1); };

  // options: [[code key, label shown], ...]; value: the selected code key
  function seg(options, value, onPick, disabled) {
    const box = el("div", "seg"); box.setAttribute("role", "radiogroup");
    options.forEach(([k, label]) => {
      const b = el("button", null, label); b.type = "button"; b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(k === value)); if (disabled && disabled.includes(k)) b.disabled = true;
      b.onclick = () => onPick(k); box.appendChild(b);
    });
    return box;
  }
  const opts = (prefix, keys) => keys.map(k => [k, LB[prefix + k]]);
  function row(label, hint, control, topic) {  // topic: id of the explanation card opened by clicking the label
    const r = el("div", "row"), q = el("div", "q", label);
    if (topic) { q.dataset.explain = topic; q.classList.add("explainable"); q.title = UI.row_tooltip; }
    if (hint) q.appendChild(el("span", "hint", hint));
    r.append(q, control); return r;
  }

  function renderControls() {
    const sit = cfg.situation; sit.textContent = "";
    sit.appendChild(row(UI["row.arrival"], null, seg(opts("arrival.", ["online", "offline"]), state.arrival,
      k => { state.arrival = k; changed(); }), "arrival"));
    if (state.arrival === "online") {
      sit.appendChild(row(UI["row.machines"], LB["busyhint." + state.busy], seg(opts("busy.", Object.keys(C.BUSY)), state.busy,
        k => { state.busy = k; if (k !== "Normal") state.tight = false; changed(); }), "machines"));
      sit.appendChild(row(UI["row.deadlines"], state.busy === "Normal" ? null : UI["hint.tight_only_normal"],
        seg(opts("deadline.", ["Normal", "Tight"]), state.tight ? "Tight" : "Normal", k => { state.tight = k === "Tight"; changed(); },
          state.busy === "Normal" ? [] : ["Tight"]), "deadlines"));
    } else {
      sit.appendChild(row(UI["row.deadlines"], null, seg(opts("deadlines.", Object.keys(C.DEADLINES)), state.deadlines, k => { state.deadlines = k; changed(); }), "deadlines"));
      sit.appendChild(row(UI["row.jobs"], null, seg(opts("jobs.", ["equal", "unequal"]), state.equal ? "equal" : "unequal", k => { state.equal = k === "equal"; changed(); }), "importance_jobs"));
    }
    cfg.examples.textContent = "";
    exampleNames.forEach(name => {
      const b = el("button", "chip", name); b.type = "button";
      b.onclick = () => { Object.keys(state.imp).forEach(k => { state.imp[k] = D.examples[name][k] || 0; }); changed(); };
      cfg.examples.appendChild(b);
    });
    cfg.factors.textContent = "";
    D.factors.forEach(f => {
      const t = TX.factors[f.key];
      cfg.factors.appendChild(row(t.label, t.hint, seg([0, 2, 5].map(v => [v, LB["level." + v]]), levelKey(state.imp[f.key]),
        v => { state.imp[f.key] = v; changed(); }), "f_" + f.key));
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
      lane.innerHTML = cfg.laneHTML; lane.dataset.explain = "racer"; lane.dataset.idx = o.i; lane.title = UI.bar_tooltip;
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
    if (!result.winner) { cfg.verdict.textContent = UI.nothing; return; }
    const w = result.winner, kind = w.family === "H" ? UI["kind.rule"] : UI["kind.ai"];
    cfg.verdict.append(el("b", null, fill(UI.winner, { title: w.title })), el("span", null, " - " + kind));
    if (result.note) cfg.verdict.appendChild(el("div", "note", result.note));
  }

  function current() { return C.run(C.presetFor(state), state.imp); }
  let finished = false;  // a finished race stays on screen until step 1 or 2 changes
  function changed() { running++; finished = false; lastRes = null; window.Explain.close(); renderControls(); clearRace(UI.idle); }  // any change to steps 1 or 2 resets the race

  function finish(res) {
    const w = field.find(f => f.r === res.winner);
    if (w) w.lane.classList.add("win");
    verdict(res); finished = true;
  }
  function start() {
    finished = false;
    const res = current(), id = ++running;
    if (!res.racers.length) { clearRace(UI.nothing); return; }
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
    if (data && data.finished && res.racers.length) { build(res); update(1); finish(res); } else clearRace(UI.idle);
  }
  // Clicking anything marked data-explain opens its card; a racer card gets that racer's numbers, the others the current race context.
  function context(target) {
    const res = lastRes || current();
    if (target.dataset.explain === "racer") return { racer: res.racers[+target.dataset.idx], ctx: res.ctx };
    return res.ctx;
  }
  window.RaceUI = { init(c) {
    cfg = c;
    document.querySelectorAll("[data-ui]").forEach(e => { e.textContent = UI[e.dataset.ui]; });  // static texts of the page template
    window.Explain.init(c.explain, () => context({ dataset: {} }));
    c.verdict.dataset.explain = "winner"; c.verdict.title = UI.winner_tooltip;
    document.addEventListener("click", e => {
      const t = e.target.closest("[data-explain]");
      if (t && !t.closest(".seg") && !t.closest("#explain")) window.Explain.open(t.dataset.explain, context(t));
    });
    window.claude?.hot?.snapshot?.(() => ({ state: JSON.parse(JSON.stringify(state)), finished }));
    window.claude?.hot?.ready ? window.claude.hot.ready(boot) : boot(window.claude?.hot?.data ?? {});
  } };
})();
