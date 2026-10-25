/* race_ui.js - shared interaction for the browser demo variants: controls, state, race rendering and animation.
   A variant supplies the containers and CSS; RaceUI.init(cfg) wires everything. Uses RaceCore (race_core.js). */
(function () {
  const C = window.RaceCore, D = C.data;
  const state = { arrival: "online", busy: "Busy", tight: false, deadlines: "Medium", equal: false,
                  imp: Object.assign({ late: 0, delay: 0, wait: 0, flow: 0, energy: 0 }, D.examples["Never be late"]) };
  let cfg, running = 0;
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const levelName = v => (v === 0 ? "Ignore" : v <= 2 ? "Some" : "A lot");

  function seg(options, value, onPick, disabled) {
    const box = el("div", "seg"); box.setAttribute("role", "radiogroup");
    options.forEach(o => {
      const b = el("button", null, o); b.type = "button"; b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(o === value)); if (disabled && disabled.includes(o)) b.disabled = true;
      b.onclick = () => onPick(o); box.appendChild(b);
    });
    return box;
  }
  function row(label, hint, control) {
    const r = el("div", "row"), q = el("div", "q", label);
    if (hint) q.appendChild(el("span", "hint", hint));
    r.append(q, control); return r;
  }

  function renderControls() {
    const sit = cfg.situation; sit.textContent = "";
    sit.appendChild(row("Jobs arrive", null, seg(["Over time", "Known in advance"], state.arrival === "online" ? "Over time" : "Known in advance",
      o => { state.arrival = o === "Over time" ? "online" : "offline"; changed(); })));
    if (state.arrival === "online") {
      const hints = { Quiet: "about half used", Normal: "about 3/4 used", Busy: "nearly full", Overloaded: "more work than capacity" };
      sit.appendChild(row("Machines", hints[state.busy], seg(Object.keys(C.BUSY), state.busy, o => { state.busy = o; if (o !== "Normal") state.tight = false; changed(); })));
      sit.appendChild(row("Deadlines", state.busy === "Normal" ? null : "tight only tested at Normal",
        seg(["Normal", "Tight"], state.tight ? "Tight" : "Normal", o => { state.tight = o === "Tight"; changed(); }, state.busy === "Normal" ? [] : ["Tight"])));
    } else {
      sit.appendChild(row("Deadlines", null, seg(Object.keys(C.DEADLINES), state.deadlines, o => { state.deadlines = o; changed(); })));
      sit.appendChild(row("Jobs", null, seg(["All equal", "Some matter more"], state.equal ? "All equal" : "Some matter more", o => { state.equal = o === "All equal"; changed(); })));
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
        seg(["Ignore", "Some", "A lot"], levelName(state.imp[f.key]), o => { state.imp[f.key] = C.LEVELS[o]; changed(); })));
    });
  }

  function lanes(result, fraction) {
    cfg.lanes.textContent = "";
    result.racers.forEach(r => {
      const lane = el("div", "lane k-" + (r.family === "H" ? "rule" : r.kind) + (r === result.winner ? " win" : ""));
      lane.innerHTML = cfg.laneHTML;
      lane.querySelector(".t").textContent = r.title; lane.querySelector(".d").textContent = r.detail;
      const w = r.percent * fraction;
      lane.querySelector(".pct").textContent = Math.round(w) + "%";
      cfg.place(lane, w);
      cfg.lanes.appendChild(lane);
    });
  }

  function verdict(result) {
    cfg.verdict.textContent = "";
    if (!result.winner) { cfg.verdict.textContent = "Nothing to compare for this combination - choose something that matters."; return; }
    const w = result.winner, kind = w.family === "H" ? "classic rule" : "AI scheduler";
    cfg.verdict.append(el("b", null, "Winner: " + w.title), el("span", null, " - " + kind));
    if (result.note) cfg.verdict.appendChild(el("div", "note", result.note));
  }

  function current() { return C.run(C.presetFor(state), state.imp); }
  function changed() { running++; renderControls(); const res = current(); lanes(res, 1); verdict(res); }

  function start() {
    const res = current(), id = ++running; verdict(res);
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { lanes(res, 1); return; }
    const t0 = performance.now(), dur = 3200;
    (function frame(now) {
      if (id !== running) return;
      const t = Math.min((now - t0) / dur, 1); lanes(res, 1 - Math.pow(1 - t, 2));
      if (t < 1) requestAnimationFrame(frame);
    })(t0);
  }

  window.RaceUI = { init(c) { cfg = c; renderControls(); c.start.onclick = start; const res = current(); lanes(res, 1); verdict(res); } };
})();
