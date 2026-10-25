/* explain.js - the "click anything to see the maths" cards of the browser demo.
   All sentences come from app/web/wording.md (exported as RACE_DATA.text by tools/analysis/make_demo_artifacts.py); this file holds
   only the formulas (MathML, rendered by the browser, no library) and the logic that fills the numbers in. Every formula is copied
   from the project code or docs; the wording file names the source in each card. */
(function () {
  const D = window.RACE_DATA, TX = D.text, TP = TX.templates, LB = TX.labels;

  // ---- markup in the wording file: **bold** *italic* `code` a_{j} x^{2} and {{formula}} slots ------------------------------
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function inline(s) {
    s = esc(s);
    s = s.replace(/\^\{([^}]*)\}/g, (m, x) => "<sup>" + x.replace(/\*/g, "&#42;") + "</sup>");
    s = s.replace(/([A-Za-zλρσφℓπθγΦΔĈ])_\{([^}]*)\}/g, "<i>$1</i><sub>$2</sub>");
    return s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\*([^*]+)\*/g, "<i>$1</i>").replace(/`([^`]+)`/g, "<code>$1</code>");
  }
  let slots = {};
  const slot = n => (slots[n] != null ? slots[n] : F[n] != null ? F[n] : `<span class="missing">[unknown slot ${n}]</span>`);
  function md(text) {
    return text.split(/\n\s*\n/).map(p => {
      p = p.trim(); if (!p) return "";
      const only = p.match(/^\{\{(\w+)\}\}$/); if (only) return slot(only[1]);
      return "<p>" + inline(p).replace(/\{\{(\w+)\}\}/g, (_, n) => slot(n)) + "</p>";
    }).join("");
  }
  const fill = (tpl, v) => tpl.replace(/\{(\w+)\}/g, (m, k) => (v[k] != null ? v[k] : m));

  // ---- MathML helpers --------------------------------------------------------------------------------------------------
  const I = s => `<mi>${s}</mi>`, N = s => `<mn>${s}</mn>`, O = s => `<mo>${s}</mo>`, T = s => `<mtext>${s}</mtext>`;
  const R = (...x) => `<mrow>${x.join("")}</mrow>`;
  const sub = (a, b) => `<msub>${a}${b}</msub>`, sup = (a, b) => `<msup>${a}${b}</msup>`, frac = (a, b) => `<mfrac>${a}${b}</mfrac>`;
  const sum = (u, b) => R(`<munder>${O("∑")}${u}</munder>`, b);
  const ind = (...x) => R(`<mi mathvariant="double-struck">1</mi>`, O("["), ...x, O("]"));
  const M = (...x) => `<math display="block">${x.join("")}</math>`;
  const wj = sub(I("w"), I("j")), pj = sub(I("p"), I("j")), dj = sub(I("d"), I("j")), aj = sub(I("a"), I("j")),
        Cj = sub(I("C"), I("j")), sj = sub(I("s"), I("j")), Tj = sub(I("T"), I("j")), tt = I("t");
  const pos = x => R(O("("), x, sub(O(")"), O("+")));                     // (x)_+  = max(0, x)
  const maxOf = (...x) => R(I("max"), O("("), x.join(O(",")), O(")"));
  const note = k => T("   " + TP[k]);                                      // short annotation after a formula
  const slack = R(dj, O("−"), pj, O("−"), tt);
  const key = sub(I("key"), I("j")), prio = sub(I("priority"), I("j"));

  // ---- formulas (filled into the wording with {{name}}) ------------------------------------------------------------------
  const F = {
    tard: M(Tj, O("="), maxOf(N(0), R(Cj, O("−"), dj))),
    J: M(I("J"), O("="), sum(I("j"), R(wj, sup(Tj, N(2))))),
    late: M(sub(I("N"), T("late")), O("="), sum(I("j"), ind(Tj, O(">"), N(0)))),
    wait: M(sub(I("W"), T("mean")), O("="), frac(N(1), I("n")), sum(I("j"), R(O("("), sj, O("−"), aj, O(")")))),
    flow: M(sub(I("F"), T("mean")), O("="), frac(N(1), I("n")), sum(I("j"), R(O("("), Cj, O("−"), aj, O(")")))),
    energy: M(I("E"), O("="), sum(R(I("t"), O(","), I("m")), ind(T(TP["math.energy_cond"])))),
    rho: M(I("ρ"), O("="), sub(I("max"), I("r")), frac(R(I("λ"), I("E"), O("["), pj, sub(I("u"), R(I("j"), I("r"))), O("]")), R(I("M"), sub(I("C"), I("r"))))),
    dueOff: M(dj, O("∼"), I("U"), O("["), R(I("L"), O("("), N("1"), O("−"), T("TF"), O("−"), frac(T("RDD"), N(2)), O(")")), O(","), R(I("L"), O("("), N("1"), O("−"), T("TF"), O("+"), frac(T("RDD"), N(2)), O(")")), O("]")),
    dueOn: M(dj, O("="), aj, O("+"), I("U"), O("["), T(TP["math.slack_range"]), O("]")),
    score: M(I("S"), O("="), sum(I("m"), frac(R(sub(I("λ"), I("m")), sub(I("x"), I("m"))), sub(I("x"), R(I("m"), O(","), T("ref")))))),
    pct: M(T("bar"), O("="), N("100"), O("·"), I("exp"), O("("), O("−"), N("4"), O("·"), frac(R(I("S"), O("−"), sup(I("S"), O("*"))), sup(I("S"), O("*"))), O(")")),
  };
  const phi = atc => M(sub(I("φ"), I("j")), O("="), O("("), pj, O(","), dj, O(","), wj, O(","), sub(I("u"), R(I("j"), N("1"))), O(","), O("…"), O(","), sub(I("u"), R(I("j"), I("R"))), O(","), T("scheduled"), atc ? R(O(","), sub(I("I"), R(I("j"), T("ATC")))) : "", O(")"));
  const policy = atc => phi(atc) +
    M(sub(I("h"), I("j")), O("="), sub(I("W"), N(2)), T(" ReLU"), O("("), sub(I("W"), N(1)), sub(I("φ"), I("j")), O("+"), sub(I("b"), N(1)), O(")"), O("+"), sub(I("b"), N(2)), note("math.policy.shared")) +
    M(sub(I("z"), I("j")), O("="), sup(I("v"), O("⊤")), T(" tanh"), O("("), I("V"), sub(I("h"), I("j")), O("+"), I("c"), O(")"), O("+"), sub(I("c"), N(0))) +
    M(I("π"), O("("), I("j"), O("|"), I("s"), O(")"), O("="), frac(R(I("exp"), O("("), sub(I("z"), I("j")), O(")")), R(sum(I("k"), R(I("exp"), O("("), sub(I("z"), I("k")), O(")"))))), note("math.policy.over")) + md(TP["math.policy.note"]);
  F.policy = policy(false); F.policy_atc = policy(true);
  F.opt1 = M(sub(I("a"), I("t")), O("∈"), O("{"), O("("), I("ρ"), O(","), I("π"), O(")"), O(":"), I("ρ"), O("∈"), T("rules"), O(","), I("π"), O("∈"), T("placements"), O("}"), O("∪"), O("{"), T("idle"), O("}")) +
    M(sub(I("j"), O("*")), O("="), sub(I("argmin"), I("j")), sub(I("key"), I("ρ")), O("("), I("j"), O(")"), note("math.opt1.machine")) + md(TP["math.opt1.note"]);
  F.opt0 = M(sub(I("a"), I("t")), O("∈"), O("{"), O("("), I("j"), O(","), I("m"), O(")"), O("}"), O("∪"), O("{"), T("idle"), O("}")) +
    M(I("π"), O("("), I("j"), O(","), I("m"), O("|"), I("s"), O(")"), O("∝"), I("exp"), O("("), sub(I("z"), R(I("j"), I("m"))), O(")"), note("math.opt0.z")) + md(TP["math.opt0.note"]);
  F.ppo = M(sup(I("L"), T("CLIP")), O("("), I("θ"), O(")"), O("="), sub(I("E"), I("t")), O("["), I("min"), O("("), sub(I("r"), I("t")), O("("), I("θ"), O(")"), sub(I("Â"), I("t")), O(","), I("clip"), O("("), sub(I("r"), I("t")), O("("), I("θ"), O(")"), O(","), N("1"), O("−"), I("ε"), O(","), N("1"), O("+"), I("ε"), O(")"), sub(I("Â"), I("t")), O(")"), O("]")) +
    M(sub(I("r"), I("t")), O("("), I("θ"), O(")"), O("="), frac(R(sub(I("π"), I("θ")), O("("), sub(I("a"), I("t")), O("|"), sub(I("s"), I("t")), O(")")), R(sub(I("π"), T("old")), O("("), sub(I("a"), I("t")), O("|"), sub(I("s"), I("t")), O(")"))));
  F.reward = M(sub(I("r"), I("t")), O("="), O("−"), frac(R(I("Δ"), sub(I("J"), I("t"))), I("c")), O("+"), I("γ"), I("Φ"), O("("), sub(I("s"), R(I("t"), O("+"), N(1))), O(")"), O("−"), I("Φ"), O("("), sub(I("s"), I("t")), O(")")) +
    M(I("Φ"), O("("), sub(I("s"), I("t")), O(")"), O("="), O("−"), sum(R(I("j"), T(" " + TP["math.unfinished"])), R(wj, O("["), sup(pos(R(sub(I("Ĉ"), I("j")), O("−"), dj)), N(2)), O("−"), sup(pos(R(tt, O("−"), dj)), N(2)), O("]")))) + md(TP["math.reward.note"]);
  F.copy = M(sup(I("L"), T("copy")), O("("), I("θ"), O(")"), O("="), O("−"), sub(I("E"), R(I("s"), O("∼"), T(TP["math.visited"]))), O("["), I("log"), sub(I("π"), I("θ")), O("("), sup(I("a"), O("*")), O("("), I("s"), O(")"), O("|"), I("s"), O(")"), O("]"));

  // formulas of the classic rules, keyed like the RULES section of the wording file
  const RULE_FORMULA = {
    EDF: M(key, O("="), dj, note("math.smallest")),
    EDFL: M(key, O("="), O("("), dj, O(","), O("−"), pj, O(")"), note("math.smallest")),
    LST: M(key, O("="), slack, note("math.smallest")),
    SPT: M(key, O("="), pj, note("math.smallest")),
    LPT: M(key, O("="), O("−"), pj, note("math.smallest")),
    FCFS: M(key, O("="), aj, note("math.smallest")),
    WSPT: M(prio, O("="), frac(wj, pj), note("math.largest")),
    ATC: M(sub(I("I"), I("j")), O("="), frac(wj, pj), I("exp"), O("("), O("−"), frac(R(maxOf(N(0), slack)), R(I("k"), sub(I("p"), T("mean")))), O(")"), note("math.largest_k")),
    COVERT: M(prio, O("="), frac(wj, pj), maxOf(N(0), R(N(1), O("−"), frac(maxOf(N(0), slack), R(I("k"), pj)))), note("math.largest_k")),
    WMDD: M(key, O("="), frac(maxOf(pj, R(dj, O("−"), tt)), wj), note("math.smallest")),
    WLST: M(key, O("="), R(I("σ"), T(" / "), wj), T("  " + TP["math.wlst_if"] + "   "), R(I("σ"), O("·"), wj), T("  " + TP["math.wlst_else"] + "      " + TP["math.wlst_with"] + " "), slack),
    WEDF: M(key, O("="), T(TP["math.wedf_scaled"] + " "), R(dj, O("−"), tt)),
    MDC: M(sub(I("ΔC"), I("j")), O("="), wj, O("["), sup(pos(R(sub(I("ℓ"), I("j")), O("+"), N(1))), N(2)), O("−"), sup(pos(sub(I("ℓ"), I("j"))), N(2)), O("]"), O(","), T("    "), sub(I("ℓ"), I("j")), O("="), R(tt, O("+"), pj, O("−"), dj)) +
      M(prio, O("="), frac(sub(I("ΔC"), I("j")), pj), note("math.mdc_tie")),
  };

  // ---- helpers -----------------------------------------------------------------------------------------------------------
  const fmt = (x, d = 0) => (x == null || Number.isNaN(x) ? "n/a" : x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
  const section = (h, html, open) => ({ h, html, open: !!open });

  function topicCard(id, ctx, extra) {
    const t = TX.topics[id];
    slots = Object.assign({}, extra);
    const sections = t.sections.map(s => section(s.h, md(s.text), s.open));
    if (t.metric) sections.push(section(TP["factor.section.insore"], md(fill(TP["factor.insore"], { metric: t.metric })), false));
    return { kicker: t.kicker, title: t.title, sections, related: t.related };
  }
  function weightsTable(ctx) {
    const imp = (ctx && ctx.importance) || {};
    const rows = Object.keys(TX.factors).map(k => `<tr><td>${esc(TX.factors[k].label)}</td><td>${esc(LB["level." + (imp[k] >= 3 ? 5 : imp[k] > 0 ? 2 : 0)])}</td><td>${fmt(imp[k] || 0)}</td></tr>`).join("");
    return `<table class="mini"><tr><th>${esc(LB["objective.table.thing"])}</th><th>${esc(LB["objective.table.setting"])}</th><th>${esc(LB["objective.table.lambda"])}</th></tr>${rows}</table>`;
  }

  // the card opened by clicking a bar: exact performance, what the model is, its equations
  function racerCard(ctx) {
    const r = ctx.racer, c = ctx.ctx, meta = r.meta || {}, isRule = r.family === "H";
    // rows follow the order of the rows in step 2 (What matters to you), not the data's alphabetical order
    const order = D.factors.flatMap(f => Object.keys(f.metrics)).map(mk => D.metrics.indexOf(mk));
    const rows = order.map(j => {
      const mk = D.metrics[j];
      const active = c.weights[j] > 0, ratio = c.refs[j] ? r.v[j] / c.refs[j] : null, sd = meta.sd ? meta.sd[j] : null;
      return `<tr class="${active ? "on" : ""}"><td>${esc(LB["metric." + mk])}</td><td>${fmt(r.v[j], 2)}${sd != null ? ` ± ${fmt(sd, 2)}` : ""}</td><td>${fmt(c.refs[j], 2)}</td><td>${active ? fmt(ratio, 3) : "-"}</td><td>${fmt(c.weights[j], 0)}</td><td>${active ? fmt(c.weights[j] * ratio, 3) : "-"}</td></tr>`;
    }).join("");
    const head = ["measured", "value", "bestrule", "ratio", "weight", "contrib"].map(k => `<th>${esc(TP["racer.table." + k])}</th>`).join("");
    const terms = order.map(j => c.weights[j] > 0 && c.refs[j] ? `${fmt(c.weights[j], 0)} × ${fmt(r.v[j] / c.refs[j], 3)}` : null).filter(Boolean).join(" + ");
    const gap = (r.score - c.best) / (c.best > 0 ? c.best : ((c.worst - c.best) || 1));
    const runs = meta.n > 1 ? fill(TP["racer.runs.many"], { n: meta.n }) : TP["racer.runs.one"];
    const ruleName = k => (TX.rules[k] ? TX.rules[k].name : k);
    let what, math, deeper, kicker, title;
    if (isRule) {
      const rule = TX.rules[meta.rule] || { name: meta.rule, plain: "", cite: "" }, plc = TX.placements[meta.plc] || { plain: "", cite: "" };
      what = md(fill(TP["racer.rule.what"], { name: rule.name, plain: rule.plain, placement: plc.plain }));
      math = (RULE_FORMULA[meta.rule] || "") + md(TP["racer.rule.math"]);
      deeper = md(fill(TP["racer.rule.deeper"], { cite: rule.cite, pcite: plc.cite }));
      kicker = TP["racer.kicker.rule"]; title = fill(TP["racer.title.rule"], { rule: rule.name, placement: meta.plc || "" });
    } else {
      const opt = TX.options[meta.opt] || { name: "A learned model", plain: "" }, copies = meta.kind === "copied" || meta.kind === "trained", scorer = meta.opt === 2 || meta.opt === 3;
      let w = md(fill(TP["racer.ai.what"], { optname: opt.name, optplain: opt.plain }));
      if (scorer) w += md(fill(meta.win ? TP["racer.ai.window"] : TP["racer.ai.allslots"], { win: meta.win, plc: meta.plc }));
      w += md((copies ? (meta.exp ? fill(TP["racer.ai.copied"], { rule: ruleName(meta.exp) }) : TP["racer.ai.copied.generic"]) : fill(TP["racer.ai.scratch"], { algo: meta.algo })) +
              (meta.steps ? " " + fill(TP["racer.ai.budget"], { steps: meta.steps }) : ""));
      what = w;
      math = (scorer ? (meta.opt === 3 || meta.atc ? F.policy_atc : F.policy) : meta.opt === 1 ? F.opt1 : meta.opt === 0 ? F.opt0 : md(TP["racer.ai.math.other"])) + (copies ? F.copy : F.ppo);
      deeper = md(copies ? TP["racer.ai.deeper.copy"] : TP["racer.ai.deeper.scratch"]) + F.reward +
        md(fill(TP["racer.ai.deeper.tail"], { algonote: TP["racer.ai.algonote." + (meta.algo === "A2C" ? "A2C" : "PPO")] }));
      kicker = TP["racer.kicker.ai"]; title = fill(TP["racer.title.ai"], { title: r.title, detail: r.detail });
    }
    const table = md(fill(TP["racer.results"], { inst: meta.i || 50, runs })) + `<table class="mini"><tr>${head}</tr>${rows}</table>`;
    const sumHtml = md(fill(TP["racer.sum"], { terms: terms || "0", score: fmt(r.score, 3), best: fmt(c.best, 3), gap: fmt(gap * 100, 1), gapfrac: fmt(gap, 3), pct: fmt(r.percent, 1) }));
    const card = topicCard("racer", ctx, { racer_table: table, racer_sum: sumHtml, racer_what: what, racer_math: math, racer_deeper: deeper });
    return Object.assign(card, { kicker, title, related: isRule ? ["objective", "bar", "kind_rule"] : ["objective", "bar", meta.kind === "copied" ? "kind_copied" : "kind_own"] });
  }

  function build(id, data) {
    if (id === "racer") return racerCard(data);
    return topicCard(id, data, id === "objective" ? { weights_table: weightsTable(data) } : {});
  }

  // ---- the slide-over card ----------------------------------------------------------------------------------------------
  let root, body, hist = [];
  function render(id, data) {
    const spec = build(id, data || {});
    body.innerHTML = `<div class="eyebrow">${esc(spec.kicker)}</div><h2>${esc(spec.title)}</h2>` +
      spec.sections.map(s => `<details class="sec" ${s.open ? "open" : ""}><summary>${esc(s.h)}</summary><div class="sbody">${s.html}</div></details>`).join("") +
      `<div class="related"><span class="eyebrow">${esc(TX.ui.related)}</span>${(spec.related || []).map(x => `<button type="button" class="chip" data-go="${x}">${esc(LB["chip." + x] || (TX.topics[x] ? TX.topics[x].title : x))}</button>`).join("")}</div>`;
    body.scrollTop = 0;
    root.querySelector(".back").hidden = hist.length < 2;
  }
  function open(id, data) { hist.push({ id, data }); render(id, data); root.classList.add("open"); root.setAttribute("aria-hidden", "false"); }
  function close() { root.classList.remove("open"); root.setAttribute("aria-hidden", "true"); hist = []; }
  function back() { if (hist.length > 1) { hist.pop(); const h = hist[hist.length - 1]; render(h.id, h.data); } }
  function init(el, getCtx) {
    root = el;
    root.innerHTML = `<div class="dhead"><button type="button" class="back" hidden>${esc(TX.ui.back)}</button><button type="button" class="x" aria-label="${esc(TX.ui.close)}">✕</button></div><div class="dbody"></div>`;
    body = root.querySelector(".dbody");
    root.querySelector(".x").onclick = close; root.querySelector(".back").onclick = back;
    root.addEventListener("click", e => { const g = e.target.closest("[data-go]"); if (g) open(g.dataset.go, getCtx()); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
  }
  window.Explain = { init, open, close };
})();
