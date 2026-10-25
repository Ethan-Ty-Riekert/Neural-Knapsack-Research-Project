/* training_shared.js - pieces shared by the training demo pages (training_ui.js: first layout; flow_ui.js: animated layouts):
   DOM helpers, the formulas used by the cards, reference scores (random order and the best classic rule) per mode, and the learning-curve
   drawing. Needs training_core.js, mathml.js and window.RACE_DATA.text (training_wording.md). */
(function () {
  const C = window.TrainingCore;
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const fill = (tpl, v) => tpl.replace(/\{(\w+)\}/g, (m, k) => (v[k] != null ? v[k] : m));
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // formulas for the explanation cards: {{J}} {{reward}} {{score}} {{policy}} {{ret}} {{reinforce}} {{critic}} {{advantage}}
  function formulas(F) {
    const { I, N, O, T, R, sub, sup, frac, sum, M } = MathML;
    const wj = sub(I("w"), I("j")), Tj = sub(I("T"), I("j")), phi = sub(I("φ"), I("j")), th = I("θ");
    const grad = R(I("∇"), I("log"), sub(I("π"), th), O("("), sub(I("a"), I("t")), O("|"), sub(I("s"), I("t")), O(")"));
    return Object.assign(F, {
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
  }

  // reference scores on 200 fresh job sets (random order, and the best of the classic rules), per mode
  const refCache = {};
  function refs(online) {
    const key = online ? "online" : "offline";
    if (!refCache[key]) {
      const sets = C.testSets(200, 424242, online);
      const rules = Object.entries(C.RULES).map(([k, f]) => [k, C.meanCost(sets, f)]).sort((a, b) => a[1] - b[1]);
      refCache[key] = { sets, random: C.meanCost(sets, C.randomChooser(C.makeRng(7))), rule: { name: rules[0][0], cost: rules[0][1] } };
    }
    return refCache[key];
  }

  // learning curve: o = { hist: {kind: [cost per episode]}, active, refs, labels: {random, rule}, colours: {kind: css colour}, empty }
  function drawCurve(cv, o) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const padL = 36, padR = 8, padT = 8, padB = 20, maxY = o.maxY || 200, maxX = C.TRAIN_EPISODES, WINDOW = 25;
    const X = e => padL + (w - padL - padR) * e / maxX, Y = c => padT + (h - padT - padB) * (1 - Math.min(c, maxY) / maxY);
    g.font = "11px " + css("--body"); g.fillStyle = css("--muted"); g.strokeStyle = css("--rule"); g.lineWidth = 1;
    const ticks = o.ticks || [0, 50, 100, 150, 200];
    ticks.forEach(v => { g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(w - padR, Y(v)); g.stroke(); g.textAlign = "right"; g.fillText(v, padL - 4, Y(v) + 4); });
    [0, 200, 400, 600].forEach(v => { g.textAlign = "center"; g.fillText(v, X(v), h - 5); });
    const line = (cost, dash, label) => {
      g.strokeStyle = css("--rulebar"); g.lineWidth = 2; g.setLineDash(dash); g.beginPath(); g.moveTo(padL, Y(cost)); g.lineTo(w - padR, Y(cost)); g.stroke(); g.setLineDash([]);
      g.fillStyle = css("--rulebar"); g.textAlign = "right"; g.fillText(label, w - padR - 2, Y(cost) - 3);
    };
    line(o.refs.random, [6, 4], o.labels.random);
    line(o.refs.rule.cost, [], o.labels.rule);
    let any = false;
    Object.entries(o.hist).forEach(([kind, hist]) => {
      if (!hist.length) return;
      any = true;
      const colour = o.colours[kind], active = kind === o.active;
      g.globalAlpha = active ? 0.35 : 0.18; g.fillStyle = colour;
      hist.forEach((c, i) => { g.beginPath(); g.arc(X(i + 1), Y(c), 1.8, 0, 6.3); g.fill(); });
      g.globalAlpha = 1; g.strokeStyle = colour; g.lineWidth = active ? 3 : 2; g.beginPath();
      hist.forEach((c, i) => { const s = Math.max(0, i - WINDOW + 1), avg = hist.slice(s, i + 1).reduce((a, b) => a + b, 0) / (i + 1 - s); i ? g.lineTo(X(i + 1), Y(avg)) : g.moveTo(X(1), Y(avg)); });
      g.stroke();
    });
    if (!any && o.empty) { g.fillStyle = css("--muted"); g.textAlign = "center"; g.font = "14px " + css("--body"); g.fillText(o.empty, w / 2, h / 2); }
  }

  window.TrainingShared = { el, fill, css, formulas, refs, drawCurve };
})();
