/* race_core.js - the demo's scoring and race selection in the browser (mirror of app/scoring.py: FACTORS, EXAMPLES,
   race_entries). Needs window.RACE_DATA (written by tools/analysis/make_demo_artifacts.py from the results archive). */
(function () {
  const D = window.RACE_DATA;
  const STEEPNESS = 4.0, NOISE = 0.05;
  const BUSY = { Quiet: "on_rho050", Normal: "on_rho075", Busy: "on_rho095", Overloaded: "on_rho110" };
  const DEADLINES = { Relaxed: "tf02", Medium: "tf05", Tight: "tf08" };
  const LEVELS = { Ignore: 0, Some: 2, "A lot": 5 };

  function presetFor(s) {
    if (s.arrival === "offline") return "off_" + DEADLINES[s.deadlines] + (s.equal ? "_w1" : "");
    return s.tight && BUSY[s.busy] === "on_rho075" ? "on_rho075_tight" : BUSY[s.busy];
  }

  function weights(importance) {  // {factor key: level 0-5} -> weight per metric column
    const w = D.metrics.map(() => 0);
    D.factors.forEach(f => Object.entries(f.metrics).forEach(([m, x]) => { w[D.metrics.indexOf(m)] += (importance[f.key] || 0) * x; }));
    return w;
  }

  function reference(rows, j) {  // best rule on metric j; mean over rules when that is 0 (as scoring.reference_values)
    const v = rows.filter(r => r.family === "H" && r.v[j] != null).map(r => r.v[j]);
    if (!v.length) return null;
    const best = Math.min(...v);
    return best || (v.reduce((a, b) => a + Math.abs(b), 0) / v.length) || 1;
  }

  function run(preset, importance) {
    const rows = (D.presets[preset] || []).map(r => ({ family: r[0], title: r[1], detail: r[2], kind: r[3], model: r[4], v: r[5] }));
    const w = weights(importance);
    const refs = D.metrics.map((_, j) => reference(rows, j));
    const scored = rows.map(r => {
      let s = 0, ok = true;
      w.forEach((x, j) => { if (!x) return; if (r.v[j] == null || !refs[j]) ok = false; else s += x * r.v[j] / refs[j]; });
      return ok ? Object.assign({}, r, { score: s }) : null;
    }).filter(Boolean).sort((a, b) => a.score - b.score);
    if (!scored.length) return { racers: [], note: null };
    const best = scored[0].score, worst = scored[scored.length - 1].score;
    const scale = best > 0 ? best : ((worst - best) || 1);
    scored.forEach(r => { r.percent = 100 * Math.exp(-STEEPNESS * (r.score - best) / scale); });
    const pick = (pred, n) => { const seen = new Set(), out = []; for (const r of scored) { if (pred(r) && !seen.has(r.model)) { seen.add(r.model); out.push(r); if (out.length === n) break; } } return out; };
    const rules = pick(r => r.family === "H", 2), copied = pick(r => r.kind === "copied", 1), scratch = pick(r => r.kind === "scratch", 2);
    let racers = rules.concat(copied, scratch);
    const hasAI = copied.length + scratch.length > 0;
    if (!hasAI) racers = pick(r => r.family === "H", 4);
    racers.sort((a, b) => a.score - b.score);
    let note = null;
    const aiBest = scored.find(r => r.family === "R"), ruleBest = scored.find(r => r.family === "H");
    if (!hasAI) note = "We did not train an AI scheduler for this exact setting, so four classic rules race instead.";
    else if (aiBest) {
      const gap = (aiBest.score - ruleBest.score) / Math.abs(ruleBest.score || 1);
      if (gap >= 0) note = "No AI scheduler beat the best classic rule here. That is the usual result in this project: the AI schedulers mostly learned to copy a good rule and finish close to it.";
      else if (gap > -NOISE) note = "The best AI scheduler is only " + (-gap * 100).toFixed(1) + "% ahead of the best classic rule, which is within the run-to-run noise of this project (gaps under about 5% were not significant).";
    }
    return { racers, note, winner: racers.slice().sort((a, b) => b.percent - a.percent)[0] };
  }

  window.RaceCore = { BUSY, DEADLINES, LEVELS, presetFor, run, data: D };
})();
