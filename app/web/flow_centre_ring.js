/* flow_centre_ring.js - centre box, version B: "the decision ring". Four stations (State, Action, Reward, Next state) around a ring; a
   glowing token goes round the ring once per decision. Under the ring a row of dots records the episode: one dot per decision, blue if
   the job will be on time, orange if late, so the chain of decisions builds up while the ring keeps turning. */
window.FlowCentre = {
  css: `
  .ringwrap { height: 100%; display: grid; grid-template-rows: 1fr auto; gap: .3rem; }
  .ring { position: relative; margin: auto; width: min(100%, 17rem); aspect-ratio: 1.5; }
  .ring svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .ring .orbit { fill: none; stroke: var(--rule); stroke-width: 3; stroke-dasharray: 2 7; stroke-linecap: round; }
  .st { position: absolute; transform: translate(-50%, -50%); border: 2px solid var(--rule); background: var(--surface); border-radius: .7rem; padding: .25rem .5rem; text-align: center; font-size: .7rem; line-height: 1.15; min-width: 5.2rem; transition: border-color .25s, box-shadow .25s, background .25s; }
  .st b { display: block; font-size: .66rem; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
  .st.on { border-color: var(--accent); box-shadow: 0 0 0 5px color-mix(in srgb, var(--accent) 18%, transparent); }
  .st.bad.on { border-color: var(--scratch); box-shadow: 0 0 0 5px color-mix(in srgb, var(--scratch) 22%, transparent); }
  .ringmid { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); font-size: .8rem; color: var(--muted); text-align: center; font-weight: 600; }
  .rtoken { position: absolute; width: 1rem; height: 1rem; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent) 25%, transparent); transform: translate(-50%, -50%); transition: left .55s ease-in-out, top .55s ease-in-out; }
  .trail { display: flex; gap: .3rem; flex-wrap: wrap; justify-content: center; min-height: 1.4rem; }
  .trail i { width: 1.1rem; height: 1.1rem; border-radius: 50%; background: var(--accent); animation: dotpop .35s ease; display: inline-flex; align-items: center; justify-content: center; color: #fff; font-style: normal; font-size: .55rem; font-weight: 700; }
  .trail i.late { background: var(--scratch); }
  .trail i.end { background: var(--ink); width: auto; padding: 0 .5rem; border-radius: 1rem; }
  @keyframes dotpop { from { transform: scale(.2); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  `,
  create(root) {
    const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
    const UI = window.RACE_DATA.text.ui, fill = (t, v) => t.replace(/\{(\w+)\}/g, (m, k) => (v[k] != null ? v[k] : m));
    root.innerHTML = "";
    const wrap = el("div", "ringwrap"), ring = el("div", "ring"), trail = el("div", "trail");
    ring.innerHTML = `<svg viewBox="0 0 300 200" preserveAspectRatio="none"><ellipse class="orbit" cx="150" cy="100" rx="118" ry="70"></ellipse></svg>`;
    // station centres as fractions of the ring box: state (top), action (right), reward (bottom), next state (left)
    const pos = { state: [50, 12], action: [84, 50], reward: [50, 88], next: [16, 50] }, names = { state: "loop.state", action: "loop.action", reward: "loop.reward", next: "loop.next" };
    const st = {};
    Object.keys(pos).forEach(k => { const s = el("div", "st", `<b>${UI[names[k]]}</b><span class="v"></span>`); s.style.left = pos[k][0] + "%"; s.style.top = pos[k][1] + "%"; ring.append(s); st[k] = s; });
    const mid = el("div", "ringmid"), tok = el("div", "rtoken"); ring.append(mid, tok); wrap.append(ring, trail); root.append(wrap);
    const move = k => { tok.style.left = pos[k][0] + "%"; tok.style.top = pos[k][1] + "%"; };
    const on = (k, bad) => { Object.values(st).forEach(s => s.classList.remove("on", "bad")); st[k].classList.add("on"); if (bad) st[k].classList.add("bad"); move(k); };
    const wait = ms => new Promise(r => setTimeout(r, ms / (api.speed || 1)));  // api.speed is set by flow_ui.js (x1, x3, x10)
    let n = 0;
    const api = {
      speed: 1,
      reset() { Object.values(st).forEach(s => { s.classList.remove("on", "bad"); s.querySelector(".v").textContent = ""; }); trail.textContent = ""; mid.textContent = ""; move("state"); n = 0; },
      get out() { return st.state; },
      get in() { return st.action; },
      async state(info) { n = info.k; mid.textContent = fill(UI["ring.decision"], { k: info.k + 1, n: info.n }); on("state"); st.state.querySelector(".v").textContent = fill(UI["chain.node"], { t: info.t, n: info.waiting }); st.action.querySelector(".v").textContent = ""; st.reward.querySelector(".v").textContent = ""; st.next.querySelector(".v").textContent = ""; await wait(500); },
      async action(info) { st.action.querySelector(".v").textContent = fill(UI["chain.action"], { job: info.job }) + " · " + info.pct + "%"; on("action"); await wait(550); },
      async reward(info) { st.reward.querySelector(".v").textContent = info.reward; on("reward", info.late); trail.append(el("i", info.late ? "late" : "", info.reward)); await wait(550); },
      async next(info) {
        st.next.querySelector(".v").textContent = info.done ? fill(UI["chain.end"], { cost: info.cost }) : fill(UI["chain.node"], { t: info.t, n: info.n });
        on("next"); await wait(550);
        if (info.done) trail.append(el("i", "end", fill(UI["chain.end"], { cost: info.cost }))); else { move("state"); await wait(300); }
      },
      snap(ep) { api.reset(); ep.steps.forEach(s => trail.append(el("i", s.cost > 0 ? "late" : "", s.cost > 0 ? s.reward.toFixed(1) : "0"))); trail.append(el("i", "end", fill(UI["chain.end"], { cost: ep.cost }))); mid.textContent = ""; },
    };
    move("state");
    return api;
  },
};
