/* flow_centre_chain.js - centre box, version A: "the chain of the episode". Each decision adds a state circle, then an arrow carrying the
   action and, below it, the reward that popped up; the chain grows to the right and scrolls to follow the latest decision. */
window.FlowCentre = {
  css: `
  .chainwrap { height: 100%; overflow: hidden; position: relative; display: flex; align-items: center; }
  .chainrail { display: flex; align-items: center; transition: transform .5s ease; padding: 0 .6rem; }
  .cnode { flex: none; width: 5rem; border: 2px solid var(--rule); border-radius: 50%; aspect-ratio: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; background: var(--surface); font-size: .66rem; line-height: 1.1; padding: .2rem; animation: pop .35s ease; }
  .cnode b { font-family: var(--display); font-size: 1rem; color: var(--accent); }
  .cnode.active { border-color: var(--accent); box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 18%, transparent); }
  .cnode.end { background: var(--ink); color: var(--bg); border-color: var(--ink); } .cnode.end b { color: var(--bg); }
  .cedge { flex: none; width: 5.4rem; height: 4.6rem; position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; animation: pop .35s ease; }
  .cedge .act { font-size: .68rem; font-weight: 700; color: var(--ink); position: absolute; top: .2rem; }
  .cedge .line { width: 100%; height: 3px; background: var(--rulebar); position: relative; border-radius: 2px; }
  .cedge .line::after { content: ""; position: absolute; right: -1px; top: -5px; border: 6px solid transparent; border-left: 9px solid var(--rulebar); border-right: 0; }
  .cedge .rew { position: absolute; bottom: .1rem; font-size: .8rem; font-weight: 700; padding: .05rem .45rem; border-radius: 1rem; background: color-mix(in srgb, var(--ink) 8%, transparent); }
  .cedge .rew.good { color: var(--win); } .cedge .rew.bad { color: #fff; background: var(--scratch); }
  .cdot { position: absolute; left: 0; top: 50%; width: .8rem; height: .8rem; margin-top: -.4rem; border-radius: 50%; background: var(--accent); animation: travel .8s ease-in-out forwards; }
  @keyframes pop { from { transform: scale(.4); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  @keyframes travel { from { left: 0; } to { left: calc(100% - .8rem); } }
  `,
  create(root) {
    const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
    root.innerHTML = ""; const wrap = el("div", "chainwrap"), rail = el("div", "chainrail"); wrap.append(rail); root.append(wrap);
    let node = null, edge = null;
    const UI = window.RACE_DATA.text.ui, fill = (t, v) => t.replace(/\{(\w+)\}/g, (m, k) => (v[k] != null ? v[k] : m));
    const follow = () => { rail.style.transform = `translateX(${Math.min(0, wrap.clientWidth - rail.scrollWidth - 4)}px)`; };
    const addNode = (k, text, end) => { const n = el("div", "cnode" + (end ? " end" : ""), `<b>${end ? "■" : "s<sub>" + k + "</sub>"}</b><span>${text}</span>`); rail.append(n); follow(); return n; };
    const api = {
      speed: 1,  // set by flow_ui.js (x1, x3, x10)
      reset() { rail.textContent = ""; rail.style.transform = "none"; node = edge = null; },
      get out() { return node || root; },
      get in() { return edge || node || root; },
      async state(info) {
        if (info.k === 0) node = addNode(0, fill(UI["chain.node"], { t: info.t, n: info.waiting }));
        rail.querySelectorAll(".cnode").forEach(n => n.classList.remove("active")); node.classList.add("active");
      },
      async action(info) {
        edge = el("div", "cedge", `<span class="act">${fill(UI["chain.action"], { job: info.job })}</span><div class="line"></div>`);
        rail.append(edge); follow(); edge.querySelector(".line").append(el("i", "cdot"));
        await new Promise(r => setTimeout(r, 350 / (api.speed || 1)));
      },
      async reward(info) { edge.append(el("span", "rew " + (info.late ? "bad" : "good"), info.reward)); },
      async next(info) {
        node = addNode(info.done ? 0 : edge.parentNode.querySelectorAll(".cnode").length, info.done ? fill(UI["chain.end"], { cost: info.cost }) : fill(UI["chain.node"], { t: info.t, n: info.n }), info.done);
        edge = null; if (!info.done) { rail.querySelectorAll(".cnode").forEach(n => n.classList.remove("active")); node.classList.add("active"); }
      },
      snap(ep) {  // a whole finished episode at once
        api.reset();
        ep.steps.forEach((st, k) => {
          addNode(k, fill(UI["chain.node"], { t: st.state.t, n: st.state.waiting.length }));
          const e = el("div", "cedge", `<span class="act">${fill(UI["chain.action"], { job: st.job + 1 })}</span><div class="line"></div><span class="rew ${st.cost > 0 ? "bad" : "good"}">${st.cost > 0 ? st.reward.toFixed(1) : "0"}</span>`);
          rail.append(e);
        });
        addNode(0, fill(UI["chain.end"], { cost: ep.cost }), true);
      },
    };
    return api;
  },
};
