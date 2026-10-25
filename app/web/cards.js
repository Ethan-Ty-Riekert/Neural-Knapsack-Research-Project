/* cards.js - the generic "click anything for an explanation" card engine shared by the browser demos (race page, training demo).
   A page supplies its wording (RACE_DATA.text, parsed from its wording file by app/wording.py), a formulas object (name -> MathML
   html, filled in later by the page) and, optionally, a function that builds a special card (e.g. a racer's numbers).
   Cards.create({text, formulas}) -> { esc, md, fill, topicCard, init, open, close }.
   Text markup: **bold** *italic* `code` a_{j} x^{2} and {{name}} slots (formulas or per-card extras); see the header of the wording file. */
(function () {
  window.Cards = {
    create(cfg) {
      const TX = cfg.text, TP = TX.templates, LB = TX.labels, F = cfg.formulas;
      let slots = {}, root, body, hist = [], build = null;

      const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      function inline(s) {
        s = esc(s);
        s = s.replace(/\^\{([^}]*)\}/g, (m, x) => "<sup>" + x.replace(/\*/g, "&#42;") + "</sup>");
        s = s.replace(/([A-Za-zλρσφℓπθγΦΔĈ])_\{([^}]*)\}/g, "<i>$1</i><sub>$2</sub>");
        return s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\*([^*]+)\*/g, "<i>$1</i>").replace(/`([^`]+)`/g, "<code>$1</code>");
      }
      const slot = n => (slots[n] != null ? slots[n] : F[n] != null ? F[n] : `<span class="missing">[unknown slot ${n}]</span>`);
      function md(text) {
        return text.split(/\n\s*\n/).map(p => {
          p = p.trim(); if (!p) return "";
          const only = p.match(/^\{\{(\w+)\}\}$/); if (only) return slot(only[1]);
          return "<p>" + inline(p).replace(/\{\{(\w+)\}\}/g, (_, n) => slot(n)) + "</p>";
        }).join("");
      }
      const fill = (tpl, v) => tpl.replace(/\{(\w+)\}/g, (m, k) => (v[k] != null ? v[k] : m));

      // a card from the wording file's "### id" block; extra = per-card slot values
      function topicCard(id, extra) {
        const t = TX.topics[id];
        slots = Object.assign({}, extra);
        const sections = t.sections.map(s => ({ h: s.h, html: md(s.text), open: s.open }));
        if (t.metric) sections.push({ h: TP["factor.section.insore"], html: md(fill(TP["factor.insore"], { metric: t.metric })), open: false });
        return { kicker: t.kicker, title: t.title, sections, related: t.related };
      }

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
      // el: the slide-over element; getCtx: data for cards opened from a Related button; buildCard(id, data) -> card spec
      function init(el, getCtx, buildCard) {
        root = el; build = buildCard;
        root.innerHTML = `<div class="dhead"><button type="button" class="back" hidden>${esc(TX.ui.back)}</button><button type="button" class="x" aria-label="${esc(TX.ui.close)}">✕</button></div><div class="dbody"></div>`;
        body = root.querySelector(".dbody");
        root.querySelector(".x").onclick = close; root.querySelector(".back").onclick = back;
        root.addEventListener("click", e => { const g = e.target.closest("[data-go]"); if (g) open(g.dataset.go, getCtx()); });
        document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
      }
      return { esc, md, fill, topicCard, init, open, close };
    },
  };
})();
