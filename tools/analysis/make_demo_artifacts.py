"""Build the standalone browser demos (app/web/dist/variant_*.html) from the results archive.

    python tools/analysis/make_demo_artifacts.py

Exports the numbers the race needs (per scenario: classic rules, copied and from-scratch AI schedulers, five metrics) from
Results/v2_objectives/ALL_RESULTS/data/all_results.csv, and inlines them plus app/web/{race_core.js,race_ui.js,base.css} into each template
app/web/variant_*.html (placeholders /*__DATA__*/, /*__CORE__*/, /*__BASE__*/). Names, kinds and question-to-metric mapping come from
app/scoring.py, so the browser demo and the Streamlit app cannot drift apart.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "app" / "web"
sys.path.insert(0, str(ROOT / "app"))
from scoring import EXAMPLES, FACTORS, HEURISTIC, ai_kind, friendly_name, load_results, short_label  # noqa: E402

KIND = {"copied": "copied", "copied + trained": "trained", "trial and error": "scratch"}


def export():
    df = load_results()
    metrics = sorted({m for _, ms in FACTORS.values() for m in ms})
    presets = {}
    for preset, grp in df[df["family"].isin([HEURISTIC, "RL"])].groupby("preset"):
        rows = []
        for r in grp.itertuples():
            vals = [None if getattr(r, m) != getattr(r, m) else round(float(getattr(r, m)), 3) for m in metrics]
            if r.family == HEURISTIC:
                title, detail = friendly_name(r.method, r.family)
                rows.append(["H", title, detail, "rule", r.method, vals])
            else:
                title, detail = friendly_name(r.method, r.family)
                rows.append(["R", title, detail, KIND[ai_kind(r.method)], short_label(r.method), vals])
        presets[preset] = rows
    return {"metrics": metrics, "presets": presets,
            "factors": [{"key": k, "question": q, "metrics": ms} for k, (q, ms) in FACTORS.items()],
            "examples": EXAMPLES}


def main():
    data = "window.RACE_DATA = " + json.dumps(export(), separators=(",", ":")) + ";"
    core = (WEB / "race_core.js").read_text(encoding="utf-8") + "\n" + (WEB / "race_ui.js").read_text(encoding="utf-8")
    css = (WEB / "base.css").read_text(encoding="utf-8")
    out = WEB / "dist"
    out.mkdir(exist_ok=True)
    for template in sorted(WEB.glob("variant_*.html")):
        html = template.read_text(encoding="utf-8").replace("/*__DATA__*/", data).replace("/*__CORE__*/", core).replace("/*__BASE__*/", css)
        (out / template.name).write_text(html, encoding="utf-8")
        print("wrote", out / template.name, f"{len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
