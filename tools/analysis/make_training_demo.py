"""Build the standalone "watch it learn" pages from app/web/ (no results archive needed: the world and the learners run in the browser).

    python tools/analysis/make_training_demo.py

Writes app/web/dist/training_demo.html (the first layout, kept) and app/web/dist/stage.html (the animated sketch layout: jobs, machines, the MDP ring,
network, probability graph and rewards). The animated page is also copied to app/slides/training_demo.html, the page the final deck links to.
Each page inlines base.css, the wording (app/web/training_wording.md parsed by app/wording.py) and its scripts.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "app" / "web"
sys.path.insert(0, str(ROOT / "app"))
from wording import load  # noqa: E402

COMMON = ("mathml.js", "cards.js", "training_core.js", "training_shared.js")


def page(template, scripts, replacements=()):
    data = "window.RACE_DATA = " + json.dumps({"text": load(WEB / "training_wording.md")}, separators=(",", ":")) + ";"
    core = "\n".join((WEB / f).read_text(encoding="utf-8") for f in scripts)
    html = (WEB / template).read_text(encoding="utf-8").replace("/*__DATA__*/", data).replace("/*__CORE__*/", core)
    html = html.replace("/*__BASE__*/", (WEB / "base.css").read_text(encoding="utf-8"))
    for old, new in replacements:
        html = html.replace(old, new)
    return html


def main():
    out = WEB / "dist"
    out.mkdir(exist_ok=True)
    pages = {"training_demo.html": page("training.html", COMMON + ("training_ui.js",)),
             "stage.html": page("stage.html", COMMON + ("stage_ui.js",))}
    for name, html in pages.items():
        (out / name).write_text(html, encoding="utf-8")
        print("wrote", out / name, f"{len(html) / 1024:.0f} KB")
    shutil.copyfile(out / "stage.html", ROOT / "app" / "slides" / "training_demo.html")  # the deck's "Interactive demo" button opens this copy: the current (sketch) layout


if __name__ == "__main__":
    main()
