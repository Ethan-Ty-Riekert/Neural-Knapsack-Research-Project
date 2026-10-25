"""Build the standalone "Watch it learn" page: app/web/dist/training_demo.html (and a copy next to the deck, app/slides/training_demo.html).

    python tools/analysis/make_training_demo.py

Inlines base.css, the wording (app/web/training_wording.md parsed by app/wording.py) and the scripts (mathml.js, cards.js, training_core.js,
training_ui.js) into app/web/training.html. No results archive is needed: the world and the learners run in the browser.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "app" / "web"
sys.path.insert(0, str(ROOT / "app"))
from wording import load  # noqa: E402


def main():
    data = "window.RACE_DATA = " + json.dumps({"text": load(WEB / "training_wording.md")}, separators=(",", ":")) + ";"
    core = "\n".join((WEB / f).read_text(encoding="utf-8") for f in ("mathml.js", "cards.js", "training_core.js", "training_ui.js"))
    css = (WEB / "base.css").read_text(encoding="utf-8")
    html = (WEB / "training.html").read_text(encoding="utf-8").replace("/*__DATA__*/", data).replace("/*__CORE__*/", core).replace("/*__BASE__*/", css)
    out = WEB / "dist" / "training_demo.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    shutil.copyfile(out, ROOT / "app" / "slides" / "training_demo.html")  # the deck's "Interactive training demo" button opens this copy
    print("wrote", out, f"{len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
