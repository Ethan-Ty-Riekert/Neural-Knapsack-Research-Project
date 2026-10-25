"""test_training_demo.py - consistency checks for the "Watch it learn" page (app/web/training_wording.md, training.html, training_ui.js).

Run from the repo root: python tests/test_training_demo.py
(The learners themselves are tested by: node tests/test_training_core.js)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from wording import load  # noqa: E402

W = load(ROOT / "app" / "web" / "training_wording.md")
html = (ROOT / "app" / "web" / "training.html").read_text(encoding="utf-8")
ui_js = (ROOT / "app" / "web" / "training_ui.js").read_text(encoding="utf-8")
SLOTS = {"J", "reward", "score", "policy", "ret", "reinforce", "critic", "advantage"}  # formulas built in training_ui.js

topics = W["topics"]
for tid, t in topics.items():
    assert t["kicker"] and t["title"] and t["sections"], f"card {tid!r} is incomplete"
    for r in t["related"]:
        assert r in topics, f"{tid}: related card {r!r} does not exist"
    for s in t["sections"]:
        for slot in re.findall(r"\{\{(\w+)\}\}", s["text"]):
            assert slot in SLOTS, f"{tid}: unknown formula slot {{{{{slot}}}}}"
for tid in topics:
    assert f"chip.{tid}" in W["labels"], f"card {tid!r} needs a chip.{tid} label in ## LABELS"
for tid in re.findall(r'data-explain="(\w+)"', html):
    assert tid in topics or tid == "learner", f"page links to card {tid!r} that is not in the wording file"
for key in set(re.findall(r'data-ui="([\w.]+)"', html)) | set(re.findall(r'UI(?:\.(\w+)|\["([\w.]+)"\])', ui_js) and
                                                              [a or b for a, b in re.findall(r'UI(?:\.(\w+)|\["([\w.]+)"\])', ui_js)]):
    assert key in W["ui"], f"the page uses UI text {key!r} that is missing from ## UI"
for key in set(re.findall(r'TP\["([\w.]+)"\]', ui_js)):
    assert key in W["templates"], f"the page uses template {key!r} that is missing from ## TEMPLATES"
print("training demo wording checks passed")
