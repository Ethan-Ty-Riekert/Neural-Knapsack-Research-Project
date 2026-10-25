"""wording.py - reads the demo's single wording file (app/web/wording.md) into a dict.

The file holds every sentence a visitor reads, so it can be edited without touching code. Format (see the header of the file):
  ## NAME            a list of "key: value" lines (UI, LABELS, FACTORS, NOTES, RULES, PLACEMENTS, OPTIONS, TEMPLATES)
  ### topic-id       one explanation card: "kicker:", "title:", "related:", "metric:" lines, then "--- Heading [(open)]"
                     sections whose text runs until the next heading; blank lines separate paragraphs
  // comment         ignored
Only the structure is parsed here; the light markup (**bold**, *italic*, `code`, a_{j}, {{formula}}) is turned into HTML by
the browser (race_ui / explain.js), and plain-text uses (button labels, names) read the raw strings.
"""
from pathlib import Path

PATH = Path(__file__).resolve().parent / "web" / "wording.md"
KEYED = ("UI", "LABELS", "FACTORS", "NOTES", "RULES", "PLACEMENTS", "OPTIONS", "TEMPLATES")


def _unescape(v):
    return v.replace("\\n", "\n")


def load(path=PATH):
    """-> {'ui': {...}, 'labels': {...}, 'factors': {key: {label, hint}}, 'notes': {...}, 'rules': {id: {name, plain, cite}},
    'placements': {id: {short, plain, cite}}, 'options': {n: {short, name, plain}}, 'templates': {...},
    'topics': {id: {kicker, title, related: [...], metric, sections: [{h, open, text}]}}}  (order of the file is kept)."""
    out = {k.lower(): {} for k in KEYED}
    out["topics"] = {}
    section, topic, sec = None, None, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("//"):
            continue
        if line.startswith("### "):
            section, sec = "TOPIC", None
            topic = out["topics"][line[4:].strip()] = {"kicker": "", "title": "", "related": [], "metric": "", "sections": []}
            continue
        if line.startswith("## "):
            section, topic, sec = line[3:].strip(), None, None
            if section not in KEYED:
                raise ValueError(f"unknown wording section {section!r}")
            continue
        if section == "TOPIC":
            if line.startswith("--- "):
                head = line[4:].strip()
                is_open = head.endswith("(open)")
                sec = {"h": head[:-6].strip() if is_open else head, "open": is_open, "lines": []}
                topic["sections"].append(sec)
            elif sec is not None:
                sec["lines"].append(line)
            elif ":" in line and line.split(":", 1)[0] in ("kicker", "title", "related", "metric"):
                k, v = line.split(":", 1)
                topic[k] = [x.strip() for x in v.split(",") if x.strip()] if k == "related" else v.strip()
        elif section in KEYED and ":" in line and line.strip():
            k, v = line.split(":", 1)
            out[section.lower()][k.strip()] = _unescape(v.strip())
    for t in out["topics"].values():
        for s in t["sections"]:
            s["text"] = "\n".join(s.pop("lines")).strip()
    # group dotted keys: RULES "EDF.name" -> rules["EDF"]["name"], FACTORS "late.label" -> factors["late"]["label"], etc.
    for name in ("rules", "placements", "options", "factors"):
        grouped = {}
        for k, v in out[name].items():
            item, field = k.split(".", 1)
            grouped.setdefault(item, {})[field] = v
        out[name] = grouped
    return out


if __name__ == "__main__":
    w = load()
    print({k: len(v) for k, v in w.items()})
