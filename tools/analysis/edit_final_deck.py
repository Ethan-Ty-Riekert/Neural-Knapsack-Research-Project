"""Edit app/slides/Final_Presentation.pptx IN PLACE (the user edited it by hand, so it is not rebuilt from make_pptx.py):
slide 1: the gradient background becomes slightly more transparent (panels and text untouched); slide 2: new content (objective function and what was compared)."""
import copy
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(sys.argv[1])
SRC = ROOT / "app" / "slides" / "Final_Presentation.pptx"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC
A = "http://schemas.openxmlformats.org/drawingml/2006/main"

INK, BLUE, SOFT, TINT = RGBColor(0x0E, 0x22, 0x38), RGBColor(0x2A, 0x6F, 0xD1), RGBColor(0xCB, 0xD9, 0xEE), RGBColor(0xEE, 0xF4, 0xFC)
GREY, ORANGE, WHITE, MUTED = RGBColor(0x87, 0x93, 0xA1), RGBColor(0xE0, 0x70, 0x2A), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x56, 0x67, 0x7A)
FONT, FONT_BOLD, MATH = "Segoe UI", "Segoe UI Semibold", "Cambria Math"

prs = Presentation(str(SRC))


# ---------------------------------------------------------------- slide 1: slightly transparent panels
def set_alpha(shape, alpha_pct):
    """Make the solid fill (and the outline, if it is solid) of a shape alpha_pct % opaque."""
    sp = shape._element.spPr
    for fill in sp.findall(f"{{{A}}}solidFill") + [ln.find(f"{{{A}}}solidFill") for ln in sp.findall(f"{{{A}}}ln") if ln.find(f"{{{A}}}solidFill") is not None]:
        colour = fill[0]
        for old in colour.findall(f"{{{A}}}alpha"):
            colour.remove(old)
        etree.SubElement(colour, f"{{{A}}}alpha").set("val", str(int(alpha_pct * 1000)))


s1 = prs.slides[0]
# the slide background is a gradient whose last stop is already part transparent (alpha 39 %): make that stop more transparent so the slide is lighter
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
stops = s1._element.cSld.find(f"{{{P}}}bg").iter(f"{{{A}}}gs")
last = max(stops, key=lambda g: int(g.get("pos")))
for alpha in last.iter(f"{{{A}}}alpha"):
    alpha.set("val", "8000")


# ---------------------------------------------------------------- helpers (same look as tools/analysis/make_pptx.py)
def box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None, line_w=1.0):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def rich(slide, x, y, w, h, lines, size=12, colour=INK, anchor=MSO_ANCHOR.TOP, align=None, space=4):
    """lines: list of paragraphs; a paragraph is a list of (text, opts) runs; opts: bold, colour, size, sub, sup, math."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    f = tb.text_frame
    f.word_wrap = True
    f.vertical_anchor = anchor
    f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0
    for i, para in enumerate(lines):
        p = f.paragraphs[0] if i == 0 else f.add_paragraph()
        p.space_after = Pt(space)
        if align:
            p.alignment = align
        for t, o in para:
            r = p.add_run()
            r.text = t
            r.font.name = MATH if o.get("math") else (FONT_BOLD if o.get("bold") else FONT)
            r.font.size = Pt(o.get("size", size))
            r.font.color.rgb = o.get("colour", colour)
            if o.get("sub"):
                r._r.get_or_add_rPr().set("baseline", "-25000")
            if o.get("sup"):
                r._r.get_or_add_rPr().set("baseline", "30000")
    return tb


def label_in(shape, t, size=10, colour=WHITE, bold=True):
    f = shape.text_frame
    f.word_wrap = True
    f.vertical_anchor = MSO_ANCHOR.MIDDLE
    f.margin_left = f.margin_right = Inches(0.05)
    f.margin_top = f.margin_bottom = 0
    p = f.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = t
    r.font.name = FONT_BOLD if bold else FONT
    r.font.size = Pt(size)
    r.font.color.rgb = colour


def panel(slide, x, y, w, h):
    return box(slide, x, y, w, h, WHITE, SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, 0.04, 1.25)


# ---------------------------------------------------------------- slide 2
s2 = prs.slides[1]
keep = {2, 3, 4, 5, 26, 27, 28}  # eyebrow, title, footer, page number, the "Watch it learn" panel, its text and its button
for sh in list(s2.shapes):
    if sh.shape_id not in keep:
        sh._element.getparent().remove(sh._element)
title = next(sh for sh in s2.shapes if sh.shape_id == 3)
title.text_frame.paragraphs[0].runs[0].text = "What it minimises, and what was compared"
for r in title.text_frame.paragraphs[0].runs[1:]:
    r.text = ""

B = {"bold": True}
# ---- left card: the objective (the whole combined objective; every term is examined)
panel(s2, 0.7, 1.75, 5.6, 3.55)
rich(s2, 0.95, 1.9, 5.1, 0.3, [[("WHAT IT MINIMISES", {"bold": True, "colour": BLUE, "size": 11})]])
f = {"math": True, "size": 22}
fs = dict(f, sub=True)
fp = dict(f, sup=True)
rich(s2, 0.95, 2.2, 5.2, 0.4, [[("J", f), ("λ", fs), (" = Σ w", f), ("j", fs), (" T", f), ("j", fs), ("2", fp), (" + λ", f), ("T", fs), (" Σ w", f), ("j", fs), (" T", f), ("j", fs)]])
rich(s2, 0.95, 2.65, 5.2, 0.4, [[("      + λ", f), ("U", fs), (" Σ w", f), ("j", fs), (" U", f), ("j", fs), (" + λ", f), ("E", fs), (" E", f)]])
y = 3.12
for key, desc in [("Σ w T²", "lateness squared, weighted by importance"),
                  ("λT Σ w T", "lateness itself, weighted by importance"),
                  ("λU Σ w U", "how many jobs finish late (deadlines missed)"),
                  ("λE E", "energy: how long machines are switched on")]:
    rich(s2, 0.95, y, 1.3, 0.3, [[(key, {"math": True, "size": 12, "colour": BLUE})]], space=0)
    rich(s2, 2.3, y, 3.85, 0.3, [[(desc, {"size": 10.5, "colour": MUTED})]], space=0)
    y += 0.3
rich(s2, 0.95, 4.33, 5.2, 0.3, [[("T", {"math": True, "size": 10.5}), ("j", {"math": True, "size": 10.5, "sub": True}), (" = how late job j finishes   ", {"size": 9.5, "colour": MUTED}),
                                 ("U", {"math": True, "size": 10.5}), ("j", {"math": True, "size": 10.5, "sub": True}), (" = 1 if late   ", {"size": 9.5, "colour": MUTED}),
                                 ("w", {"math": True, "size": 10.5}), ("j", {"math": True, "size": 10.5, "sub": True}), (" = importance   ", {"size": 9.5, "colour": MUTED}),
                                 ("λ", {"math": True, "size": 10.5}), (" = weight of each goal", {"size": 9.5, "colour": MUTED})]], space=0)
box(s2, 0.95, 4.68, 5.1, 0.012, SOFT)
for i, (head, body) in enumerate([("State", "jobs, free machine capacity"), ("Action", "start a job, or wait"), ("Reward", "minus the lateness cost")]):
    rich(s2, 0.95 + i * 1.74, 4.76, 1.65, 0.5, [[(head, {"bold": True, "colour": BLUE, "size": 11})], [(body, {"size": 9.5, "colour": MUTED})]], space=0)

# ---- right card: what was compared
panel(s2, 6.5, 1.75, 6.15, 3.55)
rich(s2, 6.75, 1.9, 5.6, 0.3, [[("WHAT WAS COMPARED", {"bold": True, "colour": BLUE, "size": 11})]])
for i, (head, body) in enumerate([("49 dispatching heuristics", "a priority rule plus a placement rule, e.g. EDF + First-Fit"), ("CP-SAT exact solver", "gives a proven bound on the best possible")]):
    x = 6.75 + i * 2.95
    chip = box(s2, x, 2.18, 2.85, 0.6, TINT, SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, 0.15, 1)
    rich(s2, x + 0.12, 2.22, 2.65, 0.55, [[(head, {"bold": True, "size": 11.5})], [(body, {"size": 9, "colour": MUTED})]], space=0)
rich(s2, 6.75, 2.88, 5.7, 0.25, [[("Reinforcement learning (PPO and A2C), five ways of choosing the action", {"bold": True, "colour": BLUE, "size": 9.5})]])
OPTS = [("Option 0", "Direct pick", [("job + machine, one choice", 2.7, BLUE)]),
        ("Option 1", "Rule selection", [("picks a rule", 1.35, BLUE), ("the rule does the rest", 1.35, GREY)]),
        ("Option 2", "Priority learning", [("job", 1.35, BLUE), ("machine: First-Fit", 1.35, GREY)]),
        ("Option 3", "Priority + ATC feature", [("job (+ ATC cue)", 1.35, BLUE), ("machine: First-Fit", 1.35, GREY)]),
        ("Option 4", "Action branching", [("job", 1.32, BLUE), ("machine", 1.32, BLUE)])]
y = 3.15
for opt, name, segs in OPTS:
    rich(s2, 6.75, y + 0.03, 2.6, 0.32, [[(opt + "  ", {"bold": True, "size": 11}), (name, {"size": 11, "colour": MUTED})]], anchor=MSO_ANCHOR.MIDDLE, space=0)
    x = 9.45
    for label, w, colour in segs:
        seg = box(s2, x, y + 0.03, w - 0.03, 0.3, colour, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
        label_in(seg, label, size=9, bold=False)
        x += w + (0.03 if len(segs) == 2 and w < 1.4 and label in ("job", "machine") else 0)
    y += 0.38
box(s2, 9.45, 5.07, 0.14, 0.14, BLUE, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.3)
rich(s2, 9.65, 5.04, 1.4, 0.2, [[("chosen by the network", {"size": 9, "colour": MUTED})]])
box(s2, 11.05, 5.07, 0.14, 0.14, GREY, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.3)
rich(s2, 11.25, 5.04, 1.4, 0.2, [[("chosen by a fixed rule", {"size": 9, "colour": MUTED})]])

s2.notes_slide.notes_text_frame.text = (
    "Left: what every scheduler tries to minimise, the whole combined objective. Squared lateness punishes a few very late jobs more than many slightly "
    "late ones; linear lateness, the number of late jobs and energy are the other goals, and the lambdas set how much each one counts. Every method was "
    "scored on the full objective, with the weights of each term varied, to ask when RL is competitive when no single hand-made rule targets all the goals. "
    "The three words underneath are the MDP: state, action, reward. "
    "Right: what was compared on the same 50 unseen test cases per setting. Classic methods are 49 dispatching heuristics (a priority rule that picks the "
    "next job plus a placement rule that picks the machine) and an exact solver, CP-SAT, which gives a proven bound. The RL schedulers were trained with PPO "
    "and A2C and differ in how much the network decides: Option 0 picks the job and machine directly; Option 1 only picks which dispatching rule to apply; "
    "Options 2 and 3 learn a priority for the next job and a fixed rule places it (Option 3 adds the ATC index as an input); Option 4 chooses job and machine "
    "with two separate output heads. Every design had the same tuning budget. Then the button: a toy version you can train yourself.")

try:
    prs.save(str(OUT))
    print("saved", OUT)
except PermissionError:
    alt = OUT.with_name(OUT.stem + "_new.pptx")
    prs.save(str(alt))
    print("the deck is open in PowerPoint; saved to", alt)
