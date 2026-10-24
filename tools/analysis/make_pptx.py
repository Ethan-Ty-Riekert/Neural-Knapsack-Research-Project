"""Build the editable presentation deck: app/slides/deck.pptx (16:9, six slides; charts embedded from app/slides/*.png).

    python tools/analysis/make_pptx.py        (needs python-pptx; run make_slides.py first so the chart PNGs are current)

Wording follows Future/research/2026-10-21-final-week-findings.md ("For the presentation"): no "RL beats/matches the heuristics".
Slide plan and speaking scripts: Future/research/2026-10-23-presentation-plan.md. Style: navy title slide, light content slides with
a kicker + title header, white cards, one blue accent.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
SLIDES = ROOT / "app" / "slides"
OUT = SLIDES / "deck.pptx"

NAVY, BLUE, ORANGE = RGBColor(0x0B, 0x1F, 0x3A), RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xE8, 0x7A, 0x2E)
INK, MUTED, BG, CARD, RULE = (RGBColor(0x1B, 0x25, 0x33), RGBColor(0x5F, 0x6B, 0x7A), RGBColor(0xF3, 0xF5, 0xF9),
                              RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xDD, 0xE2, 0xEA))
FONT, FONT_BOLD = "Segoe UI", "Segoe UI Semibold"
TOTAL = 6

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def text(slide, x, y, w, h, paragraphs, size=18, colour=INK, bold=False, align=None, anchor=MSO_ANCHOR.TOP, space=8):
    """paragraphs: str or (str, dict(size, colour, bold)); one text box."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = tb.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, item in enumerate(paragraphs):
        t, opt = (item, {}) if isinstance(item, str) else item
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = t
        p.space_after = Pt(opt.get("space", space))
        if align:
            p.alignment = align
        run = p.runs[0]
        is_bold = opt.get("bold", bold)
        run.font.name = FONT_BOLD if is_bold else FONT
        run.font.size = Pt(opt.get("size", size))
        run.font.color.rgb = opt.get("colour", colour)
    return tb


def content_slide(number, kicker, title, foot=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    box(s, 0, 0, 13.333, 0.12, NAVY)
    text(s, 0.7, 0.42, 9, 0.3, [kicker.upper()], size=12, colour=BLUE, bold=True)
    text(s, 0.7, 0.75, 11.9, 1.0, [title], size=28, colour=NAVY, bold=True, anchor=MSO_ANCHOR.TOP)
    if foot:
        text(s, 0.7, 6.98, 10.6, 0.3, [foot], size=10.5, colour=MUTED)
    text(s, 11.3, 6.98, 1.35, 0.3, [f"{number} / {TOTAL}"], size=10.5, colour=MUTED, align=PP_ALIGN.RIGHT)
    return s


def card(slide, x, y, w, h, accent=BLUE):
    c = box(slide, x, y, w, h, CARD, RULE, MSO_SHAPE.ROUNDED_RECTANGLE, 0.03)
    box(slide, x + 0.28, y, 0.9, 0.06, accent)  # short accent rule on the top edge
    return c


def stat(slide, x, y, w, value, label, colour=BLUE):
    text(slide, x, y, w, 0.7, [value], size=34, colour=colour, bold=True)
    text(slide, x, y + 0.72, w, 0.8, [label], size=13, colour=MUTED)


def bullets(slide, x, y, w, h, heading, items, size=16):
    text(slide, x, y, w, 0.5, [heading], size=20, colour=NAVY, bold=True)
    text(slide, x, y + 0.62, w, h, items, size=size, colour=INK, space=11)


# 1. title
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = NAVY
box(s, 0.8, 1.55, 0.09, 2.5, BLUE)
text(s, 1.15, 1.45, 10.8, 2.6, ["Can reinforcement learning beat the best scheduling rule?"], size=44, colour=CARD, bold=True)
text(s, 1.15, 4.15, 10.2, 1.5, ["We trained neural networks to schedule cloud jobs. They learned to copy good rules of thumb, "
                                "but on fresh test instances none was significantly better than the best rule."],
     size=20, colour=RGBColor(0xB9, 0xC6, 0xD8))
box(s, 0, 6.55, 13.333, 0.95, RGBColor(0x08, 0x17, 0x2B))
text(s, 1.15, 6.82, 11.5, 0.4, ["Ty Riekert   |   Curtin University, Year 3 research project   |   Supervisors: Elham Mardaneh, Tony Mathew"],
     size=13, colour=RGBColor(0x9A, 0xAB, 0xC2))

# 2. why it matters (plain language; relevance to other fields)
s = content_slide(2, "Why it matters", "Deciding what to run first, and where, shows up everywhere",
                  "Our test case: cloud computing jobs on a group of machines. The same structure appears in the fields on the right (by analogy; we tested only the cloud case).")
card(s, 0.7, 1.95, 5.75, 4.7)
bullets(s, 1.0, 2.3, 5.2, 4.0, "The problem",
        ["Work arrives over time, each piece with a deadline and a cost if it is late",
         "Something must decide which piece goes next and on which machine",
         "Good choices mean fewer late jobs and less energy spent keeping machines on"], size=16)
card(s, 6.75, 1.95, 5.9, 4.7, accent=ORANGE)
text(s, 7.05, 2.3, 5.3, 0.5, ["Same problem, other fields"], size=20, colour=NAVY, bold=True)
fields = [("Data centres", "which task runs where, and when machines can be switched off"),
          ("Hospitals", "which operation or scan comes next, in which room"),
          ("Factories and logistics", "which order is built or delivered first"),
          ("Cloud cost and energy", "a business can trade lateness against running cost")]
for i, (head, body) in enumerate(fields):
    y = 3.0 + i * 0.9
    box(s, 7.05, y + 0.05, 0.12, 0.62, ORANGE)
    text(s, 7.35, y, 5.1, 0.35, [head], size=15, colour=NAVY, bold=True, space=2)
    text(s, 7.35, y + 0.34, 5.1, 0.5, [body], size=13, colour=MUTED)

# 3. what we did
s = content_slide(3, "What we did", "We compared classic rules with a neural network that learns to schedule",
                  "Rules named in the report: EDF (earliest deadline first), MDC, COVERT, WLST, ATC. PPO is the reinforcement learning method used.")
steps = [("1", "Classic rules", "Simple, fast recipes such as \"do the most urgent job first\". Used in practice today."),
         ("2", "A learned scheduler", "A neural network trained by trial and error (reinforcement learning), or first taught by copying a rule, then improved."),
         ("3", "A fair test", "Both tried on 50 fresh test cases that no design decision had ever seen, scored on lateness, late jobs and energy.")]
for i, (num, head, body) in enumerate(steps):
    x = 0.7 + i * 4.1
    card(s, x, 1.95, 3.85, 3.3)
    box(s, x + 0.3, 2.3, 0.55, 0.55, BLUE, shape=MSO_SHAPE.OVAL)
    text(s, x + 0.3, 2.3, 0.55, 0.55, [num], size=18, colour=CARD, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, 3.05, 3.3, 0.5, [head], size=19, colour=NAVY, bold=True)
    text(s, x + 0.3, 3.6, 3.3, 1.6, [body], size=14, colour=MUTED)
box(s, 0.7, 5.55, 11.95, 1.1, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
text(s, 1.05, 5.78, 11.3, 0.7, [("The question: can the learned scheduler beat the best classic rule?", dict(size=19, colour=CARD, bold=True))],
     anchor=MSO_ANCHOR.MIDDLE)

# 4. result
s = content_slide(4, "What we found", "No learned scheduler was significantly better than the best classic rule",
                  "Online, 10 machines, load 0.95; 50 fresh test cases; exploratory results; 10 training runs per learned setting.")
card(s, 0.7, 1.95, 8.35, 4.85)
s.shapes.add_picture(str(SLIDES / "deck_assets" / "result_simple.png"), Inches(0.85), Inches(2.1), width=Inches(8.05))
stat(s, 9.5, 2.1, 3.2, "0 of 3", "learned setups significantly better than the best rule", colour=ORANGE)
stat(s, 9.5, 3.65, 3.2, "Close", "copies of a rule finish near it; extra training did not help")
stat(s, 9.5, 5.2, 3.2, "Wide", "learning from scratch varied a lot between runs")

# 5. demo
s = content_slide(5, "Try it", "Which scheduler is best for what you care about?")
steps = [("1", "Choose the environment", "How busy the machines are, how tight the deadlines are, online or offline"),
         ("2", "Choose what matters", "Energy, lateness, or your own weighted mix of metrics"),
         ("3", "Press Start", "The two best rules race the two best learned models; 100% = the best score of that scenario")]
for i, (num, head, body) in enumerate(steps):
    x = 0.7 + i * 4.1
    card(s, x, 1.95, 3.85, 2.75)
    box(s, x + 0.3, 2.3, 0.55, 0.55, BLUE, shape=MSO_SHAPE.OVAL)
    text(s, x + 0.3, 2.3, 0.55, 0.55, [num], size=18, colour=CARD, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, 3.05, 3.3, 0.5, [head], size=19, colour=NAVY, bold=True)
    text(s, x + 0.3, 3.6, 3.3, 1.0, [body], size=14, colour=MUTED)
box(s, 0.7, 5.05, 11.95, 1.55, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
text(s, 1.05, 5.3, 11.3, 1.1, [("Things to notice", dict(size=14, colour=RGBColor(0x8F, 0xB8, 0xEE), bold=True)),
                               ("The winner changes with the goal and with the load. Learned models are mostly copies of a rule, "
                                "so they finish beside it. What would you weight?", dict(size=17, colour=CARD))])

# 6. development, limits, next
s = content_slide(6, "What I learned", "Skills built this year, honest limits, and what comes next",
                  "Details and per-seed values: Future/research/2026-10-21-final-week-findings.md")
card(s, 0.7, 1.95, 4.0, 4.7)
bullets(s, 1.0, 2.3, 3.5, 4.0, "Skills developed",
        ["Reinforcement learning and imitation (PPO, behaviour cloning)",
         "Experiment design: fresh test sets, paired comparisons, pre-set wording rules",
         "Research software: queued training runs, reproducible results archive",
         "Communicating negative results honestly"], size=15.5)
card(s, 4.95, 1.95, 3.9, 4.7, accent=ORANGE)
bullets(s, 5.25, 2.3, 3.4, 4.0, "Honest limits",
        ["Few training runs per setting (3-10)", "Best rule was picked on the original test set",
         "Extra training tried at one tuned setting only", "Exploratory, not proof that RL cannot win"], size=15.5)
card(s, 9.1, 1.95, 3.55, 4.7)
bullets(s, 9.4, 2.3, 3.05, 4.0, "Next",
        ["One objective that weights all metrics together", "Explain why from-scratch learning is unstable",
         "[Add your own career goal here]"], size=15.5)

prs.save(OUT)
print("wrote", OUT)
