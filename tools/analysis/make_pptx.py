"""WARNING (2026-10-26): the deck app/slides/Final_Presentation.pptx has been edited by hand and by tools/analysis/edit_final_deck.py;
running this script rebuilds it from scratch and OVERWRITES those edits (a copy of the hand-edited deck is in app/slides/backup/).

Build the final presentation: app/slides/Final_Presentation.pptx (16:9, four slides, editable; speaker notes included).

    python tools/analysis/make_slides.py      (first: draws the two result charts into app/slides/deck_assets/)
    python tools/analysis/make_pptx.py        (needs python-pptx)

Story: 1 the problem (numbers, relevance) -> 2 the method (the reward loop, link to an interactive training demo) -> 3 the results
(two graphs, link to the Scheduler Race page) -> 4 skills built, link to career, what is next. Wording follows
Future/research/2026-10-21-final-week-findings.md ("For the presentation"): no "AI beats / matches the rules".
Links are relative (scheduler_race.html, training_demo.html): keep those files in the same folder as the deck.
"""
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "app" / "slides" / "deck_assets"
OUT = ROOT / "app" / "slides" / "Final_Presentation.pptx"
RACE_LINK, TRAIN_LINK = "scheduler_race.html", "training_demo.html"

INK, BLUE, SOFT, TINT = RGBColor(0x0E, 0x22, 0x38), RGBColor(0x2A, 0x6F, 0xD1), RGBColor(0xCB, 0xD9, 0xEE), RGBColor(0xEE, 0xF4, 0xFC)
GREY, ORANGE, WHITE, MUTED = RGBColor(0x87, 0x93, 0xA1), RGBColor(0xE0, 0x70, 0x2A), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x56, 0x67, 0x7A)
FONT, FONT_BOLD = "Segoe UI", "Segoe UI Semibold"
TOTAL = 4

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


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


def text(slide, x, y, w, h, paragraphs, size=18, colour=INK, bold=False, align=None, anchor=MSO_ANCHOR.TOP, space=6):
    """paragraphs: str or (str, dict(size, colour, bold)); one text box without margins."""
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
        is_bold = opt.get("bold", bold)
        for run in p.runs:  # every run: a line break inside the text makes a second run
            run.font.name = FONT_BOLD if is_bold else FONT
            run.font.size = Pt(opt.get("size", size))
            run.font.color.rgb = opt.get("colour", colour)
    return tb


def label_in(shape, t, size=16, colour=WHITE, bold=True):
    """Centre a label inside a shape."""
    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.08)
    p = frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = t
    for run in p.runs:
        run.font.name = FONT_BOLD if bold else FONT
        run.font.size = Pt(size)
        run.font.color.rgb = colour


def new_slide(number, kicker, title, notes):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    text(s, 0.7, 0.4, 9, 0.3, [kicker.upper()], size=12, colour=BLUE, bold=True)
    text(s, 0.7, 0.72, 11.9, 1.0, [title], size=30, colour=INK, bold=True)
    text(s, 0.7, 7.05, 8, 0.3, ["Ethan Ty Riekert  |  Curtin University  |  Supervisors: Elham Mardaneh, Tony Mathew"], size=10, colour=MUTED)
    text(s, 11.3, 7.05, 1.35, 0.3, [f"{number} / {TOTAL}"], size=10, colour=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def panel(slide, x, y, w, h, fill=WHITE, line=SOFT):
    return box(slide, x, y, w, h, fill, line, MSO_SHAPE.ROUNDED_RECTANGLE, 0.04, 1.25)


def button(slide, x, y, w, h, label, link=None, size=17):
    b = box(slide, x, y, w, h, BLUE, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.3)
    label_in(b, label, size=size)
    if link:
        b.click_action.hyperlink.address = link
    return b


def stat(slide, x, y, w, value, caption, colour=BLUE):
    text(slide, x, y, w, 0.75, [value], size=38, colour=colour, bold=True)
    text(slide, x, y + 0.78, w, 0.7, [caption], size=14, colour=MUTED)


# ---------------------------------------------------------------- 1. the problem
s = new_slide(1, "The problem", "Every cloud job is a decision: what runs next, and where?",
              "Open with the picture. Each block is a job on a machine; the orange ones finish after their deadline and cost money. "
              "Then the three numbers: nearly every business already runs on the cloud, an hour of downtime is very expensive, and the market "
              "is still growing. The same decision (which job next, on which machine) appears in hospitals, factories and deliveries, which "
              "is why a better scheduler matters beyond computing. Finish on the question: can reinforcement learning, a computer teaching "
              "itself by reward, schedule better than the best hand-made rule? [1] DigitalDefynd, 2026.")
LX, LW, LY = 1.75, 5.5, 2.15                      # schedule drawing: lane x, lane width, first lane y
panel(s, 0.7, 1.75, 6.9, 3.65, fill=TINT)
lanes = [[(0.00, .18, 0), (.20, .22, 0), (.44, .20, 0), (.66, .22, 1)], [(0.00, .30, 0), (.32, .18, 0), (.52, .24, 0), (.78, .14, 1)],
         [(0.00, .14, 0), (.16, .30, 0), (.48, .20, 0), (.70, .20, 1)], [(0.00, .24, 0), (.26, .16, 0), (.44, .30, 0), (.76, .18, 1)]]
for i, lane in enumerate(lanes):
    y = LY + i * 0.68
    text(s, 0.95, y + 0.1, 0.7, 0.3, [f"M{i + 1}"], size=13, colour=MUTED, bold=True)
    box(s, LX, y, LW, 0.5, WHITE, SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    for start, length, late in lane:
        box(s, LX + LW * start + 0.03, y + 0.06, LW * length - 0.06, 0.38, ORANGE if late else BLUE, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.25)
dx = LX + LW * 0.78
box(s, dx, LY - 0.18, 0.035, 4 * 0.68 + 0.05, INK)
text(s, dx - 0.5, LY - 0.45, 1.2, 0.3, ["deadline"], size=12, colour=INK, bold=True, align=PP_ALIGN.CENTER)
text(s, LX, LY + 2.78, LW, 0.3, ["time  →"], size=12, colour=MUTED, align=PP_ALIGN.RIGHT)
text(s, 0.95, 4.95, 6.4, 0.4, ["A schedule: each block is a job on a machine. Orange jobs finish late."], size=13, colour=MUTED)
for k, (value, caption, colour) in enumerate([("94%", "of enterprises already use cloud services [1]", BLUE),
                                              ("$8,000 / min", "average cost of cloud downtime (AUD) [1]", ORANGE),
                                              ("$1.7 trillion", "expected global cloud market by 2033 (AUD) [1]", BLUE)]):
    stat(s, 8.1, 1.75 + k * 1.2, 4.5, value, caption, colour)
panel(s, 0.7, 5.65, 11.95, 1.15, fill=TINT, line=TINT)
text(s, 1.0, 5.82, 11.4, 0.9, [("Same decision, many fields: data centres, hospitals, factories, deliveries.", dict(size=16, colour=MUTED)),
                               ("Our question: can reinforcement learning schedule better than the best hand-made rule?", dict(size=20, bold=True))], space=4)
text(s, 0.7, 6.82, 11, 0.22, ["[1] DigitalDefynd, “30 Surprising Cloud Computing Facts & Statistics,” 2026."], size=9, colour=MUTED)

# ---------------------------------------------------------------- 2. the method
s = new_slide(2, "The method", "The scheduler is a neural network that learns from reward",
              "Explain the loop. The agent is a neural network. It sees the state (which jobs are waiting, how close their deadlines are, how full "
              "the machines are), chooses an action (which job to start next, or wait), and the environment returns a reward: minus the lateness "
              "cost. Repeat millions of times and it improves. We tried three things: classic rules, a network that learns on its own, and a "
              "network that first copies a rule. Then invite the audience to the interactive training demo (link to be added when the demo is built) "
              "and point to the cost formula on the right: weight times lateness squared, with optional late-job and energy terms.")
panel(s, 0.7, 1.75, 7.4, 3.55, fill=TINT, line=TINT)
agent = box(s, 0.95, 2.3, 2.5, 1.7, BLUE, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.12)
label_in(agent, "Agent\nneural network", 18)
env = box(s, 5.35, 2.3, 2.5, 1.7, INK, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.12)
label_in(env, "Environment\n10 machines + jobs", 18)
a1 = box(s, 3.55, 2.35, 1.7, 0.55, SOFT, None, MSO_SHAPE.RIGHT_ARROW)
a2 = box(s, 3.55, 3.4, 1.7, 0.55, SOFT, None, MSO_SHAPE.LEFT_ARROW)
text(s, 3.3, 2.95, 2.2, 0.3, ["action: start a job"], size=13, colour=INK, bold=True, align=PP_ALIGN.CENTER)
text(s, 3.3, 3.98, 2.2, 0.3, ["new state + reward"], size=13, colour=INK, bold=True, align=PP_ALIGN.CENTER)
for k, (head, body) in enumerate([("State", "waiting jobs, deadlines, machine loads"), ("Action", "which job to start next (or wait)"),
                                  ("Reward", "minus the lateness cost")]):
    x = 0.95 + k * 2.4
    text(s, x, 4.45, 2.3, 0.3, [head], size=15, colour=BLUE, bold=True)
    text(s, x, 4.78, 2.25, 0.6, [body], size=13, colour=MUTED)
panel(s, 8.4, 1.75, 4.25, 1.75)
text(s, 8.7, 1.9, 3.7, 0.3, ["WHAT IT TRIES TO MINIMISE"], size=11, colour=BLUE, bold=True)
text(s, 8.7, 2.2, 3.7, 0.5, ["J = Σ w × (lateness)²"], size=24, colour=INK, bold=True)
text(s, 8.7, 2.85, 3.7, 0.6, ["Late jobs hurt more the later they are; important jobs count more. Optional terms: late jobs, energy."], size=12, colour=MUTED)
panel(s, 8.4, 3.65, 4.25, 1.65, fill=TINT, line=TINT)
text(s, 8.7, 3.8, 3.7, 0.3, ["THREE KINDS OF SCHEDULER"], size=11, colour=BLUE, bold=True)
text(s, 8.7, 4.1, 3.75, 1.2, ["Classic rules, made by hand", "AI that learns on its own", "AI that first copies a rule"], size=14, space=3)
panel(s, 0.7, 5.55, 11.95, 1.25)
text(s, 1.0, 5.8, 7.0, 0.8, [("Watch it learn", dict(size=22, bold=True)), ("Train a tiny scheduler yourself and see its reward rise.", dict(size=14, colour=MUTED))], space=3)
button(s, 8.4, 5.78, 3.95, 0.8, "Interactive training demo →", TRAIN_LINK, 17)

# ---------------------------------------------------------------- 3. the results
s = new_slide(3, "What we found", "The AI learned to match the best rules, but did not beat them",
              "Left graph: busy machines, 50 fresh test cases the models had never seen. Lower is better. The grey bar is the best classic rule. "
              "The AI that copies a rule lands close to it; extra training did not help; learning completely on its own was worse and varied a lot "
              "between runs (the dots are separate training runs). Right graph: the same story in five scenarios, cost relative to the best rule. "
              "Be honest about the limits: 3 to 10 training runs per bar, exploratory results, no AI scheduler was significantly better than the "
              "best rule. Then send the audience to the Scheduler Race page to choose their own priorities (late jobs, waiting, energy) and see who wins.")
for k, (head, img) in enumerate([("Busy machines: cost on 50 fresh test cases", "final_chart1.png"), ("Five scenarios: cost relative to the best rule", "final_chart2.png")]):
    x = 0.7 + k * 6.1
    panel(s, x, 1.75, 5.85, 3.95)
    text(s, x + 0.25, 1.88, 5.4, 0.3, [head], size=13, colour=INK, bold=True)
    s.shapes.add_picture(str(ASSETS / img), Inches(x + 0.2), Inches(2.22), width=Inches(5.45))
text(s, 0.7, 5.85, 8.0, 0.9, [("No AI scheduler was significantly better than the best classic rule.", dict(size=18, bold=True)),
                              ("Exploratory results: 50 fresh test cases per scenario, 3 to 10 training runs per bar (dots).", dict(size=12, colour=MUTED))], space=3)
button(s, 9.0, 5.9, 3.65, 0.8, "Race the models →", RACE_LINK, 18)

# ---------------------------------------------------------------- 4. skills, career, next
s = new_slide(4, "What I built and where it leads", "From following a plan to leading the next project",
              "Skills: I turned a real cloud problem into mathematics, learned reinforcement and imitation learning, designed experiments with fresh "
              "test data, and reported a null result honestly. Link to the future: in second year I made quantum vibrations visible in MATLAB; "
              "this year I made scheduling decisions visible and interactive. The thread is the same, using computation to make hidden dynamics "
              "understandable to non-experts. Machine learning and mathematical optimisation are the direction I want to build a career in. "
              "Next: train for trade-offs between late jobs and energy, test forecasts of arriving jobs, explain why learning from scratch is "
              "unstable, and take a more active role in leading a project. [Replace the last bullet with your own next step.]")
cols = [("Skills built", BLUE, [("Modelling", "a cloud problem turned into mathematics"), ("Machine learning", "reinforcement and imitation learning"),
                                ("Research rigour", "fresh test data, honest reporting"), ("Communication", "interactive tools for non-experts")]),
        ("Where it leads", ORANGE, [("My direction", "machine learning and mathematical optimisation"), ("The thread", "computation that makes hidden dynamics visible"),
                                    ("Audience", "from experts to curious visitors")]),
        ("What is next", BLUE, [("Trade-offs", "train for late jobs versus energy"), ("Forecasts", "can predicting arrivals help?"),
                                ("Understanding", "why learning on its own is unstable"), ("Leading", "[my next step: own a project]")])]
for k, (head, colour, items) in enumerate(cols):
    x = 0.7 + k * 4.05
    panel(s, x, 1.75, 3.85, 3.55)
    box(s, x + 0.3, 1.75, 0.9, 0.07, colour)
    text(s, x + 0.3, 1.98, 3.3, 0.4, [head], size=20, colour=INK, bold=True)
    text(s, x + 0.3, 2.55, 3.3, 2.7, [item for lead, body in items for item in ((lead, dict(size=15, bold=True, space=0)), (body, dict(size=13, colour=MUTED, space=8)))], size=14)
steps = [("Year 2", "CO₂ and quantum vibrations\nMATLAB visualisation"), ("Year 3", "Cloud scheduling with RL\nPython, interactive demos"), ("Next", "Lead a project\nmachine learning + optimisation")]
for k, (head, body) in enumerate(steps):
    x = 0.7 + k * 4.05
    c = box(s, x, 5.6, 3.85, 1.2, INK if k == 2 else TINT, None, MSO_SHAPE.PENTAGON)
    text(s, x + 0.3, 5.7, 3.1, 1.0, [(head, dict(size=15, bold=True, colour=WHITE if k == 2 else BLUE, space=2)), (body, dict(size=12, colour=WHITE if k == 2 else INK))], space=2)

prs.save(OUT)
race_page = ROOT / "app" / "web" / "dist" / "variant_a.html"  # the deck's button opens this copy: keep it in the deck's folder
if race_page.exists():
    shutil.copyfile(race_page, OUT.parent / RACE_LINK)
print("wrote", OUT)
