"""Streamlit demo: describe your situation in plain language, say what matters, press Start, watch the schedulers race.

    streamlit run app/app.py

Tab 1: three steps (situation, what matters, start). The top 2 classic rules and 3 AI schedulers race (1 that copied a rule, 2 that learned by trial and error) on a bar scale where
100% is the best score of the scenario. Tab 2: the slides (app/slides/). Everything shown comes from the archive of tested scenarios
(app/scoring.py explains the mapping from questions to the recorded metrics).
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import (EXAMPLES, FACTORS, HEURISTIC, disambiguate, factor_weights, friendly_name, load_results,  # noqa: E402
                     metric_label, DISPLAY_METRICS, presets, race_entries, score, short_label)

SLIDES = sorted((Path(__file__).resolve().parent / "slides").glob("*.png"))
BUSY = {"Quiet": "on_rho050", "Normal": "on_rho075", "Busy": "on_rho095", "Overloaded": "on_rho110"}
BUSY_HELP = {"Quiet": "machines are about half used", "Normal": "about three quarters used",
             "Busy": "nearly full", "Overloaded": "more work arrives than the machines can finish"}
DEADLINES_OFFLINE = {"Relaxed": "tf02", "Medium": "tf05", "Tight": "tf08"}
RACE_SECONDS, RACE_FRAMES = 3.5, 45
NOISE_MARGIN = 0.05  # score gaps below this are called noise (the project's +/-5% equivalence margin)
HEUR_COLOUR, RL_COLOUR = "#8a8a85", "#2a78d6"

st.set_page_config(page_title="Which scheduler wins?", layout="wide")
df = load_results()


def apply_example(example):
    for key in FACTORS:
        st.session_state[f"f_{key}"] = EXAMPLES[example].get(key, 0)


def scenario_preset(arrival, busy, tight, offline_deadlines, equal_importance):
    if arrival.startswith("Everything"):
        return f"off_{DEADLINES_OFFLINE[offline_deadlines]}" + ("_w1" if equal_importance else "")
    return "on_rho075_tight" if tight and BUSY[busy] == "on_rho075" else BUSY[busy]


def race_html(entries, fraction):
    """Horizontal bars with a plain-language title and detail line; the better ones pull ahead during the animation."""
    rows = []
    for _, r in entries.iterrows():
        colour = HEUR_COLOUR if r["family"] == HEURISTIC else RL_COLOUR
        width = r["percent"] * fraction
        rows.append(
            f'<div style="margin:14px 0"><div style="font-size:18px;font-weight:600">{r["title"]}'
            f'<span style="float:right">{width:.0f}%</span></div>'
            f'<div style="font-size:13px;opacity:.7;margin-bottom:4px">{r["detail"]}</div>'
            f'<div style="background:rgba(128,128,128,.22);border-radius:8px;height:30px">'
            f'<div style="width:{width:.1f}%;background:{colour};height:30px;border-radius:8px"></div></div></div>')
    return "".join(rows)


tab_race, tab_slides = st.tabs(["Which scheduler wins?", "Slides"])

with tab_race:
    st.title("Which scheduler wins for you?")
    st.write("A *scheduler* decides which job runs next and on which machine. Describe your situation and what matters to you, "
             "then watch the best classic rules race three AI schedulers: one that copied a rule and two different ones that learned by trial and error, without copying.")

    st.header("1. Your situation")
    c1, c2, c3 = st.columns(3)
    with c1:
        arrival = st.radio("How do jobs reach you?", ["Jobs keep arriving over time", "Everything is known in advance"])
    online = arrival.startswith("Jobs")
    with c2:
        if online:
            busy = st.select_slider("How busy are the machines?", list(BUSY), value="Busy")
            st.caption(BUSY_HELP[busy])
        else:
            offline_deadlines = st.select_slider("How tight are the deadlines?", list(DEADLINES_OFFLINE), value="Medium")
    with c3:
        if online:
            tight = st.checkbox("Tight deadlines", disabled=BUSY[busy] != "on_rho075",
                                help="We only tested tight deadlines at 'Normal' busyness.")
            if BUSY[busy] != "on_rho075":
                st.caption("Tight deadlines were only tested when machines are 'Normal'.")
        else:
            equal_importance = st.radio("Are all jobs equally important?", ["Yes, all equal", "No, some matter more"]) .startswith("Yes")
    if online:
        offline_deadlines, equal_importance = None, None
    else:
        busy, tight = None, False

    st.header("2. What matters to you?")
    st.caption("Start from an example, or set how important each thing is (0 = ignore, 5 = very important).")
    if "f_late" not in st.session_state:
        apply_example("Never be late")
    for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(example, on_click=apply_example, args=(example,), use_container_width=True)
    importance = {}
    for col, (key, (question, _)) in zip(st.columns(len(FACTORS)), FACTORS.items()):
        importance[key] = col.slider(question, 0, 5, key=f"f_{key}")

    preset = scenario_preset(arrival, busy, tight, offline_deadlines, equal_importance)
    weights = factor_weights(importance)
    st.header("3. Start")
    start = st.button("Start the race", type="primary", disabled=not any(weights.values()) or preset not in presets(df))
    if not any(weights.values()):
        st.caption("Give at least one thing an importance above 0.")

    if start:
        sub = df[(df["preset"] == preset) & (df["family"].isin([HEURISTIC, "RL"]))]
        scored = score(sub, weights, normalise=True)
        entries = race_entries(scored)
        if entries.empty:
            st.warning("Nothing to compare for this combination - try a different importance setting.")
            st.stop()
        if not (entries["family"] == "RL").any():  # no AI scheduler was trained for this setting: race four rules instead
            entries = race_entries(scored, per_family=4, ai_mix=())
        parts = [friendly_name(m, f) for m, f in zip(entries["method"], entries["family"])]
        entries["title"], entries["detail"] = [p[0] for p in parts], [p[1] for p in parts]
        st.session_state["race"] = (entries, scored)
        slot = st.empty()
        for i in range(RACE_FRAMES + 1):
            t = i / RACE_FRAMES
            slot.markdown(race_html(entries, 1 - (1 - t) ** 2), unsafe_allow_html=True)  # ease-out
            time.sleep(RACE_SECONDS / RACE_FRAMES)

    if "race" in st.session_state:
        entries, scored = st.session_state["race"]
        if not start:
            st.markdown(race_html(entries, 1.0), unsafe_allow_html=True)
        winner = entries.sort_values("percent", ascending=False).iloc[0]
        kind = "AI scheduler" if winner["family"] == "RL" else "classic rule"
        st.subheader(f"Winner: {winner['title']} - {kind}")
        st.caption("Grey bars are classic rules, blue bars are AI schedulers. 100% = the best result we found for this situation and "
                   "what you chose; the bars are stretched so small differences show (about 5% worse shows as ~82%).")
        rl_scores = scored[scored["family"] == "RL"]["score"]
        heur_scores = scored[scored["family"] == HEURISTIC]["score"]
        if len(rl_scores):
            gap = (rl_scores.min() - heur_scores.min()) / abs(heur_scores.min() or 1.0)
            if gap >= 0:
                st.info("No AI scheduler beat the best classic rule here. That is the usual result in this project: the AI "
                        "schedulers mostly learned to copy a good rule and finish close to it.")
            elif gap > -NOISE_MARGIN:
                st.info(f"The best AI scheduler is only {-gap * 100:.1f}% ahead of the best classic rule, which is within the "
                        "run-to-run noise of this project (gaps under about 5% were not significant).")
        else:
            st.info("We did not train an AI scheduler for this exact setting, so four classic rules race instead.")
        n_scratch = int((entries["family"] == "RL").sum() - (entries["detail"].str.startswith(("learned by watching",))).sum())
        if len(rl_scores) and n_scratch < 2:
            st.caption(f"Only {n_scratch} AI scheduler(s) that learned without copying were trained for this setting, so fewer than three AI schedulers race.")
        with st.expander("What does this show, and what did we test?"):
            st.markdown(
                "- Scores come from 50 test cases per setting (generated job sets). The best classic rule was picked on those same "
                "cases, so rules look a little better here than on new cases. Results are exploratory.\
"
                "- **Tested:** how busy the machines are (4 levels), tight deadlines (at normal busyness), online versus offline, "
                "offline deadline tightness (3 levels), and equal versus unequal job importance (offline). Always 10 machines.\
"
                "- **Not tested:** different numbers of machines or job sizes, and hard limits such as 'no job may wait more than X'. "
                "Late jobs are never refused; they are counted, so asking to avoid late jobs finds the schedulers that keep them fewest.\
"
                "- Each AI scheduler was trained 3-10 times; the bar shows the average of those training runs.")
        with st.expander("Full ranking (for the curious)"):
            view = scored.head(25).copy()
            view["model"] = disambiguate([short_label(m) for m in view["method"]])
            used = [m for m in DISPLAY_METRICS if weights.get(m)]
            st.dataframe(view[["model", "family", "n_seeds", "score"] + used].rename(columns={m: metric_label(m) for m in used}),
                         use_container_width=True, hide_index=True)

with tab_slides:
    if SLIDES:
        idx = st.slider("Slide", 1, len(SLIDES), 1) if len(SLIDES) > 1 else 1
        st.image(str(SLIDES[idx - 1]), use_container_width=True)
    else:
        st.info("Put slide images (PNG) in app/slides/ - shown here in filename order.")
