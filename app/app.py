"""Streamlit demo: pick a preset and a weighting over metrics, see the best models.

    streamlit run app/app.py

Tab 1 shows the slides (images in app/slides/), tab 2 is the metric-weighting explorer.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import (DISPLAY_METRICS, best_by_family, load_results, metric_label, presets, score,  # noqa: E402
                     short_label, QUICK_OBJECTIVES, HEURISTIC)

SLIDES = sorted((Path(__file__).resolve().parent / "slides").glob("*.png"))
PRESET_NOTES = {"off": "offline: whole job set known up front", "on": "online: jobs arrive over time"}
DEFAULT_PRESET = "on_rho095"  # the scenario where RL is closest to the best heuristic
PRESETS_OF_INTEREST_HELP = "Presets differ in load / deadline tightness; scores are only comparable within one."

st.set_page_config(page_title="Neural Knapsack results explorer", layout="wide")
df = load_results()

tab_slides, tab_explore = st.tabs(["Slides", "Explore models"])

with tab_slides:
    if SLIDES:
        idx = st.slider("Slide", 1, len(SLIDES), 1) if len(SLIDES) > 1 else 1
        st.image(str(SLIDES[idx - 1]), use_container_width=True)
    else:
        st.info("Put slide images (PNG) in app/slides/ - shown here in filename order.")

with tab_explore:
    left, right = st.columns([1, 2])
    with left:
        all_presets = presets(df)
        preset = st.selectbox("Scenario (preset)", all_presets, help=PRESETS_OF_INTEREST_HELP,
                              index=all_presets.index(DEFAULT_PRESET) if DEFAULT_PRESET in all_presets else 0)
        st.caption(PRESET_NOTES["off" if preset.startswith("off") else "on"])
        normalise = st.checkbox("Normalise metrics by the best heuristic", value=True,
                                help="Weight 1 on two metrics = equal importance relative to a good heuristic.")
        families = st.multiselect("Show method families", sorted(df["family"].unique()),
                                  default=sorted(df["family"].unique()))
        top_n = st.slider("Models shown", 3, 25, 10)
        st.markdown("**What matters? (weights)**")

        def apply_quick():
            """Set every weight slider from the chosen one-click objective (sliders stay editable afterwards)."""
            quick = QUICK_OBJECTIVES[st.session_state["quick"]]
            if quick is not None:
                for m in DISPLAY_METRICS:
                    st.session_state[f"w_{m}"] = quick.get(m, 0.0)

        st.selectbox("Quick objective", list(QUICK_OBJECTIVES), key="quick", on_change=apply_quick)
        weights = {}
        for metric in DISPLAY_METRICS:
            weights[metric] = st.slider(metric_label(metric), 0.0, 5.0, 1.0 if metric == "objective_J" else 0.0, 0.25,
                                        key=f"w_{metric}")
    with right:
        sub = df[(df["preset"] == preset) & (df["family"].isin(families))]
        if not any(weights.values()):
            st.warning("Give at least one metric a weight above zero.")
        else:
            scored = score(sub, weights, normalise)
            scored["model"] = scored["method"].map(short_label)
            best = best_by_family(scored)
            cols = st.columns(max(len(best), 1))
            for col, (fam, row) in zip(cols, sorted(best.items())):
                col.metric(f"Best {fam}", f"{row['score']:.3f}", help=row["method"])
                col.caption(short_label(row["method"]))
            if "RL" in best and HEURISTIC in best:
                gap = (best["RL"]["score"] - best[HEURISTIC]["score"]) / abs(best[HEURISTIC]["score"]) * 100
                st.subheader(f"Best RL vs best heuristic: {gap:+.1f}% ({'RL ahead' if gap < 0 else 'heuristic ahead'})")
            view = scored.head(top_n)[["model", "family", "n_seeds", "score"] + [m for m in DISPLAY_METRICS
                                                                                    if weights[m]]]
            view = view.rename(columns={m: metric_label(m) for m in DISPLAY_METRICS})
            st.dataframe(view, use_container_width=True, hide_index=True)
            chart = scored.head(top_n).set_index("model")["score"]
            st.bar_chart(chart.sort_values(ascending=False), horizontal=True)

            st.subheader("Trade-off view")
            c1, c2 = st.columns(2)
            x_metric = c1.selectbox("Horizontal axis", DISPLAY_METRICS, index=0, format_func=metric_label)
            y_metric = c2.selectbox("Vertical axis", DISPLAY_METRICS, index=DISPLAY_METRICS.index("active_machine_ticks"),
                                    format_func=metric_label)
            points = scored[["model", "family", x_metric, y_metric]].dropna().rename(
                columns={x_metric: metric_label(x_metric), y_metric: metric_label(y_metric)})
            st.scatter_chart(points, x=metric_label(x_metric), y=metric_label(y_metric), color="family")
            st.caption("Every point is one method on this scenario; lower-left is better when both axes are "
                       "lower-is-better metrics.")
