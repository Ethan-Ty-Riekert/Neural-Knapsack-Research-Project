"""Streamlit demo: pick a preset and a weighting over metrics, see the best models.

    streamlit run app/app.py

Tab 1 is the metric-weighting explorer (the landing view), tab 2 shows the slides (images in app/slides/).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import (DISPLAY_METRICS, best_by_family, load_results, metric_label, presets, score,  # noqa: E402
                     short_label, QUICK_OBJECTIVES, HEURISTIC, disambiguate)

SLIDES = sorted((Path(__file__).resolve().parent / "slides").glob("*.png"))
PRESET_NOTES = {"off": "offline: whole job set known up front", "on": "online: jobs arrive over time"}
DEFAULT_PRESET = "on_rho095"  # the scenario where RL is closest to the best heuristic
PRESETS_OF_INTEREST_HELP = "Presets differ in load / deadline tightness; scores are only comparable within one."

st.set_page_config(page_title="Neural Knapsack results explorer", layout="wide")
df = load_results()

DATA_CAPTION = ("Data: the 50-instance OLD test set of each scenario (the same instances informed many project decisions and "
                "the best heuristic of each scenario was picked on them, so heuristics look better here than they would on "
                "new instances). Fresh-instance results are in the findings document. Exploratory.")

TRY_THIS = [  # (button label, scenario, quick objective): the scripted demo prompts (presentation plan, section 3d)
    ("1. Minimise lateness (load 0.95)", "on_rho095", "Squared tardiness (the training objective J)"),
    ("2. Save energy instead", "on_rho095", "Energy saving (fewest active machine-ticks)"),
    ("3. Same goal under overload", "on_rho110", "Squared tardiness (the training objective J)"),
]


def apply_try(scenario, objective):
    """Jump to a scenario and a one-click objective (sliders follow)."""
    st.session_state["preset"] = scenario
    st.session_state["quick"] = objective
    apply_quick()


def apply_quick():
    """Set every weight slider from the chosen one-click objective (sliders stay editable afterwards)."""
    quick = QUICK_OBJECTIVES[st.session_state["quick"]]
    if quick is not None:
        for m in DISPLAY_METRICS:
            st.session_state[f"w_{m}"] = quick.get(m, 0.0)


tab_explore, tab_slides = st.tabs(["Explore models", "Slides"])

with tab_slides:
    if SLIDES:
        idx = st.slider("Slide", 1, len(SLIDES), 1) if len(SLIDES) > 1 else 1
        st.image(str(SLIDES[idx - 1]), use_container_width=True)
    else:
        st.info("Put slide images (PNG) in app/slides/ - shown here in filename order.")

with tab_explore:
    st.markdown("### Pick what matters to you - see which scheduler wins")
    st.caption("Try this:")
    for col, (label, scenario, objective) in zip(st.columns(len(TRY_THIS)), TRY_THIS):
        if scenario in presets(df):
            col.button(label, on_click=apply_try, args=(scenario, objective), use_container_width=True)
    left, right = st.columns([1, 2])
    with left:
        all_presets = presets(df)
        if "preset" not in st.session_state and DEFAULT_PRESET in all_presets:
            st.session_state["preset"] = DEFAULT_PRESET
        preset = st.selectbox("Scenario (preset)", all_presets, help=PRESETS_OF_INTEREST_HELP, key="preset")
        st.caption(PRESET_NOTES["off" if preset.startswith("off") else "on"])
        normalise = st.checkbox("Normalise metrics by the best heuristic", value=True,
                                help="Each metric is divided by the best heuristic's value on THAT metric (a different heuristic "
                                     "per metric, picked on the same instances), so a weight of 1 on two metrics means equal "
                                     "importance relative to those per-metric best values, which no single method reaches. "
                                     "Metrics where higher is better enter with a negative sign.")
        families = st.multiselect("Show method families", sorted(df["family"].unique()),
                                  default=sorted(df["family"].unique()))
        top_n = st.slider("Models shown", 3, 25, 10)
        st.markdown("**What matters? (weights)**")

        st.selectbox("Quick objective", list(QUICK_OBJECTIVES), key="quick", on_change=apply_quick)
        weights = {}
        for metric in DISPLAY_METRICS:
            weights[metric] = st.slider(metric_label(metric), 0.0, 5.0, 1.0 if metric == "objective_J" else 0.0, 0.25,
                                        key=f"w_{metric}")
    with right:
        st.caption(DATA_CAPTION)
        sub = df[(df["preset"] == preset) & (df["family"].isin(families))]
        if not any(weights.values()):
            st.warning("Give at least one metric a weight above zero.")
        else:
            scored = score(sub, weights, normalise)
            scored["model"] = disambiguate([short_label(m) for m in scored["method"]])
            best = best_by_family(scored)
            cols = st.columns(max(len(best), 1))
            for col, (fam, row) in zip(cols, sorted(best.items())):
                col.metric(f"Best {fam}", f"{row['score']:.3f}", help=row["method"])
                col.caption(short_label(row["method"]))
            rl_rows = scored[scored["family"] == "RL"]
            if len(rl_rows) and HEURISTIC in best:
                ref = best[HEURISTIC]["score"]
                med = rl_rows["score"].median()
                if abs(ref) > 1e-9:
                    st.subheader(f"Median RL model vs best heuristic: {(med - ref) / abs(ref) * 100:+.1f}%")
                else:
                    st.subheader("Median RL model vs best heuristic: not defined (reference score is zero)")
                st.caption(f"{len(rl_rows)} RL model groups on this scenario (median over groups; the best group alone is "
                           f"{(rl_rows['score'].min() - ref) / abs(ref) * 100 if abs(ref) > 1e-9 else float('nan'):+.1f}%, a "
                           "best-of-many pick). Differences are NOT significance-tested; model groups with few seeds are "
                           "indicative only.")
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
