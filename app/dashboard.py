"""Predictive maintenance dashboard.

Run from the project root:  streamlit run app/dashboard.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from src import conformal, maintenance as mt
from src.data import SUBSETS
from src.features import prepare, rolling_features, split_engines

COLORS = {"Healthy": "#2e9e44", "Degrading": "#f0a202", "Critical": "#d1342f"}

st.set_page_config(page_title="Engine RUL Dashboard", layout="wide")


# ---------- model (trained once per subset, then cached) ----------

@st.cache_resource(show_spinner="Training the model for this subset, about a minute...")
def build(subset):
    """Same setup as models/run_phase6.py: 60% train, 20% calibration, 20% engines to replay."""
    train, _, sensors = prepare(subset)
    feats = rolling_features(train, sensors)
    cols = [c for c in feats.columns if c not in ("unit", "cycle", "RUL")]

    tr, rest = split_engines(train, val_frac=0.4)
    cal, ev = split_engines(rest, val_frac=0.5)
    f_tr, f_cal, f_ev = (feats[feats.unit.isin(d.unit)] for d in (tr, cal, ev))

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(f_tr[cols], f_tr.RUL)
    q = conformal.fit(rf.predict(f_cal[cols]), f_cal.RUL.to_numpy())

    pred = rf.predict(f_ev[cols])
    lower, upper = conformal.interval(pred, q)
    f_ev = f_ev.assign(pred=pred, lower=lower, upper=upper)
    sensors_ev = train[train.unit.isin(ev.unit)]          # normalized sensor values, for plotting
    return rf, cols, f_ev, sensors_ev, shap.TreeExplainer(rf)


# ---------- sidebar ----------

st.sidebar.title("Settings")
subset = st.sidebar.selectbox("Dataset", SUBSETS)
rf, cols, f_ev, sensors_ev, explainer = build(subset)

unit = st.sidebar.selectbox("Engine", sorted(f_ev.unit.unique()))
engine = f_ev[f_ev.unit == unit].reset_index(drop=True)
raw = sensors_ev[sensors_ev.unit == unit].reset_index(drop=True)
last = int(engine.cycle.max())

margin = st.sidebar.slider("Safety margin (cycles)", 5, 40, mt.LEAD,
                           help="Service the engine when the lower end of the 90% interval drops below this.")
cycle = st.sidebar.slider("Cycle", 1, last, min(30, last))
play = st.sidebar.button("▶ Play from this cycle to failure")
show_true = st.sidebar.checkbox("Show true RUL", value=True)


# ---------- drawing ----------

def recommendation(lower):
    if lower < margin:
        return "Service now"
    return f"Service within {lower - margin:.0f} cycles"


def rul_chart(c):
    e = engine[engine.cycle <= c]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(e.cycle, e.lower, e.upper, color="grey", alpha=0.25, label="90% interval")
    ax.plot(e.cycle, e.pred, color="#1f4e79", lw=2, label="predicted RUL")
    if show_true:
        ax.plot(e.cycle, e.RUL, "k--", lw=1, label="true RUL (capped)")
    ax.axhline(margin, color=COLORS["Critical"], lw=1, ls=":", label="safety margin")
    ax.set(xlim=(0, last), ylim=(0, 150), xlabel="cycle", ylabel="RUL (cycles)")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def sensor_chart(c):
    e = raw[raw.cycle <= c]
    fig, ax = plt.subplots(figsize=(5, 3))
    for s in ["s4", "s11", "s9"]:
        ax.plot(e.cycle, e[s].rolling(5, min_periods=1).mean(), label=s)
    ax.set(xlim=(0, last), xlabel="cycle", ylabel="normalized value")
    ax.set_title("Top sensors (5-cycle average)", fontsize=10)
    ax.legend(fontsize=8)
    return fig


def shap_chart(c):
    row = engine.loc[engine.cycle == c, cols]
    contrib = pd.Series(explainer.shap_values(row)[0], index=cols)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])[::-1]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(top.index, top.values, color=[COLORS["Critical"] if v < 0 else "#1f4e79" for v in top])
    ax.set_xlabel("effect on predicted RUL (cycles)")
    ax.set_title("Why this prediction? (red = closer to failure)", fontsize=10)
    return fig


def show(c, explain=True):
    now = engine[engine.cycle == c].iloc[0]
    state = mt.health_state(now.pred, now.lower, lead=margin)

    with top.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cycle", f"{c} / {last}")
        m2.metric("Predicted RUL", f"{now.pred:.0f} cycles", f"90%: {now.lower:.0f} - {now.upper:.0f}", delta_color="off")
        m3.markdown(f"**Status**<br><span style='font-size:28px;color:{COLORS[state]}'>● {state}</span>",
                    unsafe_allow_html=True)
        m4.markdown(f"**Recommendation**<br><span style='font-size:22px'>{recommendation(now.lower)}</span>",
                    unsafe_allow_html=True)

    chart.pyplot(rul_chart(c))
    if explain:                      # skipped while playing, to keep the replay fast
        left.pyplot(sensor_chart(c))
        right.pyplot(shap_chart(c))
    else:
        right.info("Sensors and explanation appear when the replay stops.")
    plt.close("all")


# ---------- page ----------

st.title("Turbofan Engine Health Monitor")
st.caption(f"{subset}, engine {unit}. Random Forest with 90% conformal prediction intervals.")

tab_replay, tab_compare = st.tabs(["Engine replay", "Model comparison"])

with tab_replay:
    top = st.empty()
    chart = st.empty()
    col_left, col_right = st.columns(2)
    left, right = col_left.empty(), col_right.empty()

    if play:
        for c in range(cycle, last, 5):    # 5 cycles per frame
            show(c, explain=False)
            time.sleep(0.05)
        show(last)
    else:
        show(cycle)

    # when would this engine be serviced, and how does that compare with a fixed schedule?
    below = np.flatnonzero(engine.lower.to_numpy() < margin)
    service = int(below[0]) + 1 if len(below) else None
    if service is None or service >= last:
        st.error(f"With a {margin}-cycle margin this engine would fail before being serviced.")
    else:
        st.success(f"With a {margin}-cycle margin this engine is serviced at cycle {service}, "
                   f"{last - service} cycles before it would have failed.")

with tab_compare:
    reports = ROOT / "reports"
    st.subheader("Prediction accuracy (test set, RUL capped at 125)")
    acc = pd.concat([pd.read_csv(reports / "baseline_results.csv"), pd.read_csv(reports / "deep_results.csv")])
    st.dataframe(acc.pivot(index="model", columns="subset", values="RMSE").round(2))

    st.subheader("Maintenance cost per 1000 cycles (evaluation engines)")
    p6 = pd.read_csv(reports / "phase6_results.csv").set_index("subset")
    cost = p6[["reactive_cost", "fixed_cost", "predictive_cost", "fixed_failures",
               "predictive_failures", "saving_vs_fixed_%"]].round(2)
    cost.columns = ["Reactive", "Fixed schedule", "Predictive", "Failures (fixed)",
                    "Failures (predictive)", "Saving vs fixed (%)"]
    st.dataframe(cost)
    st.caption("Costs: planned maintenance = 1, failure = 10. The predictive policy uses a 10-cycle safety margin.")
