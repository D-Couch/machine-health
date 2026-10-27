from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from machine_health.alert_policy import _first_persistent_alert
from machine_health.data import attach_rul, load_fd001
from machine_health.features import FEATURES, causal_features
from machine_health.model import load_artifact

REPORTS = ROOT / "reports"
DATA = ROOT / "data/raw/train_FD001.txt"
ARTIFACT = ROOT / "artifacts/fd001.joblib"
SPLIT = REPORTS / "fd001_split.json"
TEMPORAL_REPLAY = ROOT / "data/processed/temporal_validation_replay.csv"
TEMPORAL_SPLIT = REPORTS / "temporal_split.json"
COLORS = {"navy": "#10243A", "blue": "#2864DC", "teal": "#16A89A",
          "orange": "#E28A3B", "red": "#D65757", "muted": "#66758A",
          "grid": "#E8EDF3"}

st.set_page_config(page_title="Machine Health | FD001", page_icon="⚙", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
  .stApp { background: #f4f7fb; color: #15263c; }
  .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3.5rem; }
  [data-testid="stHeader"] { background: rgba(244,247,251,.88); }
  [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e6ebf2; }
  [data-testid="stMetric"] { background: #fff; padding: 1rem 1.1rem; border: 1px solid #e5ebf3; border-radius: 14px; box-shadow: 0 2px 10px rgba(16,36,58,.035); }
  [data-testid="stMetricLabel"] p { color: #66758a; font-size: .82rem; }
  [data-testid="stMetricValue"] { color: #10243a; font-weight: 700; }
  .hero { background: linear-gradient(118deg,#10243a 0%,#183b5b 64%,#20556a 100%); border-radius: 22px; padding: 2rem 2.2rem; color: #f7fbff; margin: .2rem 0 1.2rem; box-shadow: 0 12px 34px rgba(16,36,58,.16); }
  .hero-grid { display: grid; grid-template-columns: minmax(0,1fr) 270px; gap: 2rem; align-items: center; }
  .hero-kicker { color: #81d9ce; text-transform: uppercase; letter-spacing: .14em; font-size: .73rem; font-weight: 700; }
  .hero h1 { color: #fff; font-size: clamp(2rem,4vw,3.25rem); line-height: 1.08; margin: .55rem 0 .75rem; }
  .hero p { color: #d5e3ee; max-width: 760px; font-size: 1.05rem; line-height: 1.6; margin: 0; }
  .hero-stat { margin-top: .35rem; padding: 1.15rem 1.3rem; border: 1px solid rgba(255,255,255,.2); border-radius: 16px; background: rgba(255,255,255,.075); }
  .hero-stat span { display: block; color: #c6d7e5; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
  .hero-stat strong { display: block; color: #fff; font-size: 2.25rem; line-height: 1.2; margin: .3rem 0; }
  .hero-stat small { color: #d5e3ee; }
  .trust-note { border-left: 4px solid #16a89a; background: #eaf7f5; padding: .8rem 1rem; border-radius: 0 10px 10px 0; color: #234b50; margin: .5rem 0 1rem; }
  .section-kicker { color: #2864dc; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 750; margin-bottom: -.35rem; }
  .small-note { color: #66758a; font-size: .88rem; }
  .model-card { min-height: 170px; background: #fff; border: 1px solid #e5ebf3; border-radius: 16px; padding: 1.1rem 1.25rem; }
  .model-card h3 { margin: .1rem 0 .55rem; color: #10243a; }
  .model-card p { color: #5c6d81; line-height: 1.5; }
  .pill { display: inline-block; padding: .22rem .55rem; border-radius: 999px; background: #e9efff; color: #2856ad; font-size: .72rem; font-weight: 700; margin-right: .35rem; }
  div[data-testid="stTabs"] button { font-weight: 650; }
  @media (max-width: 760px) { .hero-grid { grid-template-columns: 1fr; gap: 1rem; } .hero { padding: 1.4rem; } }
</style>
""", unsafe_allow_html=True)


def read_report(filename: str) -> dict | None:
    path = REPORTS / filename
    return json.loads(path.read_text()) if path.exists() else None


def chart_layout(fig: go.Figure, *, height: int = 390, title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin={"l": 12, "r": 12, "t": 54, "b": 12},
                      font={"family": "Inter, sans-serif", "color": COLORS["navy"]},
                      legend={"orientation": "h", "y": -0.22},
                      hoverlabel={"bgcolor": "#10243a", "font_color": "white"})
    fig.update_xaxes(showgrid=False, linecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


@st.cache_data(show_spinner="Loading local FD001 histories…")
def load_history(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = attach_rul(load_fd001(Path(path)))
    return frame, causal_features(frame)


@st.cache_resource
def load_tree_artifact(path: str) -> dict:
    return load_artifact(Path(path))


def render_header(test_report: dict | None) -> None:
    if test_report:
        mae = test_report["primary_model"]["mae_cycles"]
        improvement = (1 - mae / test_report["constant_baseline"]["mae_cycles"]) * 100
        stat = f"""<div class="hero-stat"><span>Frozen FD001 test · 100 engines</span>
        <strong>{mae:.1f} <small>cycles MAE</small></strong>
        <small>{improvement:.1f}% lower MAE than the training-mean baseline</small></div>"""
    else:
        stat = "<div class=\"hero-stat\"><span>NASA C-MAPSS · FD001</span><strong>RUL</strong><small>Remaining useful life, measured in cycles</small></div>"
    st.markdown(f"""
    <div class="hero"><div class="hero-kicker">Machine health · predictive maintenance case study</div>
      <div class="hero-grid"><div class="hero-copy"><h1>From sensor history<br>to remaining-life estimates.</h1>
      <p>An end-to-end ML engineering project on NASA’s simulated turbofan fleet: engine-safe splits, causal features, temporal models, calibrated uncertainty, and an interactive replay.</p></div>
      {stat}</div></div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='trust-note'><b>Research demonstration.</b> C-MAPSS is simulated benchmark data. These estimates are not validated maintenance recommendations. Official test results are frozen and are not used for model selection.</div>", unsafe_allow_html=True)


def render_overview(validation: dict | None, temporal: dict | None,
                    alert: dict | None, test_report: dict | None) -> None:
    st.markdown('<div class="section-kicker">At a glance</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("FD001 training engines", "100", help="Full simulated run-to-failure trajectories.")
    b.metric("Original model-development split", "80 / 20", help="Training engines / held-out validation engines; split before preprocessing.")
    c.metric("TCN validation design", "12 + 8", help="12 model-selection engines plus 8 disjoint calibration engines.")
    d.metric("Alert threshold", f"{alert['selected_threshold_cycles']} cycles" if alert else "—",
             help="Selected on complete validation engines for the gradient-boosted baseline.")

    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.subheader("Engine-level evaluation design")
        st.caption("Every trajectory stays within one group. Model scaling and fitting use only training engines.")
        sankey = go.Figure(go.Sankey(
            arrangement="fixed",
            node={"pad": 22, "thickness": 22,
                  "label": ["FD001 training\n100 engines", "Model development\n80 engines",
                            "Validation holdout\n20 engines", "Fit + scale\n72 engines",
                            "Early stop\n8 engines", "Model comparison\n12 engines",
                            "Interval calibration\n8 engines"],
                  "x": [.01, .34, .34, .91, .91, .91, .91],
                  "y": [.48, .22, .75, .06, .36, .63, .91],
                  "color": [COLORS["navy"], "#3e6b8d", "#3e6b8d", COLORS["blue"],
                            COLORS["blue"], COLORS["teal"], COLORS["teal"]],
                  "line": {"color": "white", "width": 1}},
            link={"source": [0, 0, 1, 1, 2, 2], "target": [1, 2, 3, 4, 5, 6],
                  "value": [80, 20, 72, 8, 12, 8],
                  "color": ["rgba(40,100,220,.22)", "rgba(22,168,154,.25)",
                            "rgba(40,100,220,.32)", "rgba(40,100,220,.16)",
                            "rgba(22,168,154,.36)", "rgba(22,168,154,.20)"]}))
        chart_layout(sankey, height=390)
        st.plotly_chart(sankey, width="stretch", key="partition_sankey")
    with right:
        st.subheader("Frozen test set")
        if test_report:
            st.metric("Official test engines", str(test_report["n_engines"]),
                      help="100 truncated trajectories with paired endpoint RUL labels.")
            st.metric("Primary endpoint MAE", f"{test_report['primary_model']['mae_cycles']:.2f} cycles")
            ci = test_report["primary_model"]["engine_bootstrap_95_ci"]["mae_cycles"]
            st.caption(f"Engine-bootstrap 95% interval: {ci['lower_95']:.2f}–{ci['upper_95']:.2f} cycles.")
        st.markdown("""
        <div class="model-card"><span class="pill">ONE-TIME EVALUATION</span>
        <h3>Separate by design</h3>
        <p>The official test partition is separate from training and validation. Its report is preserved; test results do not select features, models, or alert thresholds.</p></div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("One pipeline, three engineering questions")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("<div class='model-card'><span class='pill'>PREDICT</span><h3>Can the model estimate RUL?</h3><p>Compare a training-derived constant, histogram gradient boosting, and a causal temporal CNN on engine-held-out data.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='model-card'><span class='pill'>CALIBRATE</span><h3>How wide should uncertainty be?</h3><p>Build engine-block conformal intervals on calibration engines disjoint from model-selection engines.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='model-card'><span class='pill'>REPLAY</span><h3>When would an alert fire?</h3><p>Replay causal predictions through recorded sensor histories and inspect the validation-only alert rule.</p></div>", unsafe_allow_html=True)
    if validation and temporal:
        tree_m1 = validation["hist_gradient_boosting"]["mae_cycles"]
        tcn_m3 = temporal["tcn"]["score"]["windows"]["mae_cycles"]
        st.caption(f"The headline validation scores use different engine groups: Milestone 1 tree MAE {tree_m1:.2f} cycles across 20 engines; TCN comparison MAE {tcn_m3:.2f} cycles across 12 different validation engines. See Evaluation for matched comparisons.")


def render_replay(split: dict, alert: dict | None, temporal: dict | None) -> None:
    st.markdown('<div class="section-kicker">Recorded operations view</div>', unsafe_allow_html=True)
    st.header("Replay an engine")
    st.caption("Scrub through observed cycles. The dashed trajectory is withheld ground truth shown for retrospective evaluation.")
    available = DATA.exists() and ARTIFACT.exists() and SPLIT.exists()
    if not available:
        st.info("Replay requires the local FD001 training file and trained baseline artifact. The overview and evaluation views remain available from committed reports.")
        st.code(".venv/bin/machine-health prepare\n.venv/bin/machine-health train\n.venv/bin/machine-health replay")
        return

    frame, features = load_history(str(DATA))
    tree_artifact = load_tree_artifact(str(ARTIFACT))
    model_choices = ["Histogram gradient boosting"]
    if TEMPORAL_REPLAY.exists() and TEMPORAL_SPLIT.exists():
        model_choices.append("Temporal convolutional network")
    model_name = st.selectbox("Model", model_choices, key="replay_model")
    if model_name == "Temporal convolutional network":
        temporal_predictions = pd.read_csv(TEMPORAL_REPLAY)
        temporal_split = json.loads(TEMPORAL_SPLIT.read_text())
        engine_ids = temporal_split["model_selection_engines"]
    else:
        engine_ids = split["validation_engines"]
    engine_id = st.selectbox("Held-out validation engine", engine_ids, key="replay_engine")
    trajectory = frame[frame.engine_id == engine_id].copy()
    if model_name == "Temporal convolutional network":
        series = temporal_predictions[temporal_predictions.engine_id == engine_id].set_index("cycle")
        aligned = series.reindex(trajectory.cycle)
        trajectory["prediction"] = aligned.prediction.to_numpy()
        trajectory["lower"] = aligned.lower.to_numpy()
        trajectory["upper"] = aligned.upper.to_numpy()
    else:
        engine_features = features.loc[trajectory.index]
        trajectory["prediction"] = np.maximum(0, tree_artifact["model"].predict(engine_features[FEATURES]))

    cycle = st.slider("Replay through cycle", int(trajectory.cycle.min()), int(trajectory.cycle.max()),
                      int(trajectory.cycle.max()), key="replay_cycle")
    shown = trajectory[trajectory.cycle <= cycle]
    current = shown.iloc[-1]
    a, b, c, d = st.columns(4)
    a.metric("Predicted RUL", f"{current.prediction:.1f} cycles")
    b.metric("Observed cycle", f"{int(current.cycle)}")
    c.metric("Evaluation actual RUL", f"{int(current.rul)} cycles",
             help="Withheld full-trajectory target; retrospective evaluation only.")
    d.metric("Absolute error", f"{abs(current.prediction - current.rul):.1f} cycles")

    fig = go.Figure()
    if model_name == "Temporal convolutional network":
        fig.add_trace(go.Scatter(x=shown.cycle, y=shown.lower, mode="lines", line={"width": 0},
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=shown.cycle, y=shown.upper, mode="lines", line={"width": 0},
                                 fill="tonexty", fillcolor="rgba(40,100,220,.16)",
                                 name="Nominal 80% interval"))
    fig.add_trace(go.Scatter(x=shown.cycle, y=shown.prediction, mode="lines",
                             name="Predicted RUL", line={"color": COLORS["blue"], "width": 3}))
    fig.add_trace(go.Scatter(x=shown.cycle, y=shown.rul, mode="lines",
                             name="Actual RUL · evaluation only",
                             line={"color": COLORS["orange"], "dash": "dash", "width": 2}))
    if model_name == "Histogram gradient boosting" and alert:
        threshold = alert["selected_threshold_cycles"]
        fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["red"],
                      annotation_text=f"Validation alert threshold · {threshold} cycles")
        policy = alert["policy"]
        alert_data = trajectory[["cycle", "prediction"]].rename(columns={"prediction": "prediction"})
        alert_index = _first_persistent_alert(alert_data, threshold)
        alert_cycle = None if alert_index is None else int(trajectory.iloc[alert_index].cycle)
        if alert_cycle is not None and alert_cycle <= cycle:
            fig.add_vline(x=alert_cycle, line_dash="dot", line_color=COLORS["teal"],
                          annotation_text="First persistent alert")
        st.caption(f"Alert policy: {policy['persistence_cycles']} consecutive cycles at/below threshold, after {policy['minimum_history_cycles']} cycles of history. Threshold was selected on validation engines for the tree model.")
    fig.update_layout(xaxis_title="Observed cycle", yaxis_title="Remaining useful life (cycles)",
                      hovermode="x unified")
    chart_layout(fig, height=450, title=f"Engine trajectory · {engine_id}")
    st.plotly_chart(fig, width="stretch", key="rul_replay_chart")

    sensors = st.multiselect("Sensor channels", [f"sensor_{i}" for i in range(1, 22)],
                             default=["sensor_2", "sensor_3", "sensor_4"], key="sensor_channels")
    if sensors:
        sensor_fig = px.line(shown, x="cycle", y=sensors, color_discrete_sequence=[COLORS["blue"], COLORS["teal"], COLORS["orange"]])
        chart_layout(sensor_fig, height=340, title="Observed sensor traces")
        st.plotly_chart(sensor_fig, width="stretch", key="sensor_replay_chart")


def render_evaluation(validation: dict | None, temporal: dict | None,
                      alert: dict | None, test_report: dict | None) -> None:
    st.markdown('<div class="section-kicker">Measured evidence</div>', unsafe_allow_html=True)
    st.header("Results, by evaluation boundary")
    st.caption("Each panel labels its partition, aggregation unit, and engine count. Scores from different engine groups are not mixed into one leaderboard.")

    if test_report:
        st.subheader("01 · Frozen official test · endpoint evaluation")
        st.caption("100 truncated test engines · raw endpoint labels · one equally weighted score per engine · report evaluated once under the frozen protocol.")
        a, b, c = st.columns(3)
        a.metric("Gradient boosting · MAE", f"{test_report['primary_model']['mae_cycles']:.2f} cycles")
        b.metric("Gradient boosting · RMSE", f"{test_report['primary_model']['rmse_cycles']:.2f} cycles")
        ci = test_report["primary_model"]["engine_bootstrap_95_ci"]["mae_cycles"]
        c.metric("Engine bootstrap 95% CI · MAE", f"{ci['lower_95']:.2f}–{ci['upper_95']:.2f}")
        records = pd.DataFrame(test_report["endpoint_per_engine"])
        left, right = st.columns([1.2, .8], gap="large")
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=records.actual_rul_cycles, y=records.predicted_rul_cycles,
                                     mode="markers", text=records.engine_id,
                                     marker={"size": 9, "color": records.absolute_error_cycles,
                                             "colorscale": "Blues", "showscale": True,
                                             "colorbar": {"title": "Absolute<br>error"}},
                                     hovertemplate="%{text}<br>Actual: %{x:.1f} cycles<br>Predicted: %{y:.1f} cycles<extra></extra>",
                                     name="Test engine"))
            limit = max(records.actual_rul_cycles.max(), records.predicted_rul_cycles.max()) * 1.04
            fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines",
                                     line={"color": COLORS["muted"], "dash": "dash"},
                                     name="Perfect prediction"))
            fig.update_layout(xaxis_title="Actual endpoint RUL (cycles)",
                              yaxis_title="Predicted endpoint RUL (cycles)",
                              hovermode="closest")
            chart_layout(fig, height=410, title="Prediction vs. ground truth · one point per engine")
            st.plotly_chart(fig, width="stretch", key="test_endpoint_scatter")
        with right:
            band_rows = []
            for band, value in test_report["by_actual_rul_band"].items():
                band_rows.append({"Actual RUL": band, "Engines": value["n_engines"],
                                  "Gradient boost": value["hist_gradient_boosting"]["mae_cycles"],
                                  "Mean constant": value["constant_baseline"]["mae_cycles"]})
            band_frame = pd.DataFrame(band_rows).melt(id_vars=["Actual RUL", "Engines"],
                                                      var_name="Model", value_name="MAE (cycles)")
            fig = px.bar(band_frame, x="Actual RUL", y="MAE (cycles)", color="Model",
                         barmode="group", text_auto=".1f",
                         color_discrete_map={"Gradient boost": COLORS["blue"], "Mean constant": "#A9B6C5"})
            chart_layout(fig, height=410, title="Endpoint MAE by actual RUL band")
            st.plotly_chart(fig, width="stretch", key="test_band_errors")
        st.markdown(f"<p class='small-note'>Secondary all-window metric: {test_report['all_window_secondary']['n_windows']:,} observed cycles. Gradient boosting MAE {test_report['all_window_secondary']['hist_gradient_boosting']['mae_cycles']:.2f} cycles; the longer trajectories contribute more rows to this diagnostic.</p>", unsafe_allow_html=True)
        st.divider()

    if validation:
        st.subheader("02 · Original held-out validation · 20 complete engines")
        st.caption("Window-weighted RUL errors on 3,644 cycles. All 20 engines run to failure; endpoint targets are zero by construction.")
        val_models = pd.DataFrame([
            {"Model": "Training-mean constant", **validation["constant_baseline"]},
            {"Model": "Histogram gradient boosting", **validation["hist_gradient_boosting"]},
        ])
        metric_col, chart_col = st.columns([.65, 1.35], gap="large")
        with metric_col:
            st.dataframe(val_models.rename(columns={"mae_cycles": "MAE (cycles)", "rmse_cycles": "RMSE (cycles)"}).round(2),
                         hide_index=True, width="stretch")
            st.metric("Complete-run endpoint MAE", f"{validation['endpoint']['mae_cycles']:.2f} cycles",
                      help="Weak check: each complete run-to-failure endpoint has actual RUL 0.")
        with chart_col:
            band_rows = [{"Actual RUL": band, "MAE (cycles)": metrics["mae_cycles"]}
                         for band, metrics in validation["by_rul_range"].items()]
            fig = px.bar(pd.DataFrame(band_rows),
                         x="Actual RUL", y="MAE (cycles)", text_auto=".1f",
                         color_discrete_sequence=[COLORS["blue"]])
            chart_layout(fig, height=340, title="Gradient-boosted error by remaining-life range")
            st.plotly_chart(fig, width="stretch", key="validation_band_mae")
        st.divider()

    if temporal:
        score = temporal["tcn"]["score"]
        tree = temporal["hist_gradient_boosting"]["score"]
        st.subheader("03 · Matched temporal-model comparison · 12 validation engines")
        st.caption("Tree and TCN are scored on the exact same 12 model-selection engines. Eight different validation engines are reserved for interval calibration.")
        model_rows = []
        for label, result in [("Gradient boosting", tree), ("Causal TCN", score)]:
            model_rows.append({"Model": label, "Window MAE": result["windows"]["mae_cycles"],
                               "Window RMSE": result["windows"]["rmse_cycles"],
                               "Endpoint MAE": result["endpoint"]["mae_cycles"]})
        st.dataframe(pd.DataFrame(model_rows).round(2), hide_index=True, width="stretch")
        interval = score["prediction_intervals"]
        coverage_col, width_col = st.columns(2, gap="large")
        with coverage_col:
            coverage_rows = [{"Actual RUL band": band,
                              "Cycle coverage": metrics["mean_engine_cycle_coverage"],
                              "Engine-band all-cycle coverage": metrics["engine_band_simultaneous_coverage"]}
                             for band, metrics in interval["by_actual_rul_band"].items()]
            coverage_frame = pd.DataFrame(coverage_rows).melt(id_vars="Actual RUL band",
                                                              var_name="Measure", value_name="Coverage")
            fig = px.bar(coverage_frame, x="Actual RUL band", y="Coverage", color="Measure",
                         barmode="group", range_y=[0, 1.05], text_auto=".0%",
                         color_discrete_map={"Cycle coverage": COLORS["teal"],
                                             "Engine-band all-cycle coverage": COLORS["blue"]})
            fig.add_hline(y=.8, line_dash="dash", line_color=COLORS["orange"],
                          annotation_text="Nominal 80%")
            chart_layout(fig, height=370, title="Empirical interval coverage · 12 engines")
            st.plotly_chart(fig, width="stretch", key="interval_coverage_by_rul")
        with width_col:
            width_rows = [{"Actual RUL band": band, "Mean interval width": metrics["mean_width_cycles"]}
                          for band, metrics in interval["by_actual_rul_band"].items()]
            fig = px.bar(pd.DataFrame(width_rows), x="Actual RUL band", y="Mean interval width",
                         text_auto=".1f", color_discrete_sequence=[COLORS["blue"]])
            fig.update_layout(yaxis_title="Mean width (cycles)")
            chart_layout(fig, height=370, title="Interval width by actual RUL band")
            st.plotly_chart(fig, width="stretch", key="interval_width_by_rul")
        endpoint = interval["endpoint"]
        x, y, z = st.columns(3)
        x.metric("Nominal interval coverage", "80%")
        y.metric("Observed endpoint coverage", f"{endpoint['coverage']:.0%} · {endpoint['engines']} engines")
        z.metric("Mean endpoint interval width", f"{endpoint['mean_width_cycles']:.1f} cycles")
        st.warning("Only 8 engines calibrate these intervals. The observed 12-engine coverage is descriptive, not a guarantee of real-world coverage; simulator shift and the small calibration sample are not captured.")
        st.divider()

    if alert:
        st.subheader("04 · Validation alert policy · 20 complete engines")
        st.caption("Selected for the gradient-boosted model on full validation lifecycles; it does not establish full-lifecycle performance on truncated test engines.")
        selected = alert["selected_validation_result"]
        a, b, c, d = st.columns(4)
        a.metric("Selected threshold", f"{alert['selected_threshold_cycles']} cycles")
        b.metric("Timely detections", f"{selected['timely_detection_engines']}/{selected['engines']}")
        c.metric("Early false alarms", f"{selected['early_false_alarm_engines']}/{selected['engines']}")
        d.metric("Missed failures", f"{selected['missed_failures']}/{selected['engines']}")
        candidates = pd.DataFrame(alert["candidate_results"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=candidates.early_false_alarm_rate, y=candidates.timely_detection_rate,
                                 mode="lines+markers", text=candidates.threshold_cycles,
                                 marker={"size": 9, "color": candidates.threshold_cycles,
                                         "colorscale": "Tealgrn", "showscale": True,
                                         "colorbar": {"title": "Threshold<br>(cycles)"}},
                                 hovertemplate="Threshold: %{text} cycles<br>Early false alarm: %{x:.0%}<br>Timely detection: %{y:.0%}<extra></extra>",
                                 name="Validation policy candidates"))
        fig.add_trace(go.Scatter(x=[selected["early_false_alarm_rate"]],
                                 y=[selected["timely_detection_rate"]], mode="markers",
                                 marker={"size": 16, "color": COLORS["orange"], "symbol": "star"},
                                 name="Selected · 20 cycles"))
        fig.update_layout(xaxis_title="Engines with early false alarm", yaxis_title="Timely detections",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_range=[-.02, 1.03], yaxis_range=[-.02, 1.03])
        chart_layout(fig, height=370, title="Validation tradeoff · each point is a threshold")
        st.plotly_chart(fig, width="stretch", key="alert_tradeoff")


def render_methods(temporal: dict | None) -> None:
    st.markdown('<div class="section-kicker">System design</div>', unsafe_allow_html=True)
    st.header("A causal, engine-safe workflow")
    st.caption("Prediction at cycle t can only use observations available at or before t. Engine identity and lifetime labels are excluded from inference.")
    diagram = r"""
    digraph machine_health {
      graph [rankdir=LR, bgcolor="transparent", pad="0.25", nodesep="0.42", ranksep="0.72"];
      node [shape=box, style="rounded,filled", fontname="Arial", fontsize=12, margin="0.18,0.12", color="#dce5ef", penwidth=1.2];
      edge [fontname="Arial", fontsize=10, color="#7890a8", penwidth=1.5, arrowsize=.7];
      sensors [label="Multivariate sensor history\ncycle · settings · 21 sensors", fillcolor="#10243a", fontcolor="white", color="#10243a"];
      split [label="Engine-level split\nfit · early stop · validation", fillcolor="#eaf0f8", fontcolor="#10243a"];
      causal [label="Causal features\ntrailing windows · reset per engine", fillcolor="#e9efff", fontcolor="#214d9e"];
      tree [label="Gradient-boosted trees\nraw RUL in cycles", fillcolor="#e9efff", fontcolor="#214d9e"];
      sequence [label="30-cycle sequence\ntrain-only normalization", fillcolor="#e3f5f2", fontcolor="#14685f"];
      tcn [label="Temporal CNN\nengine-level early stopping", fillcolor="#e3f5f2", fontcolor="#14685f"];
      output [label="RUL replay\nvalidation interval · alert policy", fillcolor="#10243a", fontcolor="white", color="#10243a"];
      frozen [label="Official FD001 test\n100 truncated engines · frozen report", fillcolor="#fff2e3", fontcolor="#8a541c", color="#eed8bb"];
      sensors -> split;
      split -> causal;
      causal -> tree;
      split -> sequence;
      sequence -> tcn;
      tree -> output;
      tcn -> output;
      frozen -> output [style=dashed, label="one-time score", color="#c38d52"];
    }
    """
    st.graphviz_chart(diagram, width="stretch")

    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Gradient-boosted baseline")
        st.markdown("""
        - Input: current cycle and operating settings plus 5, 10, and 20-cycle rolling sensor means and standard deviations.
        - Rolling features use current/past observations only and reset at engine boundaries.
        - Median imputation and model are saved together in the sklearn artifact.
        - Official test evaluation is endpoint-weighted: one score per truncated engine.
        """)
    with b:
        st.subheader("Temporal model")
        st.markdown("""
        - Input: last 30 raw cycles; four residual causal Conv1D blocks with kernel 3 and dilations 1, 2, 4, and 8.
        - Feature and target scaling are fit on 72 training engines; 8 other training engines drive early stopping.
        - 12 held-out validation engines compare models; 8 separate validation engines calibrate intervals.
        - Intervals use per-engine maximum absolute residuals within predicted-RUL bands.
        """)
    st.subheader("What to keep in mind")
    st.markdown("""
    - **Simulation domain:** FD001 is simulated turbofan degradation, not field data.
    - **Small engine sample:** the TCN comparison has 12 engines and its calibration has 8. Observed coverage is noisy.
    - **Interval width:** longer-life ranges have wider intervals; an 80% nominal target does not mean 80% guaranteed real-world coverage.
    - **Endpoint metric:** complete validation runs end at failure, so actual endpoint RUL is zero by construction. The frozen test uses truncated endpoint labels.
    - **Recorded replay:** it is retrospective and has no live telemetry integration.
    - **No maintenance advice:** model output must not be used for operational decisions.
    """)
    if temporal:
        with st.expander("Experiment provenance"):
            st.json({"code_revision": temporal["code_revision"],
                     "data_sha256": temporal["data_sha256"],
                     "parent_split_sha256": temporal["split_manifest_sha256"],
                     "tcn_artifact_sha256": temporal["tcn_artifact_sha256"],
                     "seed": temporal["seed"], "python": temporal["python"],
                     "torch": temporal["torch"], "code_dirty_at_run": temporal["code_dirty"]})


validation = read_report("validation_metrics.json")
temporal_report = read_report("temporal_model_v1.json")
alert_report = read_report("validation_alert_policy.json")
test_report = read_report("official_test_v1.json")

render_header(test_report)
st.sidebar.markdown("### MACHINE HEALTH")
st.sidebar.caption("NASA C-MAPSS · FD001")
st.sidebar.markdown("**Project status**  \nMilestones 1–3 · CPU-first")
st.sidebar.caption("All RUL estimates are measured in cycles.")
overview_tab, replay_tab, evidence_tab, methods_tab = st.tabs(
    ["Project overview", "Engine replay", "Evaluation evidence", "Methods & limitations"])
with overview_tab:
    render_overview(validation, temporal_report, alert_report, test_report)
with replay_tab:
    if SPLIT.exists():
        render_replay(json.loads(SPLIT.read_text()), alert_report, temporal_report)
    else:
        st.info("The split manifest is missing. Run the project training workflow to enable recorded replay.")
with evidence_tab:
    render_evaluation(validation, temporal_report, alert_report, test_report)
with methods_tab:
    render_methods(temporal_report)

st.markdown("<div class='small-note'>Machine Health · NASA C-MAPSS FD001 · Research and engineering demonstration</div>", unsafe_allow_html=True)
