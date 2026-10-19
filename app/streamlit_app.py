from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from machine_health.data import attach_rul, load_fd001
from machine_health.features import FEATURES, causal_features
from machine_health.model import load_artifact

st.set_page_config(page_title="Machine Health | FD001", page_icon="✈️", layout="wide")
st.title("Machine Health")
st.caption("NASA C-MAPSS FD001 · recorded run-to-failure replay · research demonstration")
artifact_path = ROOT / "artifacts/fd001.joblib"
data_path = ROOT / "data/raw/train_FD001.txt"
split_path = ROOT / "reports/fd001_split.json"
if not all(p.exists() for p in (artifact_path, data_path, split_path)):
    st.error("Run the documented prepare, train, and evaluate-validation commands before opening the replay.")
    st.stop()

@st.cache_data
def data_and_features(path: str):
    frame = attach_rul(load_fd001(Path(path)))
    return frame, causal_features(frame)

@st.cache_resource
def model(path: str):
    return load_artifact(Path(path))

frame, features = data_and_features(str(data_path))
artifact = model(str(artifact_path))
split = json.loads(split_path.read_text())
engine_ids = split["validation_engines"]
temporal_replay_path = ROOT / "data/processed/temporal_validation_replay.csv"
temporal_split_path = ROOT / "reports/temporal_split.json"
model_options = ["Histogram gradient boosting"]
if temporal_replay_path.exists() and temporal_split_path.exists():
    model_options.append("Temporal convolutional network")
selected_model = st.sidebar.selectbox("Model", model_options)
if selected_model == "Temporal convolutional network":
    temporal_predictions = pd.read_csv(temporal_replay_path)
    temporal_split = json.loads(temporal_split_path.read_text())
    engine_ids = temporal_split["model_selection_engines"]
engine = st.sidebar.selectbox("Held-out validation engine", engine_ids)
trajectory = frame[frame.engine_id == engine].copy()
if selected_model == "Temporal convolutional network":
    engine_predictions = temporal_predictions[temporal_predictions.engine_id == engine].set_index("cycle")
    aligned = engine_predictions.reindex(trajectory.cycle)
    trajectory["Predicted RUL"] = aligned.prediction.to_numpy()
    trajectory["Interval lower"] = aligned.lower.to_numpy()
    trajectory["Interval upper"] = aligned.upper.to_numpy()
else:
    engine_features = features.loc[trajectory.index]
    predictions = artifact["model"].predict(engine_features[FEATURES])
    trajectory["Predicted RUL"] = predictions.clip(min=0)
cycle = st.sidebar.slider("Replay through cycle", int(trajectory.cycle.min()), int(trajectory.cycle.max()), int(trajectory.cycle.max()))
shown = trajectory[trajectory.cycle <= cycle]
current = shown.iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Predicted remaining cycles", f"{current['Predicted RUL']:.1f}")
col2.metric("Evaluation actual RUL", f"{int(current.rul)}", help="Withheld validation ground truth shown for retrospective evaluation only.")
col3.metric("Absolute error", f"{abs(current['Predicted RUL'] - current.rul):.1f} cycles")
chart = go.Figure()
if selected_model == "Temporal convolutional network":
    chart.add_trace(go.Scatter(x=shown.cycle, y=shown["Interval lower"], mode="lines",
                               line={"width": 0}, showlegend=False, hoverinfo="skip"))
    chart.add_trace(go.Scatter(x=shown.cycle, y=shown["Interval upper"], mode="lines",
                               line={"width": 0}, fill="tonexty", fillcolor="rgba(37, 99, 235, 0.18)",
                               name="80% calibrated interval"))
chart.add_trace(go.Scatter(x=shown.cycle, y=shown["Predicted RUL"], mode="lines",
                           name="Predicted RUL", line={"color": "#2563eb", "width": 3}))
chart.add_trace(go.Scatter(x=shown.cycle, y=shown.rul, mode="lines",
                           name="Actual RUL (evaluation only)", line={"dash": "dash", "color": "#f97316"}))
chart.update_layout(title="RUL prediction over recorded engine history", xaxis_title="Cycle",
                    yaxis_title="RUL (cycles)", hovermode="x unified")
chart.add_vline(x=cycle, line_dash="dot", line_color="#ef4444")
st.plotly_chart(chart, width="stretch")
sensors = st.multiselect("Sensor traces", [f"sensor_{i}" for i in range(1, 22)], default=["sensor_2", "sensor_3", "sensor_4"])
if sensors:
    sensor_chart = px.line(shown, x="cycle", y=sensors, title="Observed sensor history")
    st.plotly_chart(sensor_chart, width="stretch")

test_report_path = ROOT / "reports/official_test_v1.json"
temporal_report_path = ROOT / "reports/temporal_model_v1.json"
if temporal_report_path.exists():
    temporal_report = json.loads(temporal_report_path.read_text())
    st.divider()
    st.subheader("Temporal model comparison and calibrated intervals")
    st.caption("Validation only: 12 model-selection engines and 8 disjoint calibration engines. These estimates do not use the official test partition.")
    comparison_rows = []
    for key, label in [("hist_gradient_boosting", "Histogram gradient boosting"),
                       ("tcn", "Causal temporal CNN")]:
        score = temporal_report[key]["score"]
        comparison_rows.append({"Model": label,
                                "Window MAE (cycles)": score["windows"]["mae_cycles"],
                                "Window RMSE (cycles)": score["windows"]["rmse_cycles"],
                                "Endpoint MAE (cycles)": score["endpoint"]["mae_cycles"]})
    st.dataframe(pd.DataFrame(comparison_rows).round(3), hide_index=True, width="stretch")
    intervals = temporal_report["tcn"]["score"]["prediction_intervals"]
    interval_coverage = intervals["endpoint"].get("coverage")
    if interval_coverage is not None:
        st.metric("TCN endpoint interval coverage (12 model-selection engines)",
                  f"{interval_coverage:.1%}",
                  help="Nominal 80% interval; the calibration and model-selection engine groups are disjoint.")

if test_report_path.exists():
    report = json.loads(test_report_path.read_text())
    st.divider()
    st.subheader("Frozen official test endpoint evaluation")
    st.caption("100 truncated NASA test engines · raw endpoint RUL labels · one score per engine. The test set is not used for model or alert-policy selection.")
    first, second, third = st.columns(3)
    first.metric("Endpoint MAE", f"{report['primary_model']['mae_cycles']:.2f} cycles")
    second.metric("Endpoint RMSE", f"{report['primary_model']['rmse_cycles']:.2f} cycles")
    ci = report["primary_model"]["engine_bootstrap_95_ci"]["mae_cycles"]
    third.metric("MAE engine bootstrap 95% CI", f"{ci['lower_95']:.2f}–{ci['upper_95']:.2f}")
    with st.expander("Endpoint error by actual RUL range"):
        band_rows = []
        for band, values in report["by_actual_rul_band"].items():
            score = values["hist_gradient_boosting"]
            band_rows.append({"Actual RUL (cycles)": band, "Engines": values["n_engines"],
                              "MAE (cycles)": score["mae_cycles"], "RMSE (cycles)": score["rmse_cycles"]})
        st.dataframe(pd.DataFrame(band_rows), hide_index=True, width="stretch")

alert_report = ROOT / "reports/validation_alert_policy.json"
if alert_report.exists():
    alert = json.loads(alert_report.read_text())
    selected = alert["selected_validation_result"]
    st.subheader("Validation alert policy")
    st.caption("Selected on complete held-out validation trajectories only; retrospective lifecycle estimates, not observed full-lifecycle test alerts.")
    a, b, c = st.columns(3)
    a.metric("Alert threshold", f"{alert['selected_threshold_cycles']} cycles")
    b.metric("Timely detections", f"{selected['timely_detection_engines']}/{selected['engines']}")
    c.metric("Missed failures", f"{selected['missed_failures']}/{selected['engines']}")
model_caption = ("causal temporal CNN · 30-cycle history · validation-only 80% engine-block intervals"
                 if selected_model == "Temporal convolutional network"
                 else f"histogram gradient boosting · seed {artifact['metadata']['seed']}")
st.caption(f"Model: {model_caption} · target: raw RUL in cycles. Recorded replay is not live telemetry or maintenance advice.")
