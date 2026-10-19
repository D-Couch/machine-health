from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
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
engine = st.sidebar.selectbox("Held-out validation engine", engine_ids)
trajectory = frame[frame.engine_id == engine].copy()
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
chart = px.line(shown, x="cycle", y="Predicted RUL", title="RUL prediction over recorded engine history")
chart.add_scatter(x=shown.cycle, y=shown.rul, mode="lines", name="Actual RUL (evaluation only)", line={"dash": "dash"})
chart.add_vline(x=cycle, line_dash="dot", line_color="#ef4444")
st.plotly_chart(chart, use_container_width=True)
sensors = st.multiselect("Sensor traces", [f"sensor_{i}" for i in range(1, 22)], default=["sensor_2", "sensor_3", "sensor_4"])
if sensors:
    sensor_chart = px.line(shown, x="cycle", y=sensors, title="Observed sensor history")
    st.plotly_chart(sensor_chart, use_container_width=True)

test_report_path = ROOT / "reports/official_test_v1.json"
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
        st.dataframe(pd.DataFrame(band_rows), hide_index=True, use_container_width=True)

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
st.caption(f"Model: histogram gradient boosting · seed {artifact['metadata']['seed']} · target: raw RUL in cycles. Recorded replay is not live telemetry or maintenance advice.")
