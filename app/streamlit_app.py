from __future__ import annotations

import json
import sys
from pathlib import Path

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
st.caption(f"Model: histogram gradient boosting · seed {artifact['metadata']['seed']} · target: raw RUL in cycles. Recorded replay is not live telemetry or maintenance advice.")
