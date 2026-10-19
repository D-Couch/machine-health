# Machine Health — FD001 RUL baseline

A reproducible CPU-first predictive-maintenance vertical slice using NASA's simulated C-MAPSS turbofan histories. The app replays held-out, complete validation engines and displays the withheld RUL as retrospective evaluation information. It is not live telemetry or maintenance advice.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/machine-health prepare
.venv/bin/machine-health train
.venv/bin/machine-health evaluate-validation
.venv/bin/machine-health select-alert-policy
.venv/bin/machine-health replay
.venv/bin/streamlit run app/streamlit_app.py
```

To use an existing file, run `machine-health prepare --file /path/to/train_FD001.txt`. Parser expects the standard 26 whitespace-separated columns.

## Data and protocol

Source: [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), direct archive [download](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip). NASA describes the dataset as C-MAPSS simulator output provided by NASA Ames PCoE and requests repository/donor acknowledgment. No separate redistribution license was identified on the repository page, so the raw archive is kept local and ignored by Git. Acquisition stores UTC retrieval time and SHA-256 hashes in `data/raw/provenance.json`.

Training RUL is raw, uncapped cycles: for engine cycle `t`, `max_cycle - t`; failure-cycle target is zero. Engines are deterministically split 80/20 before model fitting (seed 41); all cycles from an engine stay together. Trailing rolling means and standard deviations use only current/past readings; statistics are reset per engine. The model uses cycle, raw operating settings, and causal trailing statistics. Median imputation is included in the serialized sklearn pipeline.

The validation comparison reports both all-window and final-cycle endpoint metrics. Endpoint RUL on complete validation trajectories is zero by construction because the trajectory ends at failure; the endpoint metric therefore measures how near the model prediction is to zero at the recorded failure endpoint and should not be mistaken for truncated-test endpoint performance. The dashboard is useful for trajectory replay and evaluation visibility, while window metrics provide the more substantive regression comparison.

## Current real-data results

Run `machine-health evaluate-validation`; details are in `reports/validation_metrics.md` and JSON. On the fixed 80/20 split, 20 held-out engines and 3,644 cycles:

| Model | Window MAE | Window RMSE |
|---|---:|---:|
| Training-mean constant | 50.464 cycles | 59.822 cycles |
| Histogram gradient boosting | 26.032 cycles | 33.679 cycles |

Across the 20 full-trajectory final cycles, endpoint MAE is 2.987 cycles and RMSE is 4.782 cycles. This unusual near-zero endpoint target follows directly from complete run-to-failure validation records and is not a meaningful estimate for truncated test records.

## Frozen official test results

The protocol in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) was committed before test scoring. The official FD001 test set was scored once; the report refuses overwrite. On 100 truncated test engines with raw endpoint labels, the frozen gradient-boosted model achieved endpoint MAE **20.922 cycles** and RMSE **29.316 cycles** (engine-bootstrap 95% intervals: MAE 16.855–25.015, RMSE 24.314–33.903). The training-group mean baseline scored MAE 42.230 and RMSE 54.597 cycles. Per-engine values and actual-RUL bands are in [reports/official_test_v1.md](reports/official_test_v1.md). These simulated benchmark results are not validated aircraft maintenance predictions.

## Validation alert policy

The threshold is selected only on 20 held-out complete validation engines. The policy requires 20 cycles of history and three consecutive predictions at or below threshold. It targets an actionable 30-cycle horizon, selects maximum timely detection subject to at most a 10% early false-alarm-engine rate, and counts first persistent alerts after RUL reaches zero as misses. On this split, selected threshold is **20 cycles**, with 20/20 timely detections, 0/20 early false alarms, 0 missed failures, and mean timely lead 11.75 cycles. This is validation-only retrospective lifecycle evidence; the truncated official test set is not used for full-lifecycle alert claims. Full threshold comparison is in `reports/validation_alert_policy.md`.

## API, experiment index, and container

FastAPI uses the dashboard's saved artifact and shared causal feature builder. It validates strict cycle order and rejects extra fields such as RUL labels or engine IDs. See [docs/API.md](docs/API.md). Launch with `.venv/bin/uvicorn machine_health.api:app --host 127.0.0.1 --port 8000`.

Training and evaluation calls append provenance/config/metric rows to the local SQLite index at `data/experiments.sqlite`; inspect it with `.venv/bin/machine-health list-runs`. The database is local-only and ignored by Git.

For a container demo, first prepare data and train locally, then run `podman compose up --build`. The Compose file bind-mounts the local dataset, model artifact, and reports read-only. GitHub Actions runs CPU tests and lint only; it does not acquire NASA data or train a model.

## Development checks

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests app
```

See [docs/STATUS.md](docs/STATUS.md) for host details, checks, and remaining limitations.
