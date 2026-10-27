# Machine Health — predictive maintenance benchmark and planning prototype

A reproducible CPU-first predictive-maintenance project with two linked goals: rigorous ML evaluation on NASA's simulated C-MAPSS turbofan histories, and a prototype for a reliability or maintenance planner reviewing a simulated validation fleet. The workflow prioritizes engines from prediction-only snapshots, then lets a reviewer inspect sensor and model evidence. It is designed to focus limited review attention with traceable evidence, but does not measure that benefit. It is not live telemetry, a real-fleet system, or maintenance advice.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.5,<3'
.venv/bin/machine-health prepare
.venv/bin/machine-health train
.venv/bin/machine-health evaluate-validation
.venv/bin/machine-health select-alert-policy
.venv/bin/machine-health train-temporal
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

## Maintenance planning prototype

The Fleet review view presents the 20 held-out validation engines (or the 12 disjoint TCN model-selection engines when TCN replay artifacts are present) at a shared observed cycle. The default snapshot is cycle 120, which is available for all 20 original validation engines; the selector permits other common observed cycles. All cycle values refer to simulated benchmark cycles, not hours or field operating time.

Queue status and ordering use only model predictions through the selected snapshot, the explicit threshold/persistence settings, and the review horizon. The queue builder accepts no RUL labels or engine metadata. Retrospective actual RUL and prediction error are separated in the asset detail. Threshold, persistence, and horizon controls change the queue scenario and its retrospective validation scoring only; they do not retrain models or modify saved validation results. Review workload counts unique benchmark engines with a first persistent alert and is not a staffing-capacity estimate. The alert scenario uses one first alert per complete validation engine: timely means actual RUL 1 through the selected horizon, early false alarm means actual RUL above it, and no alert or alert only at failure counts as missed. Repeated exploration can overfit the validation cohort. The evaluation tab's saved alert curve is an exploratory retrospective comparison on the same validation cohort used to select the documented threshold, not independent confirmation or an operational optimum. The frozen official test sensors and labels are not loaded by this workflow.

This demonstrates a maintenance-planning interaction on benchmark data. It does not establish economic benefit, real-fleet performance, or suitability for aircraft or other operational maintenance decisions.

## API, experiment index, and container

FastAPI uses the dashboard's saved artifact and shared causal feature builder. It validates strict cycle order and rejects extra fields such as RUL labels or engine IDs. See [docs/API.md](docs/API.md). Launch with `.venv/bin/uvicorn machine_health.api:app --host 127.0.0.1 --port 8000`.

Training and evaluation calls append provenance/config/metric rows to the local SQLite index at `data/experiments.sqlite`; inspect it with `.venv/bin/machine-health list-runs`. The database is local-only and ignored by Git.

For a container demo, first prepare data and train locally, then run `podman compose up --build`. The Compose file bind-mounts the local dataset, model artifact, and reports read-only. GitHub Actions runs CPU tests and lint only; it does not acquire NASA data or train a model.

## Visual walkthrough

Launch `.venv/bin/streamlit run app/streamlit_app.py` for the five-part project experience:

- **Fleet review:** prediction-ranked validation queue at a common observation cycle, scenario controls, and an asset evidence view through the selected snapshot. Withheld labels are isolated in a retrospective evaluation expander.
- **Local disposition history:** record inspect, monitor, defer, or dismiss with an optional rationale for a selected validation engine. Events include snapshot, prediction, policy, and artifact context and are stored in ignored `data/review_dispositions.sqlite` (override with `MACHINE_HEALTH_REVIEW_DB`). This is a local prototype log, not CMMS integration or model feedback.
- **Project overview:** an engine-level partition diagram, project summary, and frozen test result.
- **Engine replay:** scrub a validation engine by cycle, inspect sensor traces, predictions, and (for the TCN) its interval band. Validation truth is labeled as retrospective evaluation data.
- **Evaluation evidence:** per-engine official endpoint errors, RUL-band breakdowns, matched validation model comparison, interval coverage/width, and alert tradeoffs.
- **Methods & limitations:** data flow, model inputs, split design, interval method, and benchmark caveats.

The overview and evidence panels use the small committed JSON reports and work without the ignored NASA data or model artifacts. Fleet review and replay need the local FD001 training file and corresponding trained artifacts. Every chart identifies its split and aggregation to keep unlike validation scores separate from frozen test scores.

## Temporal model and intervals

`machine-health train-temporal` trains a compact causal TCN from 30-cycle raw sensor histories. It uses 72 original training engines for fitting, 8 disjoint training engines for early stopping, 12 of the existing validation engines for model comparison, and the remaining 8 validation engines for interval calibration. Tree and TCN point metrics use the same 12 model-selection engines. The official test set is not loaded.

The current validation-only comparison is in `reports/temporal_model_v1.md`. On the 12 model-selection engines, TCN window MAE/RMSE is 20.701/28.217 cycles versus 23.653/30.502 for the gradient-boosted baseline. Complete-path endpoint targets are zero by construction, so endpoint metrics there are not operational RUL evidence. Nominal 80% intervals use an engine-block maximum-residual conformal method grouped by *predicted* RUL band; with only 8 calibration engines, intervals are coarse and can be wide. Empirical model-selection endpoint coverage was 12/12 with mean interval width 12.67 cycles. Read [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the method, results by RUL range, assumptions, and limitations.

The Streamlit dashboard offers the TCN replay and calibrated interval when temporal experiment outputs exist.

## Development checks

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests app
```

See [docs/STATUS.md](docs/STATUS.md) for host details, checks, and remaining limitations.
