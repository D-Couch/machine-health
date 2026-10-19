# Machine Health — FD001 RUL baseline

A reproducible CPU-first predictive-maintenance vertical slice using NASA's simulated C-MAPSS turbofan histories. The app replays held-out, complete validation engines and displays the withheld RUL as retrospective evaluation information. It is not live telemetry or maintenance advice.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/machine-health prepare
.venv/bin/machine-health train
.venv/bin/machine-health evaluate-validation
.venv/bin/machine-health replay
.venv/bin/streamlit run app/streamlit_app.py
```

To use an existing file, run `machine-health prepare --file /path/to/train_FD001.txt`. Parser expects the standard 26 whitespace-separated columns.

## Data and protocol

Source: [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), direct archive [download](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip). NASA describes the dataset as C-MAPSS simulator output provided by NASA Ames PCoE and requests repository/donor acknowledgment. No separate redistribution license was identified on the repository page, so the raw archive is kept local and ignored by Git. Acquisition stores UTC retrieval time and SHA-256 hashes in `data/raw/provenance.json`.

Training RUL is raw, uncapped cycles: for engine cycle `t`, `max_cycle - t`; failure-cycle target is zero. The official test partition is reserved and was not loaded or evaluated. Engines are deterministically split 80/20 before model fitting (seed 41); all cycles from an engine stay together. Trailing rolling means and standard deviations use only current/past readings; statistics are reset per engine. The model uses cycle, raw operating settings, and causal trailing statistics. Median imputation is included in the serialized sklearn pipeline.

The validation comparison reports both all-window and final-cycle endpoint metrics. Endpoint RUL on complete validation trajectories is zero by construction because the trajectory ends at failure; the endpoint metric therefore measures how near the model prediction is to zero at the recorded failure endpoint and should not be mistaken for truncated-test endpoint performance. The dashboard is useful for trajectory replay and evaluation visibility, while window metrics provide the more substantive regression comparison.

## Current real-data results

Run `machine-health evaluate-validation`; details are in `reports/validation_metrics.md` and JSON. On the fixed 80/20 split, 20 held-out engines and 3,644 cycles:

| Model | Window MAE | Window RMSE |
|---|---:|---:|
| Training-mean constant | 50.464 cycles | 59.822 cycles |
| Histogram gradient boosting | 26.032 cycles | 33.679 cycles |

Across the 20 full-trajectory final cycles, endpoint MAE is 2.987 cycles and RMSE is 4.782 cycles. This unusual near-zero endpoint target follows directly from complete run-to-failure validation records and is not a meaningful estimate for truncated test records. No official test results are reported.

## Development checks

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests app
```

See [docs/STATUS.md](docs/STATUS.md) for host details, checks, and remaining limitations.
