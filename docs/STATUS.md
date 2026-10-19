# Machine Health status — 2026-10-19

## Milestone 1

Implemented real NASA C-MAPSS FD001 ingestion, deterministic engine-held-out split, raw-cycle RUL labeling, causal per-engine rolling sensor features, a training-mean constant and histogram-gradient-boosting pipeline, persisted model artifact/schema/provenance, machine-readable and readable validation reports, replay CSV generation, and Streamlit validation-engine replay.

### Environment and data

- Executing in `/var/home/dylan/src/machine-health` on Fedora Linux 6.17.12-300.fc43.x86_64, x86_64, Intel Core i9-13900KF (24 cores / 32 logical CPUs), 31 GiB RAM (22 GiB available at inspection).
- Python 3.14.8. Isolated `.venv`; project and dev dependencies installed from `pyproject.toml`.
- NVIDIA GPU is not available to this process: `nvidia-smi` cannot communicate with the driver. Training ran on CPU.
- NASA PCoE archive retrieved 2026-10-19 from the direct URL in `data/raw/provenance.json`. Verified nested archive contains `train_FD001.txt`; parsed 20,631 rows, 100 engines, 26 standard source columns. Archive and extracted file hashes are in local provenance JSON. Original data is ignored by Git.
- NASA repository page describes C-MAPSS simulator provenance and requests repository/donor acknowledgment. No distinct redistribution license was found; raw archive remains local.
- Git metadata is unusable: the visible `.git` entry is not a repository, and Git reports “not a git repository”. No commit was made.

### Real validation results

Split seed 41: 80 training engines and 20 held-out full run-to-failure validation engines. Model targets are uncapped raw RUL in cycles. The official test partition was not acquired or evaluated.

| Model | Window MAE | Window RMSE |
|---|---:|---:|
| Training-derived mean constant | 50.464 cycles | 59.822 cycles |
| Histogram gradient boosting | 26.032 cycles | 33.679 cycles |

There are 3,644 validation windows. Endpoint MAE/RMSE are 2.987/4.782 cycles on 20 complete trajectories; every true endpoint target is zero by label construction, so this is not comparable to truncated official-test endpoint performance. Details and per-engine values are in `reports/validation_metrics.{json,md}`. Split manifest is `reports/fd001_split.json`; model artifact is `artifacts/fd001.joblib`; replay data is `data/processed/validation_replay.csv`.

### Checks run

- `.venv/bin/pytest -q` — 5 passed: parser/schema, RUL semantics, deterministic disjoint engine split, feature causality and engine isolation, artifact round-trip.
- Real data commands completed: `.venv/bin/machine-health train`; `.venv/bin/machine-health evaluate-validation`; `.venv/bin/machine-health replay`.
- Ruff autofix applied; final lint and Streamlit startup smoke check pending at this status write.

### Launch

From the project folder: `.venv/bin/streamlit run app/streamlit_app.py`. Prepare/train/evaluate once first using the commands in README. Dashboard replay computes predictions from the saved artifact and shows actual validation RUL only as retrospective evaluation information.

### Limitations and next work

- No official test evaluation (reserved for Milestone 2), no alert-policy analysis, API, uncertainty, or calibrated intervals.
- Endpoint results on complete run-to-failure trajectories are a weak endpoint check, not operational lead-time evidence.
- GPU unavailable; required CPU run is successful.
- Repair/initialize repository Git metadata if local milestone commits are desired.
- `.venv/bin/ruff check src tests app` — passed after import cleanup.
- Streamlit server started headless on port 8765; `curl http://127.0.0.1:8765/` returned HTTP 200. Server then stopped.
