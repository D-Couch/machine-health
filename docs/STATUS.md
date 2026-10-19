# Machine Health status — 2026-10-19

## Milestones 1 and 2

Implemented a CPU-first NASA C-MAPSS FD001 pipeline with engine-held-out validation, causal rolling features, persisted sklearn artifact, Streamlit replay, strict FastAPI inference, validation-only alert threshold selection, local run provenance index, and a frozen one-time official test report. See README for the workflow and `docs/EVALUATION_PROTOCOL.md` for the test boundary.

## Environment and data

- Executing in `/var/home/dylan/src/machine-health` on Fedora Linux 6.17.12-300.fc43.x86_64, x86_64, Intel Core i9-13900KF (24 cores / 32 logical CPUs), 31 GiB RAM. Python 3.14.8 in the project `.venv`; dependencies are locked in `requirements-lock.txt`.
- NVIDIA GPU is not available to this process (`nvidia-smi` could not communicate with a driver); training and inference are CPU.
- Official NASA archive and FD001 train/test inputs are local and ignored by Git. Train has 20,631 rows and 100 engines. The official test has 100 truncated trajectories with paired endpoint labels. Data hashes and retrieval provenance are local under `data/raw/provenance.json`.
- NASA PCoE repository asks for dataset and donor acknowledgment. No distinct redistribution license was identified; raw data stays local.
- Git is a normal checkout on `main`, tracking `origin/main` at the initial baseline commit. The frozen protocol is committed as `a2223a5` before test scoring. No remote changes or pushes were made in this work.

## Real results

Fixed seed-41 split: 80 training engines and 20 held-out complete validation engines, separated before feature fitting. Targets are raw RUL cycles; trailing windows are causal and reset for each engine.

| Model | Validation window MAE | Validation window RMSE |
|---|---:|---:|
| Training-group mean constant | 50.464 cycles | 59.822 cycles |
| Histogram gradient boosting | 26.032 cycles | 33.679 cycles |

There are 3,644 validation windows. Complete-trajectory endpoint MAE/RMSE are 2.987/4.782 cycles; all endpoint labels are zero by construction, so these figures are not representative of truncated operation.

The frozen official test evaluation was run once after committing the protocol. On 100 engine endpoints, gradient boosting scored MAE 20.922 cycles and RMSE 29.316 (10,000-engine-bootstrap 95% intervals 16.855–25.015 and 24.314–33.903). The training-group mean constant scored 42.230/54.597. Secondary all-window metrics are 34.143/44.228 for boosting and 52.819/66.309 for the constant. See `reports/official_test_v1.md`.

Protocol deviation: the evaluator implementation was uncommitted when that one-time score was produced. The original report is preserved unchanged; `reports/official_test_v1_provenance_addendum.md` captures the dirty-tree state and source hashes. No test result was used to change features, model, or alert policy, and the official test has not been rescored.

Alert policy uses only the 20 complete validation engines: threshold 20 cycles, minimum history 20 cycles, three consecutive threshold crossings, 30-cycle actionable horizon. Result: 20/20 timely alerts, zero early false alarms, zero misses, mean lead 11.75 cycles. This is retrospective validation lifecycle evidence, not test lifecycle evidence.

## Implemented interfaces

- Streamlit: replay complete validation engines and show validation/test summaries.
- FastAPI: strict `POST /v1/predict` schema; requires 20–1,000 ordered observations and rejects label/engine metadata. Uses the shared causal feature builder and serialized model.
- SQLite experiment index: `data/experiments.sqlite`, ignored by Git; `machine-health list-runs` inspects recorded evaluations.
- Podman image build succeeded (`localhost/machine-health:milestone2`). Runtime container HTTP check was unavailable because Podman cannot initialize its runtime state in this sandbox (`/run/user/1000/libpod` is read-only). No GPU or cloud service is required.

## Checks run

- `.venv/bin/ruff check src tests app` — passed.
- `.venv/bin/pytest -q` — 10 passed. One Starlette deprecation warning notes that its TestClient currently uses deprecated `httpx` integration.
- Real artifact API smoke via FastAPI TestClient — `/health` returned `ok`; `/v1/predict` returned a finite nonnegative prediction for 25 real training observations. TestClient lifecycle was entered so artifact loading ran.
- `.venv/bin/machine-health evaluate-validation` — completed and reproduced the recorded validation endpoint result; `list-runs` shows validation, alert policy, and official test evaluation entries with artifact/data/report hashes.
- Official test evaluation was not rerun. Container build passed; container runtime smoke not verified due to the Podman sandbox limitation above.

## Launch and next steps

From the project directory, after local dependencies/data/artifact are present:

```bash
.venv/bin/streamlit run app/streamlit_app.py
.venv/bin/uvicorn machine_health.api:app --host 127.0.0.1 --port 8000
podman compose up --build
```

Recreate the local environment with `python3 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt`. Acquire and prepare data, then train with the README commands. The official test result is a frozen benchmark and must not be fed back into model or alert tuning. Further work can focus on uncertainty/calibration and broader subsets under a new validation design.
