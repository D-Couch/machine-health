# Machine Health status — 2026-10-27

## Milestones 1–6

Implemented a CPU-first NASA C-MAPSS FD001 pipeline with engine-held-out validation, causal rolling features, persisted sklearn artifact, Streamlit replay, strict FastAPI inference, validation-only alert threshold selection, local run provenance index, and a frozen one-time official test report. See README for the workflow and `docs/EVALUATION_PROTOCOL.md` for the test boundary.

Milestone 3 adds a compact causal PyTorch TCN, early stopping on training-group engines, a separate engine-level model-selection/calibration split within validation, nominal 80% engine-block conformal intervals, a model comparison report, and a model card. The five-tab dashboard walkthrough now includes the validation fleet review, project overview, replay, evaluation evidence, and methods/limitations. Report-driven views render without local raw data or model artifacts. The official test partition was not loaded during Milestone 3.

Milestone 4 adds a reliability-planner framing and an as-of validation fleet queue. The default common snapshot is cycle 120; status and ranking use only prediction histories and policy settings. The asset view links ranked engines to prediction/interval evidence, sensors through the snapshot, and the model artifact hash. A separate validation-only retrospective what-if scorer reports first-alert workload, timely events, early false alarms, misses, and lead time. Milestone 4 acceptance checks passed; no trained model or frozen report was changed. The official test raw data was absent in an integration smoke copy, and the workflow rendered without it.

Milestone 5 adds a local-only disposition form for inspect/monitor/defer/dismiss, optional rationale, and per-validation-engine append-only SQLite history. Each event preserves the as-of snapshot, prediction and optional interval, policy settings, model, and artifact hash; it stores no actual RUL and does not feed predictions, reports, or model training. The database is ignored by Git and can be redirected with `MACHINE_HEALTH_REVIEW_DB`. A structured author self-review and limitations are recorded in `docs/USABILITY_REVIEW.md`; external practitioner review remains outstanding.

Milestone 6 adds a true fresh-clone demo path driven by committed reports, a short walkthrough and architecture diagram in `docs/DEMO.md`, a project-aligned Streamlit color theme, and three real browser captures from the local FD001 app. The sidebar now describes the project as a portfolio prototype instead of showing stale milestone progress.

## Environment and data

- Executing in `/var/home/dylan/src/machine-health` on Fedora Linux 6.17.12-300.fc43.x86_64, x86_64, Intel Core i9-13900KF (24 cores / 32 logical CPUs), 31 GiB RAM. Python 3.14.8 in the project `.venv`; dependencies are locked in `requirements-lock.txt`.
- NVIDIA GPU is not available to this process (`nvidia-smi` could not communicate with a driver); training and inference are CPU.
- Official NASA archive and FD001 train/test inputs are local and ignored by Git. Train has 20,631 rows and 100 engines. The official test has 100 truncated trajectories with paired endpoint labels. Data hashes and retrieval provenance are local under `data/raw/provenance.json`.
- NASA PCoE repository asks for dataset and donor acknowledgment. No distinct redistribution license was identified; raw data stays local.
- Git is a normal checkout on `main`, tracking `origin/main`. The frozen protocol preceded test scoring. Latest pushed commit before current Milestone 4 work is `0cad66d` on `origin/main`.

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

The TCN and tree comparison use the same 12 model-selection engines; 8 other original validation engines are reserved for calibration. The original 80 training engines are split into 72 fitting and 8 early-stopping engines. On the 12 comparison engines:

| Model | Window MAE | Window RMSE | Endpoint MAE |
|---|---:|---:|---:|
| Histogram gradient boosting | 23.653 cycles | 30.502 cycles | 1.910 cycles |
| Causal TCN | 20.701 cycles | 28.217 cycles | 0.179 cycles |

The TCN selected epoch 9 (17 epochs ran, limit 60, patience 8). The complete validation paths end at failure, so endpoint targets are zero by construction and endpoint scores are not operational truncated-life evidence. TCN intervals use eight disjoint calibration engines and per-engine maximum absolute residuals within predicted-RUL bands. On the 12 model-selection engines, endpoint coverage was 12/12 with mean width 12.67 cycles. Mean cycle coverage by actual-RUL band was 99.2%–99.8%; mean widths ranged 25.19–164.72 cycles. Engine-band simultaneous coverage ranged 83.3%–91.7%. These are small-sample descriptive checks, not evidence of exact nominal coverage. Full per-engine and RUL-band details are in `reports/temporal_model_v1.md`; model assumptions are in `docs/MODEL_CARD.md`.

## Implemented interfaces

- Streamlit: five-tab portfolio walkthrough with a validation fleet queue, as-of asset review, scenario policy scoring, engine replay, evaluation evidence, and methods. The queue consumes prediction histories only; label-bearing outcomes enter a separate validation-only retrospective scorer and asset evidence panel. Report-driven overview/evidence still work on a fresh clone without local data/artifacts; fleet review/replay explain their local prerequisites.
- FastAPI: strict `POST /v1/predict` schema; requires 20–1,000 ordered observations and rejects label/engine metadata. Uses the shared causal feature builder and serialized model.
- SQLite experiment index: `data/experiments.sqlite`, ignored by Git; `machine-health list-runs` inspects recorded evaluations.
- TCN checkpoint: `artifacts/fd001_tcn.pt`, local-only and ignored by Git. Replay predictions are in ignored `data/processed/temporal_validation_replay.csv`.
- Podman image `localhost/machine-health:milestone3` builds with the lockfile and CPU PyTorch. An approved temporary container run served the dashboard and Streamlit health endpoint with HTTP 200; the container was stopped afterward. No GPU or cloud service is required.

## Checks run

- `.venv/bin/ruff check src tests app` — passed after adding TCN code/tests.
- `.venv/bin/pytest -q` — 15 passed. One Starlette deprecation warning notes that its TestClient currently uses deprecated `httpx` integration.
- Installed CPU-only PyTorch 2.14.1 in `.venv` from the official PyTorch CPU wheel index; no CUDA package/driver was added.
- `.venv/bin/machine-health train-temporal` — completed on real NASA training/validation engines, early-stopping at epoch 9. Saved the TCN checkpoint, disjoint split manifest, validation JSON/Markdown report, and replay CSV. The official test partition was not opened.
- The TCN report records code revision, dirty state, source hashes, Python/platform, package versions, CPU threads, data hash, parent split hash, and checkpoint SHA-256. Checkpoint reload reproduced 192 predictions for a held-out engine within 7.4e-06 cycles of the saved replay.
- Synthetic TCN tests cover deterministic engine partitioning, causal sequence construction, engine reset/padding, model output, calibration finite-sample bound, and interval band assignment. Synthetic test values are not reported as NASA results.
- Streamlit `AppTest` rendered all four sections, nine Plotly visualizations, the architecture diagram, and TCN replay mode (12 eligible engines). A temporary no-data project copy rendered overview/evidence and correctly gated replay.
- `podman build -t machine-health:milestone3 .` — passed. Temporary local dashboard container returned HTTP 200 at `/` and `/_stcore/health`; container was stopped.
- Real artifact API smoke via FastAPI TestClient — `/health` returned `ok`; `/v1/predict` returned a finite nonnegative prediction for 25 real training observations. TestClient lifecycle was entered so artifact loading ran.
- `.venv/bin/machine-health evaluate-validation` — completed and reproduced the recorded validation endpoint result; `list-runs` shows validation, alert policy, and official test evaluation entries with artifact/data/report hashes.
- Official test evaluation was not rerun. The `localhost/machine-health:milestone3` image build and temporary runtime HTTP smoke both passed; data/model volumes are local-only.
- Milestone 4 unit tests verify validation-only engine namespaces, prediction-only queue inputs, label-column rejection, contiguous as-of history, future-cycle invariance, and separate retrospective policy scoring.
- `.venv/bin/ruff check src tests app` — passed after adding the planner queue and scenario scorer.
- `.venv/bin/pytest -q` — 18 passed. One existing Starlette `TestClient`/`httpx` deprecation warning.
- Streamlit `AppTest` rendered all five tabs without exceptions in gradient-boosting and TCN modes. The queue had 20 and 12 rows respectively; TCN rows included interval bounds. A non-default threshold/persistence/horizon scenario changed the queue and retrospective metrics without changing saved reports.
- At the default shared snapshot (cycle 120) on the real 20-engine FD001 validation cohort, the tree queue displayed 1 Review, 1 Watch, and 18 Routine assets. This is an as-of prediction queue example, not a performance, workload-cost, or operational-impact claim. The default scenario's retrospective metrics matched the existing selected-policy report (20/20 timely, 0 early false alarms, 0 missed).
- A temporary fresh-clone copy with committed reports but no raw data/artifacts rendered all five tabs and correctly gated fleet review/replay. Headless Streamlit `/` and `/_stcore/health` both returned HTTP 200; server stopped after smoke check.
- No raw official-test sensors or endpoint labels were loaded, and no official-test score was rerun during Milestone 4 work.
- `.venv/bin/ruff check src tests app` — passed after Milestone 5 changes.
- `.venv/bin/pytest -q` — 25 passed; one existing Starlette `TestClient`/`httpx` deprecation warning.
- Streamlit `AppTest` rendered all five tabs with no exceptions. A disposition plus rationale submitted to a temporary SQLite database, showed a saved-event success message, and appeared in the selected engine's history. The temporary database was outside the project.
- Review-log tests confirm append-only event ordering, validation-engine filtering, provenance capture, absence of actual-RUL fields, and rejection of invalid namespace/disposition/hash/number/rationale/interval inputs.
- The official test data, labels, frozen report, trained artifacts, and validation reports were not changed or evaluated during Milestone 5.
- Fresh-clone AppTest using a temporary copy with committed source/reports and no raw data or model artifacts — all five tabs rendered without exceptions; committed overview/evaluation metrics appeared and the Fleet/Replay views correctly showed local prerequisites.
- Headless Chromium browser inspection against the actual local FD001 app — HTTP 200, no page errors, selected Fleet/Overview/Evaluation tabs captured in `docs/images/`, and no horizontal overflow at 1440px or 390px viewport widths. Screenshots use actual committed reports and local trained artifacts, not fixtures.
- `.streamlit/config.toml` applies the dashboard's blue/teal palette consistently to Streamlit controls and tabs. `README.md` now separates the quick report-only demo from data acquisition, baseline training, and optional temporal-model reproduction.
- Milestone 6 did not load official-test raw sensors/labels, rerun its score, or change any metric/report/artifact. Chromium/Playwright were installed only in the ignored local `.venv`/`/tmp` for this visual review; they are not runtime dependencies.

## Launch and next steps

From the project directory, after local dependencies/data/artifact are present:

```bash
.venv/bin/streamlit run app/streamlit_app.py
.venv/bin/uvicorn machine_health.api:app --host 127.0.0.1 --port 8000
podman compose up --build
```

Recreate the local environment with `python3 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt && .venv/bin/pip install -e '.[dev]'`. Acquire and prepare data, then train with the README commands. The official test result is a frozen benchmark and must not be fed back into model or alert tuning. Future uncertainty or broader-subset work requires a new documented validation design and should follow the product workflow milestones unless a concrete user need prioritizes it.

The dashboard shows a TCN option and calibrated interval when `train-temporal` outputs are present. A remaining limitation is that the current run uses only 8 calibration engines; intervals are coarse and can be wide. Model-selection uses 12 engines, so score uncertainty is substantial. No TCN official-test metric is claimed or available. Dashboard changes are included in `5c83d75`, pushed to `origin/main`.

## Roadmap revision — 2026-10-27

Milestones 1–6 are complete. The former Milestone 4 combined visual polish with optional input monitoring and deployment notes, but did not specify a concrete maintenance user or decision. `PROJECT_BRIEF.md` now replaces that with three sequenced milestones: Milestone 4 builds a benchmark-backed fleet-to-asset inspection-planning workflow; Milestone 5 adds a local human-review prototype and task-based usability checks; Milestone 6 makes the product and evidence easy to evaluate from a fresh clone. Broader operational relevance, data drift work, and any deployment are separately gated on a concrete need and suitable data. The official test remains frozen and excluded from interactive product exploration.

Milestone 4 acceptance was checked through prediction-only queue unit tests, separate retrospective scorer tests, app rendering in baseline/TCN modes, non-default policy controls, a report-only fresh-clone simulation, a temporary runtime without official-test sensors/labels, and HTTP health checks. Milestone 5 acceptance is checked through review-log tests and temporary-database app interaction. Milestone 6 acceptance is checked through the report-only fresh-clone test, browser captures, and viewport/error checks above. No external practitioner review has been run; that remains the principal usability limitation rather than a claim of user validation.
