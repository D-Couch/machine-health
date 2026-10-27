# Machine Health status — 2026-10-27

## Milestones 1–6

Implemented a CPU-first NASA C-MAPSS FD001 pipeline with engine-held-out validation, causal rolling features, persisted sklearn artifact, Streamlit replay, strict FastAPI inference, validation-only alert threshold selection, local run provenance index, and a frozen one-time official test report. See README for the workflow and `docs/EVALUATION_PROTOCOL.md` for the test boundary.

Milestone 3 adds a compact causal PyTorch TCN, early stopping on training-group engines, a separate engine-level model-selection/calibration split within validation, nominal 80% engine-block conformal intervals, a model comparison report, and a model card. The five-tab dashboard walkthrough now includes the validation fleet review, project overview, replay, evaluation evidence, and methods/limitations. Report-driven views render without local raw data or model artifacts. The official test partition was not loaded during Milestone 3.

Milestone 4 adds a reliability-planner framing and an as-of validation fleet queue. The default common snapshot is cycle 120; status and ranking use only prediction histories and policy settings. The asset view links ranked engines to prediction/interval evidence, sensors through the snapshot, and the model artifact hash. A separate validation-only retrospective what-if scorer reports first-alert workload, timely events, early false alarms, misses, and lead time. Milestone 4 acceptance checks passed; no trained model or frozen report was changed. The official test raw data was absent in an integration smoke copy, and the workflow rendered without it.

Milestone 5 adds a local-only disposition form for inspect/monitor/defer/dismiss, optional rationale, and per-validation-engine append-only SQLite history. Each event preserves the as-of snapshot, prediction and optional interval, policy settings, model, and artifact hash; it stores no actual RUL and does not feed predictions, reports, or model training. The database is ignored by Git and can be redirected with `MACHINE_HEALTH_REVIEW_DB`. A structured author self-review and limitations are recorded in `docs/USABILITY_REVIEW.md`; external practitioner review remains outstanding.

Milestone 6 adds a true fresh-clone demo path driven by committed reports, a short walkthrough and architecture diagram in `docs/DEMO.md`, a project-aligned Streamlit color theme, and three real browser captures from the local FD001 app. The sidebar now describes the project as a portfolio prototype instead of showing stale milestone progress.

## Broader operational relevance — initial input-support audit

After the planned milestones, an initial exploratory slice of the separately gated operational-relevance work adds `machine-health audit-input-support`. It derives per-feature 1st–99th percentile bounds from training engines only and reports how often held-out validation features fall outside those bounds at fixed observed cycles. It is an offline diagnostic, not a live monitor or alert policy, and it reads no RUL or official-test data. Method and limitations are documented in `docs/INPUT_SUPPORT_AUDIT.md`.

On the real held-out FD001 validation cohort, cycle 120 had 15/2,600 feature values outside the training bounds (0.58%), affecting 5/20 engines; cycle 60 had 14/2,600 (0.54%), affecting 9/20; and cycle 180 had 27/1,430 (1.89%), affecting 4/11 engines with that cycle. These are descriptive feature-support counts from simulated complete trajectories, not operational warning rates or evidence of accuracy loss. The next useful gate is a concrete equipment context and representative authorized telemetry before turning such signals into API warnings or decision workflow behavior.

## Environment and data

- Executing in `/var/home/dylan/src/machine-health` on Bazzite Linux 44, x86_64, Python 3.14.8 in the project `.venv`; dependencies are locked in `requirements-lock.txt`.
- The host NVIDIA driver is loaded and the host RTX 4090 is healthy. The default Codex command sandbox (`workspace-write`) hides `/dev/nvidia*` and `/dev/dri/*`, so ordinary commands cannot see the GPU and PyTorch correctly falls back to CPU. An approved escalated command can access the device. The project's `.venv` now has CUDA-enabled PyTorch 2.14.1+cu130; the CI lock remains CPU-only. TCN training and inference automatically select CUDA when PyTorch can access the device, with CPU fallback otherwise.
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
- Reproducibility follow-up: `.github/workflows/ci.yml` now installs the pinned CPU-only `requirements-lock.txt`, installs the project editable with `--no-deps`, and keys pip caching to the lockfile. Its existing pytest suite includes the FastAPI health/predict/input-validation smoke with a temporary fixture artifact. CI still does not download NASA data or train benchmark models. The hosted GitHub Actions workflow has not been dispatched from this local session.
- `.venv/bin/ruff check src tests app` — passed after adding the input-support audit.
- `.venv/bin/pytest -q` — 29 passed; one existing Starlette `TestClient`/`httpx` deprecation warning.
- `.venv/bin/machine-health audit-input-support` — generated `reports/input_support_v1.json` and `.md` from local NASA training/validation data. Inputs use the fixed engine split and training-only feature quantiles; report provenance includes dataset/split and source hashes. Official-test files/labels and RUL labels were not used.

## GPU and CPU runtime selection — 2026-10-27

The optional causal TCN selects CUDA at runtime when PyTorch can access it, otherwise CPU. Checkpoint normalization tensors stay on CPU for NumPy preprocessing, while model weights and inference/training batches move to the selected device. Checkpoint and validation report provenance record the selected device, CUDA runtime, and GPU name; the CLI prints its selected device. Scikit-learn histogram gradient boosting remains CPU based. `scripts/install_torch.sh` installs PyTorch 2.14.1 CUDA 13.0 when `nvidia-smi` lists a GPU and verifies CUDA access; on a host without a GPU and without an existing CUDA build it installs PyTorch 2.14.1 CPU. If a restricted command sandbox hides GPU devices while a CUDA build already exists, the installer preserves that build. The host RTX 4090 and driver are healthy, but ordinary Codex `workspace-write` commands cannot access device nodes; GPU-dependent commands must use the approved escalated execution path when available. This retry rule is in this project's `AGENTS.md`. CI remains CPU-only through `requirements-lock.txt`.

Real GPU validation: `.venv/bin/machine-health train-temporal --checkpoint artifacts/fd001_tcn_gpu.pt --output reports/temporal_model_gpu_v1.json --replay data/processed/temporal_validation_replay_gpu.csv` ran with approved host GPU access. Its report records PyTorch 2.14.1+cu130, CUDA 13.0, and the NVIDIA GeForce RTX 4090. It trained on the original training partition, used 8 distinct training engines for early stopping, scored on 12 validation model-selection engines, and calibrated on 8 disjoint validation engines. Window MAE/RMSE were 21.902/28.528 cycles versus 23.653/30.502 for the matched gradient-boosted baseline. Early stopping selected epoch 5 (13 epochs run). The complete-trajectory endpoint MAE/RMSE were 0.915/1.900 cycles; these endpoints are zero by construction and are not truncated-life evidence. The official test partition was not opened or scored.

The saved GPU checkpoint reloaded onto CUDA and reproduced all 128 replay rows for engine `train:FD001:39` within 6.8e-06 cycles maximum absolute error. The normal restricted process sees the installed CUDA wheel but reports `cuda_available: false` and selects CPU; the same environment and code select CUDA when run with approved host access. The CPU lock and default CI path remain usable.

Checks: `.venv/bin/ruff check src tests app` passed; `.venv/bin/pytest -q` passed (30 tests, one existing Starlette/httpx deprecation warning); `bash -n scripts/install_torch.sh` passed. The installer preserves CUDA PyTorch in the restricted sandbox and detects the RTX 4090 under approved host access. The CLI banner and report both confirmed CUDA selection for the real training run. Implementation commit: `179a9b0` (`Enable CUDA TCN with CPU fallback`). The GPU report and this status update are prepared for a separate local evidence commit. No official-test evaluation was run.

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
