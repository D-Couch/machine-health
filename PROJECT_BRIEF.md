# Machine Health — project brief

## Purpose
Build an end-to-end predictive maintenance portfolio project for applied AI/ML and ML engineering roles. Estimate remaining useful life (RUL) from multivariate equipment sensor histories, replay predictions over time, and evaluate maintenance alerts. Emphasize reproducibility, honest evaluation, usable software, and clear explanations. This complements the owner's MRI/quantum ML, Magic rules assistant, and network diagnostics projects by adding industrial time-series ML.

## Product
A user selects an engine and replays its recorded sensor history. The interface shows observed sensor traces, predicted cycles remaining, model selection, and alert status. Evaluation views show withheld ground truth and errors separately from live-style predictions. Later versions add calibrated prediction intervals and alert-policy comparisons. Recorded replay is not live telemetry. Predictions are a simulation-based research demonstration, not validated equipment maintenance advice.

## Data
Start with NASA C-MAPSS FD001 only. Official starting point: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
Verify current availability, documentation, column meanings, ground-truth conventions, attribution, and redistribution terms before implementing acquisition. Document the source URL, retrieval date, and checksums. Do not assume NASA hosting automatically grants unrestricted redistribution. Keep original datasets out of Git; provide acquisition and local-import commands. If the source is unavailable, support a verified local archive and report the blocker. Synthetic fixtures may test code but must never be presented as NASA performance results.

The training trajectories run to failure; official test trajectories are truncated, with endpoint RUL labels. Derive labels using the documented conventions and verify off-by-one behavior with hand-worked examples. Engine IDs must be namespaced by partition and subset because numeric IDs can repeat. Use cycles as units, never translate them into hours without evidence.

## Practical stack
Use a Python package with pyproject.toml, pandas/NumPy, scikit-learn, pytest, and Ruff. Use a constant training-derived RUL baseline and gradient-boosted trees on causal rolling features first. Prefer a straightforward scikit-learn boosting implementation before adding another framework. Use Streamlit with clear Plotly charts for the first dashboard. Add FastAPI, PyTorch for a compact temporal convolutional network, SQLite experiment records, and Docker in subsequent milestones. CPU training must work for baseline models; GPU support is optional and verified independently. No paid APIs, Kubernetes, distributed streaming, or mandatory cloud services. Keep notebooks optional; core behavior belongs in importable modules and CLI commands.

The owner's expected desktop: Bazzite Linux, i9-13900KF, RTX 4090, 32 GB RAM. Inspect the actual execution host; remote runtimes may differ. Use an isolated virtual environment or appropriate container. Avoid changing the host OS or GPU drivers for this project.

## Evaluation rules
1. Split complete training engines into deterministic train and validation groups BEFORE fitting transforms or constructing overlapping windows. Commit split manifests and seeds. Partition validation engines further for calibration only if uncertainty is implemented, or adopt a documented disjoint engine-level calibration split.
2. Reserve the official test partition for the frozen evaluation protocol. Choose features, models, thresholds, and hyperparameters on training/validation data. Record every official test evaluation and do not tune against it. If a final result exposes a bug, preserve the original report, explain the correction, and disclose that test data has been inspected.
3. Fit all transforms, feature selection, constant baselines, and imputation on training engines only. A prediction at cycle t can use only observations at or before t: no centered rolling windows, whole-trajectory normalization, future interpolation, engine lifetime, or remaining-life labels as features. Do not use engine identity as a predictive feature.
4. Use complete training-engine trajectories to construct RUL targets, but never expose their terminal lifetime as an inference input. For official test replay, reconstructed earlier RUL labels may be used only for offline scoring and displayed ground truth, not feature construction or inference.
5. State whether the training target is raw or capped RUL; make any cap explicit and fixed before test evaluation. Report official endpoint metrics against raw ground truth as the primary result, with any capped-label metrics clearly separate. Avoid claims based on a favorable cap alone.
6. Report endpoint MAE and RMSE in cycles, plus a documented asymmetric late-prediction penalty if implemented. Separate per-window results from endpoint results and avoid letting long trajectories silently dominate scores. Break down errors by remaining-life range. Use engine-level aggregation or bootstrap resampling for uncertainty on metrics when appropriate.
7. Select alert thresholds and persistence rules using validation engines. Measure event-level false alarms, missed failures, and lead time over held-out full run-to-failure validation engines. Define an actionable horizon, persistence, minimum-history rules, and denominators. C-MAPSS test records do not observe the full path to failure; do not claim observed full-lifecycle test alert performance from truncated records. Reconstructed labels can support explicitly labeled retrospective estimates only.
8. Later prediction intervals must be calibrated on separate engines and evaluated for empirical coverage and width, including by RUL range. Describe assumptions and distribution-shift limitations. Do not manufacture confidence bands from arbitrary percentages or treat correlated windows as independent calibration examples.
9. Record configurations, data hashes, splits, code revision/dirty state, package versions, metrics, and runtime. A neural model is not required to outperform trees. Publish unfavorable comparisons honestly.

## Milestones and acceptance criteria

### Milestone 1 — complete baseline application
Deliver a small working vertical slice, not just scaffolding:
- Installable package, verified data acquisition or local import, validation, and a deterministic engine split.
- Correct RUL labels and causal rolling features with meaningful leakage tests.
- A training-derived constant baseline and gradient-boosted model trained on real FD001 data.
- Validation comparison saved as machine-readable metrics and a readable report. Keep official test evaluation for Milestone 2.
- Model artifact including preprocessing and metadata; loading it yields reproducible predictions.
- Dashboard that loads trained artifacts and replays at least one held-out validation engine. Label withheld actual RUL as evaluation information. Never train on every page refresh.
- CLI commands for prepare, train, evaluate-validation, and replay data generation, or an equally simple documented interface.
- README with exact installation/run commands, dataset provenance, actual results, limitations, and status.
- Passing tests for parsing, labels, engine isolation, causal feature behavior, artifact round-trip, and one small end-to-end fixture. Dependency installation and actual test execution are part of this milestone.
- If real data access is blocked, deliver the local-import path and clearly report Milestone 1 as incomplete; do not substitute synthetic results.

### Milestone 2 — evaluation and inference engineering
- Freeze the protocol, then run official test endpoint evaluation with raw RUL labels.
- Engine-level error analysis and validated maintenance alert policy on full validation trajectories.
- FastAPI inference using the exact shared preprocessing/artifact pipeline. Validate request schema and sequence ordering; no training/test targets accepted as model inputs.
- SQLite experiment index with artifact paths and configurations.
- CPU-only CI for tests/lint, an API smoke test, and a containerized reproducible baseline demo. CI must not download the full dataset or train large models.

### Milestone 3 — temporal model and uncertainty
- Compact temporal convolutional network in PyTorch using the same splits and a comparable input-information budget. Match endpoint eligibility and clearly disclose different history lengths.
- Validation-based early stopping and bounded experiments. Save checkpoints, seeds, and comparisons.
- Separate calibration engines, interval coverage/width evaluation, and dashboard intervals.
- A concise model card and technical report covering limitations, selected approach, operational tradeoffs, and reproducibility.

### Milestone 4 — portfolio polish
- Clear architecture figure, polished dashboard, screenshots from real runs, and a short reproducible demo.
- Inspect the rendered interface and fix broken, crowded, or misleading views.
- Optional local input-quality monitoring: missing sensors, invalid ranges, feature drift. Distinguish input drift from measured accuracy loss; ground-truth performance monitoring requires delayed labels.
- Cloud deployment instructions may be added after the local workflow works. Do not incur hosting costs as part of this brief.

## Suggested structure
src/machine_health/ for data, features, models, evaluation, artifacts, CLI and later API; app/ for dashboard; configs/; tests/; docs/; reports/ for small versioned results; data/ and artifacts/ ignored by Git. Adapt this structure if a simpler arrangement is clearer. Avoid empty modules and speculative abstractions.

## Working method
Inspect the environment, record a short plan, and implement Milestone 1. Make routine implementation decisions independently. Maintain docs/STATUS.md with completed work, tested commands, current blockers, and the next concrete task. Future sessions should be able to continue from repository files. Do not mark a milestone complete based on compilation, a mock, or documentation alone.
