# Machine Health — project brief

## Purpose
Build an end-to-end predictive maintenance portfolio project for applied AI/ML and ML engineering roles. Estimate remaining useful life (RUL) from multivariate equipment sensor histories, replay predictions over time, and evaluate maintenance alerts. Emphasize reproducibility, honest evaluation, usable software, and clear explanations. This complements the owner's MRI/quantum ML, Magic rules assistant, and network diagnostics projects by adding industrial time-series ML.

## Product
The project has two linked outcomes: a rigorous, reproducible ML benchmark and a benchmark-backed maintenance decision-support prototype. The product perspective is a reliability or maintenance planner reviewing a simulated fleet, prioritizing assets for inspection, and deciding whether to inspect, monitor, defer, or dismiss an alert. The interface should connect fleet-level prioritization to an asset's sensor history, prediction, uncertainty, alert rationale, and evaluation evidence.

NASA C-MAPSS is a narrow simulated benchmark, not a representative industrial fleet. Fleet and workflow views must label the data as simulated benchmark engines. The product prototype demonstrates how model outputs could fit into a planning workflow; it does not claim deployment readiness, measured business savings, aircraft maintenance suitability, or validated maintenance advice. Keep measured benchmark outcomes separate from user-entered or illustrative workload/cost assumptions. Recorded replay is not live telemetry.

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

### Milestone 4 — maintenance decision-support workflow
Make the project's user, decision, and workflow explicit, then reshape the existing dashboard around that job. Keep the current model and evaluation work as the evidence layer; this milestone does not require another model.
- Write a concise product brief naming the maintenance/reliability planner, the inspection-prioritization decision, the planning horizon, the supported actions, the intended benefit, and explicit non-goals. Align the README and dashboard opening view with it.
- Add a fleet overview and prioritized review queue over an explicitly identified validation cohort. Show scope and eligibility. Queue ordering may use predictions and documented policy inputs only; ground-truth RUL and outcomes are retrospective evaluation evidence, never ranking inputs.
- Add an asset review view linking priority to prediction, uncertainty where available, observed sensor history, alert rationale, model/artifact version, and relevant validation evidence.
- Add a policy explorer for threshold, persistence, and planning-horizon tradeoffs. Report measured validation alert counts, lead time, false alarms, misses, and review workload with clear denominators. Any interactive curves are explicitly retrospective validation analysis, not blind deployment results; do not label a threshold "optimal" based on repeated exploration. Keep the official test set frozen and out of interactive tuning.
- Clearly distinguish measured benchmark results from illustrative assumptions. Do not invent costs, avoided downtime, ROI, or operational impact.
- Preserve the existing replay and methods/evaluation views, adapting navigation and language to the fleet-to-asset workflow.

Acceptance criteria:
- A reviewer can identify the intended user and decision, start at the simulated fleet, inspect a prioritized asset, understand the evidence and alert rationale, and explore policy tradeoffs.
- Every fleet view says which benchmark cohort and model outputs it uses; every retrospective outcome is visibly labeled as evaluation data.
- Tests verify that queue ordering does not consume ground-truth labels, policy scoring uses labels only for explicitly retrospective validation outcomes, and the official test partition is never loaded by the interactive workflow. Existing benchmark reports remain reproducible and unchanged.
- No user-facing copy implies live telemetry, real-fleet validation, maintenance advice, or measured economic benefit.

### Milestone 5 — human review loop and usability
Demonstrate the human handoff without implying an integration with a real maintenance system.
- Let a reviewer record a local prototype disposition: inspect, monitor, defer, or dismiss, with an optional rationale.
- Show disposition and alert history for the selected benchmark engine. Label this as local demonstration state; it is not production CMMS data or model feedback.
- Define task-based usability checks: find urgent assets, explain a priority, choose a disposition, compare policies, and distinguish evidence from assumptions.
- Review the experience with maintenance/reliability practitioners if available. Record task observations and resulting changes. If no practitioners are available, perform a structured self-review and explicitly state that external usability review remains outstanding.

Acceptance criteria:
- The reviewer can complete the defined workflow and understand what is recorded, where it is stored, and what the prediction supports.
- Usability findings are recorded without claiming user validation when no target users participated.
- Prototype dispositions do not alter model predictions, evaluation results, or the frozen official-test protocol.

### Milestone 6 — portfolio readiness and reproducibility
- Provide a clear architecture figure, polished rendered dashboard, screenshots captured from real project reports/artifacts, and a short reproducible demo path.
- Inspect the rendered interface and fix broken, crowded, inaccessible, or misleading views.
- Make the fresh-clone walkthrough useful from committed reports without local NASA data or model files; document the additional local prerequisites for replay and retraining.
- Connect displayed results to versioned reports, splits, configuration, and model artifacts. State the benchmark scope and limitations where reviewers encounter results.
- Keep local launch simple. Optional cloud deployment instructions may be added only after the local workflow works; no hosting costs are required.

Acceptance criteria:
- A reviewer can follow the documented demo and understand the product workflow and ML evidence without hidden local files or author narration.
- Screenshots and demo results correspond to actual recorded runs; no synthetic fixture result is presented as benchmark performance.
- Relevant tests, lint, and fresh-clone smoke checks pass, and docs/STATUS.md records exact commands and remaining limitations.

### Future, separately gated — broader operational relevance
Do not treat this as a commitment to deploy the current model. Consider broader datasets or fault modes, local input-quality monitoring, delayed-label performance monitoring, real maintenance-system integration, or deployment only when a concrete user need and suitable data/access are available. Any new evaluation requires a new documented validation design. Distinguish input drift from measured accuracy loss; accuracy monitoring requires labels. No paid cloud services are required by this brief.

## Suggested structure
src/machine_health/ for data, features, models, evaluation, artifacts, CLI and later API; app/ for dashboard; configs/; tests/; docs/; reports/ for small versioned results; data/ and artifacts/ ignored by Git. Adapt this structure if a simpler arrangement is clearer. Avoid empty modules and speculative abstractions.

## Working method
Inspect the environment, record a short plan, and implement Milestone 1. Make routine implementation decisions independently. Maintain docs/STATUS.md with completed work, tested commands, current blockers, and the next concrete task. Future sessions should be able to continue from repository files. Do not mark a milestone complete based on compilation, a mock, or documentation alone.
