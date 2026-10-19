# FD001 frozen evaluation protocol — v1

**Frozen:** 2026-10-19, before official test labels are loaded for scoring.

## Scope and data boundaries

- Subset: official NASA C-MAPSS FD001 test trajectories and their paired `RUL_FD001.txt` endpoint labels.
- Training and model selection use only the committed seed-41 engine split: 80 train engines and 20 complete validation engines. Test engines are not used for fitting, threshold choice, features, or model selection.
- Sensor feature schema and causal trailing windows (5, 10, 20 cycles) are the Milestone 1 schema. Features at cycle t use observations through t only, reset per engine.
- Target is raw uncapped remaining cycles. For official test engine i with endpoint label `r_i` and last observed cycle `c_i`, earlier cycle t has retrospective scoring label `r_i + c_i - t`. This reconstructed label is only for scoring; neither labels nor endpoint lifetime enter inference.

## Frozen estimators and artifacts

- Primary estimator: the already-trained `artifacts/fd001.joblib`, histogram gradient boosting, random seed 41, trained on the fixed 80-engine training group. Predictions are clipped at zero for reporting.
- Comparator: constant prediction equal to the training-group mean raw RUL saved in the same artifact.
- Artifact SHA-256: `2fa1a538304661af10d98206d72b16dd7f79717fb7e03d1feb06e194c2476138`.
- Training file SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- Split manifest SHA-256: `b3dc2188f4b937ad28c64bad00ad2dbda7c21fc7d89902754626e73335f4f2d6`.

## Metrics, aggregation, and uncertainty

1. Primary: endpoint MAE and RMSE in cycles over the 100 official test engines, one equally weighted observation per engine.
2. Report each engine's true endpoint RUL, prediction, and absolute error; show endpoint errors grouped by actual RUL bands `[0,20)`, `[20,50)`, `[50,100)`, `[100,+inf)` with engine counts, MAE, and RMSE.
3. Report 95% percentile bootstrap intervals for endpoint MAE/RMSE by resampling engines with replacement (10,000 replicates, seed 2026). These intervals describe test-engine sampling variability only; they do not account for distribution shift or simulator limitations.
4. Report all-window metrics separately as secondary diagnostics, equally weighting observed cycles (and explicitly state this weights long trajectories more). No test-derived threshold or model changes are permitted.

## Change control

This protocol and its source commit precede test scoring. Any discovered defect after scoring must be documented as a protocol deviation; preserve the first report and label any corrected rescore as a later evaluation. No parameter tuning or feature changes may use these results.
