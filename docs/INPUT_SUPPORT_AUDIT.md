# Input support audit

The audit is an initial step toward input-quality monitoring. It checks whether held-out FD001 validation feature values fall outside simple per-feature bands derived from training engines. It does not alter model predictions, issue operator alerts, infer accuracy loss, or evaluate a maintenance decision.

Run it after the local NASA FD001 training file and split manifest are present:

```bash
.venv/bin/machine-health audit-input-support
```

The command builds the same causal features as training, fits each feature's 1st and 99th percentile bounds on the 80 training engines, and scores held-out validation engines at fixed cycles 60, 120, and 180. It does not read RUL labels or load official test sensors or labels. JSON and Markdown outputs include the data/split hashes, source hashes, selected engine groups, number of engines available at each snapshot, per-feature coverage, and per-engine detail.

## Current FD001 observation

The recorded run is in [`reports/input_support_v1.md`](../reports/input_support_v1.md). At cycle 120, 15 of 2,600 feature values (0.58%) were outside their training 1st–99th percentile bands, affecting 5 of the 20 validation engines. At cycle 60, 14/2,600 (0.54%) were outside, affecting 9/20 engines. At cycle 180, 27/1,430 (1.89%) were outside across 4 of the 11 validation engines with a cycle-180 observation.

This shows that the summary can surface differences in feature support in this dataset. It does not establish that a real sensor stream has drifted or that predictions have become less accurate. The reference consists of correlated observations from complete simulator trajectories; one outlying feature is not a failure condition.

## Limits and next evidence needed

- The 1st–99th percentile bands are descriptive choices, not calibrated operational thresholds.
- The 130 input features are correlated. The fraction of feature cells outside univariate bands is not a multivariate probability or confidence score.
- The validation cohort has 20 engines and shares the benchmark development context. It is not independent field evidence.
- Input support and input integrity differ: the API already rejects malformed schemas, non-finite values, short sequences, and non-increasing cycles. This report considers whether well-formed causal features resemble the training feature ranges.
- No accuracy or maintenance-utility conclusion is possible without appropriately matched delayed outcomes and a new evaluation design.
- Before exposing warnings in an API or planning workflow, obtain representative sensor history and operating limits from a concrete equipment context, then measure false-warning behavior on separate engines/assets.
