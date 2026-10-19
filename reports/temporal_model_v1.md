# FD001 temporal model and interval evaluation

Validation-only report. Official test data and labels were not loaded or used.

Model-selection engines: 12; calibration engines: 8.

## Point prediction comparison

| Model | Window MAE | Window RMSE | Endpoint MAE | Endpoint RMSE |
|---|---:|---:|---:|---:|
| Histogram gradient boosting | 23.653 | 30.502 | 1.910 | 2.599 |
| Causal TCN | 20.701 | 28.217 | 0.179 | 0.549 |

TCN early stopping selected epoch 9 of at most 60, using engine-held-out early-stopping data from the original training group.

## Provenance

Code revision: `fcb7f198c94736d904c1f67251c6ce8e3e52b386`; working tree dirty at run time: `True`. Component source SHA-256 hashes and dirty paths are recorded in the JSON report. TCN artifact SHA-256: `dd59f2af019c10b2e7252e8466479e3e9c1f8abc422e93bcf994e8ad386b6ab0`.
Runtime: Python 3.14.8, PyTorch 2.14.1+cpu, CPU platform `Linux-7.2.9-ogc3.1.fc44.x86_64-x86_64-with-glibc2.43`.

## Prediction intervals

Nominal 80% split-conformal intervals use calibration engines disjoint from model-selection engines. Residuals are grouped by predicted RUL band; each calibration engine contributes its maximum absolute residual in a band, preserving engine-level dependence.

| Actual RUL band | Engines | Engine-band simultaneous coverage | Mean cycle coverage | Mean width (cycles) |
|---|---:|---:|---:|---:|
| 0-19 | 12 | 0.917 | 0.992 | 25.19 |
| 20-49 | 12 | 0.917 | 0.992 | 62.25 |
| 50-99 | 12 | 0.917 | 0.998 | 130.88 |
| 100+ | 12 | 0.833 | 0.994 | 164.72 |

Endpoint coverage: 1.000 over 12 engines; mean endpoint interval width 12.67 cycles.

## Limitations

Only eight calibration engines are available. Engine-block conformal radii use each engine's maximum absolute residual within a predicted-RUL band; finite 80% quantiles are unavailable when a band has too few distinct engines. Coverage assumes exchangeable engines and does not address simulator-to-real shift.

TCN history is 30 raw cycles; the tree baseline uses causal summaries through 20 cycles. The TCN trains on windows from 72 fit engines and early-stops on 8 other training engines. All reports are validation results only.
