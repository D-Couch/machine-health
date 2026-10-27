# FD001 temporal model and interval evaluation

Validation-only report. Official test data and labels were not loaded or used.

Model-selection engines: 12; calibration engines: 8.

## Point prediction comparison

| Model | Window MAE | Window RMSE | Endpoint MAE | Endpoint RMSE |
|---|---:|---:|---:|---:|
| Histogram gradient boosting | 23.653 | 30.502 | 1.910 | 2.599 |
| Causal TCN | 21.902 | 28.528 | 0.915 | 1.900 |

TCN early stopping selected epoch 5 of at most 60, using engine-held-out early-stopping data from the original training group.

## Provenance

Code revision: `179a9b0ab965bc09061b8e6cdb448b6f0339bf49`; working tree dirty at run time: `True`. Component source SHA-256 hashes and dirty paths are recorded in the JSON report. TCN artifact SHA-256: `e3b6246c57b54c9800a3169b9f57d52600035945a0b724957d1a3315bcacf050`.
Runtime: Python 3.14.8, PyTorch 2.14.1+cu130, device `cuda` (NVIDIA GeForce RTX 4090), platform `Linux-7.2.9-ogc3.1.fc44.x86_64-x86_64-with-glibc2.43`.

## Prediction intervals

Nominal 80% split-conformal intervals use calibration engines disjoint from model-selection engines. Residuals are grouped by predicted RUL band; each calibration engine contributes its maximum absolute residual in a band, preserving engine-level dependence.

| Actual RUL band | Engines | Engine-band simultaneous coverage | Mean cycle coverage | Mean width (cycles) |
|---|---:|---:|---:|---:|
| 0-19 | 12 | 1.000 | 1.000 | 31.57 |
| 20-49 | 12 | 1.000 | 1.000 | 80.96 |
| 50-99 | 12 | 1.000 | 1.000 | 141.72 |
| 100+ | 12 | 0.917 | 0.991 | 165.59 |

Endpoint coverage: 1.000 over 12 engines; mean endpoint interval width 14.93 cycles.

## Limitations

Only eight calibration engines are available. Engine-block conformal radii use each engine's maximum absolute residual within a predicted-RUL band; finite 80% quantiles are unavailable when a band has too few distinct engines. Coverage assumes exchangeable engines and does not address simulator-to-real shift.

TCN history is 30 raw cycles; the tree baseline uses causal summaries through 20 cycles. The TCN trains on windows from 72 fit engines and early-stops on 8 other training engines. All reports are validation results only.
