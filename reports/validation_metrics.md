# FD001 validation report

NASA C-MAPSS FD001; engine-held-out validation, not official test

Held-out engines: train:FD001:1, train:FD001:100, train:FD001:14, train:FD001:23, train:FD001:30, train:FD001:31, train:FD001:33, train:FD001:38, train:FD001:39, train:FD001:42, train:FD001:47, train:FD001:58, train:FD001:60, train:FD001:63, train:FD001:70, train:FD001:74, train:FD001:8, train:FD001:80, train:FD001:93, train:FD001:94
Scored windows: 3644

| Model | Window MAE (cycles) | Window RMSE (cycles) | Endpoint MAE (cycles) | Endpoint RMSE (cycles) |
|---|---:|---:|---:|---:|
| Training mean constant | 50.464 | 59.822 | — | — |
| Histogram gradient boosting | 26.032 | 33.679 | — | — |
| Endpoint only (20 engines) | — | — | 2.987 | 4.782 |

The endpoint score is the final observed cycle of each complete held-out validation trajectory. Window metrics weight each recorded cycle equally. The official test partition was not used.

## Endpoint errors

| Engine | Actual RUL | Prediction | Absolute error |
|---|---:|---:|---:|
| train:FD001:1 | 0 | 4.13 | 4.13 |
| train:FD001:100 | 0 | 14.07 | 14.07 |
| train:FD001:14 | 0 | 1.59 | 1.59 |
| train:FD001:23 | 0 | 4.20 | 4.20 |
| train:FD001:30 | 0 | 4.96 | 4.96 |
| train:FD001:31 | 0 | 0.78 | 0.78 |
| train:FD001:33 | 0 | 0.85 | 0.85 |
| train:FD001:38 | 0 | 0.30 | 0.30 |
| train:FD001:39 | 0 | 0.00 | 0.00 |
| train:FD001:42 | 0 | 11.91 | 11.91 |
| train:FD001:47 | 0 | 1.45 | 1.45 |
| train:FD001:58 | 0 | 0.20 | 0.20 |
| train:FD001:60 | 0 | 2.61 | 2.61 |
| train:FD001:63 | 0 | 4.70 | 4.70 |
| train:FD001:70 | 0 | 2.73 | 2.73 |
| train:FD001:74 | 0 | 0.00 | 0.00 |
| train:FD001:8 | 0 | 3.67 | 3.67 |
| train:FD001:80 | 0 | 0.00 | 0.00 |
| train:FD001:93 | 0 | 0.10 | 0.10 |
| train:FD001:94 | 0 | 1.51 | 1.51 |
