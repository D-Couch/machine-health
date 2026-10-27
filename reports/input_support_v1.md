# FD001 validation input-support audit

NASA C-MAPSS FD001; descriptive input support audit

**Interpretation:** Feature values outside the training 1st–99th percentile bounds are a descriptive input-support signal, not evidence of prediction error or a production alert rate.

Bounds use training-only per-feature 1st and 99th percentiles across 16,987 rows and 80 engines. Validation features are scored at fixed observed-cycle snapshots; no RUL labels or official-test files are read.

| Snapshot cycle | Engines available | Feature values outside bounds | Engines with ≥1 outside feature |
|---:|---:|---:|---:|
| 60 | 20 | 0.54% (14/2600) | 9 |
| 120 | 20 | 0.58% (15/2600) | 5 |
| 180 | 11 | 1.89% (27/1430) | 4 |

## Validation engine detail at cycle 120

| Engine | Outside feature bounds | Features |
|---|---:|---|
| train:FD001:1 | 0/130 | — |
| train:FD001:8 | 0/130 | — |
| train:FD001:14 | 0/130 | — |
| train:FD001:23 | 1/130 | sensor_13_std_5 |
| train:FD001:30 | 1/130 | sensor_14_std_5 |
| train:FD001:31 | 0/130 | — |
| train:FD001:33 | 0/130 | — |
| train:FD001:38 | 0/130 | — |
| train:FD001:39 | 10/130 | sensor_2_mean_5, sensor_4_std_20, sensor_7_std_10, sensor_9_mean_10, sensor_9_mean_20, sensor_12_std_20, sensor_14_mean_5, sensor_14_mean_10, sensor_14_mean_20, sensor_15_mean_10 |
| train:FD001:42 | 0/130 | — |
| train:FD001:47 | 0/130 | — |
| train:FD001:58 | 0/130 | — |
| train:FD001:60 | 0/130 | — |
| train:FD001:63 | 2/130 | sensor_17_std_5, sensor_17_std_10 |
| train:FD001:70 | 0/130 | — |
| train:FD001:74 | 0/130 | — |
| train:FD001:80 | 0/130 | — |
| train:FD001:93 | 1/130 | sensor_3_std_10 |
| train:FD001:94 | 0/130 | — |
| train:FD001:100 | 0/130 | — |

## Limits

- The reference is derived from full run-to-failure NASA simulator training trajectories.
- Validation contains only 20 complete engines and is not representative of field telemetry.
- Univariate quantile bounds ignore feature dependence and do not measure model accuracy.
- Snapshot diagnostics are descriptive and must not be interpreted as operational alarm thresholds.

## Provenance

- Data SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`
- Split manifest SHA-256: `b3dc2188f4b937ad28c64bad00ad2dbda7c21fc7d89902754626e73335f4f2d6`
- Code revision: `2d59908224c34f49d98ecc5a7e2a6eb547be36dd`
- Validation engines: train:FD001:1, train:FD001:100, train:FD001:14, train:FD001:23, train:FD001:30, train:FD001:31, train:FD001:33, train:FD001:38, train:FD001:39, train:FD001:42, train:FD001:47, train:FD001:58, train:FD001:60, train:FD001:63, train:FD001:70, train:FD001:74, train:FD001:8, train:FD001:80, train:FD001:93, train:FD001:94
- Source SHA-256: `src/machine_health/input_monitoring.py` `5e931657e7d674422a51db63e26b3cca624cac24c9868fba839488db20854fc1`, `src/machine_health/features.py` `481a4a7d73a8c17ab1ecf0628f84c892d13c20ea68dfb265888ff802bb3b7638`, `src/machine_health/data.py` `bd6a45ea449f345bfb86039304c525ba1217c69d9fd7b7202310289dea036845`, `src/machine_health/cli.py` `78a9ddcba7176c2cbcf348e5cb3132696090a0aed8bdeed7501c8b72e10c517d`
