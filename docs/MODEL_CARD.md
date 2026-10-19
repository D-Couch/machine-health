# Model card: FD001 RUL baselines and temporal CNN

## Intended use

This is a reproducible research and portfolio demonstration for estimating turbofan remaining useful life (RUL) from NASA C-MAPSS simulated sensor histories. It supports offline engine-history replay and method comparison. It is not live telemetry, an aircraft maintenance system, or a safety decision tool.

## Data and target

The experiments use C-MAPSS FD001: 100 simulated run-to-failure training engines. The target is uncapped raw RUL in cycles, derived as each training engine's last cycle minus the current cycle. Engine IDs never enter the model. Original data is excluded from Git; source attribution and local hashes are documented in the project status and acquisition provenance.

## Models

- Histogram gradient boosting uses current cycle and settings plus causal 5, 10, and 20 cycle rolling sensor means and standard deviations.
- The temporal convolutional network (TCN) takes a 30-cycle trailing sequence of raw cycle, operating settings, and 21 sensor readings. It uses 32 channels, four residual causal convolution blocks with kernel size 3 and dilations 1, 2, 4, and 8, followed by a scalar RUL head. Its receptive field is 31 sequence positions; left padding gives early histories less than 30 observed cycles.
- The TCN uses a different but comparable history representation from the tree baseline. It has access to the last 30 raw cycles; the tree has summary statistics through the last 20 cycles. Both are limited to current and past data.

## Training and evaluation design

The original deterministic seed-41 split has 80 training engines and 20 held-out validation engines. For the TCN, the 80 training engines are divided by engine into 72 fitting engines and 8 early-stopping engines (seed 31415). The 20 validation engines are divided into 12 model-selection engines and 8 calibration engines using the same fixed seed. All partitions are disjoint by engine. Feature scaling and target scaling are fit only on the 72 fitting engines. Early stopping monitors MAE on the 8 training-group early-stopping engines (60 epoch limit, patience 8).

On the 12 model-selection engines, the tree baseline has window MAE/RMSE 23.653/30.502 cycles; the TCN has 20.701/28.217. Complete-trajectory endpoint MAE is 1.910 and 0.179 cycles respectively, but those endpoint targets are zero by construction and should not be interpreted as truncated-life performance. The 12-engine comparison is smaller than the project's original 20-engine Milestone 1 validation report.

## Uncertainty method

The TCN has nominal 80% split-conformal intervals. Eight disjoint calibration engines provide residuals. For each predicted-RUL band (0–19, 20–49, 50–99, 100+), one score per calibration engine is the maximum absolute residual over that engine's cycles whose predicted RUL falls in the band. The finite-sample order statistic is used; if there are too few distinct calibration engines to support a finite requested quantile, the interval is unavailable. Endpoints use the same cycle-wise predicted-band radius.

On the 12 model-selection engines, endpoint interval coverage was 12/12 (100%) with mean width 12.67 cycles. Actual-RUL-band mean cycle coverage ranged from 99.2% to 99.8%, with mean widths from 25.19 to 164.72 cycles. Engine-band simultaneous coverage ranged from 83.3% to 91.7%. These small-sample empirical values do not establish nominal coverage. Intervals are often wide, and the 8-engine calibration sample yields coarse quantiles.

## Limitations and risks

- C-MAPSS is a simulator benchmark; performance and uncertainty may not transfer to real engines or other operating conditions.
- There are only 20 held-out engines total, divided between model comparison and calibration. Error bars and coverage estimates are therefore unstable.
- Windows within an engine are correlated. Calibration uses engine-level block maxima, but reported cycle coverage remains descriptive; no independent-window uncertainty is claimed.
- Full validation trajectories end at failure, making their endpoint labels zero by construction. Endpoint MAE on these complete paths is weak evidence for operational RUL performance.
- The official FD001 test set was previously evaluated once under the frozen protocol for the tree baseline. This TCN milestone did not load or use official test sensors or labels, and its results do not change the frozen test report.
- No model output should be used for maintenance decisions.

## Reproducibility

Run `machine-health train-temporal` after `prepare` and the existing baseline `train` command. The command saves `artifacts/fd001_tcn.pt`, `reports/temporal_split.json`, `reports/temporal_model_v1.json` and Markdown, and a validation replay CSV. The checkpoint, local data, and replay file are ignored by Git; the small manifests and reports are versioned.
