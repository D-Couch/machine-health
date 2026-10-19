from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from machine_health.data import sha256
from machine_health.features import FEATURES
from machine_health.model import git_revision

RUL_BANDS = ((0, 20, "0-19"), (20, 50, "20-49"), (50, 100, "50-99"), (100, np.inf, "100+"))
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026


def _metrics(actual, predicted) -> dict[str, float]:
    return {
        "mae_cycles": float(mean_absolute_error(actual, predicted)),
        "rmse_cycles": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def bootstrap_interval(actual, predicted) -> dict[str, dict[str, float]]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(actual), size=(BOOTSTRAP_REPLICATES, len(actual)))
    errors = predicted[samples] - actual[samples]
    maes = np.abs(errors).mean(axis=1)
    rmses = np.sqrt(np.square(errors).mean(axis=1))
    return {
        "mae_cycles": {"lower_95": float(np.quantile(maes, 0.025)), "upper_95": float(np.quantile(maes, 0.975))},
        "rmse_cycles": {"lower_95": float(np.quantile(rmses, 0.025)), "upper_95": float(np.quantile(rmses, 0.975))},
    }


def evaluate_official_test(frame, features, artifact: dict, test_file: Path, labels_file: Path,
                           protocol_file: Path, artifact_file: Path, output: Path) -> dict:
    """Run the frozen, engine-level official test evaluation exactly once."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen first report: {output}")
    if not frame["engine_id"].str.startswith("test:FD001:").all():
        raise ValueError("Official test evaluator requires namespaced FD001 test engines")
    if "rul" not in frame or "rul" in features.columns:
        raise ValueError("Retrospective labels must be present only in the scoring frame")
    predictions = np.maximum(0, artifact["model"].predict(features[FEATURES]))
    scored = frame[["engine_id", "unit", "cycle", "rul"]].copy()
    scored["prediction"] = predictions
    endpoints = scored.sort_values("cycle").groupby("engine_id", sort=True).tail(1).sort_values("unit")
    actual = endpoints["rul"].to_numpy(dtype=float)
    predicted = endpoints["prediction"].to_numpy(dtype=float)
    constant_prediction = np.full(len(actual), float(artifact["constant_prediction"]))
    result = {
        "evaluation_id": "fd001_official_test_v1",
        "protocol": "docs/EVALUATION_PROTOCOL.md v1",
        "dataset": "NASA C-MAPSS FD001 official test; endpoint labels",
        "evaluated_utc": datetime.now(UTC).isoformat(),
        "code_revision": git_revision(),
        "n_engines": len(endpoints),
        "target": "raw uncapped RUL in cycles",
        "primary_aggregation": "one endpoint per engine, equally weighted",
        "primary_model": {"name": "histogram gradient boosting", **_metrics(actual, predicted),
                          "engine_bootstrap_95_ci": bootstrap_interval(actual, predicted)},
        "constant_baseline": {"name": "training-group mean constant", **_metrics(actual, constant_prediction),
                              "engine_bootstrap_95_ci": bootstrap_interval(actual, constant_prediction)},
        "by_actual_rul_band": {},
        "endpoint_per_engine": [],
        "all_window_secondary": {
            "aggregation": "all observed test cycles equally weighted; longer trajectories contribute more rows",
            "hist_gradient_boosting": _metrics(scored.rul, scored.prediction),
            "constant_baseline": _metrics(scored.rul, np.full(len(scored), artifact["constant_prediction"])),
            "n_windows": len(scored),
        },
        "inputs": {"test_sensors_sha256": sha256(test_file), "endpoint_labels_sha256": sha256(labels_file),
                   "artifact_sha256": sha256(artifact_file),
                   "protocol_sha256": sha256(protocol_file)},
        "test_feedback_used_for_model_selection": False,
    }
    for low, high, label in RUL_BANDS:
        selection = (actual >= low) & (actual < high)
        if selection.any():
            result["by_actual_rul_band"][label] = {
                "n_engines": int(selection.sum()),
                "hist_gradient_boosting": _metrics(actual[selection], predicted[selection]),
                "constant_baseline": _metrics(actual[selection], constant_prediction[selection]),
            }
    for row in endpoints.itertuples(index=False):
        result["endpoint_per_engine"].append({
            "engine_id": row.engine_id,
            "unit": int(row.unit),
            "last_observed_cycle": int(row.cycle),
            "actual_rul_cycles": int(row.rul),
            "predicted_rul_cycles": float(row.prediction),
            "signed_error_cycles": float(row.prediction - row.rul),
            "absolute_error_cycles": float(abs(row.prediction - row.rul)),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# FD001 official test evaluation — frozen v1",
        "",
        "Protocol: [`EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md), committed before scoring.",
        "Official test metrics are reported once. No test feedback was used for model, feature, or threshold selection.",
        "",
        f"Engines: {len(endpoints)} · target: raw RUL cycles · endpoint is the final observed test cycle per engine.",
        "",
        "## Primary endpoint results",
        "",
        "| Model | MAE (cycles) | RMSE (cycles) | MAE engine bootstrap 95% CI | RMSE engine bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_key, display in (("primary_model", "Histogram gradient boosting"), ("constant_baseline", "Training-group mean constant")):
        metric = result[model_key]
        ci = metric["engine_bootstrap_95_ci"]
        lines.append(f"| {display} | {metric['mae_cycles']:.3f} | {metric['rmse_cycles']:.3f} | "
                     f"[{ci['mae_cycles']['lower_95']:.3f}, {ci['mae_cycles']['upper_95']:.3f}] | "
                     f"[{ci['rmse_cycles']['lower_95']:.3f}, {ci['rmse_cycles']['upper_95']:.3f}] |")
    lines.extend(["", "Bootstrap resamples engines with replacement (10,000 replicates, seed 2026). Intervals express engine sampling variability, not distribution-shift uncertainty.",
                  "", "## Error by actual endpoint RUL", "", "| RUL band (cycles) | Engines | Boosting MAE | Boosting RMSE | Constant MAE | Constant RMSE |", "|---|---:|---:|---:|---:|---:|"])
    for band, item in result["by_actual_rul_band"].items():
        boost, constant = item["hist_gradient_boosting"], item["constant_baseline"]
        lines.append(f"| {band} | {item['n_engines']} | {boost['mae_cycles']:.3f} | {boost['rmse_cycles']:.3f} | "
                     f"{constant['mae_cycles']:.3f} | {constant['rmse_cycles']:.3f} |")
    lines.extend(["", "## Per-engine endpoint predictions", "", "| Engine | Last cycle | Actual RUL | Predicted RUL | Signed error | Absolute error |", "|---|---:|---:|---:|---:|---:|"])
    for row in result["endpoint_per_engine"]:
        lines.append(f"| {row['engine_id']} | {row['last_observed_cycle']} | {row['actual_rul_cycles']} | "
                     f"{row['predicted_rul_cycles']:.2f} | {row['signed_error_cycles']:.2f} | {row['absolute_error_cycles']:.2f} |")
    secondary = result["all_window_secondary"]
    secondary_summary = (f"{secondary['n_windows']} observed cycles; longer engines have more weight. "
                  f"Boosting MAE/RMSE: {secondary['hist_gradient_boosting']['mae_cycles']:.3f}/"
                  f"{secondary['hist_gradient_boosting']['rmse_cycles']:.3f} cycles. "
                  f"Constant MAE/RMSE: {secondary['constant_baseline']['mae_cycles']:.3f}/"
                  f"{secondary['constant_baseline']['rmse_cycles']:.3f} cycles.")
    lines.extend(["", "## Secondary all-window diagnostics", "", secondary_summary,
                  "", "These are simulated C-MAPSS benchmark results, not validated aircraft maintenance predictions."])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return result
