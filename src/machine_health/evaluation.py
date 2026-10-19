from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from machine_health.features import FEATURES


def _metrics(y, prediction) -> dict:
    return {"mae_cycles": float(mean_absolute_error(y, prediction)),
            "rmse_cycles": float(np.sqrt(mean_squared_error(y, prediction)))}


def evaluate(artifact: dict, features: pd.DataFrame, labeled: pd.DataFrame, engines: list[str], out: Path) -> dict:
    mask = features.engine_id.isin(engines)
    rows = features.loc[mask]
    y = labeled.set_index(features.index).loc[rows.index, "rul"].to_numpy()
    prediction = np.maximum(0, artifact["model"].predict(rows[FEATURES]))
    constant = np.full(len(y), artifact["constant_prediction"])
    result = {
        "dataset": "NASA C-MAPSS FD001; engine-held-out validation, not official test",
        "split_engines": engines,
        "n_windows": len(y),
        "n_engines": len(engines),
        "aggregation": "window-weighted; all cycles from each held-out run-to-failure engine",
        "constant_baseline": _metrics(y, constant),
        "hist_gradient_boosting": _metrics(y, prediction),
        "endpoint": {},
        "by_rul_range": {},
    }
    details = []
    for engine in engines:
        idx = np.flatnonzero(rows.engine_id.to_numpy() == engine)
        end_idx = idx[-1]
        details.append({"engine_id": engine, "endpoint_true_rul": int(y[end_idx]),
                        "endpoint_prediction": float(prediction[end_idx]),
                        "endpoint_absolute_error": float(abs(y[end_idx] - prediction[end_idx]))})
    result["endpoint"] = _metrics([d["endpoint_true_rul"] for d in details],
                                  [d["endpoint_prediction"] for d in details])
    result["endpoint_per_engine"] = details
    edges = [0, 20, 50, 100, np.inf]
    labels = ["0-19", "20-49", "50-99", "100+"]
    for low, high, label in zip(edges[:-1], edges[1:], labels):
        selected = (y >= low) & (y < high)
        if selected.any():
            result["by_rul_range"][label] = _metrics(y[selected], prediction[selected])
            result["by_rul_range"][label]["n_windows"] = int(selected.sum())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    report = ["# FD001 validation report", "", result["dataset"], "",
              f"Held-out engines: {', '.join(engines)}", f"Scored windows: {len(y)}", "",
              "| Model | Window MAE (cycles) | Window RMSE (cycles) | Endpoint MAE (cycles) | Endpoint RMSE (cycles) |",
              "|---|---:|---:|---:|---:|"]
    for name, metrics in (("Training mean constant", result["constant_baseline"]),
                           ("Histogram gradient boosting", result["hist_gradient_boosting"])):
        report.append(f"| {name} | {metrics['mae_cycles']:.3f} | {metrics['rmse_cycles']:.3f} | — | — |")
    report.extend([f"| Endpoint only ({len(engines)} engines) | — | — | {result['endpoint']['mae_cycles']:.3f} | {result['endpoint']['rmse_cycles']:.3f} |",
                   "", "The endpoint score is the final observed cycle of each complete held-out validation trajectory. Window metrics weight each recorded cycle equally. The official test partition was not used.",
                   "", "## Endpoint errors", "", "| Engine | Actual RUL | Prediction | Absolute error |", "|---|---:|---:|---:|"])
    report.extend(f"| {d['engine_id']} | {d['endpoint_true_rul']} | {d['endpoint_prediction']:.2f} | {d['endpoint_absolute_error']:.2f} |" for d in details)
    out.with_suffix(".md").write_text("\n".join(report) + "\n")
    return result
