from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from machine_health.data import attach_rul, load_fd001, sha256
from machine_health.features import FEATURES, causal_features
from machine_health.model import git_revision, load_artifact
from machine_health.run_store import record_run
from machine_health.temporal import (
    SEED,
    add_intervals,
    calibrate_by_predicted_band,
    device_metadata,
    interval_metrics,
    predict_frame,
    train_tcn,
)

ROOT = Path(__file__).resolve().parents[2]


def partition_validation_engines(validation_engines: list[str], seed: int = SEED,
                                calibration_fraction: float = 0.4) -> tuple[list[str], list[str]]:
    if len(validation_engines) < 2 or not 0 < calibration_fraction < 1:
        raise ValueError("Need at least two engines and a calibration fraction between zero and one")
    model_ids, calibration_ids = train_test_split(sorted(validation_engines),
                                                  test_size=calibration_fraction,
                                                  random_state=seed)
    model_ids, calibration_ids = sorted(model_ids), sorted(calibration_ids)
    if set(model_ids) & set(calibration_ids) or (set(model_ids) | set(calibration_ids)) != set(validation_engines):
        raise ValueError("Model-selection and calibration engines must form disjoint partitions")
    return model_ids, calibration_ids


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {"mae_cycles": float(mean_absolute_error(actual, predicted)),
            "rmse_cycles": float(np.sqrt(mean_squared_error(actual, predicted)))}


def _code_state() -> tuple[bool | None, list[str]]:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                         text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None, []
    paths = sorted({line[3:].split(" -> ")[-1] for line in status.splitlines() if len(line) > 3})
    return bool(paths), paths


def _endpoint_metrics(predictions: pd.DataFrame) -> dict:
    endpoints = predictions.sort_values("cycle").groupby("engine_id", sort=False).tail(1)
    return {**_regression_metrics(endpoints.actual_rul.to_numpy(), endpoints.prediction.to_numpy()),
            "engines": len(endpoints),
            "per_engine": [{"engine_id": row.engine_id, "actual_rul": int(row.actual_rul),
                            "prediction": float(row.prediction),
                            "absolute_error": float(abs(row.actual_rul - row.prediction))}
                           for row in endpoints.itertuples()]}


def _by_actual_rul(predictions: pd.DataFrame) -> dict:
    result = {}
    for low, high, name in [(0, 20, "0-19"), (20, 50, "20-49"), (50, 100, "50-99"),
                            (100, float("inf"), "100+")]:
        group = predictions[(predictions.actual_rul >= low) & (predictions.actual_rul < high)]
        if len(group):
            result[name] = {**_regression_metrics(group.actual_rul.to_numpy(),
                                                  group.prediction.to_numpy()),
                            "cycles": len(group), "engines": group.engine_id.nunique()}
    return result


def run_temporal_experiment(data_path: Path = ROOT / "data/raw/train_FD001.txt",
                            classic_artifact: Path = ROOT / "artifacts/fd001.joblib",
                            split_path: Path = ROOT / "reports/fd001_split.json",
                            checkpoint_path: Path = ROOT / "artifacts/fd001_tcn.pt",
                            report_path: Path = ROOT / "reports/temporal_model_v1.json",
                            replay_path: Path = ROOT / "data/processed/temporal_validation_replay.csv") -> dict:
    """Train/early-stop on train-group engines, compare and calibrate on disjoint validation groups."""
    frame = attach_rul(load_fd001(data_path))
    split = json.loads(split_path.read_text())
    model_ids, calibration_ids = partition_validation_engines(split["validation_engines"])
    subtrain_ids, early_ids = train_test_split(sorted(split["train_engines"]), test_size=0.1,
                                               random_state=SEED)
    manifest = {"seed": SEED, "parent_split": str(split_path),
                "parent_split_sha256": sha256(split_path),
                "fit_engines": sorted(subtrain_ids), "early_stopping_engines": sorted(early_ids),
                "model_selection_engines": model_ids, "calibration_engines": calibration_ids,
                "calibration_coverage": 0.8,
                "notes": "All partitions are engine-disjoint. Official test data is excluded."}
    manifest_path = report_path.parent / "temporal_split.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    code_dirty, dirty_paths = _code_state()
    source_files = ["src/machine_health/temporal.py", "src/machine_health/temporal_pipeline.py",
                    "src/machine_health/features.py", "src/machine_health/model.py",
                    "src/machine_health/data.py", "src/machine_health/pipeline.py",
                    "src/machine_health/cli.py", "app/streamlit_app.py"]
    source_hashes = {name: sha256(ROOT / name) for name in source_files}
    model, bundle, stopping = train_tcn(frame, split["train_engines"], checkpoint_path)
    model_predictions, _ = predict_frame(model, bundle, frame, model_ids)
    calibration_predictions, _ = predict_frame(model, bundle, frame, calibration_ids)
    calibration = calibrate_by_predicted_band(calibration_predictions, coverage=0.8)
    model_predictions = add_intervals(model_predictions, calibration)

    features = causal_features(frame)
    classic = load_artifact(classic_artifact)
    model_mask = features.engine_id.isin(model_ids)
    tree_prediction = np.maximum(0.0, classic["model"].predict(features.loc[model_mask, FEATURES]))
    classic_rows = frame.loc[model_mask, ["engine_id", "cycle", "rul"]].copy()
    classic_rows = classic_rows.rename(columns={"rul": "actual_rul"})
    classic_rows["prediction"] = tree_prediction
    classic_score = {"windows": _regression_metrics(classic_rows.actual_rul.to_numpy(), tree_prediction),
                     "endpoint": _endpoint_metrics(classic_rows),
                     "by_actual_rul_band": _by_actual_rul(classic_rows)}
    tcn_score = {"windows": _regression_metrics(model_predictions.actual_rul.to_numpy(),
                                                  model_predictions.prediction.to_numpy()),
                 "endpoint": _endpoint_metrics(model_predictions),
                 "by_actual_rul_band": _by_actual_rul(model_predictions),
                 "prediction_intervals": interval_metrics(model_predictions)}

    report = {
        "dataset": "NASA C-MAPSS FD001 training trajectories; validation only",
        "target": "raw uncapped RUL cycles", "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256(manifest_path), "data_sha256": sha256(data_path),
        "classic_artifact_sha256": sha256(classic_artifact),
        "tcn_artifact": str(checkpoint_path), "tcn_artifact_sha256": sha256(checkpoint_path),
        "code_revision": git_revision(), "code_dirty": code_dirty, "dirty_paths": dirty_paths,
        "source_sha256": source_hashes, "python": platform.python_version(),
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        **device_metadata(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "torch": str(torch.__version__), "seed": SEED,
        "tcn": {"architecture": bundle["metadata"]["architecture"],
                "history_cycles": bundle["history_cycles"],
                "input_features": bundle["raw_features"],
                "history_note": "TCN sees 30 raw cycles; tree baseline uses causal trailing summaries through 20 cycles. Both include current cycle, operating settings, and sensors.",
                "early_stopping": stopping, "score": tcn_score},
        "hist_gradient_boosting": {"artifact": str(classic_artifact), "score": classic_score},
        "calibration": calibration,
        "calibration_limitations": (
            "Only eight calibration engines are available. Engine-block conformal radii use each engine's maximum absolute residual within a predicted-RUL band; finite 80% quantiles are unavailable when a band has too few distinct engines. Coverage assumes exchangeable engines and does not address simulator-to-real shift."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    model_predictions.to_csv(replay_path, index=False)
    _write_report(report, report_path.with_suffix(".md"))
    record_run(ROOT / "data/experiments.sqlite", kind="temporal_validation_v1", artifact=checkpoint_path,
               data_path=data_path, report_path=report_path,
               configuration={"split_manifest": str(manifest_path), "seed": SEED,
                              "coverage": 0.8, "history_cycles": bundle["history_cycles"]},
               metrics={"tcn_window_mae_cycles": tcn_score["windows"]["mae_cycles"],
                        "tcn_endpoint_mae_cycles": tcn_score["endpoint"]["mae_cycles"],
                        "interval_endpoint_coverage": tcn_score["prediction_intervals"]["endpoint"].get("coverage")})
    return report


def _write_report(result: dict, path: Path) -> None:
    tcn, tree = result["tcn"]["score"], result["hist_gradient_boosting"]["score"]
    split = json.loads(Path(result["split_manifest"]).read_text())
    partition_summary = (f"Model-selection engines: {len(split['model_selection_engines'])}; "
                         f"calibration engines: {len(split['calibration_engines'])}.")
    lines = ["# FD001 temporal model and interval evaluation", "",
             "Validation-only report. Official test data and labels were not loaded or used.", "",
             partition_summary,
             "", "## Point prediction comparison", "",
             "| Model | Window MAE | Window RMSE | Endpoint MAE | Endpoint RMSE |",
             "|---|---:|---:|---:|---:|",
             f"| Histogram gradient boosting | {tree['windows']['mae_cycles']:.3f} | {tree['windows']['rmse_cycles']:.3f} | {tree['endpoint']['mae_cycles']:.3f} | {tree['endpoint']['rmse_cycles']:.3f} |",
             f"| Causal TCN | {tcn['windows']['mae_cycles']:.3f} | {tcn['windows']['rmse_cycles']:.3f} | {tcn['endpoint']['mae_cycles']:.3f} | {tcn['endpoint']['rmse_cycles']:.3f} |",
             "", f"TCN early stopping selected epoch {result['tcn']['early_stopping']['best_epoch']} of at most 60, using engine-held-out early-stopping data from the original training group.",
             "", "## Provenance", "",
             f"Code revision: `{result['code_revision']}`; working tree dirty at run time: `{result['code_dirty']}`. Component source SHA-256 hashes and dirty paths are recorded in the JSON report. TCN artifact SHA-256: `{result['tcn_artifact_sha256']}`.",
             f"Runtime: Python {result['python']}, PyTorch {result['torch']}, device `{result['device']}` ({result['gpu_name'] or 'CPU'}), platform `{result['platform']}`.",
             "", "## Prediction intervals", "",
             "Nominal 80% split-conformal intervals use calibration engines disjoint from model-selection engines. Residuals are grouped by predicted RUL band; each calibration engine contributes its maximum absolute residual in a band, preserving engine-level dependence.", "",
             "| Actual RUL band | Engines | Engine-band simultaneous coverage | Mean cycle coverage | Mean width (cycles) |",
             "|---|---:|---:|---:|---:|"]
    for band, metric in tcn["prediction_intervals"]["by_actual_rul_band"].items():
        lines.append(f"| {band} | {metric['engines']} | {metric['engine_band_simultaneous_coverage']:.3f} | {metric['mean_engine_cycle_coverage']:.3f} | {metric['mean_width_cycles']:.2f} |")
    end = tcn["prediction_intervals"]["endpoint"]
    lines.extend(["", f"Endpoint coverage: {end.get('coverage', float('nan')):.3f} over {end.get('engines', 0)} engines; mean endpoint interval width {end.get('mean_width_cycles', float('nan')):.2f} cycles.",
                  "", "## Limitations", "", result["calibration_limitations"],
                  "", "TCN history is 30 raw cycles; the tree baseline uses causal summaries through 20 cycles. The TCN trains on windows from 72 fit engines and early-stops on 8 other training engines. All reports are validation results only."])
    path.write_text("\n".join(lines) + "\n")
