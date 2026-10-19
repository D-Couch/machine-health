from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from machine_health.alert_policy import evaluate_alert_policy
from machine_health.data import (
    acquire,
    acquire_official_test,
    attach_rul,
    attach_test_endpoint_rul,
    load_fd001,
    write_provenance,
)
from machine_health.evaluation import evaluate
from machine_health.features import causal_features
from machine_health.frozen_evaluation import evaluate_official_test
from machine_health.model import load_artifact, train_model
from machine_health.run_store import record_run

ROOT = Path(__file__).resolve().parents[2]


def prepare(data_dir: Path = ROOT / "data" / "raw") -> Path:
    archive = data_dir / "cmapss.zip"
    extracted = data_dir / "train_FD001.txt"
    if not extracted.exists():
        extracted = acquire(data_dir)
    if archive.exists():
        write_provenance(archive, extracted, data_dir / "provenance.json")
        test_file, labels_file = acquire_official_test(archive, data_dir)
        provenance = json.loads((data_dir / "provenance.json").read_text())
        from machine_health.data import sha256
        provenance["official_test_sensors_sha256"] = sha256(test_file)
        provenance["official_test_endpoint_labels_sha256"] = sha256(labels_file)
        provenance["official_test_extracted_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        (data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return extracted


def split_engines(frame: pd.DataFrame, seed: int = 41) -> tuple[list[str], list[str]]:
    engines = sorted(frame.engine_id.unique())
    train, validation = train_test_split(engines, test_size=0.2, random_state=seed)
    return sorted(train), sorted(validation)


def train(data_path: Path, output: Path = ROOT / "artifacts" / "fd001.joblib") -> dict:
    frame = attach_rul(load_fd001(data_path))
    train_ids, validation_ids = split_engines(frame)
    features = causal_features(frame)
    train_mask = features.engine_id.isin(train_ids)
    artifact = train_model(features.loc[train_mask], frame.loc[train_mask, "rul"], output)
    split_path = ROOT / "reports" / "fd001_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"seed": 41, "train_engines": train_ids, "validation_engines": validation_ids}, indent=2) + "\n")
    record_run(ROOT / "data/experiments.sqlite", kind="train", artifact=output, data_path=data_path,
               report_path=None, configuration={"seed": 41, "train_engines": len(train_ids),
                                                "validation_engines": len(validation_ids),
                                                "target": "raw uncapped RUL cycles"}, metrics=None)
    return {"artifact": artifact, "features": features, "labeled": frame, "train_engines": train_ids, "validation_engines": validation_ids}


def evaluate_saved(data_path: Path, artifact_path: Path = ROOT / "artifacts" / "fd001.joblib",
                   split_path: Path = ROOT / "reports" / "fd001_split.json",
                   output: Path = ROOT / "reports" / "validation_metrics.json") -> dict:
    frame = attach_rul(load_fd001(data_path))
    features = causal_features(frame)
    artifact = load_artifact(artifact_path)
    split = json.loads(split_path.read_text())
    result = evaluate(artifact, features, frame, split["validation_engines"], output)
    record_run(ROOT / "data/experiments.sqlite", kind="validation", artifact=artifact_path,
               data_path=data_path, report_path=output,
               configuration={"split_manifest": str(split_path), "aggregation": "all windows and endpoint"},
               metrics={"window_mae_cycles": result["hist_gradient_boosting"]["mae_cycles"],
                        "window_rmse_cycles": result["hist_gradient_boosting"]["rmse_cycles"],
                        "endpoint_mae_cycles": result["endpoint"]["mae_cycles"],
                        "endpoint_rmse_cycles": result["endpoint"]["rmse_cycles"]})
    return result


def select_alert_policy(data_path: Path = ROOT / "data/raw/train_FD001.txt",
                        artifact_path: Path = ROOT / "artifacts/fd001.joblib",
                        split_path: Path = ROOT / "reports/fd001_split.json",
                        output: Path = ROOT / "reports/validation_alert_policy.json") -> dict:
    frame = attach_rul(load_fd001(data_path))
    split = json.loads(split_path.read_text())
    frame = frame[frame.engine_id.isin(split["validation_engines"])].copy()
    features = causal_features(frame)
    result = evaluate_alert_policy(frame, features, load_artifact(artifact_path), output)
    record_run(ROOT / "data/experiments.sqlite", kind="validation_alert_policy", artifact=artifact_path,
               data_path=data_path, report_path=output,
               configuration=result["policy"],
               metrics={"selected_threshold_cycles": result["selected_threshold_cycles"],
                        "timely_detection_rate": result["selected_validation_result"]["timely_detection_rate"],
                        "early_false_alarm_rate": result["selected_validation_result"]["early_false_alarm_rate"],
                        "miss_rate": result["selected_validation_result"]["miss_rate"]})
    return result


def evaluate_test(data_dir: Path = ROOT / "data/raw",
                  artifact_path: Path = ROOT / "artifacts/fd001.joblib",
                  output: Path = ROOT / "reports/official_test_v1.json") -> dict:
    archive = data_dir / "cmapss.zip"
    test_file, labels_file = acquire_official_test(archive, data_dir)
    frame = load_fd001(test_file, partition="test")
    labeled = attach_test_endpoint_rul(frame, labels_file)
    features = causal_features(frame)
    protocol = ROOT / "docs/EVALUATION_PROTOCOL.md"
    artifact = load_artifact(artifact_path)
    result = evaluate_official_test(labeled, features, artifact, test_file,
                                    labels_file, protocol, artifact_path, output)
    record_run(ROOT / "data/experiments.sqlite", kind="official_test_v1", artifact=artifact_path,
               data_path=test_file, report_path=output,
               configuration={"protocol": str(protocol), "target": "raw endpoint RUL cycles"},
               metrics={"endpoint_mae_cycles": result["primary_model"]["mae_cycles"],
                        "endpoint_rmse_cycles": result["primary_model"]["rmse_cycles"]})
    return result
