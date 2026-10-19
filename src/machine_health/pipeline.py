from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from machine_health.data import acquire, attach_rul, load_fd001, write_provenance
from machine_health.evaluation import evaluate
from machine_health.features import causal_features
from machine_health.model import load_artifact, train_model

ROOT = Path(__file__).resolve().parents[2]


def prepare(data_dir: Path = ROOT / "data" / "raw") -> Path:
    archive = data_dir / "cmapss.zip"
    extracted = data_dir / "train_FD001.txt"
    if not extracted.exists():
        extracted = acquire(data_dir)
    if archive.exists():
        write_provenance(archive, extracted, data_dir / "provenance.json")
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
    return {"artifact": artifact, "features": features, "labeled": frame, "train_engines": train_ids, "validation_engines": validation_ids}


def evaluate_saved(data_path: Path, artifact_path: Path = ROOT / "artifacts" / "fd001.joblib",
                   split_path: Path = ROOT / "reports" / "fd001_split.json",
                   output: Path = ROOT / "reports" / "validation_metrics.json") -> dict:
    frame = attach_rul(load_fd001(data_path))
    features = causal_features(frame)
    artifact = load_artifact(artifact_path)
    split = json.loads(split_path.read_text())
    return evaluate(artifact, features, frame, split["validation_engines"], output)
