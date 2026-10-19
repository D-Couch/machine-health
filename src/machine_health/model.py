from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CalledProcessError

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from machine_health.features import FEATURES

SEED = 41


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (CalledProcessError, FileNotFoundError):
        return "unavailable"


def train_model(train_x, train_y, output: Path) -> dict:
    estimator = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED
    ))
    estimator.fit(train_x[FEATURES], train_y)
    artifact = {
        "model": estimator,
        "features": FEATURES,
        "constant_prediction": float(np.mean(train_y)),
        "metadata": {
            "dataset": "NASA C-MAPSS FD001 training trajectories",
            "target": "raw RUL cycles, no cap",
            "feature_windows": [5, 10, 20],
            "seed": SEED,
            "training_rows": len(train_y),
            "code_revision": git_revision(),
            "python": platform.python_version(),
            "created_utc": datetime.now(UTC).isoformat(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


def load_artifact(path: Path) -> dict:
    artifact = joblib.load(path)
    if artifact.get("features") != FEATURES:
        raise ValueError("Artifact feature schema does not match this application")
    return artifact
