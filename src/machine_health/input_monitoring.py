from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from machine_health.data import load_fd001, sha256
from machine_health.features import FEATURES, causal_features
from machine_health.model import git_revision

ROOT = Path(__file__).resolve().parents[2]


def _validate_feature_table(frame: pd.DataFrame, *, partition: str) -> None:
    forbidden = {"rul", "actual_rul", "unit"} & set(frame.columns)
    if forbidden:
        raise ValueError(f"Input support audit does not accept labels or unit IDs: {sorted(forbidden)}")
    missing = {"engine_id", "cycle", *FEATURES} - set(frame.columns)
    if missing:
        raise ValueError(f"Input feature table is missing columns: {sorted(missing)}")
    prefix = f"{partition}:FD001:"
    if frame.empty or not frame.engine_id.astype(str).str.startswith(prefix).all():
        raise ValueError(f"Expected non-empty {partition} FD001 engine feature rows")
    values = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input features must be finite")
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Input contains duplicate engine/cycle rows")


def training_support_reference(
    train_features: pd.DataFrame,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict:
    """Fit descriptive univariate feature bounds using training engines only."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")
    _validate_feature_table(train_features, partition="train")
    values = train_features[FEATURES]
    lower = values.quantile(lower_quantile)
    upper = values.quantile(upper_quantile)
    return {
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "training_engines": sorted(train_features.engine_id.unique().tolist()),
        "training_rows": len(train_features),
        "features": FEATURES,
        "bounds": {
            name: {"lower": float(lower[name]), "upper": float(upper[name])}
            for name in FEATURES
        },
    }


def score_input_support(
    reference: dict,
    validation_features: pd.DataFrame,
    *,
    snapshot_cycle: int,
) -> dict:
    """Describe validation feature values outside training quantile bounds.

    This is input support coverage only. It does not estimate prediction accuracy,
    label drift, or a production alarm rate.
    """
    forbidden = {"rul", "actual_rul", "unit"} & set(validation_features.columns)
    if forbidden:
        raise ValueError(f"Input support audit does not accept labels or unit IDs: {sorted(forbidden)}")
    missing = {"engine_id", "cycle", *FEATURES} - set(validation_features.columns)
    if missing:
        raise ValueError(f"Input feature table is missing columns: {sorted(missing)}")
    if validation_features.empty or not validation_features.engine_id.astype(str).str.startswith(
        "train:FD001:"
    ).all():
        raise ValueError("Expected non-empty held-out train:FD001 validation feature rows")
    overlap = set(reference["training_engines"]) & set(validation_features.engine_id.unique())
    if overlap:
        raise ValueError(f"Training and validation engines overlap: {sorted(overlap)}")
    if not np.isfinite(validation_features[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Input features must be finite")
    if validation_features.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Input contains duplicate engine/cycle rows")
    if snapshot_cycle < 1:
        raise ValueError("Snapshot cycle must be positive")

    rows = validation_features.loc[validation_features.cycle == snapshot_cycle]
    if rows.empty:
        return {
            "snapshot_cycle": snapshot_cycle,
            "validation_engines": 0,
            "feature_cells": 0,
            "out_of_support_cells": 0,
            "out_of_support_fraction": None,
            "engines_with_any_out_of_support_feature": 0,
            "per_engine": [],
            "feature_counts": [],
        }

    lower = pd.Series({name: reference["bounds"][name]["lower"] for name in FEATURES})
    upper = pd.Series({name: reference["bounds"][name]["upper"] for name in FEATURES})
    values = rows[FEATURES]
    outside = values.lt(lower, axis="columns") | values.gt(upper, axis="columns")
    per_engine = []
    for index, (_, row) in enumerate(rows.iterrows()):
        flags = outside.iloc[index]
        per_engine.append({
            "engine_id": row.engine_id,
            "out_of_support_features": int(flags.sum()),
            "fraction_out_of_support": float(flags.mean()),
            "feature_names": flags.index[flags].tolist(),
        })
    feature_counts = outside.sum(axis=0).sort_values(ascending=False)
    cells = int(outside.size)
    outside_cells = int(outside.to_numpy().sum())
    return {
        "snapshot_cycle": int(snapshot_cycle),
        "validation_engines": int(rows.engine_id.nunique()),
        "feature_cells": cells,
        "out_of_support_cells": outside_cells,
        "out_of_support_fraction": outside_cells / cells,
        "engines_with_any_out_of_support_feature": int(
            outside.any(axis=1).sum()
        ),
        "per_engine": per_engine,
        "feature_counts": [
            {"feature": name, "engine_count": int(count)}
            for name, count in feature_counts.items() if count > 0
        ],
    }


def audit_validation_input_support(
    data_path: Path,
    split_path: Path,
    output_path: Path,
    *,
    snapshot_cycles: tuple[int, ...] = (60, 120, 180),
) -> dict:
    """Build a training-reference support audit on held-out validation features."""
    split = json.loads(split_path.read_text())
    train_engines = split["train_engines"]
    validation_engines = split["validation_engines"]
    if set(train_engines) & set(validation_engines):
        raise ValueError("Training and validation engine groups overlap")
    frame = load_fd001(data_path)
    features = causal_features(frame)
    train_features = features.loc[features.engine_id.isin(train_engines)].copy()
    validation_features = features.loc[features.engine_id.isin(validation_engines)].copy()
    if set(train_features.engine_id.unique()) != set(train_engines):
        raise ValueError("Split manifest training engines do not match the local dataset")
    if set(validation_features.engine_id.unique()) != set(validation_engines):
        raise ValueError("Split manifest validation engines do not match the local dataset")

    reference = training_support_reference(train_features)
    result = {
        "dataset": "NASA C-MAPSS FD001; descriptive input support audit",
        "interpretation": (
            "Feature values outside the training 1st–99th percentile bounds are a descriptive "
            "input-support signal, not evidence of prediction error or a production alert rate."
        ),
        "quantiles": {
            "lower": reference["lower_quantile"],
            "upper": reference["upper_quantile"],
            "reference_rows": reference["training_rows"],
            "reference_engines": len(reference["training_engines"]),
        },
        "features": FEATURES,
        "bounds": reference["bounds"],
        "snapshots": [
            score_input_support(reference, validation_features, snapshot_cycle=cycle)
            for cycle in snapshot_cycles
        ],
        "provenance": {
            "data_sha256": sha256(data_path),
            "split_manifest_sha256": sha256(split_path),
            "code_revision": git_revision(),
            "source_sha256": {
                name: sha256(ROOT / name)
                for name in (
                    "src/machine_health/input_monitoring.py",
                    "src/machine_health/features.py",
                    "src/machine_health/data.py",
                    "src/machine_health/cli.py",
                )
            },
            "validation_engines": sorted(validation_engines),
        },
        "limitations": [
            "The reference is derived from full run-to-failure NASA simulator training trajectories.",
            "Validation contains only 20 complete engines and is not representative of field telemetry.",
            "Univariate quantile bounds ignore feature dependence and do not measure model accuracy.",
            "Snapshot diagnostics are descriptive and must not be interpreted as operational alarm thresholds.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, output_path.with_suffix(".md"))
    return result


def _write_markdown(result: dict, destination: Path) -> None:
    lines = [
        "# FD001 validation input-support audit",
        "",
        result["dataset"],
        "",
        "**Interpretation:** " + result["interpretation"],
        "",
        ("Bounds use training-only per-feature 1st and 99th percentiles across "
         f"{result['quantiles']['reference_rows']:,} rows and "
         f"{result['quantiles']['reference_engines']} engines. Validation features are scored "
         "at fixed observed-cycle snapshots; no RUL labels or official-test files are read."),
        "",
        "| Snapshot cycle | Engines available | Feature values outside bounds | Engines with ≥1 outside feature |",
        "|---:|---:|---:|---:|",
    ]
    for snapshot in result["snapshots"]:
        fraction = snapshot["out_of_support_fraction"]
        outside = "n/a" if fraction is None else f"{fraction:.2%}"
        lines.append(
            f"| {snapshot['snapshot_cycle']} | {snapshot['validation_engines']} | {outside} "
            f"({snapshot['out_of_support_cells']}/{snapshot['feature_cells']}) | "
            f"{snapshot['engines_with_any_out_of_support_feature']} |"
        )
    lines.extend(["", "## Validation engine detail at cycle 120", "",
                  "| Engine | Outside feature bounds | Features |", "|---|---:|---|"])
    cycle_120 = next((item for item in result["snapshots"] if item["snapshot_cycle"] == 120), None)
    if cycle_120:
        for engine in cycle_120["per_engine"]:
            names = ", ".join(engine["feature_names"]) or "—"
            lines.append(
                f"| {engine['engine_id']} | {engine['out_of_support_features']}/"
                f"{len(result['features'])} | {names} |"
            )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(["", "## Provenance", "",
                  f"- Data SHA-256: `{result['provenance']['data_sha256']}`",
                  f"- Split manifest SHA-256: `{result['provenance']['split_manifest_sha256']}`",
                  f"- Code revision: `{result['provenance']['code_revision']}`",
                  "- Validation engines: " + ", ".join(result["provenance"]["validation_engines"]), ""])
    lines.insert(-1, "- Source SHA-256: " + ", ".join(
        f"`{name}` `{digest}`" for name, digest in result["provenance"]["source_sha256"].items()
    ))
    destination.write_text("\n".join(lines))
