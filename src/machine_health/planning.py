from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from machine_health.alert_policy import _first_persistent_alert

BASE_PREDICTION_COLUMNS = {"engine_id", "cycle", "prediction"}
INTERVAL_COLUMNS = {"lower", "upper"}


def build_validation_review_queue(
    predictions: pd.DataFrame,
    validation_engine_ids: Sequence[str],
    *,
    snapshot_cycle: int = 120,
    threshold_cycles: float = 20.0,
    persistence_cycles: int = 3,
    minimum_history_cycles: int = 20,
    review_horizon_cycles: float = 30.0,
) -> pd.DataFrame:
    """Build an as-of validation queue using predictions only, never outcome labels.

    `predictions` must contain a complete, contiguous prediction history from cycle 1
    through the common snapshot for every named held-out FD001 validation engine.
    Optional interval bounds may be supplied. Other columns are rejected so labels or
    engine metadata cannot silently become queue inputs.
    """
    expected_columns = BASE_PREDICTION_COLUMNS | (set(predictions.columns) & INTERVAL_COLUMNS)
    if set(predictions.columns) != expected_columns:
        raise ValueError(
            "Queue input must contain only engine_id, cycle, prediction, and optional lower/upper bounds"
        )
    if bool({"lower", "upper"} & set(predictions.columns)) and not INTERVAL_COLUMNS.issubset(
        predictions.columns
    ):
        raise ValueError("Prediction intervals require both lower and upper bounds")
    if predictions.empty:
        raise ValueError("Queue requires at least one prediction")
    if snapshot_cycle < 1 or threshold_cycles < 0 or review_horizon_cycles < 0:
        raise ValueError("Snapshot, threshold, and review horizon must be non-negative cycles")
    if persistence_cycles < 1 or minimum_history_cycles < 1:
        raise ValueError("Persistence and minimum history must be positive cycle counts")

    engine_ids = list(validation_engine_ids)
    if not engine_ids or len(engine_ids) != len(set(engine_ids)):
        raise ValueError("Validation engine cohort must be non-empty and unique")
    if any(not engine_id.startswith("train:FD001:") for engine_id in engine_ids):
        raise ValueError("Review queue accepts held-out FD001 training-partition validation engines only")
    if predictions["engine_id"].isna().any() or not set(predictions.engine_id).issubset(engine_ids):
        raise ValueError("Predictions contain missing or out-of-cohort engine IDs")

    numeric_columns = ["cycle", "prediction", *sorted(set(predictions.columns) & INTERVAL_COLUMNS)]
    numeric = predictions[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Predictions must contain finite numeric values")
    if (predictions["cycle"] < 1).any() or (predictions["cycle"] % 1 != 0).any():
        raise ValueError("Cycle values must be positive integers")
    if (predictions["prediction"] < 0).any():
        raise ValueError("Predicted RUL must be non-negative")
    if {"lower", "upper"}.issubset(predictions.columns) and (
        predictions["lower"] > predictions["upper"]
    ).any():
        raise ValueError("Prediction interval lower bounds must not exceed upper bounds")

    records: list[dict] = []
    for engine_id in engine_ids:
        group = predictions.loc[predictions.engine_id == engine_id].sort_values("cycle").copy()
        history = group.loc[group.cycle <= snapshot_cycle].reset_index(drop=True)
        cycles = history.cycle.to_numpy(dtype=int)
        expected_cycles = np.arange(1, snapshot_cycle + 1)
        if not np.array_equal(cycles, expected_cycles):
            raise ValueError(
                f"Engine {engine_id} must have one prediction at each cycle 1–{snapshot_cycle}"
            )

        last = history.iloc[-1]
        alert_index = _first_persistent_alert(
            history,
            threshold_cycles,
            persistence_cycles=persistence_cycles,
            minimum_history_cycles=minimum_history_cycles,
        )
        predicted_rul = float(last.prediction)
        if alert_index is not None:
            status = "Review"
        elif predicted_rul <= review_horizon_cycles:
            status = "Watch"
        else:
            status = "Routine"

        record = {
            "engine_id": engine_id,
            "snapshot_cycle": snapshot_cycle,
            "predicted_rul_cycles": predicted_rul,
            "status": status,
            "alert_recorded": alert_index is not None,
            "first_alert_cycle": (
                None if alert_index is None else int(history.iloc[alert_index].cycle)
            ),
        }
        if "lower" in history and "upper" in history:
            record["lower_cycles"] = float(last.lower)
            record["upper_cycles"] = float(last.upper)
        records.append(record)

    queue = pd.DataFrame(records)
    priority_order = {"Review": 0, "Watch": 1, "Routine": 2}
    queue["_priority"] = queue.status.map(priority_order)
    queue = queue.sort_values(
        ["_priority", "predicted_rul_cycles", "engine_id"], kind="stable"
    ).drop(columns="_priority").reset_index(drop=True)
    queue.insert(0, "rank", np.arange(1, len(queue) + 1))
    return queue
