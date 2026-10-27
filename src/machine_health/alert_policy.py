from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from machine_health.features import FEATURES

ACTIONABLE_HORIZON = 30
PERSISTENCE_CYCLES = 3
MINIMUM_HISTORY = 20
THRESHOLD_CANDIDATES = tuple(range(5, 101, 5))
MAX_EARLY_FALSE_ALARM_RATE = 0.10


def _first_persistent_alert(
    group: pd.DataFrame,
    threshold: float,
    *,
    persistence_cycles: int = PERSISTENCE_CYCLES,
    minimum_history_cycles: int = MINIMUM_HISTORY,
) -> int | None:
    if persistence_cycles < 1 or minimum_history_cycles < 1:
        raise ValueError("Persistence and minimum history must be positive cycle counts")
    below = group["prediction"].to_numpy() <= threshold
    cycles = group["cycle"].to_numpy()
    for index in range(minimum_history_cycles - 1, len(group) - persistence_cycles + 1):
        end = index + persistence_cycles
        if below[index:end].all() and all(later - earlier == 1 for earlier, later in pairwise(cycles[index:end])):
            return end - 1
    return None


def score_validation_policy_scenario(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    validation_engine_ids: list[str],
    *,
    threshold_cycles: float,
    persistence_cycles: int,
    minimum_history_cycles: int,
    actionable_horizon_cycles: float,
) -> dict:
    """Retrospectively score one user-selected policy scenario on validation only.

    Keep the prediction history and actual outcomes as separate inputs. This scorer
    must not be used by queue ranking or applied to the frozen official test set.
    """
    if set(predictions.columns) != {"engine_id", "cycle", "prediction"}:
        raise ValueError("Policy predictions must contain only engine_id, cycle, prediction")
    if set(outcomes.columns) != {"engine_id", "cycle", "actual_rul"}:
        raise ValueError("Policy outcomes must contain only engine_id, cycle, actual_rul")
    if threshold_cycles < 0 or actionable_horizon_cycles < 0:
        raise ValueError("Threshold and actionable horizon must be non-negative")
    if persistence_cycles < 1 or minimum_history_cycles < 1:
        raise ValueError("Persistence and minimum history must be positive cycle counts")
    engine_ids = list(validation_engine_ids)
    if not engine_ids or len(engine_ids) != len(set(engine_ids)) or any(
        not engine_id.startswith("train:FD001:") for engine_id in engine_ids
    ):
        raise ValueError("Policy scoring requires a unique held-out FD001 validation cohort")
    if set(predictions.engine_id) != set(engine_ids) or set(outcomes.engine_id) != set(engine_ids):
        raise ValueError("Predictions and outcomes must exactly match the validation cohort")
    for frame, value_column in ((predictions, "prediction"), (outcomes, "actual_rul")):
        if frame[["cycle", value_column]].isna().any().any() or not np.isfinite(
            frame[["cycle", value_column]].to_numpy(dtype=float)
        ).all():
            raise ValueError("Policy inputs must be finite and complete")
        if (frame.cycle < 1).any() or (frame.cycle % 1 != 0).any():
            raise ValueError("Policy cycles must be positive integers")
        if frame.duplicated(["engine_id", "cycle"]).any():
            raise ValueError("Policy inputs cannot contain duplicate engine-cycle rows")
    if (predictions.prediction < 0).any() or (outcomes.actual_rul < 0).any():
        raise ValueError("Policy predictions and actual RUL labels must be non-negative")

    outcome_index = outcomes.set_index(["engine_id", "cycle"])["actual_rul"]
    events = []
    for engine_id in engine_ids:
        group = predictions.loc[predictions.engine_id == engine_id].sort_values("cycle").reset_index(drop=True)
        cycles = group.cycle.to_numpy(dtype=int)
        if not np.array_equal(cycles, np.arange(1, len(group) + 1)):
            raise ValueError(f"Validation engine {engine_id} must have a complete run-to-failure path")
        outcome_cycles = np.sort(
            outcomes.loc[outcomes.engine_id == engine_id, "cycle"].to_numpy(dtype=int)
        )
        if not np.array_equal(cycles, outcome_cycles):
            raise ValueError(f"Validation engine {engine_id} prediction and outcome cycles must match")
        index = _first_persistent_alert(
            group, threshold_cycles, persistence_cycles=persistence_cycles,
            minimum_history_cycles=minimum_history_cycles,
        )
        if index is None:
            category, lead, alert_cycle = "missed", None, None
        else:
            alert_cycle = int(group.iloc[index].cycle)
            lead = int(outcome_index.loc[(engine_id, alert_cycle)])
            if 0 < lead <= actionable_horizon_cycles:
                category = "timely"
            elif lead > actionable_horizon_cycles:
                category = "early_false_alarm"
            else:
                category, lead = "missed", None
        events.append({"engine_id": engine_id, "category": category,
                       "lead_time_cycles": lead, "alert_cycle": alert_cycle})

    engine_count = len(events)
    alerts = sum(event["alert_cycle"] is not None for event in events)
    early_false = sum(event["category"] == "early_false_alarm" for event in events)
    timely = sum(event["category"] == "timely" for event in events)
    missed = engine_count - early_false - timely
    leads = [event["lead_time_cycles"] for event in events if event["category"] == "timely"]
    return {
        "validation_engines": engine_count,
        "alerted_engines": alerts,
        "alert_rate": alerts / engine_count,
        "early_false_alarm_engines": early_false,
        "early_false_alarm_rate": early_false / engine_count,
        "timely_detection_engines": timely,
        "timely_detection_rate": timely / engine_count,
        "missed_failures": missed,
        "miss_rate": missed / engine_count,
        "mean_timely_lead_time_cycles": None if not leads else float(np.mean(leads)),
        "events": events,
    }


def evaluate_alert_policy(frame: pd.DataFrame, features: pd.DataFrame, artifact: dict, output: Path) -> dict:
    """Select an alert threshold on complete validation engines only."""
    if not frame["engine_id"].str.startswith("train:FD001:").all():
        raise ValueError("Alert threshold selection requires held-out training-partition validation engines")
    validation_ids = sorted(frame.engine_id.unique())
    predictions = np.maximum(0, artifact["model"].predict(features[FEATURES]))
    data = frame[["engine_id", "cycle", "rul"]].copy()
    data["prediction"] = predictions
    by_engine = {key: value.sort_values("cycle").reset_index(drop=True)
                 for key, value in data.groupby("engine_id", sort=True)}
    scenarios = []
    for threshold in THRESHOLD_CANDIDATES:
        events = []
        for engine_id in validation_ids:
            group = by_engine[engine_id]
            index = _first_persistent_alert(group, threshold)
            if index is None:
                category, lead = "missed", None
            else:
                lead = int(group.iloc[index].rul)
                if 0 < lead <= ACTIONABLE_HORIZON:
                    category = "timely"
                elif lead > ACTIONABLE_HORIZON:
                    category = "early_false_alarm"
                else:
                    category = "missed"
                    lead = None
            events.append({"engine_id": engine_id, "category": category, "lead_time_cycles": lead,
                           "alert_cycle": None if index is None else int(group.iloc[index].cycle)})
        count = len(events)
        false = sum(event["category"] == "early_false_alarm" for event in events)
        timely = sum(event["category"] == "timely" for event in events)
        missed = count - false - timely
        leads = [event["lead_time_cycles"] for event in events if event["category"] == "timely"]
        scenarios.append({"threshold_cycles": threshold, "engines": count,
                          "early_false_alarm_engines": false, "early_false_alarm_rate": false / count,
                          "timely_detection_engines": timely, "timely_detection_rate": timely / count,
                          "missed_failures": missed, "miss_rate": missed / count,
                          "mean_timely_lead_time_cycles": None if not leads else float(np.mean(leads)),
                          "events": events})
    eligible = [row for row in scenarios if row["early_false_alarm_rate"] <= MAX_EARLY_FALSE_ALARM_RATE]
    if eligible:
        selected = max(eligible, key=lambda row: (row["timely_detection_rate"], -row["threshold_cycles"]))
        constraint_met = True
    else:
        selected = min(scenarios, key=lambda row: (row["early_false_alarm_rate"], -row["timely_detection_rate"]))
        constraint_met = False
    result = {
        "dataset": "NASA C-MAPSS FD001 held-out validation engines only; no official test labels used",
        "policy": {"actionable_horizon_cycles": ACTIONABLE_HORIZON,
                   "persistence_cycles": PERSISTENCE_CYCLES,
                   "minimum_history_cycles": MINIMUM_HISTORY,
                   "early_false_alarm_definition": "first persistent alert when actual RUL exceeds the 30-cycle actionable horizon",
                   "timely_detection_definition": "first persistent alert at actual RUL 1 through 30 inclusive",
                   "miss_definition": "no alert in the actionable horizon, or alert only at failure (RUL 0)",
                   "threshold_selection": "maximize timely detection rate subject to at most 10% early false-alarm engines; ties choose the lower threshold",
                   "false_alarm_constraint_met": constraint_met},
        "selected_threshold_cycles": selected["threshold_cycles"],
        "selected_validation_result": selected,
        "candidate_results": scenarios,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Validation alert-policy selection", "",
             "Threshold chosen only on the 20 held-out full run-to-failure validation engines. This is not test-set lifecycle alert evidence.", "",
             f"Selected threshold: **{selected['threshold_cycles']} cycles**; actionable horizon {ACTIONABLE_HORIZON}; persistence {PERSISTENCE_CYCLES} consecutive cycles; minimum history {MINIMUM_HISTORY} cycles.", "",
             f"Early false alarms: {selected['early_false_alarm_engines']}/{selected['engines']} engines ({selected['early_false_alarm_rate']:.1%}). Timely detections: {selected['timely_detection_engines']}/{selected['engines']} ({selected['timely_detection_rate']:.1%}). Missed failures: {selected['missed_failures']}/{selected['engines']}. Mean timely lead: {selected['mean_timely_lead_time_cycles']} cycles.", "",
             "An early false alarm is an engine's first persistent alert when its reconstructed RUL is over 30 cycles. A timely detection is first alert at RUL 1–30. Failure-cycle-only alerts count as misses. Denominator for all event rates is the number of full validation engines.", "",
             "| Threshold | Early false alarms | Timely detections | Missed | Mean timely lead (cycles) |", "|---:|---:|---:|---:|---:|"]
    for item in scenarios:
        lead = item["mean_timely_lead_time_cycles"]
        lead_text = "—" if lead is None else f"{lead:.2f}"
        lines.append(f"| {item['threshold_cycles']} | {item['early_false_alarm_engines']}/{item['engines']} | {item['timely_detection_engines']}/{item['engines']} | {item['missed_failures']} | "
                     f"{lead_text} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return result
