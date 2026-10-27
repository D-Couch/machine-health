import sqlite3

import pytest

from machine_health.review_log import list_dispositions, record_disposition


def _record(database, *, engine_id="train:FD001:3", disposition="inspect", rationale="Check trend"):
    return record_disposition(
        database, engine_id=engine_id, snapshot_cycle=120,
        disposition=disposition, rationale=rationale, model_name="hist-gradient-boosting",
        artifact_sha256="a" * 64, predicted_rul_cycles=18.5,
        interval_lower_cycles=11.0, interval_upper_cycles=25.0,
        threshold_cycles=20, persistence_cycles=3, review_horizon_cycles=30,
    )


def test_review_log_appends_context_and_filters_by_validation_engine(tmp_path):
    database = tmp_path / "nested" / "review.sqlite"
    first = _record(database)
    second = _record(database, disposition="monitor", rationale="  Wait for next review  ")
    _record(database, engine_id="train:FD001:4", disposition="defer")

    history = list_dispositions(database, engine_id="train:FD001:3")
    assert [entry["id"] for entry in history] == [second, first]
    assert history[0]["rationale"] == "Wait for next review"
    assert history[0]["predicted_rul_cycles"] == 18.5
    assert history[0]["artifact_sha256"] == "a" * 64
    assert history[0]["persistence_cycles"] == 3
    assert list_dispositions(database, engine_id="train:FD001:4")[0]["disposition"] == "defer"
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(review_dispositions)")}
    assert "actual_rul" not in columns
    assert "rul" not in columns


@pytest.mark.parametrize("kwargs", [
    {"engine_id": "test:FD001:3"},
    {"disposition": "retrain"},
    {"artifact_sha256": "not-a-hash"},
    {"predicted_rul_cycles": float("nan")},
    {"rationale": "x" * 1001},
    {"interval_lower_cycles": 5.0, "interval_upper_cycles": None},
])
def test_review_log_rejects_invalid_context(tmp_path, kwargs):
    arguments = {
        "engine_id": "train:FD001:3", "snapshot_cycle": 120,
        "disposition": "inspect", "rationale": "", "model_name": "model",
        "artifact_sha256": "a" * 64, "predicted_rul_cycles": 18.5,
        "threshold_cycles": 20, "persistence_cycles": 3,
        "review_horizon_cycles": 30,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        record_disposition(tmp_path / "review.sqlite", **arguments)
