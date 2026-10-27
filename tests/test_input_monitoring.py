import json

import pandas as pd
import pytest

from machine_health.data import COLUMNS
from machine_health.features import FEATURES
from machine_health.input_monitoring import (
    audit_validation_input_support,
    score_input_support,
    training_support_reference,
)


def _feature_table(engine_ids: list[str]) -> pd.DataFrame:
    rows = []
    for engine_id in engine_ids:
        rows.append({"engine_id": engine_id, "cycle": 120, **dict.fromkeys(FEATURES, 0.0)})
    frame = pd.DataFrame(rows)
    frame["cycle"] = 120
    frame["engine_id"] = engine_ids
    return frame


def test_support_reference_uses_training_only_and_counts_feature_cells():
    training = _feature_table(["train:FD001:1", "train:FD001:2"])
    reference = training_support_reference(training)
    validation = _feature_table(["train:FD001:3", "train:FD001:4"])
    validation.loc[0, "sensor_1_mean_5"] = 4.0

    result = score_input_support(reference, validation, snapshot_cycle=120)

    assert result["validation_engines"] == 2
    assert result["out_of_support_cells"] == 1
    assert result["feature_cells"] == 2 * len(FEATURES)
    assert result["engines_with_any_out_of_support_feature"] == 1
    assert result["per_engine"][0]["feature_names"] == ["sensor_1_mean_5"]


def test_support_audit_rejects_labels_and_wrong_engine_partition():
    reference = training_support_reference(_feature_table(["train:FD001:1"]))
    validation = _feature_table(["test:FD001:2"])
    with pytest.raises(ValueError, match="validation"):
        score_input_support(reference, validation, snapshot_cycle=120)
    validation = _feature_table(["train:FD001:2"])
    validation["rul"] = 0
    with pytest.raises(ValueError, match="labels"):
        score_input_support(reference, validation, snapshot_cycle=120)


def test_support_audit_rejects_training_engine_reuse():
    reference = training_support_reference(_feature_table(["train:FD001:1"]))
    reused = _feature_table(["train:FD001:1"])
    with pytest.raises(ValueError, match="overlap"):
        score_input_support(reference, reused, snapshot_cycle=120)


def test_real_input_support_audit_uses_split_and_writes_provenance(tmp_path):
    records = []
    for unit in range(1, 6):
        for cycle in range(1, 9):
            records.append([unit, cycle, 0.1, 0.2, 0.3,
                            *[unit * 10 + cycle + sensor for sensor in range(21)]])
    data_path = tmp_path / "train_FD001.txt"
    pd.DataFrame(records, columns=COLUMNS).to_csv(data_path, sep=" ", header=False, index=False)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({
        "train_engines": ["train:FD001:1", "train:FD001:2", "train:FD001:3", "train:FD001:4"],
        "validation_engines": ["train:FD001:5"],
    }))
    output = tmp_path / "reports" / "support.json"

    result = audit_validation_input_support(data_path, split_path, output,
                                            snapshot_cycles=(3, 5))

    assert output.exists()
    assert output.with_suffix(".md").exists()
    assert result["quantiles"]["reference_engines"] == 4
    assert [item["validation_engines"] for item in result["snapshots"]] == [1, 1]
    assert "rul" not in output.read_text().lower()
