from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine_health.data import COLUMNS, attach_rul, load_fd001
from machine_health.features import FEATURES, causal_features
from machine_health.model import load_artifact, train_model
from machine_health.pipeline import split_engines


def fixture_frame() -> pd.DataFrame:
    rows = []
    for unit in range(1, 6):
        for cycle in range(1, 9):
            rows.append([unit, cycle, 0.1, 0.2, 0.3, *[unit * 10 + cycle + j for j in range(21)]])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_parser_and_label_semantics(tmp_path):
    path = tmp_path / "train.txt"
    fixture_frame().to_csv(path, sep=" ", header=False, index=False)
    frame = load_fd001(path)
    assert frame.engine_id.nunique() == 5
    labeled = attach_rul(frame)
    first = labeled[labeled.unit == 1]
    assert first.iloc[0].rul == 7
    assert first.iloc[-1].rul == 0
    assert first.iloc[0].cycle + first.iloc[0].rul == 8


def test_parser_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 1 2\n")
    with pytest.raises(ValueError, match="Expected 26 columns"):
        load_fd001(path)


def test_split_is_deterministic_and_engine_disjoint():
    frame = fixture_frame()
    frame["engine_id"] = "train:FD001:" + frame.unit.astype(str)
    frame = attach_rul(frame)
    train, validation = split_engines(frame, seed=7)
    assert train == split_engines(frame, seed=7)[0]
    assert set(train).isdisjoint(validation)
    assert len(validation) == 1


def test_features_are_causal_and_isolated():
    frame = fixture_frame()
    frame["engine_id"] = "train:FD001:" + frame.unit.astype(str)
    before = causal_features(frame)
    changed = frame.copy()
    changed.loc[(changed.unit == 1) & (changed.cycle == 8), "sensor_2"] += 10000
    after = causal_features(changed)
    earlier = frame[(frame.unit == 1) & (frame.cycle < 8)].index
    np.testing.assert_allclose(before.loc[earlier, FEATURES], after.loc[earlier, FEATURES])
    other = frame.unit == 2
    np.testing.assert_allclose(before.loc[other, FEATURES], after.loc[other, FEATURES])


def test_artifact_round_trip(tmp_path):
    frame = fixture_frame()
    frame["engine_id"] = "train:FD001:" + frame.unit.astype(str)
    x = causal_features(frame)
    y = np.arange(len(x), dtype=float)
    path = tmp_path / "model.joblib"
    trained = train_model(x, y, path)
    loaded = load_artifact(path)
    np.testing.assert_allclose(trained["model"].predict(x[FEATURES]), loaded["model"].predict(x[FEATURES]))
