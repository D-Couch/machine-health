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


def test_official_test_label_reconstruction(tmp_path):
    from machine_health.data import attach_test_endpoint_rul

    frame = fixture_frame()
    frame = frame[frame.unit.isin([1, 2])].copy()
    frame["engine_id"] = "test:FD001:" + frame.unit.astype(str)
    labels = tmp_path / "RUL_FD001.txt"
    labels.write_text("5\n10\n")
    scored = attach_test_endpoint_rul(frame, labels)
    engine_1 = scored[scored.unit == 1]
    engine_2 = scored[scored.unit == 2]
    assert engine_1.iloc[-1].rul == 5
    assert engine_1.iloc[0].rul == 12
    assert engine_2.iloc[-1].rul == 10
    assert engine_2.iloc[0].rul == 17


def test_alert_requires_minimum_history_and_consecutive_persistence():
    from machine_health.alert_policy import _first_persistent_alert

    group = pd.DataFrame({"cycle": np.arange(1, 25), "prediction": [10.0] * 19 + [4.0] * 5})
    assert _first_persistent_alert(group, 5.0) == 21
    group.loc[20:, "cycle"] = [22, 24, 26, 28]
    assert _first_persistent_alert(group, 5.0) is None


def test_sqlite_experiment_index(tmp_path):
    from machine_health.run_store import list_runs, record_run

    artifact = tmp_path / "artifact.bin"
    source = tmp_path / "source.txt"
    artifact.write_bytes(b"artifact")
    source.write_text("source")
    database = tmp_path / "runs.sqlite"
    run_id = record_run(database, kind="fixture", artifact=artifact, data_path=source,
                        report_path=None, configuration={"seed": 3}, metrics={"mae": 1.25})
    runs = list_runs(database)
    assert runs[0]["id"] == run_id
    assert runs[0]["configuration_json"] == '{"seed": 3}'
    assert runs[0]["metrics_json"] == '{"mae": 1.25}'


def test_fastapi_shared_preprocessing_and_input_validation(tmp_path):
    from fastapi.testclient import TestClient

    from machine_health.api import create_app

    rows = []
    for unit in range(1, 6):
        for cycle in range(1, 25):
            rows.append([unit, cycle, 0.1, 0.2, 0.3, *[unit * 10 + cycle + j for j in range(21)]])
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["engine_id"] = "train:FD001:" + frame.unit.astype(str)
    x = causal_features(frame)
    artifact_path = tmp_path / "api.joblib"
    artifact = train_model(x, attach_rul(frame).rul, artifact_path)
    assert artifact["features"] == FEATURES

    observations = frame[frame.unit == 1].iloc[:20].drop(columns=["unit", "engine_id"])
    payload = {"observations": observations.to_dict(orient="records")}
    with TestClient(create_app(artifact_path)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["observations_used"] == 20
        assert response.json()["prediction_cycles"] >= 0
        bad_order = {"observations": list(payload["observations"])}
        bad_order["observations"][1]["cycle"] = 1
        assert client.post("/v1/predict", json=bad_order).status_code == 422
        with_target = {"observations": [dict(row, rul=4) for row in payload["observations"]]}
        assert client.post("/v1/predict", json=with_target).status_code == 422
        assert client.post("/v1/predict", json={"observations": payload["observations"][:10]}).status_code == 422


def test_frozen_evaluation_fixture_and_no_overwrite(tmp_path):
    from machine_health.data import attach_test_endpoint_rul
    from machine_health.frozen_evaluation import evaluate_official_test

    raw = fixture_frame()
    raw = raw[raw.unit.isin([1, 2])].copy()
    raw["engine_id"] = "test:FD001:" + raw.unit.astype(str)
    labels = tmp_path / "labels.txt"
    labels.write_text("3\n4\n")
    test_file = tmp_path / "test.txt"
    raw.drop(columns="engine_id").to_csv(test_file, sep=" ", header=False, index=False)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("synthetic fixture protocol; not NASA evaluation")
    training = fixture_frame()
    training["engine_id"] = "train:FD001:" + training.unit.astype(str)
    training_x = causal_features(training)
    artifact_file = tmp_path / "model.joblib"
    artifact = train_model(training_x, attach_rul(training).rul, artifact_file)
    test = attach_test_endpoint_rul(raw, labels)
    test_x = causal_features(raw)
    output = tmp_path / "fixture_result.json"
    result = evaluate_official_test(test, test_x, artifact, test_file, labels, protocol,
                                    artifact_file, output)
    assert result["n_engines"] == 2
    assert len(result["endpoint_per_engine"]) == 2
    assert output.exists()
    with pytest.raises(FileExistsError):
        evaluate_official_test(test, test_x, artifact, test_file, labels, protocol,
                               artifact_file, output)
