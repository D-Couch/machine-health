import numpy as np
import pandas as pd
import torch

from machine_health.temporal import (
    RAW_FEATURES,
    CausalTCN,
    _finite_sample_quantile,
    _sequences,
    add_intervals,
    calibrate_by_predicted_band,
    load_tcn,
    select_device,
)
from machine_health.temporal_pipeline import partition_validation_engines


def test_device_selection_uses_cuda_when_available_and_cpu_otherwise(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"


def _frame(engines=("train:FD001:1", "train:FD001:2"), cycles=5):
    rows = []
    for engine_number, engine_id in enumerate(engines, start=1):
        for cycle in range(1, cycles + 1):
            row = {"engine_id": engine_id, "cycle": cycle, "rul": cycles - cycle}
            row.update({name: float(engine_number * 100 + cycle) for name in RAW_FEATURES[1:]})
            rows.append(row)
    return pd.DataFrame(rows)


def test_validation_model_and_calibration_engines_are_disjoint_and_complete():
    engines = [f"train:FD001:{n}" for n in range(1, 21)]
    model, calibration = partition_validation_engines(engines)
    assert len(model) == 12
    assert len(calibration) == 8
    assert set(model).isdisjoint(calibration)
    assert set(model) | set(calibration) == set(engines)
    assert partition_validation_engines(engines) == (model, calibration)


def test_temporal_sequences_are_causal_and_reset_at_engine_boundaries():
    frame = _frame(cycles=5)
    mean = np.zeros(len(RAW_FEATURES), dtype=np.float32)
    scale = np.ones(len(RAW_FEATURES), dtype=np.float32)
    x, _, row_engines, _ = _sequences(frame, sorted(frame.engine_id.unique()), mean, scale)
    # The first row for engine 2 starts with padding, not engine 1's final sensor reading.
    first_engine_two = row_engines.index("train:FD001:2")
    assert np.all(x[first_engine_two, :, :-1].numpy() == 0)
    assert x[first_engine_two, 1, -1].item() == 201.0

    altered = frame.copy()
    altered.loc[(altered.engine_id == "train:FD001:1") & (altered.cycle == 5), "sensor_1"] = -999
    changed, _, _, _ = _sequences(altered, ["train:FD001:1"], mean, scale)
    # Earlier cycles remain identical even when a later observation is changed.
    assert torch.equal(x[3], changed[3])
    assert not torch.equal(x[4], changed[4])


def test_tcn_returns_one_prediction_per_history_and_finite_sample_quantile_is_bounded():
    model = CausalTCN().eval()
    with torch.inference_mode():
        prediction = model(torch.zeros((3, len(RAW_FEATURES), 30)))
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()
    assert _finite_sample_quantile(list(range(8)), 0.8) == 7.0
    assert _finite_sample_quantile([1, 2, 3], 0.8) is None


def test_tcn_checkpoint_round_trip(tmp_path):
    model = CausalTCN().eval()
    path = tmp_path / "tcn.pt"
    torch.save({"state_dict": model.state_dict(), "raw_features": RAW_FEATURES,
                "channels": 32, "history_cycles": 30,
                "feature_mean": torch.zeros(len(RAW_FEATURES)),
                "feature_scale": torch.ones(len(RAW_FEATURES)),
                "target_mean": 12.0, "target_scale": 4.0,
                "metadata": {"seed": 5}}, path)
    loaded, bundle = load_tcn(path)
    input_values = torch.randn((2, len(RAW_FEATURES), 30))
    with torch.inference_mode():
        assert torch.equal(model(input_values), loaded(input_values))
    assert bundle["history_cycles"] == 30


def test_calibration_uses_engine_maxima_and_assigns_intervals_by_prediction_band():
    rows = []
    for engine, offset in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1):
        for prediction in [10.0, 30.0, 70.0, 120.0]:
            rows.append({"engine_id": f"cal:{engine}", "prediction": prediction,
                         "actual_rul": prediction + offset})
    calibration = calibrate_by_predicted_band(pd.DataFrame(rows), coverage=0.8)
    assert calibration["intervals"]["0-19"]["calibration_engines"] == 5
    assert calibration["intervals"]["0-19"]["radius_cycles"] == 5.0
    eval_rows = pd.DataFrame([{"engine_id": "eval:1", "cycle": 1,
                               "prediction": 30.0, "actual_rul": 35.0}])
    result = add_intervals(eval_rows, calibration)
    assert result.loc[0, "lower"] == 25.0
    assert result.loc[0, "upper"] == 35.0
