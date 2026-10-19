from __future__ import annotations

import math
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from machine_health.data import sha256

RAW_FEATURES = ["cycle", "setting_1", "setting_2", "setting_3", *[f"sensor_{i}" for i in range(1, 22)]]
HISTORY = 30
SEED = 31415
DEVICE = torch.device("cpu")


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(4, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (2 * self.dilation, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = self.dropout(F.gelu(x))
        return F.gelu(x + residual)


class CausalTCN(nn.Module):
    """Small causal residual TCN; output depends only on the supplied history."""

    def __init__(self, input_channels: int = len(RAW_FEATURES), channels: int = 32):
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, channels, kernel_size=1)
        self.blocks = nn.Sequential(*(CausalBlock(channels, d) for d in (1, 2, 4, 8)))
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.input_projection(x))
        return self.head(hidden[:, :, -1]).squeeze(-1)


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                         text=True, stderr=subprocess.DEVNULL)
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _sequences(frame: pd.DataFrame, engines: list[str], mean: np.ndarray,
               scale: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, list[str], np.ndarray]:
    sequences, targets, row_engines, cycles = [], [], [], []
    selected = frame[frame.engine_id.isin(engines)]
    for engine_id, group in selected.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        values = ((group[RAW_FEATURES].to_numpy(dtype=np.float32) - mean) / scale).astype(np.float32)
        padded = np.pad(values, ((HISTORY - 1, 0), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, HISTORY, axis=0)
        sequences.append(np.ascontiguousarray(windows))  # (cycles, channels, history)
        targets.append(group.rul.to_numpy(dtype=np.float32))
        row_engines.extend([engine_id] * len(group))
        cycles.extend(group.cycle.to_numpy(dtype=int))
    x = torch.from_numpy(np.concatenate(sequences))
    y = torch.from_numpy(np.concatenate(targets))
    return x, y, row_engines, np.asarray(cycles, dtype=int)


def _engine_mean_and_scale(frame: pd.DataFrame, engines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[frame.engine_id.isin(engines), RAW_FEATURES].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _prediction(model: nn.Module, x: torch.Tensor, target_mean: float,
                target_scale: float, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for batch, in DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False):
            outputs.append(model(batch.to(DEVICE)).cpu().numpy())
    return np.maximum(0.0, np.concatenate(outputs) * target_scale + target_mean)


def load_tcn(path: Path) -> tuple[CausalTCN, dict]:
    bundle = torch.load(path, map_location=DEVICE, weights_only=True)
    if bundle.get("raw_features") != RAW_FEATURES or bundle.get("history_cycles") != HISTORY:
        raise ValueError("TCN checkpoint feature schema or history length does not match this application")
    if bundle["feature_mean"].shape != (len(RAW_FEATURES),) or bundle["feature_scale"].shape != (len(RAW_FEATURES),):
        raise ValueError("TCN checkpoint normalization schema is invalid")
    model = CausalTCN(input_channels=len(bundle["raw_features"]), channels=bundle["channels"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def train_tcn(frame: pd.DataFrame, train_engines: list[str],
              output: Path) -> tuple[CausalTCN, dict, dict]:
    """Fit on an engine-level subset; early-stop on different training engines."""
    from sklearn.model_selection import train_test_split

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    fit_ids, early_ids = train_test_split(sorted(train_engines), test_size=0.1, random_state=SEED)
    fit_ids, early_ids = sorted(fit_ids), sorted(early_ids)
    mean, scale = _engine_mean_and_scale(frame, fit_ids)
    fit_x, fit_y, _, _ = _sequences(frame, fit_ids, mean, scale)
    early_x, early_y, _, _ = _sequences(frame, early_ids, mean, scale)
    target_mean = float(fit_y.mean())
    target_scale = float(fit_y.std().clamp(min=1.0))
    fit_y_scaled = (fit_y - target_mean) / target_scale

    model = CausalTCN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(fit_x, fit_y_scaled), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    best_loss, best_state, best_epoch, patience = float("inf"), None, 0, 0
    max_epochs, patience_limit = 60, 8
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        early_pred = _prediction(model, early_x, target_mean, target_scale)
        early_mae = float(np.mean(np.abs(early_pred - early_y.numpy())))
        history.append({"epoch": epoch, "train_smooth_l1_scaled": float(np.mean(losses)),
                        "early_stop_mae_cycles": early_mae})
        if early_mae < best_loss - 1e-5:
            best_loss, best_epoch, patience = early_mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_limit:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    bundle = {
        "state_dict": best_state, "raw_features": RAW_FEATURES, "history_cycles": HISTORY,
        "channels": 32, "feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale),
        "target_mean": target_mean, "target_scale": target_scale,
        "metadata": {"dataset": "NASA C-MAPSS FD001 training trajectories",
                     "target": "raw uncapped RUL cycles", "seed": SEED,
                     "fit_engines": fit_ids, "early_stopping_engines": early_ids,
                     "best_epoch": best_epoch, "max_epochs": max_epochs,
                     "patience": patience_limit, "history_cycles": HISTORY,
                     "architecture": "4 residual causal Conv1d blocks, kernel 3, dilations 1/2/4/8, 32 channels",
                     "input_information": "cycle plus current/past 29 raw settings and sensor cycles",
                     "code_revision": _git_revision(), "code_dirty": _git_dirty(),
                     "temporal_module_sha256": sha256(Path(__file__)),
                     "python": platform.python_version(), "platform": platform.platform(),
                     "cpu_threads": torch.get_num_threads(),
                     "torch": str(torch.__version__), "created_utc": datetime.now(UTC).isoformat()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    bundle["artifact_sha256"] = sha256(output)
    return model, bundle, {"best_epoch": best_epoch, "early_stop_mae_cycles": best_loss,
                           "epochs_run": len(history), "history": history,
                           "fit_engines": fit_ids, "early_stopping_engines": early_ids}


def predict_frame(model: CausalTCN, bundle: dict, frame: pd.DataFrame,
                  engines: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    x, y, row_engines, cycles = _sequences(frame, engines, bundle["feature_mean"].numpy(),
                                           bundle["feature_scale"].numpy())
    predictions = _prediction(model, x, float(bundle["target_mean"]), float(bundle["target_scale"]))
    result = pd.DataFrame({"engine_id": row_engines, "cycle": cycles,
                           "actual_rul": y.numpy(), "prediction": predictions})
    return result, predictions


def _finite_sample_quantile(scores: list[float], coverage: float) -> float | None:
    """Split-conformal upper quantile; None means sample size cannot support finite coverage."""
    n = len(scores)
    if n == 0:
        return None
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        return None
    return float(np.sort(np.asarray(scores, dtype=float))[rank - 1])


def calibrate_by_predicted_band(calibration: pd.DataFrame, coverage: float = 0.8) -> dict:
    """Engine-block conformal radii by prediction band, using each engine's max residual."""
    bands = [(0, 20, "0-19"), (20, 50, "20-49"), (50, 100, "50-99"),
             (100, float("inf"), "100+")]
    result = {"nominal_coverage": coverage, "grouping": "predicted RUL band",
              "calibration_unit": "engine maximum absolute residual within band",
              "intervals": {}}
    for low, high, name in bands:
        engine_scores = []
        for _, group in calibration.groupby("engine_id"):
            in_band = (group.prediction >= low) & (group.prediction < high)
            if in_band.any():
                engine_scores.append(float(np.max(np.abs(group.loc[in_band, "actual_rul"] -
                                                               group.loc[in_band, "prediction"]))))
        radius = _finite_sample_quantile(engine_scores, coverage)
        result["intervals"][name] = {"radius_cycles": radius,
                                      "calibration_engines": len(engine_scores),
                                      "finite_sample_rank": math.ceil((len(engine_scores) + 1) * coverage),
                                      "available": radius is not None}
    return result


def add_intervals(predictions: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    result = predictions.copy()
    band_names = np.select([result.prediction < 20, result.prediction < 50,
                            result.prediction < 100], ["0-19", "20-49", "50-99"], default="100+")
    radii = calibration["intervals"]
    result["interval_radius"] = [np.nan if radii[name]["radius_cycles"] is None
                                 else radii[name]["radius_cycles"] for name in band_names]
    result["lower"] = np.maximum(0.0, result.prediction - result.interval_radius)
    result["upper"] = result.prediction + result.interval_radius
    return result


def interval_metrics(predictions: pd.DataFrame) -> dict:
    result = {"nominal_coverage": 0.8, "by_actual_rul_band": {}, "endpoint": {}}
    bands = [(0, 20, "0-19"), (20, 50, "20-49"), (50, 100, "50-99"),
             (100, float("inf"), "100+")]
    for low, high, name in bands:
        band = predictions[(predictions.actual_rul >= low) & (predictions.actual_rul < high)]
        if band.empty:
            continue
        per_engine = []
        for engine_id, group in band.groupby("engine_id"):
            available = group.dropna(subset=["lower", "upper"])
            if len(available):
                per_engine.append({"engine_id": engine_id,
                                   "all_cycles_covered": bool(((available.actual_rul >= available.lower) &
                                                                (available.actual_rul <= available.upper)).all()),
                                   "coverage_fraction": float(((available.actual_rul >= available.lower) &
                                                               (available.actual_rul <= available.upper)).mean()),
                                   "mean_width_cycles": float((available.upper - available.lower).mean()),
                                   "cycles": len(available)})
        if per_engine:
            result["by_actual_rul_band"][name] = {
                "engines": len(per_engine), "engine_band_simultaneous_coverage":
                float(np.mean([item["all_cycles_covered"] for item in per_engine])),
                "mean_engine_cycle_coverage": float(np.mean([item["coverage_fraction"] for item in per_engine])),
                "mean_width_cycles": float(np.mean([item["mean_width_cycles"] for item in per_engine])),
                "engine_details": per_engine}
    endpoints = predictions.sort_values("cycle").groupby("engine_id", sort=False).tail(1)
    endpoints = endpoints.dropna(subset=["lower", "upper"])
    if len(endpoints):
        covered = (endpoints.actual_rul >= endpoints.lower) & (endpoints.actual_rul <= endpoints.upper)
        result["endpoint"] = {"engines": len(endpoints), "coverage": float(covered.mean()),
                               "mean_width_cycles": float((endpoints.upper - endpoints.lower).mean()),
                               "mae_cycles": float(np.abs(endpoints.actual_rul - endpoints.prediction).mean())}
    return result
