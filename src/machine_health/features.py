from __future__ import annotations

import pandas as pd

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
WINDOWS = (5, 10, 20)
FEATURES = ["cycle"] + SETTINGS + [f"{sensor}_{stat}_{window}" for sensor in SENSORS for window in WINDOWS for stat in ("mean", "std")]


def causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Trailing rolling summaries within engines; no centered/future observations."""
    pieces = []
    for _, group in frame.groupby("engine_id", sort=False):
        group = group.sort_values("cycle").copy()
        data = {"cycle": group["cycle"].to_numpy()}
        for setting in SETTINGS:
            data[setting] = group[setting].to_numpy()
        for sensor in SENSORS:
            series = group[sensor]
            for window in WINDOWS:
                rolling = series.rolling(window, min_periods=1)
                data[f"{sensor}_mean_{window}"] = rolling.mean().to_numpy()
                data[f"{sensor}_std_{window}"] = rolling.std(ddof=0).fillna(0).to_numpy()
        part = pd.DataFrame(data, index=group.index)
        part["engine_id"] = group["engine_id"].to_numpy()
        pieces.append(part)
    return pd.concat(pieces).sort_index()
