from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from itertools import pairwise
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from machine_health.features import FEATURES, SENSORS, SETTINGS, causal_features
from machine_health.model import load_artifact

ROOT = Path(__file__).resolve().parents[2]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    cycle: int = Field(ge=1)
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observations: list[Observation] = Field(min_length=20, max_length=1000)


class PredictionResponse(BaseModel):
    prediction_cycles: float
    last_observed_cycle: int
    observations_used: int
    model: str


def create_app(artifact_path: Path | None = None) -> FastAPI:
    path = artifact_path or Path(os.environ.get("MACHINE_HEALTH_ARTIFACT", ROOT / "artifacts/fd001.joblib"))
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not path.exists():
            raise RuntimeError(f"Model artifact not found: {path}. Run machine-health train first.")
        app.state.artifact = load_artifact(path)
        yield

    app = FastAPI(title="Machine Health RUL API", version="0.1.0",
                  description="CPU inference for the frozen NASA C-MAPSS FD001 baseline.", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": "histogram_gradient_boosting"}

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        observations = [item.model_dump() for item in request.observations]
        cycles = [row["cycle"] for row in observations]
        if any(later <= earlier for earlier, later in pairwise(cycles)):
            raise HTTPException(status_code=422, detail="observation cycles must be strictly increasing")
        for row in observations:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise HTTPException(status_code=422, detail=f"{key} must be finite")
        frame = pd.DataFrame(observations)
        missing = (set(SETTINGS) | set(SENSORS) | {"cycle"}) - set(frame.columns)
        if missing:
            raise HTTPException(status_code=422, detail=f"missing feature values: {sorted(missing)}")
        frame["unit"] = 1
        frame["engine_id"] = "request:FD001:sequence"
        features = causal_features(frame)
        last = features.iloc[[-1]][FEATURES]
        predicted = float(app.state.artifact["model"].predict(last)[0])
        return PredictionResponse(prediction_cycles=max(0.0, predicted), last_observed_cycle=cycles[-1],
                                  observations_used=len(observations), model="histogram_gradient_boosting")

    return app


app = create_app()
