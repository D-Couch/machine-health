# Inference API

Run the frozen baseline API after creating `artifacts/fd001.joblib`:

```bash
.venv/bin/uvicorn machine_health.api:app --host 127.0.0.1 --port 8000
```

`POST /v1/predict` accepts JSON `{ "observations": [...] }` with 20–1,000 chronologically ordered records. Each record has integer `cycle`, three numeric `setting_1..3` values, and numeric `sensor_1..21` values. Cycles must be strictly increasing. Unknown fields (including `rul`, labels, engine lifetime, and unit identifiers) are rejected. The model predicts from causal rolling features over the supplied sequence and returns a nonnegative RUL estimate in cycles at the final observation. It uses the same feature builder, schema, imputer, and model artifact as the dashboard. Missing sensor values are rejected at the request schema boundary; artifact imputation remains in the shared model pipeline for compatible future input sources.

```bash
curl -sS http://127.0.0.1:8000/health
```

Interactive request schema: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
