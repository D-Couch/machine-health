from __future__ import annotations

import argparse
from pathlib import Path

from machine_health.pipeline import (
    ROOT,
    evaluate_saved,
    evaluate_test,
    prepare,
    select_alert_policy,
    train,
)
from machine_health.run_store import list_runs


def main() -> None:
    parser = argparse.ArgumentParser(prog="machine-health")
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare", help="Acquire NASA archive or validate local FD001 file")
    p_prepare.add_argument("--data", type=Path, default=ROOT / "data/raw")
    p_prepare.add_argument("--file", type=Path)
    p_train = sub.add_parser("train", help="Train constant and gradient-boosted baselines")
    p_train.add_argument("--data", type=Path, default=ROOT / "data/raw/train_FD001.txt")
    p_train.add_argument("--artifact", type=Path, default=ROOT / "artifacts/fd001.joblib")
    p_eval = sub.add_parser("evaluate-validation", help="Evaluate saved model on held-out engines")
    p_eval.add_argument("--data", type=Path, default=ROOT / "data/raw/train_FD001.txt")
    p_eval.add_argument("--artifact", type=Path, default=ROOT / "artifacts/fd001.joblib")
    p_eval.add_argument("--split", type=Path, default=ROOT / "reports/fd001_split.json")
    p_eval.add_argument("--output", type=Path, default=ROOT / "reports/validation_metrics.json")
    replay = sub.add_parser("replay", help="Generate predictions for all validation engine cycles")
    replay.add_argument("--data", type=Path, default=ROOT / "data/raw/train_FD001.txt")
    replay.add_argument("--artifact", type=Path, default=ROOT / "artifacts/fd001.joblib")
    replay.add_argument("--split", type=Path, default=ROOT / "reports/fd001_split.json")
    replay.add_argument("--output", type=Path, default=ROOT / "data/processed/validation_replay.csv")
    test_eval = sub.add_parser("evaluate-test", help="Run frozen official test evaluation exactly once")
    test_eval.add_argument("--data", type=Path, default=ROOT / "data/raw")
    test_eval.add_argument("--artifact", type=Path, default=ROOT / "artifacts/fd001.joblib")
    test_eval.add_argument("--output", type=Path, default=ROOT / "reports/official_test_v1.json")
    alert = sub.add_parser("select-alert-policy", help="Select alert threshold on validation engines only")
    alert.add_argument("--data", type=Path, default=ROOT / "data/raw/train_FD001.txt")
    alert.add_argument("--artifact", type=Path, default=ROOT / "artifacts/fd001.joblib")
    alert.add_argument("--split", type=Path, default=ROOT / "reports/fd001_split.json")
    alert.add_argument("--output", type=Path, default=ROOT / "reports/validation_alert_policy.json")
    runs = sub.add_parser("list-runs", help="List recorded train/evaluation runs from SQLite")
    runs.add_argument("--database", type=Path, default=ROOT / "data/experiments.sqlite")
    runs.add_argument("--limit", type=int, default=20)
    temporal = sub.add_parser("train-temporal", help="Train and evaluate the TCN with engine-level calibration")
    temporal.add_argument("--data", type=Path, default=ROOT / "data/raw/train_FD001.txt")
    temporal.add_argument("--artifact", type=Path, default=ROOT / "artifacts/fd001.joblib")
    temporal.add_argument("--split", type=Path, default=ROOT / "reports/fd001_split.json")
    temporal.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/fd001_tcn.pt")
    temporal.add_argument("--output", type=Path, default=ROOT / "reports/temporal_model_v1.json")
    temporal.add_argument("--replay", type=Path, default=ROOT / "data/processed/temporal_validation_replay.csv")
    args = parser.parse_args()
    if args.command == "prepare":
        path = args.file or prepare(args.data)
        from machine_health.data import load_fd001
        print(f"Validated {len(load_fd001(path))} rows from {path}")
    elif args.command == "train":
        result = train(args.data, args.artifact)
        print(f"Saved {args.artifact}; train engines={len(result['train_engines'])}, validation engines={len(result['validation_engines'])}")
    elif args.command == "evaluate-validation":
        result = evaluate_saved(args.data, args.artifact, args.split, args.output)
        print(f"Validation endpoint MAE={result['endpoint']['mae_cycles']:.3f} cycles; report={args.output.with_suffix('.md')}")
    elif args.command == "replay":
        import json

        from machine_health.data import attach_rul, load_fd001
        from machine_health.features import FEATURES, causal_features
        from machine_health.model import load_artifact
        frame = attach_rul(load_fd001(args.data))
        feat = causal_features(frame)
        split = json.loads(args.split.read_text())
        selected = feat.engine_id.isin(split['validation_engines'])
        artifact = load_artifact(args.artifact)
        output = frame.loc[selected, ['engine_id', 'unit', 'cycle', 'rul']].copy()
        output['prediction'] = artifact['model'].predict(feat.loc[selected, FEATURES])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Wrote {len(output)} validation replay rows to {args.output}")
    elif args.command == "list-runs":
        import json

        print(json.dumps(list_runs(args.database, args.limit), indent=2))
    elif args.command == "select-alert-policy":
        result = select_alert_policy(args.data, args.artifact, args.split, args.output)
        print(f"Selected threshold={result['selected_threshold_cycles']} cycles on validation engines")
    elif args.command == "train-temporal":
        from machine_health.temporal_pipeline import run_temporal_experiment

        result = run_temporal_experiment(args.data, args.artifact, args.split,
                                         args.checkpoint, args.output, args.replay)
        score = result["tcn"]["score"]
        print(f"TCN validation endpoint MAE={score['endpoint']['mae_cycles']:.3f} cycles; "
              f"report={args.output.with_suffix('.md')}")
    else:
        result = evaluate_test(args.data, args.artifact, args.output)
        print(f"Official test endpoint MAE={result['primary_model']['mae_cycles']:.3f} cycles; report={args.output.with_suffix('.md')}")
