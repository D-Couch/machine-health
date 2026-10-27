from __future__ import annotations

import math
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DISPOSITIONS = ("inspect", "monitor", "defer", "dismiss")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def record_disposition(
    database: Path,
    *,
    engine_id: str,
    snapshot_cycle: int,
    disposition: str,
    rationale: str = "",
    model_name: str,
    artifact_sha256: str,
    predicted_rul_cycles: float,
    threshold_cycles: float,
    persistence_cycles: int,
    review_horizon_cycles: float,
    interval_lower_cycles: float | None = None,
    interval_upper_cycles: float | None = None,
) -> int:
    """Append one local review decision with the prediction context shown to the user."""
    if not re.fullmatch(r"train:FD001:[1-9][0-9]*", engine_id):
        raise ValueError("Review dispositions accept validation FD001 engines only")
    if snapshot_cycle < 1 or persistence_cycles < 1:
        raise ValueError("Snapshot and persistence must be positive cycle counts")
    if disposition not in DISPOSITIONS:
        raise ValueError(f"Disposition must be one of: {', '.join(DISPOSITIONS)}")
    if not model_name.strip() or len(model_name) > 100:
        raise ValueError("Model name must contain 1–100 characters")
    if not _SHA256.fullmatch(artifact_sha256):
        raise ValueError("Artifact SHA-256 must be a 64-character lowercase hex digest")
    if not math.isfinite(predicted_rul_cycles) or predicted_rul_cycles < 0:
        raise ValueError("Predicted RUL must be finite and non-negative")
    if not math.isfinite(threshold_cycles) or threshold_cycles < 0:
        raise ValueError("Threshold must be finite and non-negative")
    if not math.isfinite(review_horizon_cycles) or review_horizon_cycles < 0:
        raise ValueError("Review horizon must be finite and non-negative")
    rationale = rationale.strip()
    if len(rationale) > 1000:
        raise ValueError("Rationale must be at most 1,000 characters")
    if (interval_lower_cycles is None) != (interval_upper_cycles is None):
        raise ValueError("Both prediction interval bounds must be provided together")
    if interval_lower_cycles is not None and interval_upper_cycles is not None:
        if not math.isfinite(interval_lower_cycles) or not math.isfinite(interval_upper_cycles):
            raise ValueError("Prediction interval bounds must be finite")
        if interval_lower_cycles > interval_upper_cycles:
            raise ValueError("Prediction interval lower bound must not exceed upper bound")

    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS review_dispositions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_utc TEXT NOT NULL,
            engine_id TEXT NOT NULL,
            snapshot_cycle INTEGER NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN ('inspect', 'monitor', 'defer', 'dismiss')),
            rationale TEXT NOT NULL,
            model_name TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            predicted_rul_cycles REAL NOT NULL,
            interval_lower_cycles REAL,
            interval_upper_cycles REAL,
            threshold_cycles REAL NOT NULL,
            persistence_cycles INTEGER NOT NULL,
            review_horizon_cycles REAL NOT NULL
        )""")
        cursor = connection.execute(
            """INSERT INTO review_dispositions (
                created_utc, engine_id, snapshot_cycle, disposition, rationale,
                model_name, artifact_sha256, predicted_rul_cycles,
                interval_lower_cycles, interval_upper_cycles, threshold_cycles,
                persistence_cycles, review_horizon_cycles
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(), engine_id, snapshot_cycle, disposition,
                rationale, model_name.strip(), artifact_sha256, predicted_rul_cycles,
                interval_lower_cycles, interval_upper_cycles, threshold_cycles,
                persistence_cycles, review_horizon_cycles,
            ),
        )
        return int(cursor.lastrowid)


def list_dispositions(
    database: Path,
    *,
    engine_id: str,
    limit: int = 50,
) -> list[dict]:
    """List the latest local review events for one validation engine."""
    if not re.fullmatch(r"train:FD001:[1-9][0-9]*", engine_id):
        raise ValueError("Review history accepts validation FD001 engines only")
    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    if not database.exists():
        return []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT * FROM review_dispositions
               WHERE engine_id = ? ORDER BY id DESC LIMIT ?""",
            (engine_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]
