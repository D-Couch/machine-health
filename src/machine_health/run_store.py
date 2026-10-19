from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from machine_health.data import sha256
from machine_health.model import git_revision


def record_run(database: Path, *, kind: str, artifact: Path, data_path: Path,
               report_path: Path | None, configuration: dict[str, Any], metrics: dict[str, Any] | None) -> int:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_utc TEXT NOT NULL,
            kind TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            data_path TEXT NOT NULL,
            data_sha256 TEXT NOT NULL,
            report_path TEXT,
            code_revision TEXT NOT NULL,
            configuration_json TEXT NOT NULL,
            metrics_json TEXT
        )""")
        cursor = connection.execute(
            """INSERT INTO experiments
            (created_utc, kind, artifact_path, artifact_sha256, data_path, data_sha256,
             report_path, code_revision, configuration_json, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(UTC).isoformat(), kind, str(artifact), sha256(artifact), str(data_path),
             sha256(data_path), None if report_path is None else str(report_path), git_revision(),
             json.dumps(configuration, sort_keys=True),
             None if metrics is None else json.dumps(metrics, sort_keys=True)),
        )
        return int(cursor.lastrowid)


def list_runs(database: Path, limit: int = 20) -> list[dict[str, Any]]:
    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    if not database.exists():
        return []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
