from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
COLUMNS = ["unit", "cycle", *[f"setting_{i}" for i in range(1, 4)], *[f"sensor_{i}" for i in range(1, 22)]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "cmapss.zip"
    if not archive.exists():
        urlretrieve(NASA_URL, archive)
    extracted = destination / "train_FD001.txt"
    with zipfile.ZipFile(archive) as bundle:
        matches = [name for name in bundle.namelist() if name.lower().endswith("train_fd001.txt")]
        if len(matches) == 1:
            extracted.write_bytes(bundle.read(matches[0]))
        else:
            nested = [name for name in bundle.namelist() if name.lower().endswith("cmapssdata.zip")]
            if len(matches) or len(nested) != 1:
                raise ValueError(f"Expected FD001 file or one nested CMAPSSData.zip; found {matches}, {nested}")
            with zipfile.ZipFile(__import__("io").BytesIO(bundle.read(nested[0]))) as inner:
                matches = [name for name in inner.namelist() if name.lower().endswith("train_fd001.txt")]
                if len(matches) != 1:
                    raise ValueError(f"Expected one train_FD001.txt in nested archive; found {matches}")
                extracted.write_bytes(inner.read(matches[0]))
    return extracted


def load_fd001(path: Path, partition: str = "train") -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected 26 columns, got {frame.shape[1]}")
    frame.columns = COLUMNS
    if frame.isna().any().any() or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError("Input contains missing or non-finite values")
    if (frame["unit"] < 1).any() or (frame["cycle"] < 1).any():
        raise ValueError("Unit and cycle identifiers must be positive")
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate unit/cycle records")
    frame["engine_id"] = partition + ":FD001:" + frame["unit"].astype(str)
    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


def attach_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw RUL at cycle t is final_cycle - t; failure cycle has RUL zero."""
    result = frame.copy()
    endpoints = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = endpoints - result["cycle"]
    return result


def write_provenance(archive: Path, extracted: Path, destination: Path) -> None:
    metadata = {
        "source": NASA_URL,
        "repository": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "archive_sha256": sha256(archive),
        "train_fd001_sha256": sha256(extracted),
        "attribution": "A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA PCoE repository, NASA Ames Research Center.",
        "terms_note": "NASA PCoE page requests acknowledgment of repository and data donors; no separate redistribution license was identified. Original data excluded from Git.",
    }
    destination.write_text(json.dumps(metadata, indent=2) + "\n")
