"""ROS log ingestion and canonical pendulum schema conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CANONICAL_COLUMNS = ["t", "theta", "omega", "u", "current", "source"]


def ingest_ros_csv(path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    """Map arbitrary ROS-exported column names to the canonical schema."""
    df = pd.read_csv(path)
    renamed = df.rename(columns=mapping)
    missing = {"t", "theta", "omega", "u"} - set(renamed.columns)
    if missing:
        raise ValueError(f"{path} missing mapped columns: {sorted(missing)}")
    if "current" not in renamed.columns:
        renamed["current"] = 0.0
    renamed["source"] = path.name
    return renamed[CANONICAL_COLUMNS].copy()
