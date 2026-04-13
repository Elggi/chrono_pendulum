"""Dataset segmentation utilities for free-decay and excitation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class SegmentConfig:
    """Segmentation thresholds and minimum segment length."""

    input_column: str = "u"
    zero_input_threshold: float = 0.03
    min_samples: int = 100


def load_csv(path: Path) -> pd.DataFrame:
    """Load a trajectory CSV and enforce required columns."""
    df = pd.read_csv(path)
    required = {"t", "theta", "omega"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def mark_free_decay(df: pd.DataFrame, cfg: SegmentConfig) -> pd.Series:
    """Return boolean mask for near-zero-input samples."""
    if cfg.input_column not in df.columns:
        raise ValueError(f"Input column '{cfg.input_column}' not found")
    return df[cfg.input_column].abs() <= cfg.zero_input_threshold


def extract_contiguous_segments(mask: Iterable[bool], min_samples: int) -> list[tuple[int, int]]:
    """Extract contiguous True spans as [start, end) index segments."""
    values = list(mask)
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(values):
        if flag and start is None:
            start = idx
        if (not flag) and start is not None:
            if idx - start >= min_samples:
                spans.append((start, idx))
            start = None
    if start is not None:
        end = len(values)
        if end - start >= min_samples:
            spans.append((start, end))
    return spans


def split_nominal_excitation(
    df: pd.DataFrame,
    cfg: SegmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split mixed trajectory into free-decay and excitation datasets."""
    m = mark_free_decay(df, cfg)
    return df[m].copy(), df[~m].copy()
