"""Pipeline B-2: sparse residual equation identification (SINDy-style)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import save_json


@dataclass
class SparseConfig:
    data_csv: Path
    threshold: float = 0.02
    out_path: Path = Path("models/sparse/residual_sindy.json")


def _library(theta: np.ndarray, omega: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, list[str]]:
    terms = [
        np.ones_like(theta),
        theta,
        omega,
        u,
        np.sin(theta),
        omega * np.abs(omega),
        u * np.abs(u),
    ]
    names = ["1", "theta", "omega", "u", "sin(theta)", "omega*abs(omega)", "u*abs(u)"]
    return np.column_stack(terms), names


def fit_sparse_residual(cfg: SparseConfig) -> Path:
    """Fit sparse equation for residual torque from simulator mismatch."""
    df = pd.read_csv(cfg.data_csv)
    req = {"theta", "omega", "u", "tau_residual"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(req - set(df.columns))}")

    x, names = _library(df["theta"].to_numpy(), df["omega"].to_numpy(), df["u"].to_numpy())
    y = df["tau_residual"].to_numpy()
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    coeff[np.abs(coeff) < cfg.threshold] = 0.0

    active = [{"term": n, "coef": float(c)} for n, c in zip(names, coeff) if c != 0.0]
    payload = {
        "method": "STLSQ",
        "threshold": cfg.threshold,
        "equation": "tau_residual = sum_i coef_i * term_i",
        "active_terms": active,
        "all_terms": [{"term": n, "coef": float(c)} for n, c in zip(names, coeff)],
        "config": asdict(cfg) | {"data_csv": str(cfg.data_csv), "out_path": str(cfg.out_path)},
    }
    save_json(cfg.out_path, payload)
    return cfg.out_path
