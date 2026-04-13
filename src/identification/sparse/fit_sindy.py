"""Pipeline B-2: sparse residual equation identification via PySINDy."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pysindy as ps

from ..common import save_json


@dataclass
class SparseConfig:
    data_csv: Path
    threshold: float = 0.02
    out_path: Path = Path("models/sparse/residual_sindy.json")


def fit_sparse_residual(cfg: SparseConfig) -> Path:
    """Fit sparse equation for residual torque from simulator mismatch."""
    df = pd.read_csv(cfg.data_csv)
    req = {"theta", "omega", "u", "tau_residual"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(req - set(df.columns))}")

    x = df[["theta", "omega", "u"]].to_numpy()
    y = df["tau_residual"].to_numpy().reshape(-1, 1)

    library = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
    optimizer = ps.SINDyPI(threshold=cfg.threshold)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=library,
        feature_names=["theta", "omega", "u"],
    )
    model.fit(x, x_dot=y)

    coefficients = model.coefficients()[0]
    names = model.get_feature_names()
    active = [{"term": n, "coef": float(c)} for n, c in zip(names, coefficients) if c != 0.0]
    payload = {
        "method": "PySINDy-SINDyPI",
        "threshold": cfg.threshold,
        "equation": "tau_residual = sum_i coef_i * term_i",
        "active_terms": active,
        "all_terms": [{"term": n, "coef": float(c)} for n, c in zip(names, coefficients)],
        "config": asdict(cfg) | {"data_csv": str(cfg.data_csv), "out_path": str(cfg.out_path)},
    }
    save_json(cfg.out_path, payload)
    return cfg.out_path
