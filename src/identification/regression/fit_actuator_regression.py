"""Pipeline B-1: interpretable regression-based actuator law fitting."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import save_json


@dataclass
class RegressionConfig:
    data_csv: Path
    out_path: Path = Path("models/actuator/actuator_regression.json")


def fit_linear_actuator(cfg: RegressionConfig) -> Path:
    """Fit tau = k_u*u + k_w*omega + k_c*sign(omega) + b."""
    df = pd.read_csv(cfg.data_csv)
    req = {"u", "omega", "tau_eff"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(req - set(df.columns))}")

    phi = np.column_stack([
        df["u"].to_numpy(),
        df["omega"].to_numpy(),
        np.sign(df["omega"].to_numpy()),
        np.ones(len(df)),
    ])
    y = df["tau_eff"].to_numpy()
    coeff, *_ = np.linalg.lstsq(phi, y, rcond=None)

    payload = {
        "equation": "tau = k_u*u + k_w*omega + k_c*sign(omega) + b",
        "k_u": float(coeff[0]),
        "k_w": float(coeff[1]),
        "k_c": float(coeff[2]),
        "b": float(coeff[3]),
        "config": asdict(cfg) | {"data_csv": str(cfg.data_csv), "out_path": str(cfg.out_path)},
    }
    save_json(cfg.out_path, payload)
    return cfg.out_path
