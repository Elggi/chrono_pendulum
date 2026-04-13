"""Validation metrics for one-step and rollout sim-to-real error."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def one_step_error(df: pd.DataFrame, pred_theta: np.ndarray, pred_omega: np.ndarray) -> dict[str, float]:
    return {
        "theta_rmse": rmse(df["theta_next"].to_numpy(), pred_theta),
        "omega_rmse": rmse(df["omega_next"].to_numpy(), pred_omega),
    }


def rollout_error(real: pd.DataFrame, sim: pd.DataFrame) -> dict[str, float]:
    return {
        "theta_rollout_rmse": rmse(real["theta"].to_numpy(), sim["theta"].to_numpy()),
        "omega_rollout_rmse": rmse(real["omega"].to_numpy(), sim["omega"].to_numpy()),
    }
