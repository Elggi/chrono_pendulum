#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str], n: int) -> np.ndarray:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    return np.full(n, np.nan, dtype=float)


def _safe_time(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    t = _pick_col(df, ["wall_elapsed", "t", "time", "time_sec"], n)
    if np.isfinite(t).sum() >= 2:
        t0 = t[np.isfinite(t)][0]
        t = t - t0
        return t
    return np.arange(n, dtype=float) * 0.01


@dataclass
class Stage2Trajectory:
    name: str
    source_csv: str
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    motor_input_a: np.ndarray


def load_trajectory(csv_path: Path) -> Stage2Trajectory:
    df = pd.read_csv(csv_path)
    n = len(df)
    if n < 4:
        raise ValueError(f"Too few samples in {csv_path}: {n}")

    t = _safe_time(df)
    theta = _pick_col(df, ["theta_real", "theta_imu_filtered_unwrapped", "theta", "theta_imu"], n)
    omega = _pick_col(df, ["omega_real", "omega_imu_filtered", "omega", "omega_imu"], n)
    alpha = _pick_col(df, ["alpha_real", "alpha_from_linear_accel_filtered", "alpha", "alpha_linear"], n)

    # Actual current input priority (A)
    current_ma = _pick_col(
        df,
        ["I_filtered_mA", "ina_current_corr_mA", "ina_current_raw_mA", "current_mA", "ina_current_signed_online_mA"],
        n,
    )
    motor_input_a = current_ma / 1000.0

    if not np.isfinite(alpha).any():
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = dt[1]
        alpha = np.gradient(omega, np.maximum(dt, 1e-6))

    good = (
        np.isfinite(t)
        & np.isfinite(theta)
        & np.isfinite(omega)
        & np.isfinite(alpha)
        & np.isfinite(motor_input_a)
    )
    if int(np.sum(good)) < 4:
        raise ValueError(f"Not enough finite samples in {csv_path}")

    t = t[good]
    t = t - t[0]
    theta = theta[good]
    omega = omega[good]
    alpha = alpha[good]
    motor_input_a = motor_input_a[good]

    return Stage2Trajectory(
        name=csv_path.stem,
        source_csv=str(csv_path),
        t=t,
        theta=theta,
        omega=omega,
        alpha=alpha,
        motor_input_a=motor_input_a,
    )


def load_trajectories(csv_paths: list[Path]) -> list[Stage2Trajectory]:
    out = []
    for p in csv_paths:
        out.append(load_trajectory(p))
    if not out:
        raise ValueError("No Stage2 trajectories loaded")
    return out

