#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps


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


def _infer_current_a(df: pd.DataFrame, n: int) -> np.ndarray:
    # Prefer explicit ampere channels first.
    for c in ["I_filtered_A", "ina_current_a", "current_A"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    # Otherwise consume mA channels and convert to A.
    current_ma = _pick_col(
        df,
        ["ina_current_signed_online_mA", "I_filtered_mA", "ina_current_corr_mA", "ina_current_raw_mA", "current_mA"],
        n,
    )
    return current_ma / 1000.0


def _compute_alpha_from_omega_smoothed(omega: np.ndarray, t: np.ndarray) -> np.ndarray:
    omega_2d = np.asarray(omega, dtype=float).reshape(-1, 1)
    t_1d = np.asarray(t, dtype=float).reshape(-1)
    if omega_2d.shape[0] != t_1d.shape[0]:
        raise ValueError("omega/time length mismatch while building alpha")
    sfd = ps.SmoothedFiniteDifference()
    if hasattr(sfd, "_differentiate"):
        alpha_2d = sfd._differentiate(omega_2d, t_1d)  # PySINDy API path
    else:
        alpha_2d = sfd(omega_2d, t_1d)
    alpha = np.asarray(alpha_2d, dtype=float).reshape(-1)
    return alpha


def load_trajectory(csv_path: Path) -> Stage2Trajectory:
    df = pd.read_csv(csv_path)
    n = len(df)
    if n < 4:
        raise ValueError(f"Too few samples in {csv_path}: {n}")

    t = _safe_time(df)
    theta = _pick_col(df, ["theta_real", "theta_imu_filtered_unwrapped", "theta", "theta_imu"], n)
    omega = _pick_col(df, ["omega_real", "omega_imu_filtered", "omega", "omega_imu"], n)

    motor_input_a = _infer_current_a(df, n)

    good = np.isfinite(t) & np.isfinite(theta) & np.isfinite(omega) & np.isfinite(motor_input_a)
    if int(np.sum(good)) < 4:
        raise ValueError(f"Not enough finite samples in {csv_path}")

    t = t[good]
    theta = theta[good]
    omega = omega[good]
    motor_input_a = motor_input_a[good]
    order = np.argsort(t)
    t = t[order]
    theta = theta[order]
    omega = omega[order]
    motor_input_a = motor_input_a[order]
    t = t - t[0]

    # Stage2 policy: never use CSV alpha for identification.
    alpha = _compute_alpha_from_omega_smoothed(omega=omega, t=t)
    alpha_good = np.isfinite(alpha)
    if int(np.sum(alpha_good)) < 4:
        raise ValueError(f"SmoothedFiniteDifference alpha has too few finite samples in {csv_path}")
    t = t[alpha_good]
    theta = theta[alpha_good]
    omega = omega[alpha_good]
    motor_input_a = motor_input_a[alpha_good]
    alpha = alpha[alpha_good]

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
