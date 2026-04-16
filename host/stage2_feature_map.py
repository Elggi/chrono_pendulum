#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Requested default Stage2 residual library (no constant term).
DEFAULT_FEATURES = [
    "motor_input",
    "theta",
    "omega",
    "theta2",
    "omega2",
    "sin_theta",
    "tanh_omega_eps",
    "motor_input_omega",
]


@dataclass
class FeatureMatrix:
    names: list[str]
    phi: np.ndarray


def build_feature_matrix(
    theta: np.ndarray,
    omega: np.ndarray,
    motor_input: np.ndarray,
    feature_names: list[str],
    *,
    eps: float,
) -> FeatureMatrix:
    th = np.asarray(theta, dtype=float).reshape(-1)
    om = np.asarray(omega, dtype=float).reshape(-1)
    mi = np.asarray(motor_input, dtype=float).reshape(-1)
    if not (len(th) == len(om) == len(mi)):
        raise ValueError(f"Feature input length mismatch: len(theta)={len(th)}, len(omega)={len(om)}, len(motor_input)={len(mi)}")
    teps = max(float(eps), 1e-9)

    cols: list[np.ndarray] = []
    names: list[str] = []
    for name in feature_names:
        key = str(name).strip()
        if key == "1":
            raise ValueError("Constant feature '1' is forbidden for Stage2 residual identification.")
        if key == "motor_input":
            col = mi
        elif key == "theta":
            col = th
        elif key == "omega":
            col = om
        elif key == "theta2":
            col = th * th
        elif key == "omega2":
            col = om * om
        elif key == "sin_theta":
            col = np.sin(th)
        elif key == "tanh_omega_eps":
            col = np.tanh(om / teps)
        elif key == "motor_input_omega":
            col = mi * om
        else:
            raise ValueError(
                f"Unknown feature '{key}'. Allowed={DEFAULT_FEATURES}. "
                "No implicit extra features are permitted."
            )
        cols.append(np.asarray(col, dtype=float))
        names.append(key)

    if not cols:
        raise ValueError("No feature columns selected.")
    phi = np.column_stack(cols).astype(float)
    return FeatureMatrix(names=names, phi=phi)
