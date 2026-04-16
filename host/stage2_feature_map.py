#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


FeatureFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _ones(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = omega, motor_input
    return np.ones_like(theta, dtype=float)


def _theta(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = omega, motor_input
    return np.asarray(theta, dtype=float)


def _omega(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = theta, motor_input
    return np.asarray(omega, dtype=float)


def _sin_theta(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = omega, motor_input
    return np.sin(np.asarray(theta, dtype=float))


def _cos_theta(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = omega, motor_input
    return np.cos(np.asarray(theta, dtype=float))


def _theta2(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = omega, motor_input
    th = np.asarray(theta, dtype=float)
    return th * th


def _omega2(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = theta, motor_input
    om = np.asarray(omega, dtype=float)
    return om * om


def _sign_omega(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = theta, motor_input
    om = np.asarray(omega, dtype=float)
    out = np.zeros_like(om)
    out[om > 0.0] = 1.0
    out[om < 0.0] = -1.0
    return out


def _motor_input(theta: np.ndarray, omega: np.ndarray, motor_input: np.ndarray) -> np.ndarray:
    _ = theta, omega
    return np.asarray(motor_input, dtype=float)


FEATURE_REGISTRY: dict[str, FeatureFn] = {
    "1": _ones,
    "theta": _theta,
    "omega": _omega,
    "sin_theta": _sin_theta,
    "cos_theta": _cos_theta,
    "theta2": _theta2,
    "omega2": _omega2,
    "sign_omega": _sign_omega,
    "motor_input": _motor_input,
}

# Conservative default library for rollout-safe Stage2 (derivative-based residual dynamics).
DEFAULT_FEATURES = ["omega", "motor_input", "omega2", "theta"]


@dataclass
class FeatureMatrix:
    names: list[str]
    phi: np.ndarray


def build_feature_matrix(
    theta: np.ndarray,
    omega: np.ndarray,
    motor_input: np.ndarray,
    feature_names: list[str],
) -> FeatureMatrix:
    cols = []
    names = []
    for name in feature_names:
        fn = FEATURE_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown feature name: {name}. available={sorted(FEATURE_REGISTRY.keys())}")
        col = np.asarray(fn(theta, omega, motor_input), dtype=float).reshape(-1)
        cols.append(col)
        names.append(name)
    if not cols:
        raise ValueError("No feature columns selected")
    phi = np.column_stack(cols).astype(float)
    return FeatureMatrix(names=names, phi=phi)
