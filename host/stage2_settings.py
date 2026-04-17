#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------
# Stage2 single-source config
# ---------------------------

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
class KnownParams:
    m_total: float
    j_total: float
    l_com: float
    g: float
    K_i: float
    b_eq: float
    tau_eq: float
    eps: float


@dataclass
class ResidualTarget:
    tau_total_target: np.ndarray
    tau_motor: np.ndarray
    tau_gravity: np.ndarray
    tau_visc: np.ndarray
    tau_coul: np.ndarray
    tau_residual_target: np.ndarray


def known_params_from_model_json(model_data: dict[str, Any] | None) -> KnownParams:
    data = model_data if isinstance(model_data, dict) else {}
    known = data.get("known", {}) if isinstance(data.get("known"), dict) else {}
    tm = data.get("torque_model", {}) if isinstance(data.get("torque_model"), dict) else {}
    motor = tm.get("motor", {}) if isinstance(tm.get("motor"), dict) else {}
    resistance = tm.get("resistance", {}) if isinstance(tm.get("resistance"), dict) else {}
    mp = motor.get("params", {}) if isinstance(motor.get("params"), dict) else {}
    rp = resistance.get("params", {}) if isinstance(resistance.get("params"), dict) else {}

    return KnownParams(
        m_total=float(known.get("mass_total_kg", 0.22)),
        j_total=float(known.get("inertia_total_kgm2", 0.00666293085698)),
        l_com=float(known.get("l_com_total_m", 0.15225390909090908)),
        g=float(known.get("gravity_mps2", 9.81)),
        K_i=float(mp.get("K_i", 0.0)),
        b_eq=float(rp.get("b_eq", 0.01922636540411224)),
        tau_eq=float(rp.get("tau_eq", 0.014448922020706325)),
        eps=max(float(rp.get("eps", 0.0005)), 1e-9),
    )


def parse_feature_list(raw_features: list[str]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    out = [str(s).strip() for s in raw_features if str(s).strip()]
    if "1" in out:
        warnings.append("feature '1' is forbidden and removed.")
        out = [s for s in out if s != "1"]
    unknown = [s for s in out if s not in DEFAULT_FEATURES]
    if unknown:
        warnings.append(f"unsupported features removed: {unknown}")
        out = [s for s in out if s in DEFAULT_FEATURES]
    if not out:
        out = list(DEFAULT_FEATURES)
        warnings.append(f"fallback to default features: {out}")
    return out, warnings


def build_feature_matrix(
    theta: np.ndarray,
    omega: np.ndarray,
    motor_input: np.ndarray,
    feature_names: list[str],
    *,
    eps: float,
) -> tuple[list[str], np.ndarray]:
    th = np.asarray(theta, dtype=float).reshape(-1)
    om = np.asarray(omega, dtype=float).reshape(-1)
    mi = np.asarray(motor_input, dtype=float).reshape(-1)
    if not (len(th) == len(om) == len(mi)):
        raise ValueError("feature input length mismatch")
    teps = max(float(eps), 1e-9)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for key in feature_names:
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
            raise ValueError(f"unknown feature '{key}'")
        names.append(key)
        cols.append(np.asarray(col, dtype=float))
    return names, np.column_stack(cols).astype(float)


def compute_residual_target(
    theta: np.ndarray,
    omega: np.ndarray,
    alpha: np.ndarray,
    motor_input: np.ndarray,
    p: KnownParams,
    target_mode: str = "greybox",
) -> ResidualTarget:
    th = np.asarray(theta, dtype=float)
    om = np.asarray(omega, dtype=float)
    al = np.asarray(alpha, dtype=float)
    mi = np.asarray(motor_input, dtype=float)
    tau_total = float(p.j_total) * al
    tau_motor = float(p.K_i) * mi
    tau_gravity = float(p.m_total * p.g * p.l_com) * np.sin(th)
    tau_visc = float(p.b_eq) * om
    tau_coul = float(p.tau_eq) * np.tanh(om / float(p.eps))
    mode = str(target_mode).strip().lower()
    if mode == "blackbox":
        # Identify external torque required to reproduce measured acceleration,
        # with gravity explicitly canceled for Chrono torque-injection usage.
        tau_residual_target = tau_total + tau_gravity
    elif mode == "greybox":
        # Subtract Stage1 nominal (motor + friction), with gravity canceled.
        tau_residual_target = tau_total + tau_visc + tau_coul + tau_gravity - tau_motor
    else:
        raise ValueError(f"Unknown target_mode='{target_mode}'. allowed=['greybox','blackbox']")
    return ResidualTarget(
        tau_total_target=tau_total,
        tau_motor=tau_motor,
        tau_gravity=tau_gravity,
        tau_visc=tau_visc,
        tau_coul=tau_coul,
        tau_residual_target=tau_residual_target,
    )


def evaluate_residual_from_terms(theta: float, omega: float, motor_input: float, residual_terms: list[dict[str, float]], eps: float) -> float:
    th = float(theta)
    om = float(omega)
    mi = float(motor_input)
    teps = max(float(eps), 1e-9)
    out = 0.0
    for term in residual_terms:
        feat = str(term.get("feature", ""))
        c = float(term.get("coeff", 0.0))
        if feat == "motor_input":
            out += c * mi
        elif feat == "theta":
            out += c * th
        elif feat == "omega":
            out += c * om
        elif feat == "theta2":
            out += c * (th ** 2)
        elif feat == "omega2":
            out += c * (om ** 2)
        elif feat == "sin_theta":
            out += c * float(np.sin(th))
        elif feat == "tanh_omega_eps":
            out += c * float(np.tanh(om / teps))
        elif feat == "motor_input_omega":
            out += c * (mi * om)
    return float(out)
