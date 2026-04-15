#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chrono_core.config import BridgeConfig
from chrono_core.model_parameter_io import extract_runtime_overrides
from stage2_dataset import Stage2Trajectory


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


def known_params_from_model_json(model_data: dict[str, Any] | None, cfg: BridgeConfig) -> KnownParams:
    known = model_data.get("known", {}) if isinstance(model_data, dict) and isinstance(model_data.get("known"), dict) else {}
    torque_model = (
        model_data.get("torque_model", {})
        if isinstance(model_data, dict) and isinstance(model_data.get("torque_model"), dict)
        else {}
    )
    resistance = torque_model.get("resistance", {}) if isinstance(torque_model.get("resistance"), dict) else {}
    res_params = resistance.get("params", {}) if isinstance(resistance.get("params"), dict) else {}

    runtime = extract_runtime_overrides(model_data, cfg)
    m_total = float(known.get("mass_total_kg", cfg.rod_mass + cfg.imu_mass))
    j_total = float(
        known.get(
            "inertia_total_kgm2",
            (cfg.rod_mass * (cfg.link_L ** 2) / 3.0) + (cfg.imu_mass * (cfg.r_imu ** 2)),
        )
    )
    l_com = float(known.get("l_com_total_m", cfg.l_com_init))
    g = float(known.get("gravity_mps2", cfg.gravity))
    K_i = float(runtime.get("K_i", cfg.K_i_init))
    b_eq = float(runtime.get("b_eq", cfg.b_eq_init))
    tau_eq = float(runtime.get("tau_eq", cfg.tau_eq_init))
    eps = float(res_params.get("eps", cfg.tanh_eps))
    return KnownParams(
        m_total=m_total,
        j_total=j_total,
        l_com=l_com,
        g=g,
        K_i=K_i,
        b_eq=b_eq,
        tau_eq=tau_eq,
        eps=max(eps, 1e-9),
    )


@dataclass
class ResidualTarget:
    tau_total_target: np.ndarray
    tau_motor: np.ndarray
    tau_gravity: np.ndarray
    tau_visc: np.ndarray
    tau_coul: np.ndarray
    tau_residual_target: np.ndarray


def build_residual_target(traj: Stage2Trajectory, p: KnownParams) -> ResidualTarget:
    theta = np.asarray(traj.theta, dtype=float)
    omega = np.asarray(traj.omega, dtype=float)
    alpha = np.asarray(traj.alpha, dtype=float)
    motor_input = np.asarray(traj.motor_input_a, dtype=float)

    tau_total = float(p.j_total) * alpha
    tau_motor = float(p.K_i) * motor_input
    tau_gravity = float(p.m_total * p.g * p.l_com) * np.sin(theta)
    tau_visc = float(p.b_eq) * omega
    tau_coul = float(p.tau_eq) * np.tanh(omega / float(p.eps))
    tau_residual_target = tau_motor - tau_gravity - tau_visc - tau_coul - tau_total

    return ResidualTarget(
        tau_total_target=tau_total,
        tau_motor=tau_motor,
        tau_gravity=tau_gravity,
        tau_visc=tau_visc,
        tau_coul=tau_coul,
        tau_residual_target=tau_residual_target,
    )

