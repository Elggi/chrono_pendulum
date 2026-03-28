import json
import math
import os

from .config import BridgeConfig


def apply_calibration_json(cfg: BridgeConfig, json_path: str | None):
    if not json_path or not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    model_init = calib.get("model_init", {})
    if not model_init and "best_params" in calib:
        # RL_fitting result schema compatibility
        model_init = calib.get("best_params", {})
        if "Rm" in model_init and "R" not in model_init:
            model_init["R"] = model_init["Rm"]
    delay = calib.get("delay", {})
    summary = calib.get("summary", {}) if isinstance(calib.get("summary", {}), dict) else {}

    cfg.J_init = float(model_init.get("J", cfg.J_init))
    cfg.b_init = float(model_init.get("b", cfg.b_init))
    cfg.tau_c_init = float(model_init.get("tau_c", cfg.tau_c_init))
    cfg.mgl_init = float(model_init.get("mgl", cfg.mgl_init))
    cfg.k_t_init = float(model_init.get("k_t", cfg.k_t_init))
    cfg.i0_init = float(model_init.get("i0", cfg.i0_init))
    cfg.R_init = float(model_init.get("R", cfg.R_init))
    cfg.k_e_init = float(model_init.get("k_e", cfg.k_e_init))
    cfg.delay_init_ms = float(delay.get("effective_control_delay_ms", cfg.delay_init_ms))
    cpr_candidates = [
        summary.get("mean_cpr"),
        calib.get("mean_cpr"),
        summary.get("cpr"),
        calib.get("cpr"),
    ]
    for cpr in cpr_candidates:
        try:
            cpr = float(cpr)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cpr) and cpr > 1.0:
            cfg.cpr = cpr
            break

    cfg.calibration_json = json_path
    return calib


def extract_radius_from_json(json_path: str | None) -> float | None:
    if not json_path or not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {}) if isinstance(data.get("summary", {}), dict) else {}
    radius_candidates = [
        summary.get("mean_radius_m"),
        summary.get("r_m"),
        data.get("mean_radius_m"),
        data.get("r_from_imu_orientation"),
    ]
    for radius in radius_candidates:
        if radius is None:
            continue
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            continue
        if math.isfinite(radius) and radius > 0.0:
            return radius
    return None
