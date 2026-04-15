import json
import math
import os

from .config import BridgeConfig


def apply_calibration_json(cfg: BridgeConfig, json_path: str | None, apply_model_init: bool = False):
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
    if not model_init and isinstance(calib.get("best_eval"), dict):
        # chrono_run_*.meta.json compatibility
        best_eval = calib.get("best_eval", {})
        if isinstance(best_eval.get("params"), dict):
            model_init = dict(best_eval["params"])
    if not model_init and isinstance(calib.get("fit_final_params"), dict):
        model_init = dict(calib["fit_final_params"])
    summary = calib.get("summary", {}) if isinstance(calib.get("summary", {}), dict) else {}
    if not isinstance(calib.get("summary"), dict):
        calib["summary"] = summary

    # Keep fresh surrogate init values unless explicitly requested otherwise.
    # This prevents calibration JSON model fields from silently overriding
    # initialization before staged regression / RL finds new parameters.
    if apply_model_init:
        cfg.l_com_init = float(model_init.get("l_com", cfg.l_com_init))
        cfg.b_eq_init = float(model_init.get("b_eq", model_init.get("b", cfg.b_eq_init)))
        cfg.tau_eq_init = float(model_init.get("tau_eq", model_init.get("tau_c", cfg.tau_eq_init)))
        cfg.K_i_init = float(model_init.get("K_i", model_init.get("K_u", model_init.get("k_u", cfg.K_i_init))))
    cfg.r_imu = float(
        model_init.get(
            "r_imu",
            summary.get("mean_radius_m", calib.get("mean_radius_m", cfg.r_imu)),
        )
    )
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

    # Extend/normalize summary for staged calibration workflows.
    summary.setdefault("mean_radius_m", float(cfg.r_imu))
    if math.isfinite(float(cfg.cpr)):
        summary.setdefault("mean_cpr", float(cfg.cpr))
    g_candidates = [
        summary.get("g_eff_mps2"),
        summary.get("gravity_mps2"),
        summary.get("gravity"),
        calib.get("gravity_mps2"),
        calib.get("gravity"),
        calib.get("g"),
    ]
    for gv in g_candidates:
        try:
            g_eff = float(gv)
        except (TypeError, ValueError):
            continue
        if math.isfinite(g_eff) and g_eff > 0.0:
            cfg.gravity = g_eff
            break
    calib["summary"] = summary

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
