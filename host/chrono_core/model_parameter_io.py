import json
import os
from typing import Any

from .config import BridgeConfig


def load_model_parameter_json(json_path: str | None) -> dict[str, Any] | None:
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_runtime_overrides(data: dict[str, Any] | None, cfg: BridgeConfig) -> dict[str, Any]:
    """
    Parse model parameter JSON and return runtime overrides for Chrono execution.
    Supports both:
      - new schema (`known`, `torque_model`)
      - legacy schema (`model_init`, `best_params`, flat keys)
    """
    if not isinstance(data, dict):
        return {}

    out: dict[str, Any] = {}

    known = data.get("known", {}) if isinstance(data.get("known"), dict) else {}
    if "gravity_mps2" in known:
        out["gravity"] = float(known["gravity_mps2"])
    if "r_imu_m" in known:
        out["r_imu"] = float(known["r_imu_m"])

    torque_model = data.get("torque_model", {}) if isinstance(data.get("torque_model"), dict) else {}
    motor = torque_model.get("motor", {}) if isinstance(torque_model.get("motor"), dict) else {}
    resistance = torque_model.get("resistance", {}) if isinstance(torque_model.get("resistance"), dict) else {}
    motor_params = motor.get("params", {}) if isinstance(motor.get("params"), dict) else {}
    res_params = resistance.get("params", {}) if isinstance(resistance.get("params"), dict) else {}

    if "K_i" in motor_params:
        out["K_i"] = float(motor_params["K_i"])
    if "b_eq" in res_params:
        out["b_eq"] = float(res_params["b_eq"])
    if "tau_eq" in res_params:
        out["tau_eq"] = float(res_params["tau_eq"])
    if isinstance(torque_model.get("residual_terms"), list):
        out["residual_terms"] = list(torque_model["residual_terms"])

    legacy = data.get("model_init", data.get("best_params", data))
    if isinstance(legacy, dict):
        if "K_i" in legacy and "K_i" not in out:
            out["K_i"] = float(legacy["K_i"])
        if "b_eq" in legacy and "b_eq" not in out:
            out["b_eq"] = float(legacy["b_eq"])
        if "tau_eq" in legacy and "tau_eq" not in out:
            out["tau_eq"] = float(legacy["tau_eq"])
        if "r_imu" in legacy and "r_imu" not in out:
            out["r_imu"] = float(legacy["r_imu"])

    # defaults
    out.setdefault("K_i", float(cfg.K_i_init))
    out.setdefault("b_eq", float(cfg.b_eq_init))
    out.setdefault("tau_eq", float(cfg.tau_eq_init))
    return out
