import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BridgeConfig

RUNTIME_CSV_COLUMNS = [
    "wall_time", "wall_elapsed", "sim_time", "mode",
    "cmd_u_raw", "cmd_u_delayed", "hw_pwm", "delay_sec_est", "tau_cmd",
    "theta", "omega", "alpha",
    "hw_enc", "hw_arduino_ms",
    "theta_real", "omega_real", "alpha_real",
    "delay_ms",
    "l_com_est", "b_eq_est", "tau_eq_est", "k_t_est", "i0_est", "R_est", "k_e_est",
    "bus_v_raw", "bus_v_filtered", "current_raw_A", "current_filtered_A", "power_raw_W",
    "tau_motor", "tau_res", "tau_visc", "tau_coul", "i_pred", "v_applied",
    "inst_cost", "best_cost_so_far",
    "imu_qw", "imu_qx", "imu_qy", "imu_qz",
    "imu_wx", "imu_wy", "imu_wz",
    "imu_ax", "imu_ay", "imu_az",
    "ls_cost", "fit_done", "fit_complete", "fit_final_params",
]

PARAM_NAMES_DEFAULT = ["l_com", "J_cm_base", "b_eq", "tau_eq", "k_t", "i0", "R", "k_e"]


@dataclass
class ReplayTrajectory:
    path: str
    t: np.ndarray
    dt: np.ndarray
    cmd_u: np.ndarray
    hw_pwm: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    bus_v: np.ndarray
    current: np.ndarray
    power: np.ndarray
    delay_sec_est: float



def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gather_csv_paths(single_csv: str | None, csv_dir: str | None):
    out = []
    if single_csv:
        out.append(single_csv)
    if csv_dir:
        for p in sorted(Path(csv_dir).glob("*.csv")):
            out.append(str(p))
    uniq = []
    seen = set()
    for p in out:
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    if not uniq:
        raise FileNotFoundError("No training csv provided. Use --csv and/or --csv_dir.")
    return uniq


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_csv_dict(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        raise ValueError(f"Empty csv: {path}")
    cols = {k: np.array([_to_float(r.get(k, "")) for r in rows], dtype=float) for k in rdr.fieldnames}
    return cols


def _pick_col(cols: dict, names: list[str], n: int, fill=np.nan):
    for name in names:
        if name in cols:
            return np.asarray(cols[name], dtype=float)
    return np.full(n, fill, dtype=float)


def _safe_gradient(x: np.ndarray, t: np.ndarray):
    if len(x) < 3:
        return np.zeros_like(x)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    pos = dt > 0
    dt[~pos] = np.median(dt[pos]) if np.any(pos) else 1e-3
    return np.gradient(x, dt)


def estimate_delay_sec(cmd_u: np.ndarray, hw_pwm: np.ndarray, dt_med: float, max_delay_sec: float = 0.2):
    if len(cmd_u) < 10 or len(hw_pwm) < 10 or not np.isfinite(cmd_u).any() or not np.isfinite(hw_pwm).any():
        return 0.0
    u = np.nan_to_num(cmd_u - np.nanmean(cmd_u))
    y = np.nan_to_num(hw_pwm - np.nanmean(hw_pwm))
    max_lag = int(max(1, round(max_delay_sec / max(dt_med, 1e-4))))
    best_lag = 0
    best = -1e18
    for lag in range(0, max_lag + 1):
        if lag >= len(u) - 2:
            break
        a = u[:-lag] if lag > 0 else u
        b = y[lag:]
        score = float(np.dot(a, b) / max(len(a), 1))
        if score > best:
            best = score
            best_lag = lag
    return float(best_lag * dt_med)


def load_replay_trajectory(path: str, delay_override: float | None = None):
    cols = load_csv_dict(path)
    n = len(next(iter(cols.values())))
    t = _pick_col(cols, ["sim_time", "wall_elapsed", "wall_time"], n)
    if "wall_time" in cols and not np.isfinite(t).all():
        wt = cols["wall_time"]
        t = wt - wt[0]
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1e-3
    cmd_u = _pick_col(cols, ["cmd_u_raw", "cmd_u", "cmd_u_delayed"], n, 0.0)
    hw_pwm = _pick_col(cols, ["hw_pwm"], n, 0.0)
    theta = _pick_col(cols, ["theta_real", "est_theta", "theta"], n, 0.0)
    omega = _pick_col(cols, ["omega_real", "est_omega", "omega"], n, np.nan)
    if not np.isfinite(omega).any():
        omega = _safe_gradient(theta, t)
    alpha = _pick_col(cols, ["alpha_real", "est_alpha", "alpha"], n, np.nan)
    if not np.isfinite(alpha).any():
        alpha = _safe_gradient(omega, t)
    bus_v = _pick_col(cols, ["bus_v_filtered", "bus_v_raw"], n, np.nan)
    bus_v[~np.isfinite(bus_v)] = 7.4
    current = _pick_col(cols, ["current_filtered_A", "current_raw_A"], n, np.nan)
    power = _pick_col(cols, ["power_raw_W"], n, np.nan)
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1e-3
    d_est = estimate_delay_sec(cmd_u, hw_pwm, dt_med)
    if delay_override is not None:
        d_est = float(delay_override)
    return ReplayTrajectory(
        path=path,
        t=t,
        dt=dt,
        cmd_u=cmd_u,
        hw_pwm=hw_pwm,
        theta=theta,
        omega=omega,
        alpha=alpha,
        bus_v=bus_v,
        current=current,
        power=power,
        delay_sec_est=d_est,
    )


def build_params_from_calibration(cfg: BridgeConfig, calibration_json: str, parameter_json: str | None = None):
    from .calibration_io import apply_calibration_json

    apply_calibration_json(cfg, calibration_json)
    params = {
        "l_com": float(cfg.l_com_init),
        "J_cm_base": float(cfg.J_cm_base),
        "b_eq": float(cfg.b_eq_init),
        "tau_eq": float(cfg.tau_eq_init),
        "k_t": float(cfg.k_t_init),
        "i0": float(cfg.i0_init),
        "R": float(cfg.R_init),
        "k_e": float(cfg.k_e_init),
    }
    if parameter_json and os.path.exists(parameter_json):
        data = load_json(parameter_json)
        src = data.get("model_init", data)
        for k in list(params.keys()) + ["delay_sec"]:
            if k in src:
                params[k] = float(src[k])
    return params


def write_runtime_csv(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(RUNTIME_CSV_COLUMNS)
        for row in rows:
            wr.writerow([row.get(c, "") for c in RUNTIME_CSV_COLUMNS])
