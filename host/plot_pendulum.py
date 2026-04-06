#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from chrono_core.signal_filter import estimate_filtered_alpha_from_omega

try:
    import pandas as pd
except Exception:
    pd = None


def find_latest_csv(folder: str):
    csvs = [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No csv files in {folder}")
    csvs.sort(key=os.path.getmtime)
    return csvs[-1]


def load_meta_if_exists(csv_path):
    meta_path = csv_path[:-4] + ".meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cpr_fallback(csv_path: str):
    folder = os.path.dirname(csv_path)
    candidates = [
        os.path.join(folder, "training_metadata.json"),
        os.path.join(os.path.dirname(folder), "training_metadata.json"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cal = meta.get("settings", {}).get("calibration_json") or meta.get("calibration_json")
            if cal and os.path.exists(cal):
                with open(cal, "r", encoding="utf-8") as cf:
                    calib = json.load(cf)
                sm = calib.get("summary", {})
                if sm.get("mean_cpr") is not None:
                    return float(sm["mean_cpr"])
        except Exception:
            continue
    return None


def moving_average(x: np.ndarray, win: int):
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")
    return y[: len(x)]


def offline_smooth(x: np.ndarray, win: int):
    w = max(5, int(win))
    if w % 2 == 0:
        w += 1
    try:
        from scipy.signal import savgol_filter  # type: ignore

        if len(x) >= w:
            return savgol_filter(x, window_length=w, polyorder=3, mode="interp")
    except Exception:
        pass
    return moving_average(x, max(3, w))


def safe_gradient(x: np.ndarray, t: np.ndarray):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(x) < 2 or len(t) < 2:
        return np.zeros_like(x, dtype=float)
    tc = t.copy()
    for i in range(1, len(tc)):
        if not np.isfinite(tc[i]) or tc[i] <= tc[i - 1]:
            tc[i] = tc[i - 1] + 1e-6
    return np.gradient(x, tc, edge_order=1)


def unwrap_and_zero(theta: np.ndarray):
    out = np.asarray(theta, dtype=float).copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    idx = np.where(finite)[0]
    unwrapped = np.unwrap(out[idx])
    unwrapped = unwrapped - unwrapped[0]
    out[idx] = unwrapped
    return out


def derive_theta_from_encoder(enc, counts_per_rev, sign=1.0, offset=0.0):
    return sign * (2.0 * np.pi / counts_per_rev) * (enc - enc[0]) + offset


def derive_theta_from_imu_quat(qx, qy, qz, qw):
    n = len(qw)
    theta = np.full(n, np.nan, dtype=float)
    valid0 = np.isfinite(qw) & np.isfinite(qx) & np.isfinite(qy) & np.isfinite(qz)
    idx = np.where(valid0)[0]
    if len(idx) == 0:
        return theta

    def rot(w, x, y, z):
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    i0 = int(idx[0])
    R0 = rot(qw[i0], qx[i0], qy[i0], qz[i0])
    prev = math.atan2(-1.0, 0.0)
    unwrapped = 0.0
    for i in range(i0, n):
        if not valid0[i]:
            continue
        R = rot(qw[i], qx[i], qy[i], qz[i])
        tip = (R0.T @ R) @ np.array([0.0, -1.0, 0.0], dtype=float)
        ang = math.atan2(float(tip[1]), float(tip[0]))
        d = ang - prev
        while d > np.pi:
            d -= 2.0 * np.pi
        while d < -np.pi:
            d += 2.0 * np.pi
        unwrapped += d
        prev = ang
        theta[i] = unwrapped

    valid = np.where(np.isfinite(theta))[0]
    if len(valid) > 0:
        theta[: valid[0]] = theta[valid[0]]
        for i in range(1, len(valid)):
            theta[valid[i - 1] : valid[i]] = theta[valid[i - 1]]
        theta[valid[-1] :] = theta[valid[-1]]
    return theta


class SimpleFrame:
    def __init__(self, columns):
        self._data = columns
        self.columns = list(columns.keys())

    def __getitem__(self, key):
        return self._data[key]

    def row(self, idx: int):
        return {k: self._data[k][idx] for k in self.columns}


def load_csv_frame(csv_path: str):
    if pd is not None:
        return pd.read_csv(csv_path)
    cols = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames or []
        for name in fieldnames:
            cols[name] = []
        for row in rdr:
            for k in fieldnames:
                v = row.get(k, "")
                try:
                    cols[k].append(float(v))
                except (TypeError, ValueError):
                    cols[k].append(np.nan)
    for k in list(cols.keys()):
        cols[k] = np.asarray(cols[k], dtype=float)
    return SimpleFrame(cols)


def col_to_numpy(df, key: str):
    if pd is not None and hasattr(df, "to_numpy"):
        return df[key].to_numpy(dtype=float)
    return np.asarray(df[key], dtype=float)


def col_any(df, keys, n_default=None):
    for k in keys:
        if k in df.columns:
            return col_to_numpy(df, k)
    if n_default is None:
        n_default = len(col_to_numpy(df, df.columns[0])) if len(df.columns) > 0 else 0
    return np.full(n_default, np.nan, dtype=float)


def normalize_theta_path(theta: np.ndarray) -> np.ndarray:
    """Canonical theta normalization: unwrap phase then zero-anchor at first finite sample [rad]."""
    return unwrap_and_zero(theta)


def robust_mad(x: np.ndarray) -> float:
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    v = x[finite]
    med = np.median(v)
    return float(np.median(np.abs(v - med)) * 1.4826)


def hf_roughness(x: np.ndarray, t: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(t)
    if np.sum(finite) < 8:
        return float("nan")
    xv = x[finite]
    tv = t[finite]
    dt = np.diff(tv)
    if len(dt) == 0:
        return float("nan")
    fs = 1.0 / max(np.median(dt), 1e-9)
    X = np.fft.rfft(xv - np.mean(xv))
    f = np.fft.rfftfreq(len(xv), d=1.0 / fs)
    if len(f) <= 2:
        return float("nan")
    cutoff = 0.35 * (fs * 0.5)
    idx = f >= cutoff
    if not np.any(idx):
        return float(0.0)
    return float(np.sum(np.abs(X[idx]) ** 2) / max(len(xv), 1))


def derivative_consistency(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> float:
    da = safe_gradient(a, t)
    finite = np.isfinite(da) & np.isfinite(b)
    if np.sum(finite) < 8:
        return float("nan")
    return float(np.sqrt(np.mean((da[finite] - b[finite]) ** 2)))


def evaluate_candidate_set(candidates: dict[str, np.ndarray], t: np.ndarray, quantity: str, rest_mask: np.ndarray | None, link_ref: np.ndarray | None):
    rows = []
    for name, sig in candidates.items():
        finite = np.isfinite(sig)
        if not np.any(finite):
            continue
        rest = rest_mask & finite if rest_mask is not None else finite
        std_rest = float(np.std(sig[rest])) if np.any(rest) else float(np.std(sig[finite]))
        mad = robust_mad(sig[rest] if np.any(rest) else sig[finite])
        rough = hf_roughness(sig, t)
        cons = float("nan")
        if link_ref is not None:
            cons = float(np.sqrt(np.nanmean((sig[finite] - link_ref[finite]) ** 2)))
        score = 0.0
        for v, w in [(std_rest, 1.0), (mad, 1.0), (rough, 0.2), (cons, 0.6)]:
            if np.isfinite(v):
                score += w * float(v)
        rows.append(
            {
                "quantity": quantity,
                "candidate": name,
                "std_rest": std_rest,
                "mad_rest": mad,
                "hf_roughness": rough,
                "consistency_rmse": cons,
                "score": score,
            }
        )
    rows = sorted(rows, key=lambda r: (np.inf if not np.isfinite(r["score"]) else r["score"]))
    winner = rows[0]["candidate"] if rows else None
    return rows, winner


def resolve_export_base(csv_path: str) -> tuple[str, bool]:
    """Return (raw_base_without_ext, is_processed_mode)."""
    path_no_ext, ext = os.path.splitext(csv_path)
    if ext.lower() != ".csv":
        return path_no_ext, False
    suffixes = [".offline_id", ".with_winners"]
    for s in suffixes:
        if path_no_ext.endswith(s):
            return path_no_ext[: -len(s)], True
    return path_no_ext, False


def imu_fixed_sign_flip(theta_imu: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Apply fixed IMU direction convention: CW/CCW inversion."""
    sign = -1.0
    return sign * theta_imu, sign, {"mode": "fixed_inversion"}


def plot_simulation(df, csv_path: str, args):
    meta = load_meta_if_exists(csv_path)
    export_base, processed_mode = resolve_export_base(csv_path)

    cpr = args.counts_per_revolution
    if cpr is None and meta is not None and meta.get("cpr_mean") is not None:
        cpr = float(meta["cpr_mean"])
    if cpr is None and meta is not None and meta.get("cpr_fixed") is not None:
        cpr = float(meta["cpr_fixed"])
    if cpr is None:
        cpr = load_cpr_fallback(csv_path)

    if "wall_elapsed" in df.columns:
        t = col_to_numpy(df, "wall_elapsed")
    elif "sim_time" in df.columns:
        # backward compatibility with old logs only
        t = col_to_numpy(df, "sim_time")
    elif "wall_time" in df.columns:
        tw = col_to_numpy(df, "wall_time")
        t = tw - tw[0]
    else:
        n = len(col_to_numpy(df, df.columns[0])) if len(df.columns) > 0 else 0
        t = np.arange(n, dtype=float)

    n = len(t)
    theta_sim = col_any(df, ["theta"], n)
    omega_sim = col_any(df, ["omega"], n)
    alpha_sim = col_any(df, ["alpha"], n)
    cmd_u = col_any(df, ["cmd_u_raw", "cmd_u"], n)
    hw_pwm = col_any(df, ["hw_pwm"], n)
    current_ma = col_any(df, ["current_mA", "ina219_current_ma", "ina_current_ma"], n)
    enc = col_any(df, ["hw_enc"], n)
    tau_cmd = col_any(df, ["tau_cmd"], n)
    tau_motor = col_any(df, ["tau_motor"], n)
    tau_visc = col_any(df, ["tau_visc"], n)
    tau_coul = col_any(df, ["tau_coul"], n)

    if not np.isfinite(theta_sim).any() and all(k in df.columns for k in ["imu_qx", "imu_qy", "imu_qz", "imu_qw"]):
        qx = col_to_numpy(df, "imu_qx")
        qy = col_to_numpy(df, "imu_qy")
        qz = col_to_numpy(df, "imu_qz")
        qw = col_to_numpy(df, "imu_qw")
        theta_sim = derive_theta_from_imu_quat(qx, qy, qz, qw)

    if not np.isfinite(omega_sim).any() and "imu_wz" in df.columns:
        omega_sim = col_to_numpy(df, "imu_wz")

    if np.isfinite(omega_sim).any():
        alpha_sim = estimate_filtered_alpha_from_omega(omega_sim, t=t)

    # ---- Provenance-safe candidates (legacy *_real paths are NOT trusted as canonical) ----
    theta_sim = normalize_theta_path(theta_sim)
    theta_imu_raw = normalize_theta_path(col_any(df, ["theta_imu"], n))
    theta_encoder_raw = normalize_theta_path(col_any(df, ["theta_encoder"], n))
    if not np.isfinite(theta_encoder_raw).any() and cpr is not None and np.isfinite(enc).any():
        theta_encoder_raw = normalize_theta_path(derive_theta_from_encoder(enc, cpr, sign=args.theta_sign, offset=args.theta_offset))
    theta_imu_online = normalize_theta_path(col_any(df, ["theta_imu_online"], n))
    theta_encoder_online = normalize_theta_path(col_any(df, ["theta_encoder_online"], n))
    theta_imu_offline = normalize_theta_path(offline_smooth(theta_imu_raw, win=max(11, args.real_theta_smooth * 5)))
    theta_encoder_offline = normalize_theta_path(offline_smooth(theta_encoder_raw, win=max(11, args.real_theta_smooth * 5)))

    omega_imu_direct = col_any(df, ["omega_imu"], n)
    omega_encoder_diff = col_any(df, ["omega_encoder"], n)
    if not np.isfinite(omega_encoder_diff).any() and np.isfinite(theta_encoder_raw).any():
        omega_encoder_diff = safe_gradient(theta_encoder_raw, t)
    omega_from_theta_imu_diff = safe_gradient(theta_imu_raw, t)
    omega_imu_online = col_any(df, ["omega_imu_online"], n)
    omega_encoder_online = col_any(df, ["omega_encoder_online"], n)
    omega_imu_offline = offline_smooth(omega_imu_direct, win=max(11, args.alpha_smooth * 5))
    omega_encoder_offline = offline_smooth(omega_encoder_diff, win=max(11, args.alpha_smooth * 5))

    alpha_from_imu_gyro_diff = col_any(df, ["alpha_imu"], n)
    alpha_from_linear_accel = col_any(df, ["alpha_linear"], n)
    alpha_from_encoder_diff = col_any(df, ["alpha_encoder"], n)
    if not np.isfinite(alpha_from_imu_gyro_diff).any() and np.isfinite(omega_imu_direct).any():
        alpha_from_imu_gyro_diff = safe_gradient(omega_imu_direct, t)
    alpha_from_omega_imu_diff = safe_gradient(omega_imu_direct, t)
    alpha_imu_online = col_any(df, ["alpha_imu_online"], n)
    alpha_linear_online = col_any(df, ["alpha_linear_online"], n)
    alpha_imu_offline = offline_smooth(alpha_from_imu_gyro_diff, win=max(11, args.alpha_smooth * 5))
    alpha_linear_offline = offline_smooth(alpha_from_linear_accel, win=max(11, args.alpha_smooth * 5))

    pwm_hw = col_any(df, ["pwm_hw", "hw_pwm"], n)
    i_raw = col_any(df, ["ina_current_raw_mA", "current_mA"], n)
    i_offset = col_any(df, ["ina_current_offset_mA"], n)
    if not np.isfinite(i_offset).any():
        offset_fallback = 26.0
        if meta is not None and isinstance(meta.get("warmup"), dict) and meta["warmup"].get("current_offset_mA") is not None:
            try:
                offset_fallback = float(meta["warmup"]["current_offset_mA"])
            except (TypeError, ValueError):
                offset_fallback = 26.0
        i_offset = np.full(n, offset_fallback, dtype=float)
    i_corr = col_any(df, ["ina_current_corr_mA"], n)
    if not np.isfinite(i_corr).any():
        i_corr = i_raw - i_offset
    i_signed = col_any(df, ["ina_current_signed_mA"], n)
    if not np.isfinite(i_signed).any():
        i_signed = np.sign(pwm_hw) * i_corr
    i_online = col_any(df, ["ina_current_signed_online_mA"], n)
    if not np.isfinite(i_online).any():
        i_online = moving_average(i_signed, max(3, args.alpha_smooth))
    i_offline = offline_smooth(i_signed, win=max(11, args.alpha_smooth * 5))

    # Quantitative candidate evaluation
    rest_mask = np.abs(pwm_hw) < max(5.0, 0.05 * np.nanmax(np.abs(pwm_hw)) if np.isfinite(pwm_hw).any() else 5.0)
    theta_candidates = {
        "theta_imu_raw_unwrapped": theta_imu_raw,
        "theta_imu_online_unwrapped": theta_imu_online,
        "theta_imu_offline_unwrapped": theta_imu_offline,
        "theta_encoder_raw_unwrapped": theta_encoder_raw,
        "theta_encoder_online_unwrapped": theta_encoder_online,
        "theta_encoder_offline_unwrapped": theta_encoder_offline,
    }
    theta_imu_raw, imu_sign, imu_sign_diag = imu_fixed_sign_flip(theta_imu_raw)
    theta_imu_online = imu_sign * theta_imu_online
    theta_imu_offline = imu_sign * theta_imu_offline
    theta_candidates["theta_imu_raw_unwrapped"] = theta_imu_raw
    theta_candidates["theta_imu_online_unwrapped"] = theta_imu_online
    theta_candidates["theta_imu_offline_unwrapped"] = theta_imu_offline
    omega_imu_direct = imu_sign * omega_imu_direct
    omega_imu_online = imu_sign * omega_imu_online
    omega_imu_offline = imu_sign * omega_imu_offline
    omega_from_theta_imu_diff = safe_gradient(theta_imu_raw, t)
    alpha_from_imu_gyro_diff = imu_sign * alpha_from_imu_gyro_diff
    alpha_from_omega_imu_diff = safe_gradient(omega_imu_direct, t)
    alpha_imu_online = imu_sign * alpha_imu_online
    alpha_imu_offline = imu_sign * alpha_imu_offline

    theta_rows, theta_winner = evaluate_candidate_set(theta_candidates, t, "theta", rest_mask, theta_sim if np.isfinite(theta_sim).any() else None)
    omega_candidates = {
        "omega_imu_direct": omega_imu_direct,
        "omega_from_theta_imu_diff": omega_from_theta_imu_diff,
        "omega_imu_online_filtered": omega_imu_online,
        "omega_imu_offline_filtered": omega_imu_offline,
        "omega_encoder_diff": omega_encoder_diff,
        "omega_encoder_online_filtered": omega_encoder_online,
        "omega_encoder_offline_filtered": omega_encoder_offline,
    }
    omega_ref = safe_gradient(theta_candidates.get(theta_winner, theta_imu_raw), t) if theta_winner else None
    omega_rows, omega_winner = evaluate_candidate_set(omega_candidates, t, "omega", rest_mask, omega_ref)
    alpha_candidates = {
        "alpha_from_imu_gyro_diff": alpha_from_imu_gyro_diff,
        "alpha_from_omega_imu_diff": alpha_from_omega_imu_diff,
        "alpha_from_linear_accel": alpha_from_linear_accel,
        "alpha_from_encoder_diff": alpha_from_encoder_diff,
        "alpha_from_imu_gyro_diff_online_filtered": alpha_imu_online,
        "alpha_from_linear_accel_online_filtered": alpha_linear_online,
        "alpha_from_imu_gyro_diff_offline_filtered": alpha_imu_offline,
        "alpha_from_linear_accel_offline_filtered": alpha_linear_offline,
    }
    omega_best = omega_candidates.get(omega_winner, omega_imu_direct)
    alpha_ref = safe_gradient(omega_best, t)
    alpha_rows, alpha_winner = evaluate_candidate_set(alpha_candidates, t, "alpha", rest_mask, alpha_ref)

    # canonical winners for downstream usage in this script
    theta_eval = theta_candidates.get(theta_winner, theta_imu_raw)
    omega_eval = omega_candidates.get(omega_winner, omega_imu_direct)
    alpha_eval = alpha_candidates.get(alpha_winner, alpha_from_imu_gyro_diff)

    t_cmd = t.copy()
    if args.apply_cmd_delay_from_meta and meta is not None and meta.get("estimated_delay_ms_final") is not None:
        t_cmd = t + 0.001 * float(meta["estimated_delay_ms_final"])

    print(f"csv  : {csv_path}")
    print(f"mode : {'processed_offline' if processed_mode else 'raw_log'}")
    print(f"CPR  : {cpr if cpr is not None else 'None'}")
    if len(t) > 1:
        dt_diag = np.diff(t)
        dt_diag = dt_diag[np.isfinite(dt_diag) & (dt_diag > 0.0)]
        if len(dt_diag) > 0:
            print(
                "sampling: "
                f"mean_dt={np.mean(dt_diag):.6f}s, std_dt={np.std(dt_diag):.6f}s, "
                f"min_dt={np.min(dt_diag):.6f}s, max_dt={np.max(dt_diag):.6f}s, "
                f"mean_f={1.0/max(np.mean(dt_diag), 1e-9):.2f}Hz"
            )
    if meta is not None and meta.get("estimated_delay_ms_final") is not None:
        print(f"delay: {meta['estimated_delay_ms_final']:.3f} ms")

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), num="Pendulum Unified Dashboard")
    ax_cmd = axes[0, 0]
    ax_theta = axes[0, 1]
    ax_err = axes[1, 0]
    ax_omega = axes[1, 1]
    ax_torque = axes[2, 0]
    ax_alpha = axes[2, 1]

    ax_cmd.plot(t_cmd, cmd_u, label="cmd_u")
    ax_cmd.plot(t, pwm_hw, label="pwm_hw")
    ax_cmd2 = ax_cmd.twinx()
    ax_cmd2.plot(t, i_raw, label="I_raw_mA", color="tab:red", alpha=0.35)
    ax_cmd2.plot(t, i_corr, label="I_offset_corrected_mA", color="tab:pink", alpha=0.8)
    ax_cmd2.plot(t, i_online, label="I_online_filtered_mA", color="tab:orange", alpha=0.9)
    ax_cmd2.plot(t, i_offline, label="I_offline_filtered_mA", color="tab:brown", alpha=0.9)
    ax_cmd.grid(True)
    lines1, labels1 = ax_cmd.get_legend_handles_labels()
    lines2, labels2 = ax_cmd2.get_legend_handles_labels()
    ax_cmd.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax_cmd.set_xlabel("time [s]")
    ax_cmd.set_title("Command / PWM / Current")
    ax_cmd2.set_ylabel("current [mA]")

    ax_theta.plot(t, theta_sim, label="theta_sim")
    for name, sig in theta_candidates.items():
        if np.isfinite(sig).any():
            ax_theta.plot(t, sig, label=name, alpha=0.75 if name == theta_winner else 0.35)
    ax_theta.grid(True)
    ax_theta.legend()
    ax_theta.set_xlabel("time [s]")
    ax_theta.set_ylabel("rad")
    ax_theta.set_title("Theta")

    ax_omega.plot(t, omega_sim, label="omega_sim")
    for name, sig in omega_candidates.items():
        if np.isfinite(sig).any():
            ax_omega.plot(t, sig, label=name, alpha=0.75 if name == omega_winner else 0.35)
    ax_omega.grid(True)
    ax_omega.legend()
    ax_omega.set_xlabel("time [s]")
    ax_omega.set_ylabel("rad/s")
    ax_omega.set_title("Omega")

    ax_alpha.plot(t, alpha_sim, label="alpha_sim")
    for name, sig in alpha_candidates.items():
        if np.isfinite(sig).any():
            ax_alpha.plot(t, sig, label=name, alpha=0.75 if name == alpha_winner else 0.35)
    ax_alpha.grid(True)
    ax_alpha.legend()
    ax_alpha.set_xlabel("time [s]")
    ax_alpha.set_ylabel("rad/s^2")
    ax_alpha.set_title("Alpha")

    e_theta = theta_sim - theta_eval if np.isfinite(theta_eval).any() else np.full(n, np.nan)
    e_omega = omega_sim - omega_eval if np.isfinite(omega_eval).any() else np.full(n, np.nan)
    e_alpha = alpha_sim - alpha_eval if np.isfinite(alpha_eval).any() else np.full(n, np.nan)
    ignore_mask = t < max(args.ignore_error_initial_sec, 0.0)
    e_theta[ignore_mask] = np.nan
    e_omega[ignore_mask] = np.nan
    e_alpha[ignore_mask] = np.nan
    for arr in (e_theta, e_omega, e_alpha):
        finite = np.isfinite(arr)
        if np.any(finite):
            thr = np.quantile(np.abs(arr[finite]), args.error_clip_quantile)
            arr[np.abs(arr) > max(thr, 1e-9)] = np.nan
    if np.isfinite(e_theta).any():
        ax_err.plot(t, np.sqrt(np.maximum(e_theta * e_theta, 0.0)), label="|e_theta|")
    if np.isfinite(e_omega).any():
        ax_err.plot(t, np.sqrt(np.maximum(e_omega * e_omega, 0.0)), label="|e_omega|")
    if np.isfinite(e_alpha).any():
        ax_err.plot(t, np.sqrt(np.maximum(e_alpha * e_alpha, 0.0)), label="|e_alpha|")
    ax_err.grid(True)
    ax_err.legend()
    ax_err.set_xlabel("time [s]")
    ax_err.set_title("Absolute tracking error")

    if np.isfinite(tau_cmd).any():
        ax_torque.plot(t, tau_cmd, label="tau_cmd(net)")
    if np.isfinite(tau_motor).any():
        ax_torque.plot(t, tau_motor, label="tau_motor")
    if np.isfinite(tau_visc).any():
        ax_torque.plot(t, tau_visc, label="tau_visc")
    if np.isfinite(tau_coul).any():
        ax_torque.plot(t, tau_coul, label="tau_coul")
    ax_torque.grid(True)
    ax_torque.legend()
    ax_torque.set_xlabel("time [s]")
    ax_torque.set_ylabel("N·m")
    ax_torque.set_title("Torque analysis")

    # Current audit / sign-mismatch diagnostics
    excited = np.abs(pwm_hw) > max(8.0, 0.1 * np.nanmax(np.abs(pwm_hw)) if np.isfinite(pwm_hw).any() else 8.0)
    transition = np.abs(safe_gradient(pwm_hw, t)) > (0.2 * np.nanmax(np.abs(safe_gradient(pwm_hw, t))) if np.isfinite(pwm_hw).any() else 0.0)
    high_pwm = np.abs(pwm_hw) > max(20.0, 0.5 * np.nanmax(np.abs(pwm_hw)) if np.isfinite(pwm_hw).any() else 20.0)
    mask_no_transition = excited & (~transition)

    def _agreement(cur):
        m = np.isfinite(cur) & np.isfinite(pwm_hw) & excited
        if np.sum(m) == 0:
            return float("nan")
        return float(np.mean(np.sign(cur[m]) == np.sign(pwm_hw[m])))

    def _agreement_mask(cur, m0):
        m = np.isfinite(cur) & np.isfinite(pwm_hw) & m0
        if np.sum(m) == 0:
            return float("nan")
        return float(np.mean(np.sign(cur[m]) == np.sign(pwm_hw[m])))

    sign_diag = {
        "agreement_excited_raw": _agreement(i_raw),
        "agreement_excited_corr": _agreement(i_corr),
        "agreement_excited_filtered": _agreement(i_online),
        "agreement_no_transition_filtered": _agreement_mask(i_online, mask_no_transition),
        "agreement_high_pwm_filtered": _agreement_mask(i_online, high_pwm),
    }

    # lag estimate by xcorr on excited segment
    lag_samples = 0
    if np.sum(excited) > 32:
        x = (pwm_hw[excited] - np.nanmean(pwm_hw[excited]))
        y = (i_online[excited] - np.nanmean(i_online[excited]))
        cc = np.correlate(x, y, mode="full")
        lag_samples = int(np.argmax(cc) - (len(x) - 1))
    dt_med = np.nanmedian(np.diff(t)) if len(t) > 1 else np.nan
    sign_diag["xcorr_lag_samples"] = lag_samples
    sign_diag["xcorr_lag_sec"] = float(lag_samples * dt_med) if np.isfinite(dt_med) else float("nan")
    rest_after_corr = i_corr[rest_mask & np.isfinite(i_corr)]
    sign_diag["rest_mean_corr_mA"] = float(np.nanmean(rest_after_corr)) if len(rest_after_corr) else float("nan")

    mismatch_state = "inconclusive"
    a_nt = sign_diag["agreement_no_transition_filtered"]
    a_hp = sign_diag["agreement_high_pwm_filtered"]
    if np.isfinite(a_nt) and np.isfinite(a_hp):
        if a_nt < 0.35 and a_hp < 0.35:
            mismatch_state = "likely sign convention / wiring inversion or bus-current semantics mismatch"
        elif a_nt > 0.7:
            mismatch_state = "likely normal with transition lag"
        else:
            mismatch_state = "likely preprocessing/sign mismatch issue"
    sign_diag["classification"] = mismatch_state

    # Save candidate/evaluation artifacts
    out_base = export_base
    metrics_rows = theta_rows + omega_rows + alpha_rows
    if (not processed_mode) and pd is not None and metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(out_base + ".signal_candidate_metrics.csv", index=False)
    summary = {
        "theta_winner": theta_winner,
        "omega_winner": omega_winner,
        "alpha_winner": alpha_winner,
        "imu_theta_sign_alignment": {
            "selected_sign": float(imu_sign),
            **imu_sign_diag,
        },
        "current_offset_applied_mA": float(np.nanmedian(i_offset[np.isfinite(i_offset)])) if np.isfinite(i_offset).any() else 0.0,
        "current_sign_diagnostics": sign_diag,
    }
    if not processed_mode:
        with open(out_base + ".signal_winner_summary.json", "w", encoding="utf-8") as fsum:
            json.dump(summary, fsum, indent=2)
    # Idempotent clean offline export: only for raw logs (never nested re-exports).
    if (not processed_mode) and pd is not None:
        df_offline = pd.DataFrame(
            {
                "wall_elapsed": t,
                "hw_pwm": pwm_hw,
                "ina_current_raw_mA": i_raw,
                "ina_current_offset_mA": i_offset,
                "current_offline_filtered": i_offline,
                "theta_winner": theta_eval,
                "omega_winner": omega_eval,
                "alpha_winner": alpha_eval,
                "theta_sim": theta_sim,
                "omega_sim": omega_sim,
                "alpha_sim": alpha_sim,
            }
        )
        df_offline.to_csv(out_base + ".offline_id.csv", index=False)

    # Quantity-specific candidate plots
    if not processed_mode:
        for qname, qcand in [("theta", theta_candidates), ("omega", omega_candidates), ("alpha", alpha_candidates)]:
            fig_q, ax_q = plt.subplots(1, 1, figsize=(12, 4))
            for cname, sig in qcand.items():
                if np.isfinite(sig).any():
                    ax_q.plot(t, sig, label=cname, alpha=0.8 if cname == summary.get(f"{qname}_winner") else 0.35)
            ax_q.set_title(f"{qname.capitalize()} candidates")
            ax_q.set_xlabel("time [s]")
            ax_q.grid(True, alpha=0.3)
            ax_q.legend(loc="upper right", fontsize=8, ncol=2)
            fig_q.tight_layout()
            fig_q.savefig(out_base + f".{qname}_candidates.png", dpi=130)
            plt.close(fig_q)

    fig_cur, ax_cur = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax_cur[0].plot(t, pwm_hw, label="pwm_hw")
    ax_cur[0].plot(t, i_raw, label="I_raw_mA", alpha=0.45)
    ax_cur[0].plot(t, i_corr, label="I_offset_corrected_mA", alpha=0.8)
    ax_cur[0].plot(t, i_online, label="I_online_filtered_mA", alpha=0.9)
    ax_cur[0].legend(loc="upper right")
    ax_cur[0].grid(True, alpha=0.3)
    ax_cur[1].plot(t, np.sign(pwm_hw), label="sign(pwm_hw)")
    ax_cur[1].plot(t, np.sign(i_online), label="sign(I_online)")
    ax_cur[1].set_title(f"sign-agreement(no-transition={sign_diag['agreement_no_transition_filtered']:.3f}, high-pwm={sign_diag['agreement_high_pwm_filtered']:.3f})")
    ax_cur[1].legend(loc="upper right")
    ax_cur[1].grid(True, alpha=0.3)
    ax_cur[1].set_xlabel("time [s]")
    fig_cur.tight_layout()
    if not processed_mode:
        fig_cur.savefig(out_base + ".current_audit.png", dpi=130)
    plt.close(fig_cur)

    print(f"winner(theta): {theta_winner}")
    print(f"winner(omega): {omega_winner}")
    print(f"winner(alpha): {alpha_winner}")
    print(f"current offset applied [mA]: {summary['current_offset_applied_mA']:.3f}")
    print(f"current sign classification: {mismatch_state}")

    fig.tight_layout()
    plt.show()


def plot_rl_summary(history_csv: str, replay_csv: str):
    if pd is None:
        raise RuntimeError("pandas is required for --history-csv/--rl-dir plotting.")
    h = pd.read_csv(history_csv)
    r = pd.read_csv(replay_csv)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), num="RL + Replay Summary")

    ep = h["episode"].to_numpy(dtype=float) if "episode" in h.columns else np.arange(len(h), dtype=float)
    if "reward" in h.columns:
        axs[0, 0].plot(ep, h["reward"].to_numpy(dtype=float), label="reward")
    axs[0, 0].set_title("Episode Reward")
    axs[0, 0].set_xlabel("episode")
    axs[0, 0].grid(True, alpha=0.3)

    if "train_loss" in h.columns:
        axs[0, 1].plot(ep, h["train_loss"].to_numpy(dtype=float), label="train_loss")
    if "val_loss" in h.columns:
        axs[0, 1].plot(ep, h["val_loss"].to_numpy(dtype=float), label="val_loss")
    axs[0, 1].set_title("Loss")
    axs[0, 1].set_xlabel("episode")
    axs[0, 1].legend(loc="best")
    axs[0, 1].grid(True, alpha=0.3)

    t = r["wall_elapsed"].to_numpy(dtype=float) if "wall_elapsed" in r.columns else r["sim_time"].to_numpy(dtype=float)
    theta_real = unwrap_and_zero(r["theta_real"].to_numpy(dtype=float))
    theta_sim = unwrap_and_zero(r["theta"].to_numpy(dtype=float))
    axs[1, 0].plot(t, theta_real, label="theta_real")
    axs[1, 0].plot(t, theta_sim, label="theta_sim")
    axs[1, 0].set_title("Replay Theta Overlay")
    axs[1, 0].set_xlabel("time [s]")
    axs[1, 0].legend(loc="best")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(t, r["omega_real"].to_numpy(dtype=float), label="omega_real")
    axs[1, 1].plot(t, r["omega"].to_numpy(dtype=float), label="omega_sim")
    axs[1, 1].set_title("Replay Omega Overlay")
    axs[1, 1].set_xlabel("time [s]")
    axs[1, 1].legend(loc="best")
    axs[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir", default="./run_logs")
    ap.add_argument("--counts-per-revolution", type=float, default=None)
    ap.add_argument("--theta-sign", type=float, default=1.0)
    ap.add_argument("--theta-offset", type=float, default=0.0)
    ap.add_argument("--alpha-smooth", type=int, default=5)
    ap.add_argument("--real-theta-smooth", type=int, default=7)
    ap.add_argument("--real-derivative-smooth", type=int, default=11)
    ap.add_argument("--recompute-real-derivatives", dest="recompute_real_derivatives", action="store_true")
    ap.add_argument("--use-logged-real-derivatives", dest="recompute_real_derivatives", action="store_false")
    ap.set_defaults(recompute_real_derivatives=True)
    ap.add_argument("--ignore-error-initial-sec", type=float, default=0.35)
    ap.add_argument("--error-clip-quantile", type=float, default=0.995)
    ap.add_argument("--apply-cmd-delay-from-meta", action="store_true")
    ap.add_argument("--history-csv", default=None, help="RL history.csv path (for RL summary plot)")
    ap.add_argument("--rl-dir", default=None, help="directory containing history.csv and replay_best.csv")
    args = ap.parse_args()

    if args.rl_dir is not None:
        history_csv = os.path.join(args.rl_dir, "history.csv")
        replay_csv = os.path.join(args.rl_dir, "replay_best.csv")
        plot_rl_summary(history_csv, replay_csv)
        return
    if args.history_csv is not None:
        if args.csv is None:
            raise SystemExit("--history-csv requires --csv replay file.")
        plot_rl_summary(args.history_csv, args.csv)
        return

    csv_path = args.csv if args.csv is not None else find_latest_csv(args.dir)
    df = load_csv_frame(csv_path)
    plot_simulation(df, csv_path, args)


if __name__ == "__main__":
    main()
