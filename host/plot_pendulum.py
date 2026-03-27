#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

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


def moving_average(x: np.ndarray, win: int):
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")
    return y[: len(x)]


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


def is_calibration_dataframe(df):
    calibration_cols = {"theta_rad", "omega_rad_s", "alpha_rad_s2", "mean_cpr_running", "r_running"}
    return calibration_cols.issubset(set(df.columns))


def plot_calibration(df, csv_path: str):
    t = col_any(df, ["time_sec", "sim_time", "wall_time"])
    theta = col_any(df, ["theta_rad", "theta"])
    omega = col_any(df, ["omega_rad_s", "omega", "imu_wz"])
    alpha = col_any(df, ["alpha_rad_s2", "alpha"])
    delay_ms = col_any(df, ["delay_ms"])
    mean_cpr = col_any(df, ["mean_cpr_running"])
    r_running = col_any(df, ["r_running"])

    print(f"csv  : {csv_path}")
    print("mode : calibration")

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), num="Calibration Data Dashboard")
    ax = axes.ravel()

    ax[0].plot(t, theta, label="theta")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("rad")
    ax[0].set_title("Theta")

    ax[1].plot(t, omega, label="omega", color="tab:orange")
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("rad/s")
    ax[1].set_title("Omega")

    ax[2].plot(t, alpha, label="alpha", color="tab:green")
    ax[2].grid(True)
    ax[2].legend()
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("rad/s²")
    ax[2].set_title("Alpha")

    ax[3].plot(t, delay_ms, label="delay_ms", color="tab:red")
    ax[3].grid(True)
    ax[3].legend()
    ax[3].set_xlabel("time [s]")
    ax[3].set_ylabel("ms")
    ax[3].set_title("Delay")

    ax[4].plot(t, mean_cpr, label="mean CPR over time", color="tab:purple")
    ax[4].grid(True)
    ax[4].legend()
    ax[4].set_xlabel("time [s]")
    ax[4].set_title("Mean CPR (running)")

    ax[5].plot(t, r_running, label="r over time", color="tab:brown")
    ax[5].grid(True)
    ax[5].legend()
    ax[5].set_xlabel("time [s]")
    ax[5].set_title("r (running)")

    fig.tight_layout()
    plt.show()


def plot_simulation(df, csv_path: str, args):
    meta = load_meta_if_exists(csv_path)

    cpr = args.counts_per_revolution
    if cpr is None and meta is not None and meta.get("cpr_mean") is not None:
        cpr = float(meta["cpr_mean"])

    if "sim_time" in df.columns:
        t = col_to_numpy(df, "sim_time")
    elif "wall_time" in df.columns:
        tw = col_to_numpy(df, "wall_time")
        t = tw - tw[0]
    else:
        n = len(col_to_numpy(df, df.columns[0])) if len(df.columns) > 0 else 0
        t = np.arange(n, dtype=float)

    n = len(t)
    theta_sim = col_any(df, ["theta", "sim_theta"], n)
    omega_sim = col_any(df, ["omega", "sim_omega"], n)
    alpha_sim = col_any(df, ["alpha", "sim_alpha"], n)
    cmd_u = col_any(df, ["cmd_u_raw", "cmd_u"], n)
    cmd_used = col_any(df, ["cmd_u_used", "cmd_u", "hw_pwm"], n)
    hw_pwm = col_any(df, ["hw_pwm"], n)
    enc = col_any(df, ["hw_enc"], n)
    ina_i = col_any(df, ["current_A", "current_ma"], n)
    ina_p = col_any(df, ["power_W", "power_mw"], n)
    p_pred = col_any(df, ["p_pred"], n)
    delay_ms = col_any(df, ["delay_ms"], n)

    if not np.isfinite(theta_sim).any() and all(k in df.columns for k in ["imu_qx", "imu_qy", "imu_qz", "imu_qw"]):
        qx = col_to_numpy(df, "imu_qx")
        qy = col_to_numpy(df, "imu_qy")
        qz = col_to_numpy(df, "imu_qz")
        qw = col_to_numpy(df, "imu_qw")
        theta_sim = derive_theta_from_imu_quat(qx, qy, qz, qw)

    if not np.isfinite(omega_sim).any() and "imu_wz" in df.columns:
        omega_sim = col_to_numpy(df, "imu_wz")

    if not np.isfinite(alpha_sim).any() and np.isfinite(omega_sim).any():
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = dt[1]
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
        alpha_sim = np.gradient(omega_sim, dt)

    theta_real = None
    omega_real = None
    alpha_real = None
    if cpr is not None and np.isfinite(enc).any():
        theta_real = derive_theta_from_encoder(enc, cpr, sign=args.theta_sign, offset=args.theta_offset)
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = dt[1]
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
        omega_real = np.gradient(theta_real, dt)
        omega_real = moving_average(omega_real, args.alpha_smooth)
        alpha_real = np.gradient(omega_real, dt)
        alpha_real = moving_average(alpha_real, args.alpha_smooth)

    t_cmd = t.copy()
    if args.apply_cmd_delay_from_meta and meta is not None and meta.get("estimated_delay_ms_final") is not None:
        t_cmd = t + 0.001 * float(meta["estimated_delay_ms_final"])

    print(f"csv  : {csv_path}")
    print("mode : simulation")
    print(f"CPR  : {cpr if cpr is not None else 'None'}")
    if meta is not None and meta.get("estimated_delay_ms_final") is not None:
        print(f"delay: {meta['estimated_delay_ms_final']:.3f} ms")

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), num="Pendulum Unified Dashboard")
    ax = axes.ravel()

    ax[0].plot(t_cmd, cmd_u, label="cmd_u")
    ax[0].plot(t, cmd_used, label="cmd_used")
    ax[0].plot(t, hw_pwm, label="hw_pwm")
    ax[0].plot(t, delay_ms, label="delay_ms")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_xlabel("time [s]")
    ax[0].set_title("Command / PWM / delay")

    ax[1].plot(t, theta_sim, label="theta sim")
    if theta_real is not None:
        ax[1].plot(t, theta_real, label="theta real(enc)")
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("rad")
    ax[1].set_title("Theta")

    ax[2].plot(t, omega_sim, label="omega sim")
    if omega_real is not None:
        ax[2].plot(t, omega_real, label="omega real")
    ax[2].grid(True)
    ax[2].legend()
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("rad/s")
    ax[2].set_title("Omega")

    ax[3].plot(t, alpha_sim, label="alpha sim")
    if alpha_real is not None:
        ax[3].plot(t, alpha_real, label="alpha real")
    ax[3].grid(True)
    ax[3].legend()
    ax[3].set_xlabel("time [s]")
    ax[3].set_ylabel("rad/s^2")
    ax[3].set_title("Alpha")

    ax[4].plot(t, ina_p, label="INA power")
    ax[4].plot(t, p_pred, label="pred power")
    ax[4].plot(t, ina_i, label="INA current", alpha=0.7)
    ax[4].grid(True)
    ax[4].legend()
    ax[4].set_xlabel("time [s]")
    ax[4].set_title("Electrical (power/current)")

    theta_for_hys = theta_real if theta_real is not None else theta_sim
    omega_for_hys = omega_real if omega_real is not None else omega_sim
    valid = np.isfinite(theta_for_hys) & np.isfinite(omega_for_hys)
    ax[5].plot(theta_for_hys[valid], omega_for_hys[valid], linewidth=1.2)
    ax[5].grid(True)
    ax[5].set_xlabel("theta [rad]")
    ax[5].set_ylabel("omega [rad/s]")
    ax[5].set_title("Hysteresis / phase portrait")

    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir", default="./run_logs")
    ap.add_argument("--mode", choices=["auto", "calibration", "simulation"], default="auto")
    ap.add_argument("--counts-per-revolution", type=float, default=None)
    ap.add_argument("--theta-sign", type=float, default=1.0)
    ap.add_argument("--theta-offset", type=float, default=0.0)
    ap.add_argument("--alpha-smooth", type=int, default=5)
    ap.add_argument("--apply-cmd-delay-from-meta", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv if args.csv is not None else find_latest_csv(args.dir)
    df = load_csv_frame(csv_path)

    mode = args.mode
    if mode == "auto":
        mode = "calibration" if is_calibration_dataframe(df) else "simulation"

    if mode == "calibration":
        plot_calibration(df, csv_path)
    else:
        plot_simulation(df, csv_path, args)


if __name__ == "__main__":
    main()
