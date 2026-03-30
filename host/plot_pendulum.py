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


def plot_simulation(df, csv_path: str, args):
    meta = load_meta_if_exists(csv_path)

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

    if not np.isfinite(alpha_sim).any() and np.isfinite(omega_sim).any():
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = dt[1]
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
        alpha_sim = np.gradient(omega_sim, dt)

    theta_real = col_any(df, ["theta_real"], n)
    omega_real = col_any(df, ["omega_real"], n)
    alpha_real = col_any(df, ["alpha_real"], n)
    theta_sim = unwrap_and_zero(theta_sim)
    theta_real = unwrap_and_zero(theta_real)
    if np.isfinite(theta_real).any():
        theta_real = moving_average(theta_real, args.real_theta_smooth)
    has_real = np.isfinite(theta_real).any() or np.isfinite(omega_real).any() or np.isfinite(alpha_real).any()
    if not has_real and cpr is not None and np.isfinite(enc).any():
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
    ax[0].plot(t, hw_pwm, label="hw_pwm")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_xlabel("time [s]")
    ax[0].set_title("Command / PWM")

    ax[1].plot(t, theta_sim, label="theta sim")
    if np.isfinite(theta_real).any():
        ax[1].plot(t, theta_real, label="theta real")
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("rad")
    ax[1].set_title("Theta")

    ax[2].plot(t, omega_sim, label="omega sim")
    if np.isfinite(omega_real).any():
        ax[2].plot(t, omega_real, label="omega real")
    ax[2].grid(True)
    ax[2].legend()
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("rad/s")
    ax[2].set_title("Omega")

    ax[3].plot(t, alpha_sim, label="alpha sim")
    if np.isfinite(alpha_real).any():
        ax[3].plot(t, alpha_real, label="alpha real")
    ax[3].grid(True)
    ax[3].legend()
    ax[3].set_xlabel("time [s]")
    ax[3].set_ylabel("rad/s^2")
    ax[3].set_title("Alpha")

    if args.recompute_real_derivatives and np.isfinite(theta_real).any():
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = dt[1]
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
        omega_real = moving_average(np.gradient(theta_real, dt), args.real_derivative_smooth)
        alpha_real = moving_average(np.gradient(omega_real, dt), args.real_derivative_smooth)

    e_theta = theta_sim - theta_real if np.isfinite(theta_real).any() else np.full(n, np.nan)
    e_omega = omega_sim - omega_real if np.isfinite(omega_real).any() else np.full(n, np.nan)
    e_alpha = alpha_sim - alpha_real if np.isfinite(alpha_real).any() else np.full(n, np.nan)
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
        ax[4].plot(t, np.sqrt(np.maximum(e_theta * e_theta, 0.0)), label="|e_theta|")
    if np.isfinite(e_omega).any():
        ax[4].plot(t, np.sqrt(np.maximum(e_omega * e_omega, 0.0)), label="|e_omega|")
    if np.isfinite(e_alpha).any():
        ax[4].plot(t, np.sqrt(np.maximum(e_alpha * e_alpha, 0.0)), label="|e_alpha|")
    ax[4].grid(True)
    ax[4].legend()
    ax[4].set_xlabel("time [s]")
    ax[4].set_title("Absolute tracking error")

    if np.isfinite(tau_cmd).any():
        ax[5].plot(t, tau_cmd, label="tau_cmd(net)")
    if np.isfinite(tau_motor).any():
        ax[5].plot(t, tau_motor, label="tau_motor")
    if np.isfinite(tau_visc).any():
        ax[5].plot(t, tau_visc, label="tau_visc")
    if np.isfinite(tau_coul).any():
        ax[5].plot(t, tau_coul, label="tau_coul")
    ax[5].grid(True)
    ax[5].legend()
    ax[5].set_xlabel("time [s]")
    ax[5].set_ylabel("N·m")
    ax[5].set_title("Torque analysis")

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
    axs[1, 0].plot(t, r["theta_real"].to_numpy(dtype=float), label="theta_real")
    axs[1, 0].plot(t, r["theta"].to_numpy(dtype=float), label="theta_sim")
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
