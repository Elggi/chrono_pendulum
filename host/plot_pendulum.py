#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict visualization-only plotter.

Filtering/sign processing is intentionally moved to runtime logging (chrono_pendulum.py)
and shared filter implementations (chrono_core.signal_filter).
This script does NOT generate derived winner exports.
"""

import argparse
import json
import os
import csv

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


class SimpleFrame:
    def __init__(self, columns):
        self._data = columns
        self.columns = list(columns.keys())

    def __getitem__(self, key):
        return self._data[key]


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


def _print_available(df):
    print("[available columns]")
    for c in df.columns:
        print(f"  - {c}")


def plot_simulation(df, csv_path: str):
    meta = load_meta_if_exists(csv_path)

    if "wall_elapsed" in df.columns:
        t = col_to_numpy(df, "wall_elapsed")
    elif "wall_time" in df.columns:
        tw = col_to_numpy(df, "wall_time")
        t = tw - tw[0]
    else:
        n = len(col_to_numpy(df, df.columns[0])) if len(df.columns) > 0 else 0
        t = np.arange(n, dtype=float)

    n = len(t)
    cmd_u = col_any(df, ["cmd_u_raw", "cmd_u"], n)
    pwm_hw = col_any(df, ["pwm_hw", "hw_pwm"], n)
    i_raw = col_any(df, ["ina_current_raw_mA", "current_mA"], n)
    i_corr = col_any(df, ["ina_current_corr_mA", "I_offset_corrected_mA"], n)
    i_filtered = col_any(df, ["ina_current_signed_online_mA", "current_offline_filtered"], n)

    theta_sim = col_any(df, ["theta"], n)
    omega_sim = col_any(df, ["omega"], n)
    alpha_sim = col_any(df, ["alpha"], n)

    theta_imu_raw = col_any(df, ["theta_imu", "theta_imu_raw_unwrapped"], n)
    theta_imu_f = col_any(df, ["theta_imu_online", "theta_imu_filtered_unwrapped", "theta_winner"], n)
    omega_imu_raw = col_any(df, ["omega_imu", "omega_imu_raw"], n)
    omega_imu_f = col_any(df, ["omega_imu_online", "omega_imu_filtered", "omega_winner"], n)
    alpha_imu_gyro = col_any(df, ["alpha_imu", "alpha_imu_gyro_diff"], n)
    alpha_imu_gyro_f = col_any(df, ["alpha_imu_online", "alpha_imu_gyro_diff_filtered", "alpha_winner"], n)
    alpha_lin = col_any(df, ["alpha_linear", "alpha_from_linear_accel"], n)
    alpha_lin_f = col_any(df, ["alpha_linear_online", "alpha_from_linear_accel_filtered"], n)

    print(f"csv: {csv_path}")
    if meta is not None and isinstance(meta.get("warmup"), dict):
        print(f"warmup theta_offset={meta['warmup'].get('theta_offset_rad')} current_offset={meta['warmup'].get('current_offset_mA')}")

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), num="Pendulum Unified Dashboard")
    ax_cmd = axes[0, 0]
    ax_theta = axes[0, 1]
    ax_err = axes[1, 0]
    ax_omega = axes[1, 1]
    ax_torque = axes[2, 0]
    ax_alpha = axes[2, 1]

    ax_cmd.plot(t, cmd_u, label="cmd_u")
    ax_cmd.plot(t, pwm_hw, label="pwm_hw")
    ax_cmd2 = ax_cmd.twinx()
    ax_cmd2.plot(t, i_raw, label="I_raw_mA", alpha=0.4)
    ax_cmd2.plot(t, i_corr, label="I_offset_corrected_mA", alpha=0.8)
    ax_cmd2.plot(t, i_filtered, label="I_filtered_mA", alpha=0.9)
    l1, n1 = ax_cmd.get_legend_handles_labels()
    l2, n2 = ax_cmd2.get_legend_handles_labels()
    ax_cmd.legend(l1 + l2, n1 + n2, loc="upper right")
    ax_cmd.grid(True)
    ax_cmd.set_title("Command / PWM / Current")
    ax_cmd.set_xlabel("time [s]")
    ax_cmd2.set_ylabel("current [mA]")

    ax_theta.plot(t, theta_sim, label="theta_sim")
    ax_theta.plot(t, theta_imu_raw, label="theta_imu_raw_unwrapped")
    ax_theta.plot(t, theta_imu_f, label="theta_imu_filtered_unwrapped")
    ax_theta.set_title("Theta")
    ax_theta.set_ylabel("rad")
    ax_theta.grid(True)
    ax_theta.legend()

    ax_omega.plot(t, omega_sim, label="omega_sim")
    ax_omega.plot(t, omega_imu_raw, label="omega_imu_raw")
    ax_omega.plot(t, omega_imu_f, label="omega_imu_filtered")
    ax_omega.set_title("Omega")
    ax_omega.set_ylabel("rad/s")
    ax_omega.grid(True)
    ax_omega.legend()

    ax_alpha.plot(t, alpha_sim, label="alpha_sim")
    ax_alpha.plot(t, alpha_imu_gyro, label="alpha_imu_gyro_diff")
    ax_alpha.plot(t, alpha_imu_gyro_f, label="alpha_imu_gyro_diff_filtered")
    ax_alpha.plot(t, alpha_lin, label="alpha_from_linear_accel")
    ax_alpha.plot(t, alpha_lin_f, label="alpha_from_linear_accel_filtered")
    ax_alpha.set_title("Alpha")
    ax_alpha.set_ylabel("rad/s^2")
    ax_alpha.grid(True)
    ax_alpha.legend(fontsize=8)

    e_theta = theta_sim - theta_imu_f
    e_omega = omega_sim - omega_imu_f
    e_alpha = alpha_sim - alpha_imu_gyro_f
    if np.isfinite(e_theta).any():
        ax_err.plot(t, np.abs(e_theta), label="|e_theta|")
    if np.isfinite(e_omega).any():
        ax_err.plot(t, np.abs(e_omega), label="|e_omega|")
    if np.isfinite(e_alpha).any():
        ax_err.plot(t, np.abs(e_alpha), label="|e_alpha|")
    ax_err.grid(True)
    ax_err.set_title("Absolute tracking error")
    ax_err.legend()

    tau_cmd = col_any(df, ["tau_cmd"], n)
    tau_motor = col_any(df, ["tau_motor"], n)
    tau_visc = col_any(df, ["tau_visc"], n)
    tau_coul = col_any(df, ["tau_coul"], n)
    ax_torque.plot(t, tau_cmd, label="tau_cmd")
    ax_torque.plot(t, tau_motor, label="tau_motor")
    ax_torque.plot(t, tau_visc, label="tau_visc")
    ax_torque.plot(t, tau_coul, label="tau_coul")
    ax_torque.grid(True)
    ax_torque.set_title("Torque analysis")
    ax_torque.legend(fontsize=8)

    for ax in [ax_theta, ax_omega, ax_alpha, ax_err, ax_torque]:
        ax.set_xlabel("time [s]")

    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir", default="./run_logs")
    ap.add_argument("--history-csv", default=None)
    ap.add_argument("--rl-dir", default=None)
    args = ap.parse_args()

    if args.rl_dir is not None or args.history_csv is not None:
        raise SystemExit("RL summary mode was removed from plot_pendulum.py. Use dedicated RL plotting utility.")

    csv_path = args.csv if args.csv is not None else find_latest_csv(args.dir)
    df = load_csv_frame(csv_path)
    if len(df.columns) == 0:
        raise SystemExit("Empty CSV or unsupported format")
    plot_simulation(df, csv_path)


if __name__ == "__main__":
    main()
