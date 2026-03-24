#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
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


def moving_average(x: np.ndarray, win: int):
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")
    return y[:len(x)]


def derive_theta_from_encoder(enc, counts_per_rev, sign=1.0, offset=0.0):
    return sign * (2.0 * np.pi / counts_per_rev) * (enc - enc[0]) + offset


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir", default="./run_logs")
    ap.add_argument("--counts-per-revolution", type=float, default=None)
    ap.add_argument("--theta-sign", type=float, default=1.0)
    ap.add_argument("--theta-offset", type=float, default=0.0)
    ap.add_argument("--alpha-smooth", type=int, default=5)
    ap.add_argument("--apply-cmd-delay-from-meta", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv if args.csv is not None else find_latest_csv(args.dir)
    df = load_csv_frame(csv_path)
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
    theta_sim = col_any(df, ["theta"], n)
    omega_sim = col_any(df, ["omega"], n)
    alpha_sim = col_any(df, ["alpha"], n)
    cmd_u = col_any(df, ["cmd_u_raw", "cmd_u"], n)
    cmd_used = col_any(df, ["cmd_u_used", "cmd_u"], n)
    hw_pwm = col_any(df, ["hw_pwm"], n)
    enc = col_any(df, ["hw_enc"], n)
    ina_v = col_any(df, ["bus_v"], n)
    ina_i = col_any(df, ["current_A", "current_ma"], n)
    ina_p = col_any(df, ["power_W", "power_mw"], n)
    v_pred = col_any(df, ["v_pred"], n)
    i_pred = col_any(df, ["i_pred"], n)
    p_pred = col_any(df, ["p_pred"], n)
    delay_ms = col_any(df, ["delay_ms"], n)

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

    if "inst_cost" in df.columns:
        inst_cost = col_to_numpy(df, "inst_cost")
        if not np.isfinite(inst_cost).any():
            best_row = None
        else:
            best_idx = int(np.nanargmin(inst_cost))
            best_row = df.iloc[best_idx] if pd is not None else df.row(best_idx)
        if best_row is not None:
            print("\n=== Best calibration point ===")
            print(f"time      : {best_row['sim_time']:.6f} s")
            print(f"inst_cost : {best_row['inst_cost']:.6e}")
            print(f"delay_ms  : {best_row['delay_ms']:.3f}")
            print(f"J         : {best_row['J_est']:.6f}")
            print(f"b         : {best_row['b_est']:.6f}")
            print(f"tau_c     : {best_row['tau_c_est']:.6f}")
            print(f"mgl       : {best_row['mgl_est']:.6f}")
            print(f"k_t       : {best_row['k_t_est']:.6f}")
            print(f"i0        : {best_row['i0_est']:.6f}")
    else:
        best_row = None

    t_cmd = t.copy()
    if args.apply_cmd_delay_from_meta and meta is not None and meta.get("estimated_delay_ms_final") is not None:
        t_cmd = t + 0.001 * float(meta["estimated_delay_ms_final"])

    print(f"csv  : {csv_path}")
    print(f"CPR  : {cpr if cpr is not None else 'None'}")
    if meta is not None and meta.get("estimated_delay_ms_final") is not None:
        print(f"delay: {meta['estimated_delay_ms_final']:.3f} ms")

    plt.figure("Command / PWM / delay", figsize=(10, 5))
    plt.plot(t_cmd, cmd_u, label="cmd_u")
    plt.plot(t, cmd_used, label="cmd_used")
    plt.plot(t, hw_pwm, label="hw_pwm")
    plt.plot(t, delay_ms, label="delay_ms")
    plt.grid(True); plt.legend(); plt.xlabel("time [s]"); plt.title("Command vs applied PWM / delay")

    plt.figure("Theta", figsize=(10, 5))
    plt.plot(t, theta_sim, label="theta sim")
    if theta_real is not None:
        plt.plot(t, theta_real, label="theta real(enc)")
    plt.grid(True); plt.legend(); plt.xlabel("time [s]"); plt.ylabel("rad"); plt.title("Theta: sim vs real")

    plt.figure("Omega", figsize=(10, 5))
    plt.plot(t, omega_sim, label="omega sim")
    if omega_real is not None:
        plt.plot(t, omega_real, label="omega real")
    plt.grid(True); plt.legend(); plt.xlabel("time [s]"); plt.ylabel("rad/s"); plt.title("Omega: sim vs real")

    plt.figure("Alpha", figsize=(10, 5))
    plt.plot(t, alpha_sim, label="alpha sim")
    if alpha_real is not None:
        plt.plot(t, alpha_real, label="alpha real")
    plt.grid(True); plt.legend(); plt.xlabel("time [s]"); plt.ylabel("rad/s^2"); plt.title("Alpha: sim vs real")

    plt.figure("Electrical", figsize=(10, 6))
    plt.subplot(311)
    plt.plot(t, ina_v, label="INA voltage")
    plt.plot(t, v_pred, label="pred voltage")
    plt.grid(True); plt.legend(); plt.ylabel("V")
    plt.subplot(312)
    plt.plot(t, ina_i, label="INA current")
    plt.plot(t, i_pred, label="pred current")
    plt.grid(True); plt.legend(); plt.ylabel("A")
    plt.subplot(313)
    plt.plot(t, ina_p, label="INA power")
    plt.plot(t, p_pred, label="pred power")
    plt.grid(True); plt.legend(); plt.ylabel("W"); plt.xlabel("time [s]")
    plt.suptitle("Electrical model vs INA219")
    plt.tight_layout()

    if best_row is not None:
        plt.figure("Online parameter convergence", figsize=(12, 8))
        plt.subplot(321)
        plt.plot(t, col_to_numpy(df, "J_est"), label="J_est")
        plt.axvline(best_row["sim_time"], color="r", linestyle="--")
        plt.grid(True); plt.legend(); plt.title("J estimate")
        plt.subplot(322)
        plt.plot(t, col_to_numpy(df, "b_est"), label="b_est")
        plt.axvline(best_row["sim_time"], color="r", linestyle="--")
        plt.grid(True); plt.legend(); plt.title("b estimate")
        plt.subplot(323)
        plt.plot(t, col_to_numpy(df, "tau_c_est"), label="tau_c_est")
        plt.axvline(best_row["sim_time"], color="r", linestyle="--")
        plt.grid(True); plt.legend(); plt.title("tau_c estimate")
        plt.subplot(324)
        plt.plot(t, col_to_numpy(df, "mgl_est"), label="mgl_est")
        plt.axvline(best_row["sim_time"], color="r", linestyle="--")
        plt.grid(True); plt.legend(); plt.title("mgl estimate")
        plt.subplot(325)
        plt.plot(t, col_to_numpy(df, "k_t_est"), label="k_t_est")
        plt.axvline(best_row["sim_time"], color="r", linestyle="--")
        plt.grid(True); plt.legend(); plt.title("k_t estimate")
        plt.subplot(326)
        plt.plot(t, col_to_numpy(df, "i0_est"), label="i0_est")
        plt.axvline(best_row["sim_time"], color="r", linestyle="--")
        plt.grid(True); plt.legend(); plt.title("i0 estimate")
        plt.tight_layout()

        plt.figure("Calibration cost", figsize=(10, 5))
        plt.plot(t, col_to_numpy(df, "inst_cost"), label="instant cost")
        plt.plot(t, col_to_numpy(df, "best_cost_so_far"), label="best cost so far")
        plt.axvline(best_row["sim_time"], color="r", linestyle="--", label="best point")
        plt.grid(True); plt.legend(); plt.xlabel("time [s]"); plt.ylabel("cost"); plt.title("Online calibration progress")

    plt.show()


if __name__ == "__main__":
    main()
