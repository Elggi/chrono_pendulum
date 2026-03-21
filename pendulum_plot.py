#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_csv(folder: str):
    csvs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No csv files in {folder}")
    csvs.sort(key=os.path.getmtime)
    return csvs[-1]


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    kernel = np.ones(int(win), dtype=float) / float(win)
    pad = int(win) // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")
    return y[:len(x)]


def get_column(df: pd.DataFrame, candidates, required=True, default_value=np.nan):
    for c in candidates:
        if c in df.columns:
            return df[c].to_numpy(dtype=float), c
    if required:
        raise KeyError(f"Missing required columns. Tried: {candidates}")
    return np.full(len(df), default_value, dtype=float), None


def is_valid_array(x: np.ndarray) -> bool:
    return np.isfinite(x).any()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to csv log")
    parser.add_argument("--dir", type=str, default="./run_logs", help="Folder to search latest csv")
    parser.add_argument(
        "--counts-per-rotation",
        type=float,
        required=True,
        help="Measured encoder counts for one full rotation from imu_viewer",
    )
    parser.add_argument(
        "--theta-sign",
        type=float,
        default=1.0,
        help="Use 1 or -1 to flip theta_real direction",
    )
    parser.add_argument(
        "--theta-offset",
        type=float,
        default=0.0,
        help="Additional theta offset [rad] added to theta_real",
    )
    parser.add_argument(
        "--alpha-smooth",
        type=int,
        default=5,
        help="Moving average window for omega_real before differentiation",
    )
    parser.add_argument(
        "--plot-title-prefix",
        type=str,
        default="",
        help='Optional title prefix, e.g. "host_pub" or "jetson_pub"',
    )
    args = parser.parse_args()

    if args.counts_per_rotation <= 0:
        raise ValueError("--counts-per-rotation must be > 0")

    csv_path = args.csv if args.csv is not None else find_latest_csv(args.dir)
    df = pd.read_csv(csv_path)

    # Required / optional columns
    t, t_name = get_column(df, ["sim_time", "time", "t"], required=True)

    cmd_u, _ = get_column(df, ["cmd_u"], required=False)
    hw_pwm, _ = get_column(df, ["hw_pwm", "pwm_applied"], required=False)

    theta_sim, _ = get_column(df, ["theta"], required=False)
    omega_sim, _ = get_column(df, ["omega"], required=False)
    alpha_sim, _ = get_column(df, ["alpha"], required=False)

    imu_wz, _ = get_column(df, ["imu_wz", "gyro_z", "angular_velocity_z"], required=False)
    hw_enc, _ = get_column(df, ["hw_enc", "enc", "encoder"], required=False)

    # Encoder count -> rad
    enc_scale = 2.0 * np.pi / args.counts_per_rotation

    # Real states
    if is_valid_array(hw_enc):
        theta_real = (hw_enc - hw_enc[0]) * enc_scale * args.theta_sign + args.theta_offset
    else:
        theta_real = np.full(len(df), np.nan, dtype=float)

    if is_valid_array(imu_wz):
        omega_real = imu_wz.copy()
        omega_real_smooth = moving_average(omega_real, args.alpha_smooth)
        alpha_real = np.gradient(omega_real_smooth, t)
    else:
        omega_real = np.full(len(df), np.nan, dtype=float)
        alpha_real = np.full(len(df), np.nan, dtype=float)

    prefix = f"{args.plot_title_prefix} | " if args.plot_title_prefix else ""

    # 1. Command vs applied PWM
    plt.figure(figsize=(10, 4))
    if is_valid_array(cmd_u):
        plt.plot(t, cmd_u, label="cmd_u")
    if is_valid_array(hw_pwm):
        plt.plot(t, hw_pwm, label="hw_pwm")
    plt.xlabel(f"{t_name} [s]")
    plt.ylabel("PWM")
    plt.title(f"{prefix}Command vs applied PWM")
    plt.grid(True)
    plt.legend()

    # 2. PWM tracking error
    plt.figure(figsize=(10, 4))
    if is_valid_array(cmd_u) and is_valid_array(hw_pwm):
        pwm_error = cmd_u - hw_pwm
        plt.plot(t, pwm_error, label="cmd_u - hw_pwm")
    plt.xlabel(f"{t_name} [s]")
    plt.ylabel("PWM error")
    plt.title(f"{prefix}PWM tracking error")
    plt.grid(True)
    plt.legend()

    # 3. Theta sim vs real
    plt.figure(figsize=(10, 4))
    if is_valid_array(theta_sim):
        plt.plot(t, theta_sim, label="theta sim")
    if is_valid_array(theta_real):
        plt.plot(t, theta_real, label="theta real")
    plt.xlabel(f"{t_name} [s]")
    plt.ylabel("rad")
    plt.title(f"{prefix}Theta: sim vs real")
    plt.grid(True)
    plt.legend()

    # 4. Omega sim vs real
    plt.figure(figsize=(10, 4))
    if is_valid_array(omega_sim):
        plt.plot(t, omega_sim, label="omega sim")
    if is_valid_array(omega_real):
        plt.plot(t, omega_real, label="omega real")
    plt.xlabel(f"{t_name} [s]")
    plt.ylabel("rad/s")
    plt.title(f"{prefix}Omega: sim vs real")
    plt.grid(True)
    plt.legend()

    # 5. Alpha sim vs real
    plt.figure(figsize=(10, 4))
    if is_valid_array(alpha_sim):
        plt.plot(t, alpha_sim, label="alpha sim")
    if is_valid_array(alpha_real):
        plt.plot(t, alpha_real, label="alpha real")
    plt.xlabel(f"{t_name} [s]")
    plt.ylabel("rad/s^2")
    plt.title(f"{prefix}Alpha: sim vs real")
    plt.grid(True)
    plt.legend()

    print(f"Loaded csv: {csv_path}")
    print(f"time column            : {t_name}")
    print(f"counts per rotation    : {args.counts_per_rotation:.6f}")
    print(f"encoder scale [rad/cnt]: {enc_scale:.9f}")
    print(f"theta_sign             : {args.theta_sign}")
    print(f"theta_offset [rad]     : {args.theta_offset:.9f}")
    print(f"alpha_smooth           : {args.alpha_smooth}")
    print()
    print("CSV columns:")
    print(list(df.columns))
    print()
    print(df.head())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
