#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> np.ndarray | None:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    return None


def _load_k_i(parameter_json: Path | None) -> float | None:
    if parameter_json is None:
        return None
    if not parameter_json.exists():
        return None
    try:
        data = json.loads(parameter_json.read_text(encoding="utf-8"))
        tm = data.get("torque_model", {}) if isinstance(data.get("torque_model"), dict) else {}
        motor = tm.get("motor", {}) if isinstance(tm.get("motor"), dict) else {}
        params = motor.get("params", {}) if isinstance(motor.get("params"), dict) else {}
        if "K_i" in params:
            return float(params["K_i"])
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Motor current/torque/speed analysis from chrono CSV")
    ap.add_argument("--csv", type=Path, required=True, help="input csv")
    ap.add_argument("--outdir", type=Path, default=None, help="output directory (default: input parent)")
    ap.add_argument("--parameter-json", type=Path, default=Path("host/model_parameter.latest.json"), help="optional parameter json to infer K_i")
    ap.add_argument("--k-i", type=float, default=None, help="override motor torque constant [Nm/A]")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    if n < 2:
        raise SystemExit(f"Not enough rows in {args.csv}")

    t = _pick_col(df, ["wall_elapsed", "t", "time", "time_sec"])
    if t is None:
        t = np.arange(n, dtype=float) * 0.01
    else:
        t = t - t[np.isfinite(t)][0]

    current_ma = _pick_col(df, ["ina_current_signed_online_mA", "I_filtered_mA", "ina_current_corr_mA", "ina_current_raw_mA", "current_mA"])
    if current_ma is None:
        current_a = _pick_col(df, ["I_filtered_A", "current_A", "ina_current_a"])
        if current_a is None:
            raise SystemExit("No current column found.")
        current_ma = current_a * 1000.0

    torque_nm = _pick_col(df, ["tau_motor", "tau_cmd"])
    if torque_nm is None:
        k_i = float(args.k_i) if args.k_i is not None else _load_k_i(args.parameter_json)
        if k_i is None:
            raise SystemExit("No torque column found and K_i unavailable. Provide --k-i or --parameter-json.")
        torque_nm = k_i * (current_ma / 1000.0)

    omega = _pick_col(df, ["omega", "omega_real", "omega_imu_filtered", "omega_imu"])
    if omega is None:
        omega = np.zeros(n, dtype=float)
    speed_rpm = omega * (60.0 / (2.0 * np.pi))

    good = np.isfinite(t) & np.isfinite(current_ma) & np.isfinite(torque_nm) & np.isfinite(speed_rpm)
    t = t[good]
    current_ma = current_ma[good]
    torque_nm = torque_nm[good]
    speed_rpm = speed_rpm[good]

    outdir = args.outdir if args.outdir is not None else args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "motor_torque_analysis.csv"
    out_png = outdir / "motor_torque_analysis.png"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "current_mA", "torque_Nm", "speed_rpm"])
        for i in range(len(t)):
            wr.writerow([float(t[i]), float(current_ma[i]), float(torque_nm[i]), float(speed_rpm[i])])

    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_l = axs[0]
        ax_r = ax_l.twinx()
        ax_l.plot(t, current_ma, color="tab:blue", label="current [mA]")
        ax_r.plot(t, torque_nm, color="tab:red", label="torque [Nm]")
        ax_l.set_ylabel("Current [mA]", color="tab:blue")
        ax_r.set_ylabel("Torque [Nm]", color="tab:red")
        ax_l.grid(alpha=0.25)
        ln1, lb1 = ax_l.get_legend_handles_labels()
        ln2, lb2 = ax_r.get_legend_handles_labels()
        ax_l.legend(ln1 + ln2, lb1 + lb2, loc="upper right")

        axs[1].plot(t, speed_rpm, color="tab:green", label="speed [RPM]")
        axs[1].set_ylabel("Speed [RPM]")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid(alpha=0.25)
        axs[1].legend()

        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] plotting failed: {exc}")

    print(f"[DONE] analysis csv: {out_csv}")
    print(f"[DONE] analysis plot: {out_png}")


if __name__ == "__main__":
    main()
