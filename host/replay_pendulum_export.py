#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

from chrono_core.calibration_io import apply_calibration_json
from chrono_core.config import BridgeConfig
from chrono_core.log_schema import PENDULUM_LOG_COLUMNS
from chrono_core.pendulum_rl_env import build_init_params, load_replay_csv, simulate_trajectory, weighted_loss, compute_error_features


def main():
    ap = argparse.ArgumentParser(description="Replay CSV export using trained pendulum parameters")
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default="")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--delay_override", type=float, default=None)
    args = ap.parse_args()

    cfg = BridgeConfig()
    calib = apply_calibration_json(cfg, args.calibration_json, apply_model_init=True)
    param = None
    if args.parameter_json:
        with open(args.parameter_json, "r", encoding="utf-8") as f:
            param = json.load(f)

    params = build_init_params(cfg, calib, param)
    traj = load_replay_csv(args.csv, cfg, delay_override=args.delay_override)
    delay = params.get(
        "delay_sec",
        (traj.delay_sec_est if args.delay_override is None else args.delay_override),
    )
    sim = simulate_trajectory(traj, params, cfg, delay_sec=delay)
    feat = compute_error_features(traj, sim)
    loss = weighted_loss(feat, {})

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(PENDULUM_LOG_COLUMNS)
        best_cost = loss
        j_rod = (1.0 / 3.0) * cfg.rod_mass * (cfg.rod_length ** 2)
        j_imu = cfg.imu_mass * (cfg.r_imu ** 2)
        j_total = j_rod + j_imu
        for i in range(len(traj.t)):
            wr.writerow([
                0.0, traj.t[i], "replay",
                traj.cmd_u[i], sim["cmd_delayed"][i], traj.hw_pwm[i], traj.delay_sec_est, sim["tau_motor"][i] - sim["tau_res"][i],
                sim["theta"][i], sim["omega"][i], sim["alpha"][i],
                "", "",
                traj.theta_real[i], traj.omega_real[i], traj.alpha_real[i],
                delay * 1000.0,
                params["l_com"], params["b_eq"], params["tau_eq"], params["K_u"],
                j_rod, j_imu, j_total,
                sim["tau_motor"][i], sim["tau_res"][i], sim["tau_visc"][i], sim["tau_coul"][i],
                loss, best_cost,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, sim["omega"][i],
                0.0, 0.0, 0.0,
                loss, 1, 1, json.dumps(params),
            ])

    print(f"saved {outp}")


if __name__ == "__main__":
    main()
