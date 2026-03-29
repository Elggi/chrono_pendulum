#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Replay a CSV with calibrated params and export SAME schema CSV for plot_pendulum.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chrono_core.csv_schema import PENDULUM_CSV_COLUMNS, make_csv_row
from chrono_core.pendulum_rl_env import initial_params_from_files, load_trajectories, simulate_trajectory


def load_param_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    p = obj.get("model_init") or obj.get("best_params") or obj
    return {
        "l_com": float(p["l_com"]),
        "J_cm_base": float(p["J_cm_base"]),
        "b_eq": float(p["b_eq"]),
        "tau_eq": float(p["tau_eq"]),
        "k_t": float(p["k_t"]),
        "i0": float(p["i0"]),
        "R": float(p["R"]),
        "k_e": float(p["k_e"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default=None)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--param_override_json", default=None, help="final_params.json from RL run")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--delay_override", type=float, default=None)
    args = ap.parse_args()

    base_params, cfg = initial_params_from_files(args.calibration_json, args.parameter_json)
    if args.param_override_json:
        base_params.update(load_param_json(args.param_override_json))

    trajectories = load_trajectories([args.csv], delay_override=args.delay_override)
    tr = trajectories[0]
    sim = simulate_trajectory(tr, base_params, cfg, tr.delay_est)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PENDULUM_CSV_COLUMNS)
        best = float("inf")
        for i in range(len(tr.t)):
            inst_cost = (sim["theta"][i] - tr.theta_real[i]) ** 2 + (sim["omega"][i] - tr.omega_real[i]) ** 2 + (sim["alpha"][i] - tr.alpha_real[i]) ** 2
            best = min(best, inst_cost)
            row = {
                "wall_time": tr.t[i],
                "wall_elapsed": tr.t[i],
                "sim_time": tr.t[i],
                "mode": "replay",
                "cmd_u_raw": tr.cmd_u[i],
                "cmd_u_delayed": sim["u_aligned"][i],
                "hw_pwm": tr.hw_pwm[i],
                "delay_sec_est": tr.delay_est,
                "tau_cmd": sim["tau_motor"][i] - sim["tau_res"][i],
                "theta": sim["theta"][i],
                "omega": sim["omega"][i],
                "alpha": sim["alpha"][i],
                "theta_real": tr.theta_real[i],
                "omega_real": tr.omega_real[i],
                "alpha_real": tr.alpha_real[i],
                "delay_ms": 1000.0 * tr.delay_est,
                "l_com_est": base_params["l_com"],
                "b_eq_est": base_params["b_eq"],
                "tau_eq_est": base_params["tau_eq"],
                "k_t_est": base_params["k_t"],
                "i0_est": base_params["i0"],
                "R_est": base_params["R"],
                "k_e_est": base_params["k_e"],
                "bus_v_raw": tr.bus_v[i],
                "bus_v_filtered": tr.bus_v[i],
                "current_raw_A": tr.current_a[i],
                "current_filtered_A": tr.current_a[i],
                "power_raw_W": tr.power_w[i],
                "tau_motor": sim["tau_motor"][i],
                "tau_res": sim["tau_res"][i],
                "tau_visc": base_params["b_eq"] * sim["omega"][i],
                "tau_coul": base_params["tau_eq"] * np.tanh(sim["omega"][i] / max(cfg.tanh_eps, 1e-9)),
                "i_pred": sim["i_pred"][i],
                "v_applied": sim["v_applied"][i],
                "inst_cost": inst_cost,
                "best_cost_so_far": best,
                "fit_done": 1,
                "fit_complete": 1,
            }
            w.writerow(make_csv_row(row))

    print(f"saved replay csv: {out}")


if __name__ == "__main__":
    main()
