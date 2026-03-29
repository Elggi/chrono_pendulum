#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from chrono_core.config import BridgeConfig
from chrono_core.pendulum_rl_env import simulate_trajectory
from chrono_core.replay_io import build_params_from_calibration, gather_csv_paths, load_replay_trajectory, write_runtime_csv


def main():
    ap = argparse.ArgumentParser(description="Replay logs with calibrated parameters and export runtime-compatible CSV")
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--csv_dir", default="")
    ap.add_argument("--outdir", default="host/rl_runs/export")
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--learn_delay", action="store_true")
    args = ap.parse_args()

    cfg = BridgeConfig()
    params = build_params_from_calibration(cfg, args.calibration_json, args.parameter_json or None)
    paths = gather_csv_paths(args.csv or None, args.csv_dir or None)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        tr = load_replay_trajectory(p, args.delay_override)
        d = params.get("delay_sec", tr.delay_sec_est) if args.learn_delay else tr.delay_sec_est
        sim = simulate_trajectory(cfg, tr, params, d)
        rows = []
        for i in range(len(tr.t)):
            rows.append({
                "wall_time": tr.t[i],
                "wall_elapsed": tr.t[i] - tr.t[0],
                "sim_time": tr.t[i],
                "mode": "offline_replay",
                "cmd_u_raw": tr.cmd_u[i],
                "cmd_u_delayed": sim["cmd_delayed"][i],
                "hw_pwm": tr.hw_pwm[i],
                "delay_sec_est": d,
                "tau_cmd": sim["tau_motor"][i] - sim["tau_res"][i],
                "theta": sim["theta"][i],
                "omega": sim["omega"][i],
                "alpha": sim["alpha"][i],
                "hw_enc": "",
                "hw_arduino_ms": "",
                "theta_real": tr.theta[i],
                "omega_real": tr.omega[i],
                "alpha_real": tr.alpha[i],
                "delay_ms": 1000.0 * d,
                "l_com_est": params["l_com"],
                "b_eq_est": params["b_eq"],
                "tau_eq_est": params["tau_eq"],
                "k_t_est": params["k_t"],
                "i0_est": params["i0"],
                "R_est": params["R"],
                "k_e_est": params["k_e"],
                "bus_v_raw": tr.bus_v[i],
                "bus_v_filtered": tr.bus_v[i],
                "current_raw_A": tr.current[i],
                "current_filtered_A": tr.current[i],
                "power_raw_W": tr.power[i],
                "tau_motor": sim["tau_motor"][i],
                "tau_res": sim["tau_res"][i],
                "tau_visc": "",
                "tau_coul": "",
                "i_pred": sim["current"][i],
                "v_applied": sim["power"][i] / (sim["current"][i] if abs(sim["current"][i]) > 1e-6 else 1.0),
                "inst_cost": "",
                "best_cost_so_far": "",
                "imu_qw": "", "imu_qx": "", "imu_qy": "", "imu_qz": "",
                "imu_wx": "", "imu_wy": "", "imu_wz": "", "imu_ax": "", "imu_ay": "", "imu_az": "",
                "ls_cost": "", "fit_done": 1, "fit_complete": 1, "fit_final_params": json.dumps(params),
            })
        out_csv = outdir / (Path(p).stem + "_replay.csv")
        write_runtime_csv(str(out_csv), rows)
        print(f"[export] {out_csv}")


if __name__ == "__main__":
    main()
