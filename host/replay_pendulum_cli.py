#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import pychrono as ch
import pychrono.irrlicht as irr

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from chrono_core.calibration_io import apply_calibration_json
from chrono_core.config import BridgeConfig
from chrono_core.dynamics import PendulumModel
from chrono_core.model_parameter_io import extract_runtime_overrides
from chrono_core.pendulum_rl_env import build_init_params, load_replay_csv, simulate_trajectory


def quat_from_theta(theta: float):
    th = float(theta)
    h = 0.5 * th
    # Explicit Z-axis quaternion avoids backend convention ambiguity.
    return float(math.cos(h)), 0.0, 0.0, float(math.sin(h))


class ReplayPublisher(Node):
    def __init__(self):
        super().__init__("replay_publisher")
        self.pub_imu = self.create_publisher(Imu, "/imu/data", 10)
        self.pub_enc = self.create_publisher(Float32, "/hw/enc", 10)

    def publish_real(self, t_sec: float, theta_real: float, omega_real: float, enc_count: float):
        msg = Imu()
        sec = int(t_sec)
        nanosec = int((t_sec - sec) * 1e9)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = "imu_link"
        qw, qx, qy, qz = quat_from_theta(theta_real)
        msg.orientation.w = qw
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.angular_velocity.z = float(omega_real)
        self.pub_imu.publish(msg)

        enc = Float32()
        enc.data = float(enc_count)
        self.pub_enc.publish(enc)


def start_imu_viewer_process():
    script_path = os.path.join(os.path.dirname(__file__), "imu_viewer.py")
    try:
        return subprocess.Popen(
            ["python3", script_path, "--imu_topic", "/imu/data", "--enc_topic", "/hw/enc"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        print(f"[WARN] Failed to start IMU viewer: {exc}")
        return None


def build_replay_series(args):
    cfg = BridgeConfig()
    if args.calibration_json:
        apply_calibration_json(cfg, args.calibration_json)

    if args.parameter_json:
        param_data = json.load(open(args.parameter_json, "r", encoding="utf-8"))
        runtime = extract_runtime_overrides(param_data, cfg)
        if "r_imu" in runtime:
            # Replay must use pivot->IMU COM radius exactly like chrono_pendulum runtime.
            cfg.r_imu = float(runtime["r_imu"])
        traj = load_replay_csv(args.csv, cfg, delay_override=args.delay_override)
        params = build_init_params(cfg, parameter_json=param_data)
        delay_sec = float(args.delay_override) if args.delay_override is not None else float(traj.delay_sec_est)
        sim = simulate_trajectory(traj, params, cfg, delay_sec=delay_sec)
        theta_sim = sim["theta"]
        theta_real = traj.theta_real
        omega_real = traj.omega_real
        pwm = traj.cmd_u
        t = traj.t
    else:
        traj = load_replay_csv(args.csv, cfg, delay_override=args.delay_override)
        theta_sim = traj.theta_real if args.use_real_as_sim else load_theta_from_csv(args.csv)
        theta_real = traj.theta_real
        omega_real = traj.omega_real
        pwm = traj.cmd_u
        t = traj.t

    if len(theta_sim) != len(theta_real):
        n = min(len(theta_sim), len(theta_real))
        theta_sim, theta_real, omega_real, pwm, t = theta_sim[:n], theta_real[:n], omega_real[:n], pwm[:n], t[:n]
    theta_sim = np.unwrap(np.asarray(theta_sim, dtype=float))
    if len(theta_sim) > 0 and len(theta_real) > 0:
        theta_sim = theta_sim + (float(theta_real[0]) - float(theta_sim[0]))
    return cfg, t, pwm, theta_sim, theta_real, omega_real


def smooth_series(x: np.ndarray, mode: str):
    if mode == "raw":
        return x
    if mode == "online":
        alpha = 0.18
        y = np.zeros_like(x, dtype=float)
        for i, v in enumerate(x):
            y[i] = (1.0 - alpha) * (y[i - 1] if i > 0 else v) + alpha * v
        return y
    # offline
    win = 21
    if len(x) < win:
        return x
    try:
        from scipy.signal import savgol_filter  # type: ignore

        return savgol_filter(x, win, 3, mode="interp")
    except Exception:
        k = np.ones(9, dtype=float) / 9.0
        return np.convolve(np.pad(x, (4, 4), mode="edge"), k, mode="valid")


def print_sampling_diagnostics(csv_path: str):
    df = pd.read_csv(csv_path)
    if "wall_elapsed" in df.columns:
        t = pd.to_numeric(df["wall_elapsed"], errors="coerce").to_numpy(dtype=float)
    elif "wall_time" in df.columns:
        t0 = pd.to_numeric(df["wall_time"], errors="coerce").to_numpy(dtype=float)
        t = t0 - t0[0]
    else:
        return
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return
    print(
        "[sampling diagnostics] "
        f"mean_dt={np.mean(dt):.6f}s, std_dt={np.std(dt):.6f}s, "
        f"min_dt={np.min(dt):.6f}s, max_dt={np.max(dt):.6f}s, "
        f"mean_f={1.0/max(np.mean(dt), 1e-9):.2f}Hz"
    )


def recompute_signed_current(csv_path: str):
    df = pd.read_csv(csv_path)
    n = len(df)
    pwm_src = df.get("hw_pwm", df.get("pwm_hw"))
    if pwm_src is None:
        pwm_src = pd.Series(np.zeros(n, dtype=float))
    pwm = pd.to_numeric(pwm_src, errors="coerce").to_numpy(dtype=float)

    i_raw_src = df.get("ina_current_raw_mA", df.get("current_mA"))
    if i_raw_src is None:
        i_raw_src = pd.Series(np.zeros(n, dtype=float))
    i_raw = pd.to_numeric(i_raw_src, errors="coerce").to_numpy(dtype=float)
    i_offset = 0.0
    if "ina_current_offset_mA" in df.columns:
        vals = pd.to_numeric(df["ina_current_offset_mA"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            i_offset = float(np.median(vals))
    i_corr = i_raw - i_offset
    i_signed = np.sign(pwm) * i_corr
    print(
        "[signed current] "
        f"offset={i_offset:.4f}mA, raw_mean={np.nanmean(i_raw):.4f}mA, "
        f"signed_mean={np.nanmean(i_signed):.4f}mA"
    )
    return i_signed


def load_theta_from_csv(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "theta" in df.columns:
        return pd.to_numeric(df["theta"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "simulated_orientation" in df.columns:
        return pd.to_numeric(df["simulated_orientation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "theta_real" in df.columns:
        return pd.to_numeric(df["theta_real"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.zeros(len(df), dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Replay CSV into Chrono sim window + IMU viewer window")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--calibration_json", default="")
    ap.add_argument("--parameter_json", default="")
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1.0=real-time)")
    ap.add_argument("--use-real-as-sim", action="store_true", help="if no parameter_json, render real theta as sim")
    ap.add_argument("--filter-mode", choices=["raw", "online", "offline"], default="raw")
    args = ap.parse_args()

    cfg, t, pwm, theta_sim, theta_real, omega_real = build_replay_series(args)
    theta_real = smooth_series(theta_real, args.filter_mode)
    omega_real = smooth_series(omega_real, args.filter_mode)
    print_sampling_diagnostics(args.csv)
    _ = recompute_signed_current(args.csv)
    if len(t) < 2:
        raise SystemExit("Not enough replay samples")

    rclpy.init()
    node = ReplayPublisher()
    viewer_proc = start_imu_viewer_process()

    model = PendulumModel(cfg)
    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(model.sys)
    vis.SetWindowSize(cfg.win_w, cfg.win_h)
    vis.SetWindowTitle("Replay Chrono Viewer")
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddCamera(ch.ChVector3d(0.7, 0.2, 0.8))
    vis.AddTypicalLights()

    enc_cpr = 2048.0
    t0_wall = time.time()
    t_base = float(t[0])

    i = 0
    imu_local = model.imu_local_on_rod()
    imu_local_vec = ch.ChVector3d(float(imu_local[0]), float(imu_local[1]), float(imu_local[2]))
    try:
        while i < len(t):
            vis.Run()
            vis.BeginScene(); vis.Render(); vis.EndScene()

            target_replay_t = (time.time() - t0_wall) * max(args.speed, 1e-6)
            while i + 1 < len(t) and (t[i + 1] - t_base) <= target_replay_t:
                i += 1

            ths = float(theta_sim[i])
            thr = float(theta_real[i])
            omr = float(omega_real[i]) if np.isfinite(omega_real[i]) else 0.0
            enc = float((thr / (2.0 * math.pi)) * enc_cpr)

            model.link.SetRot(ch.QuatFromAngleZ(ths))
            model.link.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, 0.0))
            imu_abs = model.link.TransformPointLocalToParent(imu_local_vec)
            model.imu.SetPos(imu_abs)
            model.imu.SetRot(model.link.GetRot())
            node.publish_real(float(t[i] - t_base), thr, omr, enc)
            rclpy.spin_once(node, timeout_sec=0.0)

            if i >= len(t) - 1:
                break
            time.sleep(0.001)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if viewer_proc is not None:
            try:
                os.killpg(viewer_proc.pid, signal.SIGTERM)
                viewer_proc.wait(timeout=2.0)
            except Exception:
                pass


if __name__ == "__main__":
    main()
