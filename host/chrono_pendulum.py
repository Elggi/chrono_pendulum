#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from chrono_core.chrono_rigid_pendulum import ChronoRigidPendulum, CsvLogger, TorqueController, ema_filter, load_pendulum_params


def load_calibration_radius(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    for key in ("mean_radius_m", "r_from_imu_orientation", "r_imu"):
        v = summary.get(key, data.get(key))
        if v is not None:
            return float(v)
    raise ValueError(f"Could not find IMU radius in {path}")


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


class NNResidualRuntime:
    def __init__(self, motor_json: dict):
        self.meta = motor_json.get("learned_residual", {})
        if not self.meta.get("enabled", False):
            raise RuntimeError("learned_residual not enabled in motor_torque.json")
        self.ckpt = Path(self.meta["checkpoint_path"])  # absolute recommended
        self.scaler = Path(self.meta["normalization_path"])
        self.schema = Path(self.meta["schema_path"])
        for p in (self.ckpt, self.scaler, self.schema):
            if not p.exists():
                raise FileNotFoundError(f"NN runtime file missing: {p}")

        import torch
        from mnode.models import ResidualMLP

        schema = json.loads(self.schema.read_text(encoding="utf-8"))
        scaler = json.loads(self.scaler.read_text(encoding="utf-8"))
        self.features = list(schema.get("input_features", ["theta", "omega", "input_current"]))
        self.mean = np.asarray(scaler["mean"], dtype=float)
        self.std = np.asarray(scaler["std"], dtype=float)

        self.model = ResidualMLP(in_dim=len(self.features), hidden_dim=int(schema.get("hidden_dim", 64)), out_dim=1)
        self.model.load_state_dict(torch.load(self.ckpt, map_location="cpu"))
        self.model.eval()
        self.torch = torch

        print("[NN] residual runtime active")
        print(f"[NN] checkpoint: {self.ckpt}")
        print(f"[NN] schema features: {self.features}, output=tau_learned")

    def predict_tau(self, theta: float, omega: float, current: float) -> float:
        vals = {"theta": theta, "omega": omega, "input_current": current}
        x = np.array([vals[f] for f in self.features], dtype=float)
        xn = (x - self.mean) / np.maximum(self.std, 1e-8)
        with self.torch.no_grad():
            y = self.model(self.torch.tensor(xn, dtype=self.torch.float32).view(1, -1)).item()
        return float(y)


class InputSource:
    def sample(self, t: float) -> float:
        raise NotImplementedError


class HostSignalSource(InputSource):
    def __init__(self, amp_a: float = 0.8, hz: float = 0.5):
        self.amp = float(amp_a)
        self.hz = float(hz)

    def sample(self, t: float) -> float:
        return float(self.amp * math.sin(2.0 * math.pi * self.hz * t))


class ReplayCurrent(InputSource):
    def __init__(self, csv_path: Path):
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("Replay CSV is empty")
        self.t = np.array([float(r.get("time", r.get("wall_elapsed", 0.0))) for r in rows], dtype=float)
        for c in ("input_current", "ina_current_signed_mA", "current_mA"):
            if c in rows[0]:
                scale = 0.001 if "mA" in c else 1.0
                self.i = np.array([float(r[c]) * scale for r in rows], dtype=float)
                break
        else:
            raise ValueError("Replay CSV must include input_current or signed current column")

    def sample(self, t: float) -> float:
        return float(np.interp(t, self.t, self.i))


class RosCurrentBridge(Node, InputSource):
    def __init__(self, current_topic: str, imu_topic: str):
        Node.__init__(self, "chrono_pendulum_current_bridge")
        self.current_a = 0.0
        self.latest_imu_theta = 0.0
        self.has_imu = False
        self.create_subscription(Float32, current_topic, self.cb_current, 20)
        self.create_subscription(Imu, imu_topic, self.cb_imu, 50)
        self.pub_theta = self.create_publisher(Float32, "/sim/theta", 20)
        self.pub_omega = self.create_publisher(Float32, "/sim/omega", 20)
        self.pub_tau = self.create_publisher(Float32, "/sim/tau_applied", 20)

    def cb_current(self, msg: Float32) -> None:
        value = float(msg.data)
        self.current_a = value * 0.001 if abs(value) > 50.0 else value

    def cb_imu(self, msg: Imu) -> None:
        self.latest_imu_theta = quat_to_yaw(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
        self.has_imu = True

    def sample(self, t: float) -> float:
        _ = t
        return float(self.current_a)

    def publish_state(self, theta: float, omega: float, tau: float) -> None:
        m_theta = Float32(); m_theta.data = float(theta)
        m_omega = Float32(); m_omega.data = float(omega)
        m_tau = Float32(); m_tau.data = float(tau)
        self.pub_theta.publish(m_theta)
        self.pub_omega.publish(m_omega)
        self.pub_tau.publish(m_tau)


def choose_runtime_model(args: argparse.Namespace, motor_data: dict) -> str:
    if args.runtime_model:
        return args.runtime_model
    print("Select runtime model:")
    print("  1) learned equation model")
    print("  2) learned NN residual model (.pt)")
    ch = input("Enter 1 or 2 [1]: ").strip() or "1"
    if ch == "2":
        return "nn_residual"
    return "equation"


def run(args: argparse.Namespace) -> None:
    calib_path = Path(args.calibration_json or input("Select calibration_latest.json path: ").strip())
    torque_path = Path(args.motor_torque_json or input("Select motor_torque.json path: ").strip())
    motor_data = json.loads(torque_path.read_text(encoding="utf-8"))
    runtime_model = choose_runtime_model(args, motor_data)

    imu_radius = load_calibration_radius(calib_path)
    params = load_pendulum_params(torque_path, imu_radius=imu_radius)

    nn_runtime = None
    if runtime_model == "nn_residual":
        try:
            nn_runtime = NNResidualRuntime(motor_data)
        except Exception as exc:
            raise SystemExit(f"Failed to load NN residual runtime: {exc}")

    collision = args.collision_test
    if not args.no_prompt_collision and not collision:
        collision = input("Enable collision test (drop sphere at 5s)? [y/N]: ").strip().lower() == "y"

    ros_bridge: RosCurrentBridge | None = None
    if args.mode == "offline":
        if not args.replay_csv:
            raise SystemExit("--replay_csv required in offline mode")
        source: InputSource = ReplayCurrent(Path(args.replay_csv))
    elif args.mode in ("jetson", "live") or (args.mode == "host" and args.host_input == "ros"):
        rclpy.init(args=None)
        ros_bridge = RosCurrentBridge(current_topic=args.current_topic, imu_topic=args.imu_topic)
        source = ros_bridge
    elif args.mode == "host" and args.host_input == "replay":
        if not args.replay_csv:
            raise SystemExit("--replay_csv required when --host_input replay")
        source = ReplayCurrent(Path(args.replay_csv))
    else:
        source = HostSignalSource(amp_a=args.host_amp_a, hz=args.host_hz)

    pendulum = ChronoRigidPendulum(params, enable_collision=collision)
    controller = TorqueController(params.motor)
    logger = CsvLogger(Path(args.output_dir))

    rows: list[dict] = []
    try:
        while pendulum.sys.GetChTime() < args.duration_sec:
            t = float(pendulum.sys.GetChTime())
            if ros_bridge is not None:
                rclpy.spin_once(ros_bridge, timeout_sec=0.0)

            in_warmup = bool(args.free_decay_capture and t < args.warmup_sec and ros_bridge is not None and ros_bridge.has_imu)
            if in_warmup:
                pendulum.sync_to_theta(ros_bridge.latest_imu_theta)
                i_t = 0.0
                tau_cmd = 0.0
                theta, omega, alpha = pendulum.read_state()
                pendulum.apply_torque(0.0)
                pendulum.step(args.dt)
            else:
                i_t = float(source.sample(t))
                theta, omega, alpha = pendulum.read_state()
                tau_nom = controller.compute(i_t, theta, omega)
                tau_learned = nn_runtime.predict_tau(theta, omega, i_t) if nn_runtime is not None else 0.0
                tau_cmd = tau_nom + tau_learned
                pendulum.apply_torque(tau_cmd)
                if collision and t >= 5.0:
                    pendulum.drop_collision_ball()
                pendulum.step(args.dt)
                if nn_runtime is not None and (len(rows) % 500 == 0):
                    print(f"[NN mode] t={t:.2f}s tau_nom={tau_nom:.4f} tau_nn={tau_learned:.4f} tau_cmd={tau_cmd:.4f}")

            if ros_bridge is not None:
                ros_bridge.publish_state(theta, omega, tau_cmd)

            rows.append({
                "time": t,
                "input_current": i_t,
                "commanded_torque": tau_cmd,
                "theta": theta,
                "omega": omega,
                "alpha": alpha,
                "phase": "warmup_sync" if in_warmup else "run",
                "runtime_model": runtime_model,
            })
    finally:
        if ros_bridge is not None:
            ros_bridge.destroy_node()
            rclpy.shutdown()

    logger.write(rows, ema_filter(rows, alpha=0.15))
    print(f"[OK] wrote {len(rows)} rows to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chrono rigid pendulum runtime with equation/NN mode selection")
    ap.add_argument("--mode", choices=["host", "jetson", "offline", "live"], default="host")
    ap.add_argument("--runtime_model", choices=["equation", "nn_residual"], default="")
    ap.add_argument("--calibration_json", default="")
    ap.add_argument("--motor_torque_json", default="")
    ap.add_argument("--replay_csv", default="")
    ap.add_argument("--current_topic", default="/ina219/current_ma")
    ap.add_argument("--imu_topic", default="/imu/data")
    ap.add_argument("--host_input", choices=["synthetic", "ros", "replay"], default="ros")
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--duration_sec", type=float, default=10.0)
    ap.add_argument("--host_amp_a", type=float, default=0.8)
    ap.add_argument("--host_hz", type=float, default=0.5)
    ap.add_argument("--output_dir", default="host/run_logs")
    ap.add_argument("--free_decay_capture", action="store_true")
    ap.add_argument("--warmup_sec", type=float, default=1.0)
    ap.add_argument("--collision_test", action="store_true")
    ap.add_argument("--no_prompt_collision", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
