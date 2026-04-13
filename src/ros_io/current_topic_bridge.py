"""Bridge ROS current/IMU topics to filtered files for digital-twin runtime."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from .signal_filter import CausalIIRFilter


def quaternion_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


def compute_current_offset(samples: list[tuple[float, float, float]], fallback_ma: float = 26.0) -> tuple[float, int]:
    valid = [cur for cur, pwm, omg in samples if abs(pwm) < 3.0 and abs(omg) < 0.8]
    if len(valid) < 12:
        return float(fallback_ma), len(valid)
    valid.sort()
    mid = len(valid) // 2
    return (float(valid[mid]), len(valid)) if len(valid) % 2 else (float(0.5 * (valid[mid - 1] + valid[mid])), len(valid))


class CurrentTopicBridge(Node):
    """Subscribe to current/IMU/PWM topics and persist warmup-corrected filtered signals."""

    def __init__(self, current_topic: str, imu_topic: str, pwm_topic: str, out_path: Path, log_path: Path, warmup_sec: float):
        super().__init__("current_topic_bridge")
        self.out_path = out_path
        self.log_path = log_path
        self.warmup_sec = warmup_sec
        self.t0 = time.time()
        self.phase = "warmup"
        self.current_raw_ma = 0.0
        self.pwm = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self.current_offset_ma = 0.0
        self.warmup_samples: list[tuple[float, float, float]] = []
        self.current_filter = CausalIIRFilter(alpha=0.18)
        self.theta_filter = CausalIIRFilter(alpha=0.18)
        self.omega_filter = CausalIIRFilter(alpha=0.18)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_fp = self.log_path.open("w", newline="", encoding="utf-8")
        self.csv_wr = csv.writer(self.csv_fp)
        self.csv_wr.writerow(
            [
                "wall_time",
                "phase",
                "current_raw_ma",
                "current_offset_ma",
                "current_corr_ma",
                "current_filt_ma",
                "pwm",
                "theta_rad",
                "theta_filt_rad",
                "omega_rad_s",
                "omega_filt_rad_s",
            ]
        )

        self.create_subscription(Float32, current_topic, self.on_current, 20)
        self.create_subscription(Float32, pwm_topic, self.on_pwm, 20)
        self.create_subscription(Imu, imu_topic, self.on_imu, 20)
        self.create_timer(0.02, self.on_timer)
        self.get_logger().info(f"Bridging current={current_topic}, imu={imu_topic}, pwm={pwm_topic}")
        self.get_logger().info(f"state={out_path} log={log_path}")

    def on_current(self, msg: Float32) -> None:
        self.current_raw_ma = float(msg.data)

    def on_pwm(self, msg: Float32) -> None:
        self.pwm = float(msg.data)

    def on_imu(self, msg: Imu) -> None:
        self.theta = quaternion_to_pitch(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        self.omega = float(msg.angular_velocity.y)

    def on_timer(self) -> None:
        now = time.time()
        elapsed = now - self.t0
        if self.phase == "warmup":
            self.warmup_samples.append((self.current_raw_ma, self.pwm, self.omega))
            if elapsed >= self.warmup_sec:
                self.current_offset_ma, valid_count = compute_current_offset(self.warmup_samples)
                self.phase = "run"
                self.get_logger().info(
                    f"Warmup done. offset={self.current_offset_ma:.3f}mA valid_samples={valid_count}"
                )

        current_corr = self.current_raw_ma - self.current_offset_ma
        sign_pwm = 1.0 if self.pwm > 0.0 else (-1.0 if self.pwm < 0.0 else 0.0)
        current_signed = sign_pwm * current_corr

        current_f = self.current_filter.update(current_signed)
        theta_f = self.theta_filter.update(self.theta)
        omega_f = self.omega_filter.update(self.omega)

        self.out_path.write_text(f"{now:.6f},{current_f:.6f}\n", encoding="utf-8")
        self.csv_wr.writerow(
            [
                f"{now:.6f}",
                self.phase,
                f"{self.current_raw_ma:.6f}",
                f"{self.current_offset_ma:.6f}",
                f"{current_corr:.6f}",
                f"{current_f:.6f}",
                f"{self.pwm:.6f}",
                f"{self.theta:.6f}",
                f"{theta_f:.6f}",
                f"{self.omega:.6f}",
                f"{omega_f:.6f}",
            ]
        )
        self.csv_fp.flush()

    def destroy_node(self) -> bool:
        try:
            self.csv_fp.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge INA219/IMU topics to runtime input + log")
    parser.add_argument("--topic", default="/ina219/current_ma")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--pwm-topic", default="/hw/pwm_applied")
    parser.add_argument("--out", type=Path, default=Path("/tmp/chrono_current_ma.txt"))
    parser.add_argument("--log", type=Path, default=Path("data/raw/current_bridge_log.csv"))
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = CurrentTopicBridge(
        current_topic=args.topic,
        imu_topic=args.imu_topic,
        pwm_topic=args.pwm_topic,
        out_path=args.out,
        log_path=args.log,
        warmup_sec=args.warmup_sec,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
