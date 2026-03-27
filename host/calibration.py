#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


DEFAULT_PWM_STEP = 1.0
DEFAULT_LOOP_HZ = 30.0
DEFAULT_COUNTS_PER_REV = 360.0
BRAKE_ZONE_RAD = math.radians(55.0)
BRAKE_WZ_THRESHOLD = 1.2
BRAKE_GAIN = 8.0
BRAKE_MIN_PWM = 15.0


def quat_to_rotmat_ros(q):
    w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


class SysIdNode(Node):
    def __init__(self, args):
        super().__init__("system_identification")
        self.args = args

        self.pub_cmd = self.create_publisher(Float32, args.cmd_topic, 10)

        self.imu_msg = None
        self.hw_enc = 0.0
        self.hw_pwm = 0.0

        self.create_subscription(Imu, args.imu_topic, self.cb_imu, 100)
        self.create_subscription(Float32, args.hw_enc_topic, self.cb_hw_enc, 100)
        self.create_subscription(Float32, args.hw_pwm_topic, self.cb_hw_pwm, 100)

        self.imu_R0 = None
        self.prev_tip_angle = None
        self.yaw_unwrapped = 0.0
        self.prev_full_rot_count = 0
        self.prev_rot_enc = None
        self.cpr_samples = []
        self.r_samples = []
        self.prev_time = None
        self.prev_wz = None
        self.last_alpha = 0.0
        self.cmd_last = 0.0
        self.run_start_time = time.time()
        self.samples = []

    def cb_imu(self, msg: Imu):
        self.imu_msg = msg

    def cb_hw_enc(self, msg: Float32):
        self.hw_enc = float(msg.data)

    def cb_hw_pwm(self, msg: Float32):
        self.hw_pwm = float(msg.data)

    def publish_pwm(self, pwm: float):
        msg = Float32()
        msg.data = float(pwm)
        self.pub_cmd.publish(msg)
        self.cmd_last = float(pwm)

    def spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.0)

    def update_rotation_tracking(self):
        if self.imu_msg is None:
            return

        now = time.time()
        q = self.imu_msg.orientation
        R_abs = quat_to_rotmat_ros(q)

        if self.imu_R0 is None:
            self.imu_R0 = R_abs.copy()
            tip0 = np.array([0.0, -1.0, 0.0], dtype=float)
            self.prev_tip_angle = math.atan2(float(tip0[1]), float(tip0[0]))
            self.prev_time = now
            self.prev_wz = float(self.imu_msg.angular_velocity.z)
            return

        R_rel = self.imu_R0.T @ R_abs
        tip = R_rel @ np.array([0.0, -1.0, 0.0], dtype=float)
        tip_angle = math.atan2(float(tip[1]), float(tip[0]))

        dtheta = tip_angle - self.prev_tip_angle
        while dtheta > math.pi:
            dtheta -= 2.0 * math.pi
        while dtheta < -math.pi:
            dtheta += 2.0 * math.pi

        self.yaw_unwrapped += dtheta
        self.prev_tip_angle = tip_angle

        current_full_rot = self.full_rotations_detected
        if self.prev_rot_enc is None:
            self.prev_rot_enc = self.hw_enc
            self.prev_full_rot_count = current_full_rot
        elif current_full_rot > self.prev_full_rot_count:
            for _ in range(current_full_rot - self.prev_full_rot_count):
                self.cpr_samples.append(abs(self.hw_enc - self.prev_rot_enc))
                self.prev_rot_enc = self.hw_enc
            self.prev_full_rot_count = current_full_rot

        wz = float(self.imu_msg.angular_velocity.z)
        ax = float(self.imu_msg.linear_acceleration.x)
        if self.prev_time is not None and self.prev_wz is not None:
            dt = max(now - self.prev_time, 1e-4)
            alpha = (wz - self.prev_wz) / dt
            self.last_alpha = alpha
            if abs(alpha) > 0.2:
                self.r_samples.append(abs(ax) / abs(alpha))

        self.prev_time = now
        self.prev_wz = wz

    @property
    def full_rotations_detected(self) -> int:
        return int(math.floor(abs(self.yaw_unwrapped) / (2.0 * math.pi)))

    def log_sample(self):
        now = time.time()
        mean_cpr = float(np.mean(self.cpr_samples)) if self.cpr_samples else np.nan
        r_est = float(np.mean(self.r_samples)) if self.r_samples else np.nan
        self.samples.append(
            {
                "time_sec": now - self.run_start_time,
                "theta_rad": float(self.yaw_unwrapped),
                "omega_rad_s": float(self.imu_msg.angular_velocity.z) if self.imu_msg is not None else np.nan,
                "alpha_rad_s2": float(self.last_alpha),
                "delay_ms": (1000.0 * (now - self.prev_time)) if self.prev_time is not None else np.nan,
                "mean_cpr_running": mean_cpr,
                "r_running": r_est,
                "cmd_u": float(self.cmd_last),
                "hw_pwm": float(self.hw_pwm),
                "hw_enc": float(self.hw_enc),
                "full_rotations": float(self.full_rotations_detected),
            }
        )


def wait_for_imu(node: SysIdNode, timeout_sec: float) -> None:
    t_end = time.time() + timeout_sec
    while time.time() < t_end:
        node.spin_once()
        node.update_rotation_tracking()
        if node.imu_msg is not None:
            return
        time.sleep(0.01)
    raise RuntimeError("IMU 데이터를 받지 못했습니다. 토픽 연결 상태를 확인하세요.")


def ramp_until_target_rotations(
    node: SysIdNode,
    direction: int,
    pwm_limit: float,
    pwm_step: float,
    loop_hz: float,
    target_rotations: int = 2,
):
    pwm_mag = 0.0
    start_enc = node.hw_enc
    start_yaw = node.yaw_unwrapped
    target_angle = float(target_rotations) * 2.0 * math.pi

    while True:
        node.spin_once()
        node.update_rotation_tracking()
        node.log_sample()

        signed_progress = direction * (node.yaw_unwrapped - start_yaw)
        current_rots = signed_progress / (2.0 * math.pi)
        remaining = target_angle - signed_progress

        if signed_progress >= target_angle:
            stop_start = time.time()
            while time.time() - stop_start < node.args.brake_duration_sec:
                brake_pwm = -direction * max(node.args.brake_min_pwm, 0.25 * pwm_limit)
                node.publish_pwm(brake_pwm)
                node.spin_once()
                node.update_rotation_tracking()
                node.log_sample()
                time.sleep(0.01)
            node.publish_pwm(0.0)
            break

        pwm_mag = min(pwm_mag + pwm_step, pwm_limit)
        cmd = direction * pwm_mag
        if remaining < node.args.brake_zone_rad and abs(node.prev_wz or 0.0) > node.args.brake_wz_threshold:
            brake_mag = min(
                pwm_limit,
                max(node.args.brake_min_pwm, node.args.brake_gain * abs(node.prev_wz or 0.0)),
            )
            cmd = -direction * brake_mag
        node.publish_pwm(cmd)

        print(
            f"\rdir={direction:+d} cmd={cmd:7.2f} hw_pwm={node.hw_pwm:7.2f} "
            f"rot={current_rots:4.2f}/{target_rotations} total_rot={node.full_rotations_detected}",
            end="",
            flush=True,
        )

        time.sleep(1.0 / max(loop_hz, 1.0))

    print(f"\n[INFO] full rotation {target_rotations}회 감지 -> 즉시 정지")
    end_enc = node.hw_enc
    return start_enc, end_enc, start_yaw


def hard_stop_and_check_overshoot(node: SysIdNode, settle_sec: float):
    node.publish_pwm(0.0)
    t_end = time.time() + settle_sec
    while time.time() < t_end:
        node.spin_once()
        node.update_rotation_tracking()
        node.log_sample()
        node.publish_pwm(0.0)
        time.sleep(0.02)


def save_calibration_csv(path: str, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = [
        "time_sec",
        "theta_rad",
        "omega_rad_s",
        "alpha_rad_s2",
        "delay_ms",
        "mean_cpr_running",
        "r_running",
        "cmd_u",
        "hw_pwm",
        "hw_enc",
        "full_rotations",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in samples:
            f.write(",".join(str(row.get(k, np.nan)) for k in keys) + "\n")


def run_calibration(node: SysIdNode, args):
    wait_for_imu(node, timeout_sec=args.imu_wait_sec)

    user_input = input("최대 PWM 한계값을 입력하세요 (예: 80): ").strip()
    pwm_limit = float(user_input)
    pwm_limit = abs(min(pwm_limit, args.max_pwm_hard_limit))

    print(f"[INFO] 사용자 최대 PWM={pwm_limit:.2f}, step=1.00")

    start_enc, end_enc, start_yaw = ramp_until_target_rotations(
        node=node,
        direction=1,
        pwm_limit=pwm_limit,
        pwm_step=DEFAULT_PWM_STEP,
        loop_hz=args.loop_hz,
        target_rotations=2,
    )

    hard_stop_and_check_overshoot(node, settle_sec=args.stop_settle_sec)

    forward_rotations_after_stop = (node.yaw_unwrapped - start_yaw) / (2.0 * math.pi)
    extra_rotations = max(0, int(math.floor(forward_rotations_after_stop)) - 2)
    print(f"[INFO] 정지 후 추가 full rotation={extra_rotations}")

    reverse_turns_done = 0
    for _ in range(extra_rotations):
        before = node.full_rotations_detected
        ramp_until_target_rotations(
            node=node,
            direction=-1,
            pwm_limit=pwm_limit,
            pwm_step=DEFAULT_PWM_STEP,
            loop_hz=args.loop_hz,
            target_rotations=1,
        )
        hard_stop_and_check_overshoot(node, settle_sec=args.stop_settle_sec)
        after = node.full_rotations_detected
        reverse_turns_done += max(0, after - before)

    node.publish_pwm(0.0)

    mean_cpr = float(np.mean(node.cpr_samples)) if node.cpr_samples else None
    r_est = float(np.mean(node.r_samples)) if node.r_samples else None

    print(f"[CALIB] mean CPR = {mean_cpr}")
    print(f"[CALIB] IMU Orientation 기반 r = {r_est}")

    save_calibration_csv(args.output_csv, node.samples)

    result = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_pwm_limit": pwm_limit,
        "pwm_step": DEFAULT_PWM_STEP,
        "detected_full_rotations": node.full_rotations_detected,
        "extra_rotations_after_stop": extra_rotations,
        "reverse_turns_done": reverse_turns_done,
        "mean_cpr": mean_cpr,
        "r_from_imu_orientation": r_est,
        "encoder_start": start_enc,
        "encoder_end": end_enc,
        "yaw_unwrapped_rad": node.yaw_unwrapped,
        "calibration_csv": args.output_csv,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[INFO] calibration 결과 저장: {args.output_json}")
    print(f"[INFO] calibration 로그 CSV 저장: {args.output_csv}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd-topic", default="/cmd/u")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--hw-enc-topic", default="/hw/enc")
    ap.add_argument("--hw-pwm-topic", default="/hw/pwm_applied")
    ap.add_argument("--output-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--output-csv", default="./run_logs/calibration_latest.csv")
    ap.add_argument("--loop-hz", type=float, default=DEFAULT_LOOP_HZ)
    ap.add_argument("--max-pwm-hard-limit", type=float, default=120.0)
    ap.add_argument("--counts-per-rev", type=float, default=DEFAULT_COUNTS_PER_REV)
    ap.add_argument("--imu-wait-sec", type=float, default=5.0)
    ap.add_argument("--stop-settle-sec", type=float, default=0.8)
    ap.add_argument("--brake-zone-rad", type=float, default=BRAKE_ZONE_RAD)
    ap.add_argument("--brake-wz-threshold", type=float, default=BRAKE_WZ_THRESHOLD)
    ap.add_argument("--brake-gain", type=float, default=BRAKE_GAIN)
    ap.add_argument("--brake-min-pwm", type=float, default=BRAKE_MIN_PWM)
    ap.add_argument("--brake-duration-sec", type=float, default=0.12)
    ap.add_argument("--no-imu-viewer", action="store_true")
    return ap


def maybe_launch_imu_viewer(args):
    if args.no_imu_viewer:
        return None

    viewer_path = os.path.join(os.path.dirname(__file__), "imu_viewer.py")
    if not os.path.exists(viewer_path):
        return None

    return subprocess.Popen(
        [
            sys.executable,
            viewer_path,
            "--imu_topic",
            args.imu_topic,
            "--enc_topic",
            args.hw_enc_topic,
        ]
    )


def main():
    args = build_argparser().parse_args()
    rclpy.init()

    viewer_proc = maybe_launch_imu_viewer(args)
    node = SysIdNode(args)

    try:
        run_calibration(node, args)
    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    finally:
        node.publish_pwm(0.0)
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
