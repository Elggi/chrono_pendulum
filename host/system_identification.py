#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import shutil
import argparse
import math
import subprocess
import sys
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String


MAX_CALIB_PWM = 120.0
TARGET_FULL_TURNS = 2.0


def terminal_status_line(msg: str, width: int = 180):
    term_cols = shutil.get_terminal_size((width, 20)).columns
    width = min(width, term_cols)
    print("\r\033[2K" + msg[:width].ljust(width), end="", flush=True)


def quat_to_rotmat_ros(q):
    w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


class DelayEstimator:
    def __init__(self, max_delay_ms: float = 150.0, bin_ms: float = 5.0):
        self.max_delay_s = max_delay_ms * 1e-3
        self.bin_s = max(bin_ms * 1e-3, 1e-3)
        self.cmd_hist = deque()
        self.omg_hist = deque()
        self.delay_s = 0.0

    def update(self, t_now: float, cmd_u: float, omega: float):
        self.cmd_hist.append((t_now, float(cmd_u)))
        self.omg_hist.append((t_now, float(omega)))
        t_old = t_now - max(6.0, 5.0 * self.max_delay_s)
        while self.cmd_hist and self.cmd_hist[0][0] < t_old:
            self.cmd_hist.popleft()
        while self.omg_hist and self.omg_hist[0][0] < t_old:
            self.omg_hist.popleft()

    def estimate(self):
        if len(self.cmd_hist) < 40 or len(self.omg_hist) < 40:
            return self.delay_s

        cmd_t = np.array([x[0] for x in self.cmd_hist], dtype=float)
        cmd_v = np.array([x[1] for x in self.cmd_hist], dtype=float)
        omg_t = np.array([x[0] for x in self.omg_hist], dtype=float)
        omg_v = np.array([x[1] for x in self.omg_hist], dtype=float)

        t0 = max(cmd_t[0], omg_t[0])
        t1 = min(cmd_t[-1], omg_t[-1])
        if t1 <= t0 + 1.0:
            return self.delay_s

        grid = np.arange(t0, t1, self.bin_s)
        if len(grid) < 80:
            return self.delay_s

        cmd_i = np.interp(grid, cmd_t, cmd_v)
        omg_i = np.interp(grid, omg_t, omg_v)
        cmd_i -= np.mean(cmd_i)
        omg_i -= np.mean(omg_i)
        if np.std(cmd_i) < 1e-6 or np.std(omg_i) < 1e-6:
            return self.delay_s

        max_lag = int(self.max_delay_s / self.bin_s)
        best_lag, best_score = 0, -1e9
        for lag in range(0, max_lag + 1):
            c = cmd_i[:-lag] if lag > 0 else cmd_i
            o = omg_i[lag:] if lag > 0 else omg_i
            n = min(len(c), len(o))
            if n < 30:
                continue
            sc = float(np.dot(c[:n], o[:n]))
            if sc > best_score:
                best_score = sc
                best_lag = lag

        self.delay_s = best_lag * self.bin_s
        return self.delay_s


class CalibrationNode(Node):
    def __init__(self, args):
        super().__init__("system_identification")
        self.args = args
        self.pub_cmd = self.create_publisher(Float32, args.cmd_topic, 10)
        self.pub_status = self.create_publisher(String, args.status_topic, 10)
        self.last_status = None

        self.hw_pwm = 0.0
        self.hw_enc = 0.0
        self.hw_ms = 0.0
        self.bus_v = float("nan")
        self.current_ma = float("nan")
        self.power_mw = float("nan")
        self.imu_msg = None

        self.create_subscription(String, args.status_topic, self.cb_status, 100)
        self.create_subscription(Float32, args.hw_pwm_topic, self.cb_hw_pwm, 100)
        self.create_subscription(Float32, args.hw_enc_topic, self.cb_hw_enc, 100)
        self.create_subscription(Float32, args.hw_ms_topic, self.cb_hw_ms, 100)
        self.create_subscription(Float32, args.bus_v_topic, self.cb_bus_v, 100)
        self.create_subscription(Float32, args.current_ma_topic, self.cb_current_ma, 100)
        self.create_subscription(Float32, args.power_mw_topic, self.cb_power_mw, 100)
        self.create_subscription(Imu, args.imu_topic, self.cb_imu, 100)

    def cb_status(self, msg: String):
        self.last_status = msg.data

    def cb_hw_pwm(self, msg: Float32):
        self.hw_pwm = float(msg.data)

    def cb_hw_enc(self, msg: Float32):
        self.hw_enc = float(msg.data)

    def cb_hw_ms(self, msg: Float32):
        self.hw_ms = float(msg.data)

    def cb_bus_v(self, msg: Float32):
        self.bus_v = float(msg.data)

    def cb_current_ma(self, msg: Float32):
        self.current_ma = float(msg.data)

    def cb_power_mw(self, msg: Float32):
        self.power_mw = float(msg.data)

    def cb_imu(self, msg: Imu):
        self.imu_msg = msg

    def send_status(self, text: str):
        self.last_status = text
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def publish_pwm(self, value: float):
        msg = Float32()
        msg.data = float(value)
        self.pub_cmd.publish(msg)

    def snapshot(self):
        q = self.imu_msg.orientation if self.imu_msg is not None else None
        w = self.imu_msg.angular_velocity if self.imu_msg is not None else None
        a = self.imu_msg.linear_acceleration if self.imu_msg is not None else None
        return {
            "hw_pwm": self.hw_pwm,
            "hw_enc": self.hw_enc,
            "hw_ms": self.hw_ms,
            "bus_v": self.bus_v,
            "current_ma": self.current_ma,
            "power_mw": self.power_mw,
            "imu_qx": None if q is None else q.x,
            "imu_qy": None if q is None else q.y,
            "imu_qz": None if q is None else q.z,
            "imu_qw": None if q is None else q.w,
            "imu_wx": None if w is None else w.x,
            "imu_wy": None if w is None else w.y,
            "imu_wz": None if w is None else w.z,
            "imu_ax": None if a is None else a.x,
            "imu_ay": None if a is None else a.y,
            "imu_az": None if a is None else a.z,
            "last_status": self.last_status,
        }

    def monitor_text(self, step_label: str, cmd_u: float):
        snap = self.snapshot()
        return (
            f"{self.args.role:<6} step={step_label:<18} cmd={cmd_u:7.1f} pwm={snap['hw_pwm']:7.1f} "
            f"enc={snap['hw_enc']:10.1f} V={snap['bus_v']:5.2f} I={snap['current_ma']:8.1f} "
            f"wz={0.0 if snap['imu_wz'] is None else snap['imu_wz']:8.4f}"
        )

    def _write_row(self, wr, label, cmd_raw, cmd_used, delay_ms):
        snap = self.snapshot()
        wr.writerow([
            time.time(), self.args.role, label, cmd_raw, cmd_used,
            snap["hw_pwm"], snap["hw_enc"], snap["hw_ms"],
            snap["bus_v"], snap["current_ma"], snap["power_mw"],
            snap["imu_qx"], snap["imu_qy"], snap["imu_qz"], snap["imu_qw"],
            snap["imu_wx"], snap["imu_wy"], snap["imu_wz"],
            snap["imu_ax"], snap["imu_ay"], snap["imu_az"],
            delay_ms,
            snap["last_status"],
        ])
        return snap

    def run_protocol(self, csv_path):
        start_enc, last_enc = None, None
        net_encoder_delta = 0.0
        abs_encoder_travel = 0.0

        imu_R0 = None
        prev_tip_angle = None
        yaw_unwrapped = 0.0
        abs_yaw_travel = 0.0
        max_turns_observed = 0.0
        last_dtheta = 0.0
        rev_index = 0

        t_hist, wz_hist, ay_tan_hist = [], [], []
        delay_est = DelayEstimator(max_delay_ms=self.args.delay_max_ms)
        cmd_delay_line = deque()

        def update_imu_kine():
            nonlocal imu_R0, prev_tip_angle, yaw_unwrapped, abs_yaw_travel, max_turns_observed, last_dtheta, rev_index
            if self.imu_msg is None:
                return False
            q = self.imu_msg.orientation
            R_abs = quat_to_rotmat_ros(q)
            if imu_R0 is None:
                imu_R0 = R_abs.copy()
                tip = np.array([0.0, -1.0, 0.0], dtype=float)
                prev_tip_angle = math.atan2(tip[1], tip[0])
                last_dtheta = 0.0
                rev_index = 0
                return True

            R_rel = imu_R0.T @ R_abs
            tip = R_rel @ np.array([0.0, -1.0, 0.0], dtype=float)
            ang = math.atan2(float(tip[1]), float(tip[0]))
            dtheta = ang - prev_tip_angle
            while dtheta > math.pi:
                dtheta -= 2.0 * math.pi
            while dtheta < -math.pi:
                dtheta += 2.0 * math.pi
            last_dtheta = dtheta
            yaw_unwrapped += dtheta
            abs_yaw_travel += abs(dtheta)
            prev_tip_angle = ang
            max_turns_observed = max(max_turns_observed, abs(yaw_unwrapped) / (2.0 * math.pi))
            rev_index = int(math.floor(abs(yaw_unwrapped) / (2.0 * math.pi)))

            w = self.imu_msg.angular_velocity
            a = self.imu_msg.linear_acceleration
            acc_body = np.array([a.x, a.y, a.z], dtype=float)
            tangent_body = np.array([1.0, 0.0, 0.0], dtype=float)
            ay_tan = float(np.dot(acc_body, tangent_body))
            t_hist.append(time.time())
            wz_hist.append(float(w.z))
            ay_tan_hist.append(ay_tan)
            return True

        def command_with_delay_comp(cmd_raw: float, apply_delay: bool = True):
            t_now = time.time()
            delay_s = delay_est.delay_s if apply_delay else 0.0
            cmd_delay_line.append((t_now, float(cmd_raw)))
            while cmd_delay_line and cmd_delay_line[0][0] < t_now - max(2.0, 4.0 * self.args.delay_max_ms * 1e-3):
                cmd_delay_line.popleft()
            t_target = t_now - delay_s
            cmd_used = float(cmd_raw)
            for t_cmd, u_cmd in reversed(cmd_delay_line):
                if t_cmd <= t_target:
                    cmd_used = u_cmd
                    break
            return cmd_used, 1e3 * delay_s

        def step_io(label: str, cmd_raw: float, loop_sleep: bool = True, apply_delay: bool = True):
            nonlocal start_enc, last_enc, net_encoder_delta, abs_encoder_travel
            rclpy.spin_once(self, timeout_sec=0.0)
            cmd_used, delay_ms = command_with_delay_comp(cmd_raw, apply_delay=apply_delay)
            self.publish_pwm(cmd_used)
            snap = self._write_row(wr, label, cmd_raw, cmd_used, delay_ms)
            if last_enc is not None:
                d_enc = snap["hw_enc"] - last_enc
                net_encoder_delta += d_enc
                abs_encoder_travel += abs(d_enc)
            last_enc = snap["hw_enc"]
            if start_enc is None:
                start_enc = snap["hw_enc"]

            if snap["imu_wz"] is not None:
                delay_est.update(time.time(), cmd_raw, snap["imu_wz"])
                if len(wz_hist) % max(1, int(self.args.loop_hz / max(self.args.delay_update_hz, 0.1))) == 0:
                    delay_est.estimate()
            update_imu_kine()
            terminal_status_line(self.monitor_text(label, cmd_used))
            if loop_sleep:
                time.sleep(1.0 / max(self.args.loop_hz, 1.0))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([
                "wall_time", "role", "step_label", "cmd_u_raw", "cmd_u_used",
                "hw_pwm", "hw_enc", "hw_arduino_ms",
                "bus_v", "current_ma", "power_mw",
                "imu_qx", "imu_qy", "imu_qz", "imu_qw",
                "imu_wx", "imu_wy", "imu_wz",
                "imu_ax", "imu_ay", "imu_az",
                "delay_ms",
                "status_text",
            ])
            self.send_status(f"{self.args.role}:adaptive_calibration:start")

            t_settle = time.time() + self.args.settle_sec
            while time.time() < t_settle:
                step_io("settle", 0.0)

            if imu_R0 is None and self.imu_msg is None:
                self.publish_pwm(0.0)
                self.send_status(f"{self.args.role}:adaptive_calibration:imu_missing_abort")
                raise RuntimeError("IMU orientation unavailable; aborting adaptive calibration for safety.")

            def brake_until_stop(label: str, timeout_sec: float = 2.5):
                t_end = time.time() + timeout_sec
                while time.time() < t_end:
                    omega = 0.0 if self.imu_msg is None else float(self.imu_msg.angular_velocity.z)
                    if abs(omega) <= self.args.brake_omega_stop_thresh:
                        break
                    brake_pwm = max(self.args.brake_min_pwm, min(self.args.brake_kp * abs(omega), self.args.max_calib_pwm))
                    cmd = -brake_pwm if omega > 0 else brake_pwm
                    step_io(label, cmd, apply_delay=False)

            def two_turn_segment(direction: float, seg_name: str):
                cmd_mag = self.args.sweep_pwm_start
                t_last_ramp = time.time()
                t_timeout = time.time() + self.args.segment_timeout_sec
                rev_start = rev_index
                rev_target = rev_start + 2
                while rev_index < rev_target and time.time() < t_timeout:
                    if time.time() - t_last_ramp >= self.args.sweep_hold_sec:
                        cmd_mag = min(cmd_mag + self.args.sweep_pwm_step, self.args.max_calib_pwm)
                        t_last_ramp = time.time()
                    step_io(seg_name, direction * cmd_mag)
                for _ in range(max(3, int(0.15 * self.args.loop_hz))):
                    step_io(f"{seg_name}_hold_zero", 0.0, apply_delay=False)
                brake_until_stop(f"{seg_name}_brake")

            two_turn_segment(1.0, "forward_2turn")

            t_pause = time.time() + self.args.dir_pause_sec
            while time.time() < t_pause:
                step_io("dir_pause", 0.0)

            two_turn_segment(-1.0, "reverse_2turn")

            self.publish_pwm(0.0)
            time.sleep(self.args.hard_stop_hold_sec)

            if self.args.return_to_origin:
                self.send_status(f"{self.args.role}:return_to_origin:start")
                t_until = time.time() + self.args.return_timeout_sec
                while time.time() < t_until:
                    err = -yaw_unwrapped
                    if abs(err) <= self.args.return_tol_rad:
                        break
                    cmd_mag = max(self.args.return_min_pwm, min(self.args.return_kp * abs(err), self.args.max_calib_pwm))
                    cmd = cmd_mag if err > 0 else -cmd_mag
                    step_io("return_to_origin", cmd, apply_delay=False)
                self.send_status(f"{self.args.role}:return_to_origin:done")

            for _ in range(5):
                step_io("final_zero", 0.0, loop_sleep=False, apply_delay=False)
                time.sleep(0.05)

        print()
        end_enc = self.hw_enc
        turns_abs_imu = abs_yaw_travel / (2.0 * math.pi)
        cpr_estimate = abs_encoder_travel / turns_abs_imu if turns_abs_imu > 1e-6 else None

        r_estimate = None
        if len(t_hist) >= 6:
            alpha = []
            for i in range(1, len(wz_hist)):
                dt = max(t_hist[i] - t_hist[i - 1], 1e-4)
                alpha.append((wz_hist[i] - wz_hist[i - 1]) / dt)
            pairs = []
            n = min(len(alpha), len(ay_tan_hist) - 1)
            for i in range(n):
                if abs(alpha[i]) > 0.2:
                    pairs.append(abs(ay_tan_hist[i + 1]) / abs(alpha[i]))
            if pairs:
                pairs.sort()
                r_estimate = float(pairs[len(pairs) // 2])

        return {
            "encoder_start": start_enc,
            "encoder_end": end_enc,
            "encoder_net_delta": None if start_enc is None else (end_enc - start_enc),
            "encoder_signed_travel": net_encoder_delta,
            "encoder_abs_travel": abs_encoder_travel,
            "turns_net_encoder": None if start_enc is None else (end_enc - start_enc) / max(self.args.counts_per_rev, 1e-9),
            "turns_abs_imu": turns_abs_imu,
            "turns_peak_observed": max_turns_observed,
            "revolutions_detected": rev_index,
            "counts_per_revolution_est": cpr_estimate,
            "moment_arm_r_est_m": r_estimate,
            "delay_est_ms": 1e3 * delay_est.delay_s,
        }


def make_labeled_path(folder: str, prefix: str, role: str, ext: str):
    os.makedirs(folder, exist_ok=True)
    idx = 1
    while True:
        path = os.path.join(folder, f"{prefix}{role}_{idx}{ext}")
        if not os.path.exists(path):
            return path
        idx += 1


def save_calibration_json(args, result):
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["host"], default="host")
    ap.add_argument("--cmd-topic", default="/cmd/u")
    ap.add_argument("--status-topic", default="/calibration/status")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--hw-pwm-topic", default="/hw/pwm_applied")
    ap.add_argument("--hw-enc-topic", default="/hw/enc")
    ap.add_argument("--hw-ms-topic", default="/hw/arduino_ms")
    ap.add_argument("--bus-v-topic", default="/ina219/bus_voltage_v")
    ap.add_argument("--current-ma-topic", default="/ina219/current_ma")
    ap.add_argument("--power-mw-topic", default="/ina219/power_mw")
    ap.add_argument("--loop-hz", type=float, default=25.0)
    ap.add_argument("--output-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--log-dir", default="./run_logs")
    ap.add_argument("--max-calib-pwm", type=float, default=MAX_CALIB_PWM)
    ap.add_argument("--counts-per-rev", type=float, default=360.0)
    ap.add_argument("--settle-sec", type=float, default=1.0)
    ap.add_argument("--sweep-pwm-start", type=float, default=18.0)
    ap.add_argument("--sweep-pwm-step", type=float, default=6.0)
    ap.add_argument("--sweep-hold-sec", type=float, default=0.6)
    ap.add_argument("--dir-pause-sec", type=float, default=0.8)
    ap.add_argument("--segment-timeout-sec", type=float, default=25.0)
    ap.add_argument("--hard-stop-hold-sec", type=float, default=0.2)
    ap.add_argument("--brake-kp", type=float, default=22.0)
    ap.add_argument("--brake-min-pwm", type=float, default=25.0)
    ap.add_argument("--brake-timeout-sec", type=float, default=4.0)
    ap.add_argument("--brake-omega-stop-thresh", type=float, default=0.25)
    ap.add_argument("--return-to-origin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--return-timeout-sec", type=float, default=25.0)
    ap.add_argument("--return-tol-rad", type=float, default=0.08)
    ap.add_argument("--return-kp", type=float, default=30.0)
    ap.add_argument("--return-min-pwm", type=float, default=22.0)
    ap.add_argument("--delay-max-ms", type=float, default=120.0)
    ap.add_argument("--delay-update-hz", type=float, default=5.0)
    ap.add_argument("--no-imu-viewer", action="store_true")
    return ap


def host_main(args):
    node = CalibrationNode(args)
    host_csv = make_labeled_path(args.log_dir, "calibration_", "host", ".csv")
    node.send_status("host:adaptive_calibration:started")
    print("[INFO] host adaptive calibration 시작")
    host_summary = node.run_protocol(host_csv)
    node.send_status("host:protocol:finished")
    print(f"[CALIB] CPR_est={host_summary.get('counts_per_revolution_est')}")
    print(f"[CALIB] delay_est_ms={host_summary.get('delay_est_ms')}")
    print(f"[CALIB] r_est(m)={host_summary.get('moment_arm_r_est_m')}")
    print(f"[CALIB] turns_peak_observed={host_summary.get('turns_peak_observed')}")

    result = {
        "version": 2,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "workflow": "two_turn_forward_reverse_with_delay_comp",
        "counts_per_revolution": host_summary.get("counts_per_revolution_est"),
        "r_sensor_from_center_m": host_summary.get("moment_arm_r_est_m"),
        "imu_mount": {
            "r_link_frame": host_summary.get("moment_arm_r_est_m"),
            "sensor_to_link_quat": None,
            "note": "link length is defined in chrono_pendulum BridgeConfig (visual/body settings). dynamics use estimated moment arm r from IMU.",
        },
        "delay": {
            "policy": "online_compensation_in_system_identification",
            "estimated_in_calibration": True,
            "estimated_delay_ms": host_summary.get("delay_est_ms"),
        },
        "rotation_tracking": {"host": host_summary},
        "logs": {"host_csv": host_csv},
    }
    save_calibration_json(args, result)
    node.send_status(f"host:results_saved:{args.output_json}")
    node.destroy_node()


def main():
    args = build_argparser().parse_args()
    rclpy.init()
    viewer_proc = None
    viewer_path = os.path.join(os.path.dirname(__file__), "imu_viewer.py")
    if (not args.no_imu_viewer) and os.path.exists(viewer_path):
        preflight = subprocess.run(
            [sys.executable, "-c", "import matplotlib.pyplot"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if preflight.returncode != 0:
            print("[WARN] imu_viewer skipped: matplotlib/numpy runtime mismatch. run with --no-imu-viewer or fix local python env.")
        else:
            viewer_proc = subprocess.Popen([
                sys.executable, viewer_path,
                "--imu_topic", args.imu_topic,
                "--enc_topic", args.hw_enc_topic,
            ])
    try:
        host_main(args)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
