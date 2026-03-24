#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import shutil
import argparse
from dataclasses import dataclass, asdict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String


@dataclass
class ProtocolStep:
    label: str
    pwm: float
    duration_sec: float


HOST_PROTOCOL = [
    ProtocolStep("host_zero", 0.0, 2.0),
    ProtocolStep("host_pos_60", 60.0, 1.5),
    ProtocolStep("host_zero_mid_1", 0.0, 1.0),
    ProtocolStep("host_neg_60", -60.0, 1.5),
    ProtocolStep("host_zero_mid_2", 0.0, 1.0),
    ProtocolStep("host_pos_120", 120.0, 1.5),
    ProtocolStep("host_zero_mid_3", 0.0, 1.0),
    ProtocolStep("host_neg_120", -120.0, 1.5),
    ProtocolStep("host_zero_end", 0.0, 2.0),
]

JETSON_PROTOCOL = [
    ProtocolStep("jetson_zero", 0.0, 2.0),
    ProtocolStep("jetson_pos_60", 60.0, 1.5),
    ProtocolStep("jetson_zero_mid_1", 0.0, 1.0),
    ProtocolStep("jetson_neg_60", -60.0, 1.5),
    ProtocolStep("jetson_zero_mid_2", 0.0, 1.0),
    ProtocolStep("jetson_pos_120", 120.0, 1.5),
    ProtocolStep("jetson_zero_mid_3", 0.0, 1.0),
    ProtocolStep("jetson_neg_120", -120.0, 1.5),
    ProtocolStep("jetson_pos_120_second", 120.0, 1.5),
    ProtocolStep("jetson_zero_mid_4", 0.0, 1.0),
    ProtocolStep("jetson_neg_120_second", -120.0, 1.5),
    ProtocolStep("jetson_zero_end", 0.0, 2.0),
]

MAX_CALIB_PWM = 120.0


def terminal_status_line(msg: str, width: int = 180):
    term_cols = shutil.get_terminal_size((width, 20)).columns
    width = min(width, term_cols)
    print("\r\033[2K" + msg[:width].ljust(width), end="", flush=True)


def clamp_protocol_pwm(steps, max_abs_pwm: float):
    out = []
    for s in steps:
        clamped = max(-max_abs_pwm, min(max_abs_pwm, float(s.pwm)))
        label = s.label if clamped == s.pwm else f"{s.label}_clamped"
        out.append(ProtocolStep(label=label, pwm=clamped, duration_sec=s.duration_sec))
    return out


def load_protocol_from_json(path: str, role: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if role in payload:
            raw_steps = payload[role]
        else:
            raw_steps = payload.get("steps", [])
    else:
        raw_steps = payload
    steps = []
    for i, item in enumerate(raw_steps):
        steps.append(
            ProtocolStep(
                label=str(item.get("label", f"{role}_step_{i}")),
                pwm=float(item["pwm"]),
                duration_sec=float(item["duration_sec"]),
            )
        )
    if not steps:
        raise ValueError(f"No protocol steps found in {path} for role={role}")
    return steps


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
            f"{self.args.role:<6} step={step_label:<16} cmd={cmd_u:6.1f} pwm={snap['hw_pwm']:6.1f} "
            f"enc={snap['hw_enc']:10.1f} V={snap['bus_v']:5.2f} I={snap['current_ma']:7.1f} "
            f"wz={0.0 if snap['imu_wz'] is None else snap['imu_wz']:7.4f}"
        )

    def run_protocol(self, steps, csv_path):
        start_enc = None
        last_enc = None
        net_encoder_delta = 0.0
        abs_encoder_travel = 0.0
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([
                "wall_time", "role", "step_label", "cmd_u",
                "hw_pwm", "hw_enc", "hw_arduino_ms",
                "bus_v", "current_ma", "power_mw",
                "imu_qx", "imu_qy", "imu_qz", "imu_qw",
                "imu_wx", "imu_wy", "imu_wz",
                "imu_ax", "imu_ay", "imu_az",
                "status_text",
            ])
            for step in steps:
                self.send_status(f"{self.args.role}:{step.label}:start")
                t_end = time.time() + step.duration_sec
                while time.time() < t_end:
                    self.publish_pwm(step.pwm)
                    rclpy.spin_once(self, timeout_sec=0.0)
                    snap = self.snapshot()
                    if start_enc is None:
                        start_enc = snap["hw_enc"]
                    if last_enc is not None:
                        d_enc = snap["hw_enc"] - last_enc
                        net_encoder_delta += d_enc
                        abs_encoder_travel += abs(d_enc)
                    last_enc = snap["hw_enc"]
                    wr.writerow([
                        time.time(), self.args.role, step.label, step.pwm,
                        snap["hw_pwm"], snap["hw_enc"], snap["hw_ms"],
                        snap["bus_v"], snap["current_ma"], snap["power_mw"],
                        snap["imu_qx"], snap["imu_qy"], snap["imu_qz"], snap["imu_qw"],
                        snap["imu_wx"], snap["imu_wy"], snap["imu_wz"],
                        snap["imu_ax"], snap["imu_ay"], snap["imu_az"],
                        snap["last_status"],
                    ])
                    terminal_status_line(self.monitor_text(step.label, step.pwm))
                    time.sleep(1.0 / max(self.args.loop_hz, 1.0))
                self.send_status(f"{self.args.role}:{step.label}:done")

            # Return to origin encoder point using bounded PWM.
            if start_enc is not None and self.args.return_to_origin:
                self.send_status(f"{self.args.role}:return_to_origin:start")
                t_until = time.time() + self.args.return_timeout_sec
                while time.time() < t_until:
                    rclpy.spin_once(self, timeout_sec=0.0)
                    err = start_enc - self.hw_enc
                    if abs(err) <= self.args.return_tol_counts:
                        break
                    pwm = max(self.args.return_min_pwm, min(self.args.return_kp * abs(err), self.args.max_calib_pwm))
                    cmd = pwm if err > 0 else -pwm
                    self.publish_pwm(cmd)
                    snap = self.snapshot()
                    if last_enc is not None:
                        d_enc = snap["hw_enc"] - last_enc
                        net_encoder_delta += d_enc
                        abs_encoder_travel += abs(d_enc)
                    last_enc = snap["hw_enc"]
                    wr.writerow([
                        time.time(), self.args.role, "return_to_origin", cmd,
                        snap["hw_pwm"], snap["hw_enc"], snap["hw_ms"],
                        snap["bus_v"], snap["current_ma"], snap["power_mw"],
                        snap["imu_qx"], snap["imu_qy"], snap["imu_qz"], snap["imu_qw"],
                        snap["imu_wx"], snap["imu_wy"], snap["imu_wz"],
                        snap["imu_ax"], snap["imu_ay"], snap["imu_az"],
                        snap["last_status"],
                    ])
                    terminal_status_line(self.monitor_text("return_to_origin", cmd))
                    time.sleep(1.0 / max(self.args.loop_hz, 1.0))
                self.send_status(f"{self.args.role}:return_to_origin:done")
            self.publish_pwm(0.0)
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.0)
                wr.writerow([
                    time.time(), self.args.role, "final_zero", 0.0,
                    self.hw_pwm, self.hw_enc, self.hw_ms,
                    self.bus_v, self.current_ma, self.power_mw,
                    None if self.imu_msg is None else self.imu_msg.orientation.x,
                    None if self.imu_msg is None else self.imu_msg.orientation.y,
                    None if self.imu_msg is None else self.imu_msg.orientation.z,
                    None if self.imu_msg is None else self.imu_msg.orientation.w,
                    None if self.imu_msg is None else self.imu_msg.angular_velocity.x,
                    None if self.imu_msg is None else self.imu_msg.angular_velocity.y,
                    None if self.imu_msg is None else self.imu_msg.angular_velocity.z,
                    None if self.imu_msg is None else self.imu_msg.linear_acceleration.x,
                    None if self.imu_msg is None else self.imu_msg.linear_acceleration.y,
                    None if self.imu_msg is None else self.imu_msg.linear_acceleration.z,
                    self.last_status,
                ])
                time.sleep(0.05)
        print()
        end_enc = self.hw_enc
        turns_net = None if start_enc is None else (end_enc - start_enc) / max(self.args.counts_per_rev, 1e-9)
        return {
            "encoder_start": start_enc,
            "encoder_end": end_enc,
            "encoder_net_delta": None if start_enc is None else (end_enc - start_enc),
            "encoder_signed_travel": net_encoder_delta,
            "encoder_abs_travel": abs_encoder_travel,
            "turns_net": turns_net,
            "turns_abs": abs_encoder_travel / max(self.args.counts_per_rev, 1e-9),
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
    ap.add_argument("--role", choices=["host", "jetson"], required=True)
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
    ap.add_argument("--wait-timeout-sec", type=float, default=180.0)
    ap.add_argument("--protocol-json", default=None, help="Optional JSON protocol override")
    ap.add_argument("--max-calib-pwm", type=float, default=MAX_CALIB_PWM)
    ap.add_argument("--counts-per-rev", type=float, default=360.0)
    ap.add_argument("--return-to-origin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--return-timeout-sec", type=float, default=20.0)
    ap.add_argument("--return-tol-counts", type=float, default=8.0)
    ap.add_argument("--return-kp", type=float, default=0.12)
    ap.add_argument("--return-min-pwm", type=float, default=28.0)
    return ap


def host_main(args):
    node = CalibrationNode(args)
    if args.protocol_json:
        host_protocol = load_protocol_from_json(args.protocol_json, "host")
    else:
        host_protocol = HOST_PROTOCOL
    host_protocol = clamp_protocol_pwm(host_protocol, args.max_calib_pwm)
    host_csv = make_labeled_path(args.log_dir, "calibration_", "host", ".csv")
    node.send_status("host:waiting_for_jetson")
    print("[INFO] host terminal에서 Calibration을 시작하십시오… Jetson calibration 대기중")
    t0 = time.time()
    while time.time() - t0 < args.wait_timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.2)
        terminal_status_line(node.monitor_text("waiting_for_jetson", 0.0))
        if node.last_status == "jetson:protocol:finished":
            print("\n[INFO] Jetson mode calibration finished, proceeding with Host mode calibration.")
            break
    else:
        raise TimeoutError("Timed out waiting for jetson calibration completion")

    node.send_status("host:jetson_finished_proceeding_host_protocol")
    host_summary = node.run_protocol(host_protocol, host_csv)
    node.send_status("host:protocol:finished")

    result = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "workflow": "phase1_debuggable_scaffold",
        "counts_per_revolution": None,
        "imu_mount": {
            "r_link_frame": None,
            "sensor_to_link_quat": None,
        },
        "delay": {
            "host_to_jetson_ms": None,
            "jetson_to_host_ms": None,
            "effective_control_delay_ms": 0.0,
        },
        "model_init": {
            "J": 0.01,
            "b": 0.03,
            "tau_c": 0.08,
            "mgl": 0.55,
            "k_t": 0.25,
            "i0": 0.0,
            "R": 2.0,
            "k_e": 0.02,
        },
        "protocols": {"host": [asdict(x) for x in host_protocol]},
        "calibration_algorithm": {
            "name": "rule_based_openloop_protocol_with_encoder_return",
            "details": "Not Nelder-Mead/least-squares/EKF/UKF. Fixed time-PWM sequence logs sensor data and returns to initial encoder origin.",
            "max_abs_pwm": args.max_calib_pwm,
        },
        "rotation_tracking": {"host": host_summary},
        "logs": {
            "host_csv": host_csv,
        },
    }
    save_calibration_json(args, result)
    node.send_status(f"host:results_saved:{args.output_json}")
    node.destroy_node()



def jetson_main(args):
    node = CalibrationNode(args)
    if args.protocol_json:
        jetson_protocol = load_protocol_from_json(args.protocol_json, "jetson")
    else:
        jetson_protocol = JETSON_PROTOCOL
    jetson_protocol = clamp_protocol_pwm(jetson_protocol, args.max_calib_pwm)
    jetson_csv = make_labeled_path(args.log_dir, "calibration_", "jetson", ".csv")
    node.send_status("jetson:protocol:started")
    print("[INFO] jetson -> host calibration 진행중…")
    node.run_protocol(jetson_protocol, jetson_csv)
    node.send_status("jetson:protocol:finished")
    print(f"[INFO] calibration csv saved: {jetson_csv}")
    node.destroy_node()



def main():
    args = build_argparser().parse_args()
    rclpy.init()
    try:
        if args.role == "host":
            host_main(args)
        else:
            jetson_main(args)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
