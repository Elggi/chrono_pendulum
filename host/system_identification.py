#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
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
    ProtocolStep("jetson_pos_180", 180.0, 1.5),
    ProtocolStep("jetson_zero_mid_4", 0.0, 1.0),
    ProtocolStep("jetson_neg_180", -180.0, 1.5),
    ProtocolStep("jetson_zero_end", 0.0, 2.0),
]


def terminal_status_line(msg: str, width: int = 180):
    print("\r\033[2K" + msg[:width].ljust(width), end="", flush=True)


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
        self.get_logger().info(text)
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
            f"role={self.args.role:<6} | step={step_label:<18} | cmd_u={cmd_u:7.1f} | "
            f"hw_pwm={snap['hw_pwm']:7.1f} | enc={snap['hw_enc']:11.1f} | "
            f"bus_v={snap['bus_v']:6.2f} | current_mA={snap['current_ma']:7.1f} | "
            f"imu_wz={0.0 if snap['imu_wz'] is None else snap['imu_wz']:7.4f} | "
            f"imu_ay={0.0 if snap['imu_ay'] is None else snap['imu_ay']:7.3f}"
        )

    def run_protocol(self, steps, csv_path):
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
    return ap


def host_main(args):
    node = CalibrationNode(args)
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
    node.run_protocol(HOST_PROTOCOL, host_csv)
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
        "protocols": {
            "jetson": [asdict(x) for x in JETSON_PROTOCOL],
            "host": [asdict(x) for x in HOST_PROTOCOL],
        },
        "logs": {
            "host_csv": host_csv,
        },
    }
    save_calibration_json(args, result)
    node.send_status(f"host:results_saved:{args.output_json}")
    node.destroy_node()



def jetson_main(args):
    node = CalibrationNode(args)
    jetson_csv = make_labeled_path(args.log_dir, "calibration_", "jetson", ".csv")
    node.send_status("jetson:protocol:started")
    print("[INFO] jetson -> host calibration 진행중…")
    node.run_protocol(JETSON_PROTOCOL, jetson_csv)
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
