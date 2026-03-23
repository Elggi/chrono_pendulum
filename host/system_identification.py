#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


@dataclass
class ProtocolStep:
    label: str
    pwm: float
    duration_sec: float


HOST_PROTOCOL = [
    ProtocolStep("host_zero", 0.0, 2.0),
    ProtocolStep("host_pos_small", 30.0, 1.0),
    ProtocolStep("host_zero_mid", 0.0, 1.0),
    ProtocolStep("host_neg_small", -30.0, 1.0),
    ProtocolStep("host_zero_end", 0.0, 2.0),
]

JETSON_PROTOCOL = [
    ProtocolStep("jetson_zero", 0.0, 2.0),
    ProtocolStep("jetson_pos_small", 30.0, 1.0),
    ProtocolStep("jetson_zero_mid", 0.0, 1.0),
    ProtocolStep("jetson_neg_small", -30.0, 1.0),
    ProtocolStep("jetson_prbs_placeholder", 45.0, 1.0),
    ProtocolStep("jetson_zero_end", 0.0, 2.0),
]


class CalibrationNode(Node):
    def __init__(self, args):
        super().__init__("system_identification")
        self.args = args
        self.pub_cmd = self.create_publisher(Float32, args.cmd_topic, 10)
        self.pub_status = self.create_publisher(String, args.status_topic, 10)
        self.last_status = None
        self.create_subscription(String, args.status_topic, self.cb_status, 100)

    def cb_status(self, msg: String):
        self.last_status = msg.data

    def send_status(self, text: str):
        self.get_logger().info(text)
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def publish_pwm(self, value: float):
        msg = Float32()
        msg.data = float(value)
        self.pub_cmd.publish(msg)

    def run_protocol(self, steps):
        for step in steps:
            self.send_status(f"{self.args.role}:{step.label}:start")
            t_end = time.time() + step.duration_sec
            while time.time() < t_end:
                self.publish_pwm(step.pwm)
                time.sleep(1.0 / max(self.args.loop_hz, 1.0))
            self.send_status(f"{self.args.role}:{step.label}:done")
        self.publish_pwm(0.0)


def save_calibration_json(args, result):
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["host", "jetson"], required=True)
    ap.add_argument("--cmd-topic", default="/cmd/u")
    ap.add_argument("--status-topic", default="/calibration/status")
    ap.add_argument("--loop-hz", type=float, default=20.0)
    ap.add_argument("--output-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--wait-timeout-sec", type=float, default=180.0)
    return ap


def host_main(args):
    node = CalibrationNode(args)
    node.send_status("host:waiting_for_jetson")
    t0 = time.time()
    while time.time() - t0 < args.wait_timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.last_status == "jetson:protocol:finished":
            break
    else:
        raise TimeoutError("Timed out waiting for jetson calibration completion")

    node.send_status("host:jetson_finished_proceeding_host_protocol")
    node.run_protocol(HOST_PROTOCOL)
    node.send_status("host:protocol:finished")

    result = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "workflow": "phase1_scaffold",
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
    }
    save_calibration_json(args, result)
    node.send_status(f"host:results_saved:{args.output_json}")
    node.destroy_node()


def jetson_main(args):
    node = CalibrationNode(args)
    node.send_status("jetson:protocol:started")
    node.run_protocol(JETSON_PROTOCOL)
    node.send_status("jetson:protocol:finished")
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
