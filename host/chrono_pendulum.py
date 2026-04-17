#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import time
import math
import argparse
import threading
import select
import sys
import termios
import tty
import shutil
import signal
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum

import numpy as np

import pychrono as ch
import pychrono.irrlicht as irr

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Imu
from chrono_core.config import BridgeConfig
from chrono_core.utils import clamp, now_wall, terminal_status_line, sanitize_float, make_numbered_path
from chrono_core.dynamics import PendulumModel, compute_model_torque_and_electrics
from chrono_core.calibration_io import apply_calibration_json, extract_radius_from_json
from chrono_core.model_parameter_io import load_model_parameter_json, extract_runtime_overrides
from chrono_core.log_schema import PENDULUM_LOG_COLUMNS
from chrono_core.signal_filter import estimate_filtered_alpha_from_omega, CausalIIRFilter


# ============================================================
# utility
# ============================================================


class RunState(str, Enum):
    STATE_WARMUP = "STATE_WARMUP"
    STATE_FREE_DECAY_ARM = "STATE_FREE_DECAY_ARM"
    STATE_FREE_DECAY_WAIT_RELEASE = "STATE_FREE_DECAY_WAIT_RELEASE"
    STATE_RUN = "STATE_RUN"


def compute_theta_offset(theta_array: np.ndarray) -> float:
    """Robust theta offset from warmup samples [rad]."""
    x = np.asarray(theta_array, dtype=float)
    finite = np.isfinite(x)
    if np.sum(finite) < 4:
        return 0.0
    xu = np.unwrap(x[finite])
    return float(np.median(xu))


def compute_current_offset(
    current_array: np.ndarray,
    pwm_array: np.ndarray,
    omega_array: np.ndarray,
    pwm_threshold: float = 3.0,
    omega_threshold: float = 0.8,
    fallback_mA: float = 26.0,
) -> tuple[float, int]:
    cur = np.asarray(current_array, dtype=float)
    pwm = np.asarray(pwm_array, dtype=float)
    omg = np.asarray(omega_array, dtype=float)
    m = np.isfinite(cur) & np.isfinite(pwm) & np.isfinite(omg)
    m &= (np.abs(pwm) < abs(float(pwm_threshold))) & (np.abs(omg) < abs(float(omega_threshold)))
    if int(np.sum(m)) < 12:
        return float(fallback_mA), int(np.sum(m))
    return float(np.median(cur[m])), int(np.sum(m))

class KeyboardReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key_nonblocking(self, timeout=0.0):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if not rlist:
            return None
        ch1 = sys.stdin.read(1)
        if ch1 == "\x1b":
            rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
            if rlist:
                ch2 = sys.stdin.read(1)
                rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                if rlist:
                    ch3 = sys.stdin.read(1)
                    if ch2 == "[":
                        if ch3 == "A":
                            return "UP"
                        if ch3 == "B":
                            return "DOWN"
                        if ch3 == "C":
                            return "RIGHT"
                        if ch3 == "D":
                            return "LEFT"
            return "ESC"
        return ch1


class QuitWatcher:
    def __init__(self):
        self.kb = KeyboardReader() if sys.stdin.isatty() else None
        self.active = False
        self.quit_requested = False

    def __enter__(self):
        if self.kb is not None:
            self.kb.__enter__()
            self.active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active and self.kb is not None:
            self.kb.__exit__(exc_type, exc, tb)
            self.active = False

    def poll(self):
        if self.kb is None:
            return None
        key = self.kb.read_key_nonblocking(0.0)
        if key in ("q", "Q", "ESC"):
            self.quit_requested = True
        return key


class HostCommandController:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.current_u = 0.0
        self.mode = "manual"
        self.mode_t0 = now_wall()
        self.quit_requested = False
        self.kb = KeyboardReader()
        self.keyboard_active = False

    def __enter__(self):
        self.print_banner()
        self.kb.__enter__()
        self.keyboard_active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.keyboard_active:
            self.kb.__exit__(exc_type, exc, tb)
            self.keyboard_active = False

    def print_banner(self):
        print("=" * 70)
        print("Host Keyboard Controller")
        print("=" * 70)
        print("Controls:")
        print("  w / Up Arrow      : increase forward PWM")
        print("  s / Down Arrow    : increase reverse PWM")
        print("  d / Right Arrow   : fine increase (+0.5 step)")
        print("  a / Left Arrow    : fine decrease (-0.5 step)")
        print("  space             : emergency stop (0)")
        print("  x                 : set 0")
        print("  1 2 3 4           : fixed PWM presets (+60 / -60 / +120 / -120)")
        print("  5                 : sine preset")
        print("  6                 : square preset")
        print("  7                 : burst preset")
        print("  8                 : PRBS preset")
        print("  [ / ]             : decrease/increase pwm_step")
        print("  - / =             : decrease/increase pwm_max")
        print("  q                 : quit")
        print("=" * 70)
        print("Output topic:")
        print(f"  {self.cfg.topic_cmd_u}  (std_msgs/Float32, signed PWM)")
        print("=" * 70)
        print()

    def set_manual(self):
        self.mode = "manual"
        self.mode_t0 = now_wall()

    def set_mode(self, mode_name: str):
        self.mode = mode_name
        self.mode_t0 = now_wall()

    def apply_key(self, key):
        if key is None:
            return
        if key in ("w", "W", "UP"):
            self.set_manual()
            self.current_u += self.cfg.pwm_step
        elif key in ("s", "S", "DOWN"):
            self.set_manual()
            self.current_u -= self.cfg.pwm_step
        elif key in ("d", "D", "RIGHT"):
            self.set_manual()
            self.current_u += 0.5 * self.cfg.pwm_step
        elif key in ("a", "A", "LEFT"):
            self.set_manual()
            self.current_u -= 0.5 * self.cfg.pwm_step
        elif key == " " or key in ("x", "X"):
            self.set_manual()
            self.current_u = 0.0
        elif key == "1":
            self.set_manual(); self.current_u = 60.0
        elif key == "2":
            self.set_manual(); self.current_u = -60.0
        elif key == "3":
            self.set_manual(); self.current_u = 120.0
        elif key == "4":
            self.set_manual(); self.current_u = -120.0
        elif key == "5":
            self.set_mode("sin")
        elif key == "6":
            self.set_mode("square")
        elif key == "7":
            self.set_mode("burst")
        elif key == "8":
            self.set_mode("prbs")
        elif key == "[":
            self.cfg.pwm_step = max(1.0, self.cfg.pwm_step - 1.0)
        elif key == "]":
            self.cfg.pwm_step = min(100.0, self.cfg.pwm_step + 1.0)
        elif key == "-":
            self.cfg.pwm_limit = max(20.0, self.cfg.pwm_limit - 5.0)
        elif key == "=":
            self.cfg.pwm_limit = min(255.0, self.cfg.pwm_limit + 5.0)
        elif key in ("q", "Q", "ESC"):
            self.quit_requested = True

        self.current_u = clamp(self.current_u, -self.cfg.pwm_limit, self.cfg.pwm_limit)

    def get_command(self):
        t = now_wall() - self.mode_t0
        if self.mode == "manual":
            return self.current_u
        if self.mode == "sin":
            return 60.0 * math.sin(2.0 * math.pi * 0.5 * t)
        if self.mode == "square":
            return 60.0 if math.sin(2.0 * math.pi * 0.5 * t) >= 0.0 else -60.0
        if self.mode == "burst":
            tau = t % 2.0
            return 60.0 if tau < 0.30 else 0.0
        if self.mode == "prbs":
            return 60.0 * prbs_value(t, 0.25, self.cfg.prbs_seed)
        return self.current_u

    def poll(self):
        key = self.kb.read_key_nonblocking(0.0)
        self.apply_key(key)


# ============================================================
# ROS shared state
# ============================================================

class SharedROSState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cmd_u = 0.0
        self.hw_pwm = 0.0
        self.current_mA = 0.0
        self.ina_bus_voltage_v = 0.0
        self.ina_power_mw = 0.0
        self.hw_enc = 0.0
        self.hw_arduino_ms = 0.0
        self.imu_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.imu_w = np.zeros(3, dtype=float)
        self.imu_a = np.zeros(3, dtype=float)
        self.imu_has_data = False
        self.last_cmd_wall = 0.0
        self.last_hw_pwm_wall = 0.0
        self.last_hw_enc_wall = 0.0
        self.last_ina_wall = 0.0

    def snapshot(self):
        with self.lock:
            return {
                "cmd_u": self.cmd_u,
                "hw_pwm": self.hw_pwm,
                "current_mA": self.current_mA,
                "ina_bus_voltage_v": self.ina_bus_voltage_v,
                "ina_power_mw": self.ina_power_mw,
                "hw_enc": self.hw_enc,
                "hw_arduino_ms": self.hw_arduino_ms,
                "imu_q": self.imu_q.copy(),
                "imu_w": self.imu_w.copy(),
                "imu_a": self.imu_a.copy(),
                "imu_has_data": self.imu_has_data,
                "last_cmd_wall": self.last_cmd_wall,
                "last_hw_pwm_wall": self.last_hw_pwm_wall,
                "last_hw_enc_wall": self.last_hw_enc_wall,
                "last_ina_wall": self.last_ina_wall,
            }


class PendulumROSNode(Node):
    def __init__(self, cfg: BridgeConfig, shared: SharedROSState, host_mode=False):
        super().__init__("chrono_pendulum")
        self.cfg = cfg
        self.shared = shared

        self.create_subscription(Float32, cfg.topic_cmd_u, self.cb_cmd_u, 10)
        self.create_subscription(Float32, cfg.topic_hw_pwm, self.cb_hw_pwm, 10)
        self.create_subscription(Float32, cfg.topic_hw_current_ma, self.cb_hw_current_ma, 10)
        self.create_subscription(Float32, cfg.topic_hw_bus_voltage_v, self.cb_hw_bus_voltage_v, 10)
        self.create_subscription(Float32, cfg.topic_hw_power_mw, self.cb_hw_power_mw, 10)
        self.create_subscription(Float32, cfg.topic_hw_enc, self.cb_hw_enc, 10)
        self.create_subscription(Float32, cfg.topic_hw_arduino_ms, self.cb_hw_ms, 10)
        self.create_subscription(Imu, cfg.topic_hw_imu, self.cb_hw_imu, 10)

        self.pub_theta = self.create_publisher(Float32, cfg.topic_sim_theta, 10)
        self.pub_omega = self.create_publisher(Float32, cfg.topic_sim_omega, 10)
        self.pub_alpha = self.create_publisher(Float32, cfg.topic_sim_alpha, 10)
        self.pub_tau = self.create_publisher(Float32, cfg.topic_sim_tau, 10)
        self.pub_cmd_used = self.create_publisher(Float32, cfg.topic_sim_cmd_used, 10)
        self.pub_status = self.create_publisher(String, cfg.topic_sim_status, 10)
        self.pub_imu = self.create_publisher(Imu, cfg.topic_imu, 10)
        self.pub_cmd = self.create_publisher(Float32, cfg.topic_cmd_u, 10) if host_mode else None
        self.pub_debug = self.create_publisher(String, cfg.topic_debug, 10) if host_mode else None

    def cb_cmd_u(self, msg: Float32):
        with self.shared.lock:
            self.shared.cmd_u = float(msg.data)
            self.shared.last_cmd_wall = now_wall()

    def cb_hw_pwm(self, msg: Float32):
        with self.shared.lock:
            self.shared.hw_pwm = float(msg.data)
            self.shared.last_hw_pwm_wall = now_wall()

    def cb_hw_enc(self, msg: Float32):
        with self.shared.lock:
            self.shared.hw_enc = float(msg.data)
            self.shared.last_hw_enc_wall = now_wall()

    def cb_hw_current_ma(self, msg: Float32):
        with self.shared.lock:
            self.shared.current_mA = float(msg.data)
            self.shared.last_ina_wall = now_wall()

    def cb_hw_bus_voltage_v(self, msg: Float32):
        with self.shared.lock:
            self.shared.ina_bus_voltage_v = float(msg.data)

    def cb_hw_power_mw(self, msg: Float32):
        with self.shared.lock:
            self.shared.ina_power_mw = float(msg.data)

    def cb_hw_ms(self, msg: Float32):
        with self.shared.lock:
            self.shared.hw_arduino_ms = float(msg.data)

    def cb_hw_imu(self, msg: Imu):
        with self.shared.lock:
            self.shared.imu_q[:] = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
            self.shared.imu_w[:] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            self.shared.imu_a[:] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            self.shared.imu_has_data = True

            self.shared.last_ina_wall = now_wall()

    @staticmethod
    def _pub_float(pub, value):
        msg = Float32()
        msg.data = sanitize_float(value)
        pub.publish(msg)

    def publish_host_cmd(self, u, mode_name):
        if self.pub_cmd is None:
            return
        self._pub_float(self.pub_cmd, u)
        if self.pub_debug is not None:
            s = String()
            s.data = mode_name
            self.pub_debug.publish(s)

    def publish_sim(self, theta, omega, alpha, tau, cmd_used, imu_msg, status_text):
        self._pub_float(self.pub_theta, theta)
        self._pub_float(self.pub_omega, omega)
        self._pub_float(self.pub_alpha, alpha)
        self._pub_float(self.pub_tau, tau)
        self._pub_float(self.pub_cmd_used, cmd_used)
        self.pub_imu.publish(imu_msg)
        s = String()
        s.data = status_text
        self.pub_status.publish(s)


class ViewerState:
    def __init__(self, hist_len=4000):
        self.lock = threading.Lock()
        self.t = deque(maxlen=hist_len)
        self.theta = deque(maxlen=hist_len)
        self.omega = deque(maxlen=hist_len)
        self.alpha = deque(maxlen=hist_len)
        self.imu_ax = deque(maxlen=hist_len)
        self.imu_ay = deque(maxlen=hist_len)
        self.imu_az = deque(maxlen=hist_len)
        self.imu_wz = deque(maxlen=hist_len)
        self.enc = deque(maxlen=hist_len)
        self.cpr = deque(maxlen=hist_len)
        self.tipx = deque(maxlen=hist_len)
        self.tipy = deque(maxlen=hist_len)


def viewer_thread_fn(view_state: ViewerState, stop_flag: threading.Event):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig = plt.figure("Chrono IMU Viewer", figsize=(12, 7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig.tight_layout(pad=2.0)

    def update(_):
        with view_state.lock:
            t = np.array(view_state.t, dtype=float)
            theta = np.array(view_state.theta, dtype=float)
            omega = np.array(view_state.omega, dtype=float)
            alpha = np.array(view_state.alpha, dtype=float)
            imu_ax = np.array(view_state.imu_ax, dtype=float)
            imu_ay = np.array(view_state.imu_ay, dtype=float)
            imu_az = np.array(view_state.imu_az, dtype=float)
            imu_wz = np.array(view_state.imu_wz, dtype=float)
            enc = np.array(view_state.enc, dtype=float)
            cpr = np.array(view_state.cpr, dtype=float)
            tipx = np.array(view_state.tipx, dtype=float)
            tipy = np.array(view_state.tipy, dtype=float)
        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
        if len(t) > 1:
            ax1.plot(t, theta, label="theta [rad]")
            ax1.plot(t, omega, label="omega [rad/s]")
            ax1.plot(t, alpha, label="alpha [rad/s^2]")
            ax1.grid(True); ax1.legend(); ax1.set_title("Angular states")
            ax2.plot(t, imu_ax, label="ax [m/s²]")
            ax2.plot(t, imu_ay, label="ay [m/s²]")
            ax2.plot(t, imu_az, label="az [m/s²]")
            ax2.plot(t, imu_wz, label="wz [rad/s]")
            ax2.grid(True); ax2.legend(); ax2.set_title("IMU signals")
            ax3.plot(t, enc, label="enc")
            valid = np.isfinite(cpr)
            if np.any(valid):
                ax3.plot(t[valid], cpr[valid], label="CPR sample")
            ax3.grid(True); ax3.legend(); ax3.set_title("Encoder / CPR")
        if len(tipx) > 2:
            ax4.plot(tipx, tipy)
            ax4.scatter([tipx[-1]], [tipy[-1]], s=20)
            ax4.grid(True); ax4.axis("equal"); ax4.set_title("Tip trajectory")
        if len(tipx) > 0:
            ax4.plot([0.0, tipx[-1]], [0.0, tipy[-1]], "-o", alpha=0.45)
            ax4.set_xlim(-0.35, 0.35); ax4.set_ylim(-0.35, 0.35)
        if stop_flag.is_set():
            plt.close(fig)

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    fig._chrono_animation = ani
    plt.show()


def build_imu_msg(sim_t, q_wxyz, omega_xyz, acc_xyz):
    msg = Imu()
    sec = int(sim_t)
    nanosec = int((sim_t - sec) * 1e9)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = "imu_link"
    msg.orientation.w = float(q_wxyz[0])
    msg.orientation.x = float(q_wxyz[1])
    msg.orientation.y = float(q_wxyz[2])
    msg.orientation.z = float(q_wxyz[3])
    msg.angular_velocity.x = float(omega_xyz[0])
    msg.angular_velocity.y = float(omega_xyz[1])
    msg.angular_velocity.z = float(omega_xyz[2])
    msg.linear_acceleration.x = float(acc_xyz[0])
    msg.linear_acceleration.y = float(acc_xyz[1])
    msg.linear_acceleration.z = float(acc_xyz[2])
    return msg


def quat_to_rotmat(w: float, x: float, y: float, z: float):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def compute_theta_wrapped_from_imu_snapshot(snap: dict, imu_R0: np.ndarray | None, radius: float, imu_sign: float):
    """Return wrapped theta [rad], body->world0 rotation, and imu_R0."""
    if not snap.get("imu_has_data", False):
        return None, None, imu_R0
    q = snap["imu_q"]
    R_abs = quat_to_rotmat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    if imu_R0 is None:
        imu_R0 = R_abs.copy()
    R_rel = imu_R0.T @ R_abs
    tip_vec = R_rel @ np.array([0.0, -radius, 0.0], dtype=float)
    theta_wrapped = float(imu_sign * math.atan2(float(tip_vec[1]), float(tip_vec[0])))
    return theta_wrapped, R_rel, imu_R0


def gravity_world0_from_imu_anchor(imu_R0: np.ndarray | None, gravity_mps2: float) -> np.ndarray:
    g_world = np.array([0.0, -float(gravity_mps2), 0.0], dtype=float)
    if imu_R0 is None:
        return g_world
    return imu_R0.T @ g_world


def make_status_line(host_mode: bool, cmd_u_raw: float, cmd_u_used: float, hw_pwm: float, current_mA: float, mode_name: str,
                     cfg: BridgeConfig):
    if host_mode:
        return (
            f"cmd_u: {cmd_u_raw:6.1f} | used: {cmd_u_used:6.1f} | mode: {mode_name:<6} | "
            f"step: {cfg.pwm_step:4.1f} | max: {cfg.pwm_limit:5.1f}"
        )
    return (
        f"cmd_u: {cmd_u_used:6.1f} | hw_pwm: {hw_pwm:6.1f} | I[mA]: {current_mA:7.1f} | mode: external"
    )


def ros_spin_thread(node: Node, stop_flag: threading.Event):
    while rclpy.ok() and not stop_flag.is_set():
        rclpy.spin_once(node, timeout_sec=0.01)


def start_imu_viewer_process(imu_topic: str, enc_topic: str):
    import subprocess

    script_path = os.path.join(os.path.dirname(__file__), "imu_viewer.py")
    try:
        return subprocess.Popen(
            [sys.executable, script_path, "--imu_topic", imu_topic, "--enc_topic", enc_topic],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        print(f"[WARN] Failed to start IMU viewer: {exc}")
        return None




# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-imu-viewer", action="store_true")
    ap.add_argument("--duration", type=float, default=-1.0,
                    help="Run duration in seconds. <=0 means run until user quits.")
    ap.add_argument("--step", type=float, default=0.001)
    ap.add_argument(
        "--theta0-deg",
        type=float,
        default=0.0,
        help="Initial angle in degrees. + is CCW, - is CW (default: 0.0).",
    )
    ap.add_argument("--omega0", type=float, default=0.0)
    ap.add_argument("--link-mass", type=float, default=0.200)
    ap.add_argument("--link-length", type=float, default=0.285)
    ap.add_argument("--host-control", action="store_true",
                    help="Enable host-side manual command publishing.")
    ap.add_argument("--mode", choices=["host", "jetson"], default=None,
                    help="Compatibility option: host enables host-control, jetson uses external ROS input.")
    ap.add_argument("--calibration-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--parameter-json", default="", help="Exported parameter JSON containing model_init/best_params")
    ap.add_argument("--radius-json", default="./run_logs/calibration_latest.json",
                    help="JSON file containing measured radius (e.g., calibration_latest.json).")
    ap.add_argument("--l-com", type=float, default=None, help="default: link_length/2 for fresh runs")
    ap.add_argument("--b", type=float, default=None, help="default: near-zero fresh initialization")
    ap.add_argument("--tau-c", type=float, default=None, help="default: near-zero fresh initialization")
    ap.add_argument("--k-i", "--k-u", dest="k_i", type=float, default=None,
                    help="default: near-zero positive fresh initialization (motor torque constant K_i)")
    ap.add_argument("--r-imu", type=float, default=0.285, help="IMU radius from pivot [m]")
    ap.add_argument(
        "--imu-linear-accel-no-gravity-comp",
        action="store_true",
        help="Disable gravity compensation for IMU linear-accel alpha path.",
    )
    ap.add_argument("--enable-free-decay-mode", action="store_true", help="Enable release-gated free-decay startup mode.")
    ap.add_argument("--free-decay-arm-min-angle-deg", type=float, default=5.0)
    ap.add_argument("--free-decay-hold-min-sec", type=float, default=2.0)
    ap.add_argument("--free-decay-hold-gyro-threshold", type=float, default=0.15)
    ap.add_argument("--free-decay-release-gyro-threshold", type=float, default=0.35)
    ap.add_argument("--free-decay-release-delta-deg", type=float, default=0.25)
    args = ap.parse_args()

    if args.mode == "host":
        args.host_control = True
    elif args.mode == "jetson":
        args.host_control = False

    cfg = BridgeConfig()
    imu_sign = -1.0  # fixed CW/CCW convention alignment against encoder
    cfg.enable_render = not args.headless
    cfg.enable_imu_viewer = not args.no_imu_viewer
    cfg.step = args.step
    cfg.theta0_deg = args.theta0_deg
    cfg.omega0 = args.omega0
    # Keep masses fixed for physically consistent COM-based rigid body.
    cfg.rod_mass = 0.200
    cfg.imu_mass = 0.020
    cfg.rod_length = args.link_length
    cfg.link_L = args.link_length
    cfg.r_imu = args.r_imu
    if args.l_com is not None:
        print("[WARN] --l-com is deprecated and ignored. l_com is now computed from Chrono body COM.")
    cfg.b_eq_init = float(args.b) if args.b is not None else float(cfg.b_eq_init)
    cfg.tau_eq_init = float(args.tau_c) if args.tau_c is not None else float(cfg.tau_eq_init)
    cfg.K_i_init = float(args.k_i) if args.k_i is not None else float(cfg.K_i_init)

    calib = apply_calibration_json(cfg, args.calibration_json)
    param_data = load_model_parameter_json(args.parameter_json)
    runtime_overrides = extract_runtime_overrides(param_data, cfg) if param_data is not None else {}
    current_offset_mA = 0.0
    if isinstance(calib, dict):
        sm = calib.get("summary", {}) if isinstance(calib.get("summary"), dict) else {}
        for key in ("ina_current_offset_mA", "current_offset_mA"):
            if key in sm:
                try:
                    current_offset_mA = float(sm[key])
                except (TypeError, ValueError):
                    current_offset_mA = 0.0
                break
    radius_measured = extract_radius_from_json(args.radius_json)
    if radius_measured is not None:
        cfg.r_imu = float(radius_measured)
    if runtime_overrides:
        if "r_imu" in runtime_overrides:
            cfg.r_imu = float(runtime_overrides["r_imu"])
        if "gravity" in runtime_overrides:
            cfg.gravity = float(runtime_overrides["gravity"])

    log_csv = make_numbered_path(cfg.log_dir, cfg.log_prefix, ".csv")
    log_finalized_csv = log_csv[:-4] + ".finalized.csv"
    log_meta = log_csv[:-4] + ".meta.json"

    rclpy.init()
    shared = SharedROSState()
    ros_node = PendulumROSNode(cfg, shared, host_mode=args.host_control)
    ros_stop = threading.Event()
    ros_thr = threading.Thread(target=ros_spin_thread, args=(ros_node, ros_stop), daemon=True)
    ros_thr.start()

    model = PendulumModel(cfg)
    print(
        f"[inertia] J_rod={model.J_rod:.6f} kg·m^2, "
        f"J_imu={model.J_imu:.6f} kg·m^2, "
        f"J_total={model.J_total:.6f} kg·m^2"
    )
    pivot_w = model.pivot_pos_world()
    rod_com_w = model.rod_com_pos_world()
    imu_com_w = model.imu_com_pos_world()
    total_com_w = model.total_com_pos_world()
    imu_local = model.imu_local_on_rod()
    theta0_rad = math.radians(cfg.theta0_deg)
    imu_target_w = np.array(
        [
            -math.sin(theta0_rad) * cfg.r_imu,
            -math.cos(theta0_rad) * cfg.r_imu,
            cfg.motor_length / 2.0,
        ],
        dtype=float,
    )
    print(
        f"[com] pivot_world=[{pivot_w[0]:+.4f}, {pivot_w[1]:+.4f}, {pivot_w[2]:+.4f}] m | "
        f"rod_com_world=[{rod_com_w[0]:+.4f}, {rod_com_w[1]:+.4f}, {rod_com_w[2]:+.4f}] m"
    )
    print(
        f"[com] rod_COM_radius_from_pivot={model.rod_com_radius_from_pivot():.6f} m "
        f"(computed as link_L/2 = {cfg.link_L:.6f}/2)"
    )
    print(
        f"[com] total_com_world=[{total_com_w[0]:+.4f}, {total_com_w[1]:+.4f}, {total_com_w[2]:+.4f}] m | "
        f"total_l_com_from_pivot={model.total_l_com_from_pivot():.6f} m"
    )
    print(
        f"[imu-fix] imu_com_world=[{imu_com_w[0]:+.4f}, {imu_com_w[1]:+.4f}, {imu_com_w[2]:+.4f}] m | "
        f"imu_target_world_from_pivot=[{imu_target_w[0]:+.4f}, {imu_target_w[1]:+.4f}, {imu_target_w[2]:+.4f}] m | "
        f"imu_local_on_rod=[{imu_local[0]:+.4f}, {imu_local[1]:+.4f}, {imu_local[2]:+.4f}] m"
    )
    print(f"[imu-fix] imu_radius_from_pivot={model.imu_radius_from_pivot():.6f} m")
    sim_params = {
        "b_eq": float(runtime_overrides.get("b_eq", cfg.b_eq_init)),
        "tau_eq": float(runtime_overrides.get("tau_eq", cfg.tau_eq_init)),
        "K_i": float(runtime_overrides.get("K_i", cfg.K_i_init)),
        "residual_terms": list(runtime_overrides.get("residual_terms", [])),
    }
    model.update_identified_structure(sim_params)
    prev_u_eff = 0.0
    prev_du = 0.0

    viewer_proc = None
    if cfg.enable_imu_viewer:
        viewer_topic = "/imu/data"
        viewer_proc = start_imu_viewer_process(viewer_topic, cfg.topic_hw_enc)
        print(
            f"[INFO] IMU viewer started (topic={viewer_topic}, enc_topic={cfg.topic_hw_enc}) "
            f"| free_decay_mode={'on' if args.enable_free_decay_mode else 'off'}"
        )

    vis = None
    if cfg.enable_render:
        vis = irr.ChVisualSystemIrrlicht()
        vis.AttachSystem(model.sys)
        vis.SetWindowSize(cfg.win_w, cfg.win_h)
        vis.SetWindowTitle(cfg.window_title)
        vis.Initialize()
        vis.AddSkyBox()
        vis.AddCamera(ch.ChVector3d(0.7, 0.2, 0.8))
        vis.AddTypicalLights()

    host_controller = HostCommandController(cfg) if args.host_control else None
    quit_watcher = None if args.host_control else QuitWatcher()

    with open(log_csv, "w", newline="", encoding="utf-8") as f, open(log_finalized_csv, "w", newline="", encoding="utf-8") as ff:
        wr = csv.writer(f)
        wr.writerow(PENDULUM_LOG_COLUMNS)
        wr_final = csv.writer(ff)
        wr_final.writerow([
            "wall_elapsed",
            "I_filtered_mA",
            "theta_imu_filtered_unwrapped",
            "omega_imu_filtered",
            "alpha_from_linear_accel_filtered",
        ])

        wall_t0 = now_wall()
        wall_run_t0 = None
        last_model_step_wall = None
        omega_prev = model.get_omega()
        t_prev = 0.0
        enc_ref = None
        theta_encoder_prev = None
        omega_imu_prev = 0.0
        omega_encoder_prev = 0.0
        warmup_sec = 1.0
        run_state = RunState.STATE_WARMUP
        imu_R0 = None
        real_omega_hist = deque(maxlen=401)
        real_time_hist = deque(maxlen=401)
        dt_history = []
        prev_wall_elapsed = None
        online_filter_bank = {
            "theta_imu": CausalIIRFilter(alpha=0.18),
            "theta_encoder": CausalIIRFilter(alpha=0.18),
            "omega_imu": CausalIIRFilter(alpha=0.18),
            "omega_encoder": CausalIIRFilter(alpha=0.18),
            "alpha_imu": CausalIIRFilter(alpha=0.18),
            "alpha_linear": CausalIIRFilter(alpha=0.18),
            "alpha_encoder": CausalIIRFilter(alpha=0.18),
            "ina_current_signed_mA": CausalIIRFilter(alpha=0.18),
        }
        online_state = {k: 0.0 for k in online_filter_bank.keys()}
        warmup_theta_samples = []
        warmup_omega_samples = []
        warmup_current_samples = []
        warmup_pwm_samples = []
        warmup_alpha_linear_samples = []
        theta_offset_rad = 0.0
        alpha_linear_offset = 0.0
        current_offset_used_mA = float(current_offset_mA)
        theta_imu_prev_wrapped = None
        theta_imu_unwrapped_acc = 0.0
        theta_encoder_prev_wrapped = None
        theta_encoder_unwrapped_acc = 0.0
        run_limit_sec = float("inf") if args.duration <= 0.0 else float(args.duration)
        free_decay_arm_start = None
        free_decay_theta_prev = None
        free_decay_theta_hold = None
        free_decay_theta_prev_wrapped = None
        free_decay_theta_unwrapped_acc = 0.0
        free_decay_theta_filter = CausalIIRFilter(alpha=0.12)
        free_decay_omega_filter = CausalIIRFilter(alpha=0.18)

        if host_controller is not None:
            host_controller.__enter__()
            terminal_status_line("cmd_u:    0.0 | used:    0.0 | mode: manual | waiting for keyboard input", width=cfg.terminal_status_width)
            print()
        elif quit_watcher is not None:
            quit_watcher.__enter__()

        if calib is not None:
            print(f"[INFO] Loaded calibration json: {args.calibration_json}")
        if param_data is not None:
            print(f"[INFO] Loaded model-parameter json: {args.parameter_json}")
        if radius_measured is not None:
            print(f"[INFO] Loaded radius json: {args.radius_json}")
        if np.isfinite(cfg.cpr):
            print(f"[INFO] CPR from calibration json: {cfg.cpr:.3f} counts/rev")
        print(f"[INFO] Chrono rod box length (--link-length): {cfg.link_L:.6f} m")
        print(f"[INFO] Unified physical radius (pivot->IMU COM): {cfg.r_imu:.6f} m")
        if math.isfinite(run_limit_sec):
            print(f"[INFO] run limit: {run_limit_sec:.1f}s")
        else:
            print("[INFO] run limit: none (quit with q/ESC)")
        print("[INFO] alpha_real (legacy/export) source: filtered derivative of omega")
        print("[INFO] finalized training alpha source: filtered derivative of omega (compat column name kept)")
        if runtime_overrides:
            print(
                "[INFO] Runtime overrides from model-parameter json: "
                f"K_i={sim_params['K_i']:.6g}, b_eq={sim_params['b_eq']:.6g}, tau_eq={sim_params['tau_eq']:.6g}, "
                f"r_imu={cfg.r_imu:.6f}, gravity={cfg.gravity:.6f}, "
                f"residual_terms={len(sim_params.get('residual_terms', []))}"
            )
        if args.enable_free_decay_mode:
            print(
                "[INFO] free-decay startup mode: enabled "
                f"(arm_angle>={args.free_decay_arm_min_angle_deg:.2f}deg, hold>={args.free_decay_hold_min_sec:.2f}s)"
            )
        gravity_comp_enabled = not bool(args.imu_linear_accel_no_gravity_comp)
        print(
            "[INFO] alpha from linear accel: source_frame=imu_body, projection_frame=world0_xy, "
            f"radius_m={cfg.r_imu:.6f}, gravity_mps2={cfg.gravity:.6f}, "
            f"gravity_compensation={'on' if gravity_comp_enabled else 'off'}"
        )
        g0 = gravity_world0_from_imu_anchor(imu_R0=imu_R0, gravity_mps2=cfg.gravity)
        print(f"[INFO] gravity vector in world0 frame (initial estimate): [{g0[0]:.4f}, {g0[1]:.4f}, {g0[2]:.4f}] m/s^2")

        try:
            while (now_wall() - wall_t0) < run_limit_sec:
                if host_controller is not None:
                    host_controller.poll()
                    if host_controller.quit_requested:
                        break
                elif quit_watcher is not None:
                    quit_watcher.poll()
                    if quit_watcher.quit_requested:
                        break

                if vis is not None:
                    # Keep rendering loop alive regardless of vis.Run() return.
                    # User requested no auto-stop/auto-fallback on render-window state.
                    vis.Run()
                    vis.BeginScene(); vis.Render(); vis.EndScene()

                if host_controller is not None:
                    cmd_u_raw = host_controller.get_command()
                    ros_node.publish_host_cmd(cmd_u_raw, host_controller.mode)
                    mode_name = host_controller.mode
                else:
                    if quit_watcher is not None:
                        quit_watcher.poll()
                        if quit_watcher.quit_requested:
                            break
                    snap0 = shared.snapshot()
                    cmd_u_raw = snap0["cmd_u"]
                    mode_name = "external"

                snap = shared.snapshot()
                wall_now = now_wall()
                sim_t = wall_now - wall_t0

                if run_state == RunState.STATE_WARMUP:
                    if host_controller is not None:
                        ros_node.publish_host_cmd(0.0, "warmup")
                    theta_warm = float(model.get_theta())
                    omega_warm = float(model.get_omega())
                    if snap.get("imu_has_data", False):
                        theta_wrapped_imu, R_rel, imu_R0 = compute_theta_wrapped_from_imu_snapshot(
                            snap=snap,
                            imu_R0=imu_R0,
                            radius=cfg.r_imu,
                            imu_sign=imu_sign,
                        )
                        if theta_wrapped_imu is not None:
                            theta_warm = float(theta_wrapped_imu)
                        omega_warm = float(imu_sign * snap["imu_w"][2])
                        acc_body = np.asarray(snap["imu_a"], dtype=float)
                        acc_world0 = R_rel @ acc_body if R_rel is not None else acc_body
                        if gravity_comp_enabled:
                            acc_world0 = acc_world0 - gravity_world0_from_imu_anchor(imu_R0=imu_R0, gravity_mps2=cfg.gravity)
                        tangent_warm = np.array([-math.sin(theta_warm), math.cos(theta_warm), 0.0], dtype=float)
                        a_t_warm = float(np.dot(acc_world0, tangent_warm))
                        if np.isfinite(cfg.r_imu) and cfg.r_imu > 1e-6:
                            warmup_alpha_linear_samples.append(float(a_t_warm / float(cfg.r_imu)))
                    warmup_theta_samples.append(theta_warm)
                    warmup_omega_samples.append(omega_warm)
                    warmup_current_samples.append(float(snap["current_mA"]))
                    warmup_pwm_samples.append(float(snap["hw_pwm"]))
                    terminal_status_line(
                        f"[STATE_WARMUP] {sim_t:4.2f}/{warmup_sec:.2f}s | init-only collection",
                        width=cfg.terminal_status_width,
                    )
                    if sim_t >= warmup_sec:
                        theta_offset_raw = compute_theta_offset(np.asarray(warmup_theta_samples, dtype=float))
                        theta_offset_rad = float(theta_offset_raw) if args.enable_free_decay_mode else 0.0
                        if len(warmup_alpha_linear_samples) >= 4:
                            alpha_linear_offset = float(np.nanmedian(np.asarray(warmup_alpha_linear_samples, dtype=float)))
                        current_offset_used_mA, valid_cur_n = compute_current_offset(
                            np.asarray(warmup_current_samples, dtype=float),
                            np.asarray(warmup_pwm_samples, dtype=float),
                            np.asarray(warmup_omega_samples, dtype=float),
                            pwm_threshold=3.0,
                            omega_threshold=0.8,
                            fallback_mA=26.0,
                        )
                        if valid_cur_n < 12:
                            print(f"[WARN] insufficient valid warmup current samples: {valid_cur_n}")
                        theta_var = float(np.nanstd(np.unwrap(np.asarray(warmup_theta_samples, dtype=float)))) if len(warmup_theta_samples) > 8 else 0.0
                        if theta_var > 0.25:
                            print(f"[WARN] high warmup theta variance: {theta_var:.5f} rad")
                        print(
                            "[WARMUP DONE] "
                            f"warmup_duration={warmup_sec:.2f}s, theta_offset_rad={theta_offset_rad:.6f}, "
                            f"theta_offset_raw={theta_offset_raw:.6f}, "
                            f"alpha_linear_offset={alpha_linear_offset:.6f}, "
                            f"current_offset_mA={current_offset_used_mA:.6f}, number_of_valid_current_samples={valid_cur_n}"
                        )
                        if args.enable_free_decay_mode:
                            run_state = RunState.STATE_FREE_DECAY_ARM
                            wall_run_t0 = None
                            free_decay_arm_start = None
                            free_decay_theta_prev = None
                            free_decay_theta_hold = None
                            free_decay_theta_prev_wrapped = None
                            free_decay_theta_unwrapped_acc = float(theta_offset_rad)
                            free_decay_theta_filter.reset(None)
                            free_decay_omega_filter.reset(None)
                            print("[INFO] entering free-decay arming phase")
                        else:
                            run_state = RunState.STATE_RUN
                            wall_run_t0 = now_wall()
                            last_model_step_wall = wall_run_t0
                        t_prev = 0.0
                        prev_wall_elapsed = None
                        real_omega_hist.clear()
                        real_time_hist.clear()
                        theta_imu_prev_wrapped, _, imu_R0 = compute_theta_wrapped_from_imu_snapshot(
                            snap=snap,
                            imu_R0=imu_R0,
                            radius=cfg.r_imu,
                            imu_sign=imu_sign,
                        )
                        if args.enable_free_decay_mode:
                            theta_imu_unwrapped_acc = float(theta_offset_rad)
                        else:
                            theta_imu_unwrapped_acc = float(theta_imu_prev_wrapped) if theta_imu_prev_wrapped is not None else 0.0
                        theta_encoder_prev_wrapped = None
                        theta_encoder_unwrapped_acc = 0.0
                        # Reset filter state to "unseeded": first run sample becomes the seed.
                        online_state = {k: 0.0 for k in online_filter_bank.keys()}
                        for k, flt in online_filter_bank.items():
                            flt.reset(None)
                        omega_imu_prev = 0.0
                        omega_encoder_prev = 0.0
                        theta_encoder_prev = None
                        if np.isfinite(snap["hw_enc"]):
                            enc_ref = float(snap["hw_enc"])
                            if np.isfinite(cfg.cpr) and cfg.cpr > 1.0:
                                theta_encoder_prev_wrapped = 0.0
                    if cfg.realtime:
                        time.sleep(min(cfg.step, 0.01))
                    continue

                if run_state in (RunState.STATE_FREE_DECAY_ARM, RunState.STATE_FREE_DECAY_WAIT_RELEASE):
                    # Keep free-decay startup control-less regardless of host/jetson mode.
                    ros_node.publish_host_cmd(0.0, "free_decay")
                    theta_meas = float(model.get_theta())
                    omega_meas = float(model.get_omega())
                    if snap.get("imu_has_data", False):
                        theta_wrapped_imu, _, imu_R0 = compute_theta_wrapped_from_imu_snapshot(
                            snap=snap,
                            imu_R0=imu_R0,
                            radius=cfg.r_imu,
                            imu_sign=imu_sign,
                        )
                        if theta_wrapped_imu is not None:
                            if free_decay_theta_prev_wrapped is None:
                                free_decay_theta_prev_wrapped = float(theta_wrapped_imu)
                            dth_fd = float(theta_wrapped_imu - free_decay_theta_prev_wrapped)
                            while dth_fd > math.pi:
                                dth_fd -= 2.0 * math.pi
                            while dth_fd < -math.pi:
                                dth_fd += 2.0 * math.pi
                            free_decay_theta_unwrapped_acc += dth_fd
                            free_decay_theta_prev_wrapped = float(theta_wrapped_imu)
                            theta_meas = float(free_decay_theta_unwrapped_acc - theta_offset_rad)
                        omega_meas = float(imu_sign * snap["imu_w"][2])

                    theta_lp = float(free_decay_theta_filter.update(theta_meas))
                    omega_lp = float(free_decay_omega_filter.update(omega_meas))
                    # Keep simulated pendulum synchronized to measured real pendulum pose while waiting for release.
                    model.set_theta_kinematic(theta_lp, omega_lp)

                    if run_state == RunState.STATE_FREE_DECAY_ARM:
                        is_lifted = abs(math.degrees(theta_lp)) >= float(args.free_decay_arm_min_angle_deg)
                        is_steady = abs(omega_lp) <= float(args.free_decay_hold_gyro_threshold)
                        if is_lifted and is_steady:
                            if free_decay_arm_start is None:
                                free_decay_arm_start = wall_now
                            if (wall_now - free_decay_arm_start) >= float(args.free_decay_hold_min_sec):
                                free_decay_theta_hold = float(theta_lp)
                                model.set_theta_kinematic(free_decay_theta_hold, 0.0)
                                run_state = RunState.STATE_FREE_DECAY_WAIT_RELEASE
                                free_decay_theta_prev = float(theta_lp)
                                print(f"\n[INFO] armed! theta_arm={math.degrees(free_decay_theta_hold):.3f} deg")
                        else:
                            free_decay_arm_start = None
                        hold_elapsed = 0.0 if free_decay_arm_start is None else (wall_now - free_decay_arm_start)
                        terminal_status_line(
                            f"[arming...] |theta|={abs(math.degrees(theta_lp)):6.2f}/{args.free_decay_arm_min_angle_deg:.2f} deg | "
                            f"omega_lp={omega_lp: .4f}/{args.free_decay_hold_gyro_threshold:.4f} | "
                            f"hold={hold_elapsed:4.2f}/{args.free_decay_hold_min_sec:.2f}s",
                            width=cfg.terminal_status_width,
                        )
                    else:
                        dtheta_deg = 0.0 if free_decay_theta_prev is None else math.degrees(theta_lp - free_decay_theta_prev)
                        terminal_status_line(
                            f"[armed] release 대기 | dtheta={dtheta_deg: .3f}/{args.free_decay_release_delta_deg:.3f} deg | "
                            f"omega_lp={omega_lp: .4f}/{args.free_decay_release_gyro_threshold:.4f}",
                            width=cfg.terminal_status_width,
                        )
                        if (
                            abs(dtheta_deg) >= float(args.free_decay_release_delta_deg)
                            or abs(omega_lp) >= float(args.free_decay_release_gyro_threshold)
                        ):
                            # Seed run start with release instant state for real-time sim2real alignment.
                            model.set_theta_kinematic(theta_lp, omega_lp)
                            print(
                                f"\n[INFO] release 검출: t={wall_now:.6f}, "
                                f"theta={math.degrees(theta_lp):.3f} deg, omega_lp={omega_lp:.6f} rad/s"
                            )
                            run_state = RunState.STATE_RUN
                            wall_run_t0 = now_wall()
                            last_model_step_wall = wall_run_t0
                            t_prev = 0.0
                            omega_prev = model.get_omega()
                        free_decay_theta_prev = float(theta_lp)
                    if cfg.realtime:
                        time.sleep(min(cfg.step, 0.01))
                    continue

                cmd_u_used = cmd_u_raw
                if wall_run_t0 is not None:
                    sim_t = wall_now - wall_run_t0

                theta_before = model.get_theta()
                motor_input_current = float(online_state.get("ina_current_signed_mA", 0.0))
                model_out = compute_model_torque_and_electrics(
                    motor_input_current,
                    theta_before,
                    model.get_omega(),
                    float("nan"),
                    sim_params,
                    cfg,
                    cmd_u_for_duty=cmd_u_used,
                )
                model.apply_torque(model_out["tau_net"])
                step_h = float(cfg.step)
                if cfg.realtime:
                    if last_model_step_wall is None:
                        last_model_step_wall = wall_now
                    dt_wall = max(wall_now - last_model_step_wall, 0.0)
                    # Prevent apparent slow-motion when render/host loop is slower
                    # than configured dynamics step by syncing integration horizon to
                    # elapsed wall time.
                    step_h = min(max(dt_wall, cfg.step), 0.05)
                    last_model_step_wall = wall_now
                model.step(step_h)

                # wall_elapsed is the canonical timeline for runtime + replay CSVs.
                theta = model.get_theta()
                omega = model.get_omega()
                alpha = (omega - omega_prev) / max(sim_t - t_prev, cfg.step) if sim_t > 0 else 0.0
                alpha = sanitize_float(alpha)
                omega_prev = omega
                t_prev = sim_t

                _, _, a_imu, q_imu, w_imu = model.get_sensor_kinematics(sim_t, cfg.step)
                imu_msg = build_imu_msg(sim_t, q_imu, w_imu, a_imu)

                if enc_ref is None and np.isfinite(snap["hw_enc"]):
                    enc_ref = float(snap["hw_enc"])

                theta_meas_wrapped = theta
                omega_meas = omega
                alpha_meas = alpha
                R_rel = None
                if snap.get("imu_has_data", False):
                    w_imu_raw = snap["imu_w"]
                    theta_wrapped_imu, R_rel, imu_R0 = compute_theta_wrapped_from_imu_snapshot(
                        snap=snap,
                        imu_R0=imu_R0,
                        radius=cfg.r_imu,
                        imu_sign=imu_sign,
                    )
                    if theta_wrapped_imu is not None:
                        theta_meas_wrapped = float(theta_wrapped_imu)
                    omega_meas = float(imu_sign * w_imu_raw[2])

                if theta_imu_prev_wrapped is None:
                    theta_imu_prev_wrapped = float(theta_meas_wrapped)
                dth = float(theta_meas_wrapped - theta_imu_prev_wrapped)
                while dth > math.pi:
                    dth -= 2.0 * math.pi
                while dth < -math.pi:
                    dth += 2.0 * math.pi
                theta_imu_unwrapped_acc += dth
                theta_imu_prev_wrapped = float(theta_meas_wrapped)
                theta_meas = float(theta_imu_unwrapped_acc - theta_offset_rad)

                real_omega_hist.append(float(omega_meas))
                real_time_hist.append(float(sim_t))
                alpha_meas = float(
                    estimate_filtered_alpha_from_omega(
                        np.asarray(real_omega_hist, dtype=float),
                        t=np.asarray(real_time_hist, dtype=float),
                    )[-1]
                )

                dt_local = max(float(cfg.step), 1e-6)
                if prev_wall_elapsed is not None:
                    dt_history.append(float(max(sim_t - prev_wall_elapsed, 1e-6)))
                prev_wall_elapsed = float(sim_t)
                pwm_hw = float(snap["hw_pwm"])
                ina_current_raw_mA = float(snap["current_mA"])
                ina_bus_voltage_v = float(snap["ina_bus_voltage_v"])
                ina_power_mw = float(snap["ina_power_mw"])
                ina_current_corr_mA = float(ina_current_raw_mA - current_offset_used_mA)
                sign_pwm = 1.0 if pwm_hw > 0.0 else (-1.0 if pwm_hw < 0.0 else 0.0)
                ina_current_signed_mA = float(sign_pwm * ina_current_corr_mA)

                theta_imu = float(theta_meas)
                omega_imu = float(omega_meas)
                alpha_imu = float((omega_imu - omega_imu_prev) / dt_local)
                omega_imu_prev = omega_imu

                theta_encoder = float(theta_imu if not np.isfinite(snap["hw_enc"]) else 0.0)
                omega_encoder = float(omega_imu if not np.isfinite(snap["hw_enc"]) else 0.0)
                alpha_encoder = float(alpha_imu if not np.isfinite(snap["hw_enc"]) else 0.0)
                if np.isfinite(snap["hw_enc"]) and np.isfinite(cfg.cpr) and cfg.cpr > 1.0:
                    if enc_ref is None:
                        enc_ref = float(snap["hw_enc"])
                    theta_enc_wrapped = float((2.0 * math.pi / float(cfg.cpr)) * (float(snap["hw_enc"]) - float(enc_ref)))
                    if theta_encoder_prev_wrapped is None:
                        theta_encoder_prev_wrapped = theta_enc_wrapped
                    dth_e = float(theta_enc_wrapped - theta_encoder_prev_wrapped)
                    while dth_e > math.pi:
                        dth_e -= 2.0 * math.pi
                    while dth_e < -math.pi:
                        dth_e += 2.0 * math.pi
                    theta_encoder_unwrapped_acc += dth_e
                    theta_encoder_prev_wrapped = theta_enc_wrapped
                    theta_encoder = float(theta_encoder_unwrapped_acc - theta_offset_rad)
                    if theta_encoder_prev is not None:
                        omega_encoder = float((theta_encoder - theta_encoder_prev) / dt_local)
                    if theta_encoder_prev is not None:
                        alpha_encoder = float((omega_encoder - omega_encoder_prev) / dt_local)
                    theta_encoder_prev = theta_encoder
                    omega_encoder_prev = omega_encoder

                # alpha from linear acceleration (tangential component / radius)
                alpha_linear = float(alpha_imu)
                if np.isfinite(cfg.r_imu) and cfg.r_imu > 1e-6 and snap.get("imu_has_data", False):
                    acc_body = np.asarray(snap["imu_a"], dtype=float)
                    acc_world0 = (R_rel @ acc_body) if R_rel is not None else acc_body
                    if gravity_comp_enabled:
                        acc_world0 = acc_world0 - gravity_world0_from_imu_anchor(imu_R0=imu_R0, gravity_mps2=cfg.gravity)
                    theta_ref_abs = float(theta_meas_wrapped)
                    tangent = np.array([-math.sin(theta_ref_abs), math.cos(theta_ref_abs), 0.0], dtype=float)
                    a_t = float(np.dot(acc_world0, tangent))
                    alpha_linear = float(a_t / float(cfg.r_imu))
                alpha_linear = float(alpha_linear - alpha_linear_offset)

                # online low-pass filter (explicit online path)
                for k, v in {
                    "theta_imu": theta_imu,
                    "theta_encoder": theta_encoder,
                    "omega_imu": omega_imu,
                    "omega_encoder": omega_encoder,
                    "alpha_imu": alpha_imu,
                    "alpha_linear": alpha_linear,
                    "alpha_encoder": alpha_encoder,
                    "ina_current_signed_mA": ina_current_signed_mA,
                }.items():
                    online_state[k] = online_filter_bank[k].update(float(v))

                e_theta = theta - theta_meas
                e_omega = omega - omega_meas
                e_alpha = alpha - alpha_meas
                du = cmd_u_used - prev_u_eff
                d2u = du - prev_du
                prev_u_eff = cmd_u_used
                prev_du = du
                inst_cost = (
                    cfg.w_theta * (e_theta ** 2) +
                    cfg.w_omega * (e_omega ** 2) +
                    cfg.w_alpha * (e_alpha ** 2) +
                    cfg.w_du * (du ** 2) +
                    cfg.w_d2u * (d2u ** 2)
                )
                status = make_status_line(
                    host_mode=(host_controller is not None),
                    cmd_u_raw=cmd_u_raw,
                    cmd_u_used=cmd_u_used,
                    hw_pwm=snap["hw_pwm"],
                    current_mA=snap["current_mA"],
                    mode_name=mode_name,
                    cfg=cfg,
                )
                terminal_status_line(status, width=cfg.terminal_status_width)
                ros_node.publish_sim(theta, omega, alpha, model_out["tau_net"], cmd_u_used, imu_msg, status)

                wr.writerow([
                    wall_now, sim_t, mode_name,
                    cmd_u_raw, cmd_u_used, pwm_hw, model_out["tau_net"],
                    ina_current_raw_mA, ina_bus_voltage_v, ina_power_mw,
                    current_offset_used_mA, ina_current_corr_mA, ina_current_signed_mA,
                    pwm_hw,
                    theta_imu, theta_encoder,
                    omega_imu, omega_encoder,
                    alpha_imu, alpha_linear, alpha_encoder,
                    online_state["theta_imu"], online_state["theta_encoder"],
                    online_state["omega_imu"], online_state["omega_encoder"],
                    online_state["alpha_imu"], online_state["alpha_linear"], online_state["alpha_encoder"],
                    online_state["ina_current_signed_mA"],
                    theta, omega, alpha, alpha,
                    snap["hw_enc"], snap["hw_arduino_ms"],
                    theta_meas, omega_meas, alpha_meas,
                    model.J_rod, model.J_imu, model.J_total,
                    model_out["tau_motor"], model_out["tau_res"], model_out["tau_visc"], model_out["tau_coul"],
                    inst_cost, inst_cost,
                    q_imu[0], q_imu[1], q_imu[2], q_imu[3],
                    w_imu[0], w_imu[1], w_imu[2],
                    a_imu[0], a_imu[1], a_imu[2],
                ])
                wr_final.writerow([
                    sim_t,
                    online_state["ina_current_signed_mA"],
                    online_state["theta_imu"],
                    online_state["omega_imu"],
                    online_state["alpha_imu"],
                ])

                if cfg.realtime:
                    target_wall = wall_t0 + sim_t
                    dt_sleep = target_wall - now_wall()
                    if dt_sleep > 0:
                        time.sleep(min(dt_sleep, 0.01))
        finally:
            if host_controller is not None:
                host_controller.__exit__(None, None, None)
            elif quit_watcher is not None:
                quit_watcher.__exit__(None, None, None)

    if dt_history:
        dt_arr = np.asarray(dt_history, dtype=float)
        sampling_diag = {
            "dt_mean_sec": float(np.mean(dt_arr)),
            "dt_std_sec": float(np.std(dt_arr)),
            "dt_min_sec": float(np.min(dt_arr)),
            "dt_max_sec": float(np.max(dt_arr)),
            "freq_mean_hz": float(1.0 / max(np.mean(dt_arr), 1e-9)),
            "sample_count": int(len(dt_arr)),
        }
    else:
        sampling_diag = {}

    meta = {
        "log_csv": log_csv,
        "log_finalized_csv": log_finalized_csv,
        "config": asdict(cfg),
        "inertia": {
            "J_rod": float(model.J_rod),
            "J_imu": float(model.J_imu),
            "J_total": float(model.J_total),
            "rod_mass": float(cfg.rod_mass),
            "rod_length": float(cfg.rod_length),
            "imu_mass": float(cfg.imu_mass),
            "r_imu": float(cfg.r_imu),
        },
        "calibration_json": cfg.calibration_json if calib is not None else None,
        "radius_json": args.radius_json,
        "cpr_fixed": None if not np.isfinite(cfg.cpr) else float(cfg.cpr),
        "best_eval": None,
        "ls_cost": None,
        "fit_done": False,
        "fit_complete": False,
        "fit_complete_wall": None,
        "fit_final_params": None,
        "sampling_diagnostics": sampling_diag,
        "warmup": {
            "duration_sec": warmup_sec,
            "theta_offset_rad": float(theta_offset_rad),
            "alpha_linear_offset_radps2": float(alpha_linear_offset),
            "current_offset_mA": float(current_offset_used_mA),
            "imu_sign_applied": float(imu_sign),
            "warmup_theta_sample_count": int(len(warmup_theta_samples)),
            "warmup_current_sample_count": int(len(warmup_current_samples)),
        },
        "signal_policy": {
            "alpha_real_source": "filtered_domega_dt",
            "alpha_linear_source": "filtered_domega_dt_finalized_export",
            "alpha_linear_frame": "imu_body_to_world0_xy",
            "alpha_linear_gravity_compensated": bool(gravity_comp_enabled),
            "gravity_mps2_used": float(cfg.gravity),
            "radius_m_used": float(cfg.r_imu),
        },
    }
    with open(log_meta, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    print(f"saved csv  : {log_csv}")
    print(f"saved csv(finalized) : {log_finalized_csv}")
    print(f"saved meta : {log_meta}")

    if viewer_proc is not None:
        try:
            os.killpg(viewer_proc.pid, signal.SIGTERM)
            viewer_proc.wait(timeout=2.0)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.killpg(viewer_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    ros_stop.set()
    try:
        ros_thr.join(timeout=1.0)
    except Exception:
        pass
    try:
        ros_node.destroy_node()
    except Exception:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
