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
from chrono_core.pendulum_rl_env import build_init_params
from chrono_core.log_schema import PENDULUM_LOG_COLUMNS
from chrono_core.signal_filter import estimate_filtered_alpha_from_omega


# ============================================================
# utility
# ============================================================

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


def make_status_line(host_mode: bool, cmd_u_raw: float, cmd_u_used: float, hw_pwm: float, mode_name: str,
                     cfg: BridgeConfig):
    if host_mode:
        return (
            f"cmd_u: {cmd_u_raw:6.1f} | used: {cmd_u_used:6.1f} | mode: {mode_name:<6} | "
            f"step: {cfg.pwm_step:4.1f} | max: {cfg.pwm_limit:5.1f}"
        )
    return (
        f"cmd_u: {cmd_u_used:6.1f} | hw_pwm: {hw_pwm:6.1f} | mode: external"
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
    ap.add_argument("--theta0-deg", type=float, default=-10.0)
    ap.add_argument("--omega0", type=float, default=0.0)
    ap.add_argument("--link-mass", type=float, default=0.200)
    ap.add_argument("--link-length", type=float, default=0.285)
    ap.add_argument("--host-control", action="store_true",
                    help="Enable host-side manual command publishing.")
    ap.add_argument("--mode", choices=["host", "jetson"], default=None,
                    help="Compatibility option: host enables host-control, jetson uses external ROS input.")
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--calibration-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--parameter-json", default="", help="RL/exported parameter JSON containing model_init and optional delay_sec")
    ap.add_argument("--radius-json", default="./run_logs/calibration_latest.json",
                    help="JSON file containing measured radius (e.g., calibration_latest.json).")
    ap.add_argument("--l-com", type=float, default=None, help="default: link_length/2 for fresh runs")
    ap.add_argument("--b", type=float, default=None, help="default: near-zero fresh initialization")
    ap.add_argument("--tau-c", type=float, default=None, help="default: near-zero fresh initialization")
    ap.add_argument("--k-u", type=float, default=None, help="default: near-zero positive fresh initialization")
    ap.add_argument("--r-imu", type=float, default=0.285, help="IMU radius from pivot [m]")
    ap.add_argument(
        "--real-alpha-source",
        choices=["omega_diff", "tangential_accel", "blend"],
        default="omega_diff",
        help="deprecated; real alpha now always uses filtered d(omega)/dt",
    )
    args = ap.parse_args()

    if args.mode == "host":
        args.host_control = True
    elif args.mode == "jetson":
        args.host_control = False

    cfg = BridgeConfig()
    cfg.enable_render = not args.headless
    cfg.enable_imu_viewer = not args.no_imu_viewer
    cfg.step = args.step
    cfg.theta0_deg = args.theta0_deg
    cfg.omega0 = args.omega0
    # Keep masses fixed for physically consistent COM-based rigid body.
    cfg.rod_mass = 0.200
    cfg.link_mass = cfg.rod_mass
    cfg.imu_mass = 0.020
    cfg.rod_length = args.link_length
    cfg.link_L = args.link_length
    cfg.radius_m = args.link_length
    cfg.r_imu = args.r_imu
    cfg.l_com_init = float(args.l_com) if args.l_com is not None else (0.5 * float(args.link_length))
    cfg.b_eq_init = float(args.b) if args.b is not None else float(cfg.b_eq_init)
    cfg.tau_eq_init = float(args.tau_c) if args.tau_c is not None else float(cfg.tau_eq_init)
    cfg.K_u_init = float(args.k_u) if args.k_u is not None else float(cfg.K_u_init)
    cfg.delay_init_ms = args.delay_ms

    calib = apply_calibration_json(cfg, args.calibration_json)
    param_data = None
    if args.parameter_json:
        with open(args.parameter_json, "r", encoding="utf-8") as pf:
            param_data = json.load(pf)
        init_params = build_init_params(cfg, calibration=calib, parameter_json=param_data)
        if "delay_sec" in init_params:
            cfg.delay_init_ms = 1000.0 * float(init_params["delay_sec"])
    radius_measured = extract_radius_from_json(args.radius_json)
    if radius_measured is not None:
        cfg.radius_m = float(radius_measured)
        cfg.r_imu = float(radius_measured)

    log_csv = make_numbered_path(cfg.log_dir, cfg.log_prefix, ".csv")
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
    sim_params = {
        "l_com": float(cfg.l_com_init),
        "b_eq": float(cfg.b_eq_init),
        "tau_eq": float(cfg.tau_eq_init),
        "K_u": float(cfg.K_u_init),
    }
    model.update_identified_structure(sim_params)
    prev_u_eff = 0.0
    prev_du = 0.0

    viewer_proc = None
    if cfg.enable_imu_viewer:
        viewer_topic = "/imu/data"
        viewer_proc = start_imu_viewer_process(viewer_topic, cfg.topic_hw_enc)

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

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(PENDULUM_LOG_COLUMNS)

        wall_t0 = now_wall()
        omega_prev = model.get_omega()
        t_prev = 0.0
        enc_ref = None
        enc_prev = None
        theta_real_prev = None
        warmup_sec = 1.0
        imu_R0 = None
        real_omega_hist = deque(maxlen=401)
        real_time_hist = deque(maxlen=401)
        run_limit_sec = float("inf") if args.duration <= 0.0 else float(args.duration)

        if host_controller is not None:
            host_controller.__enter__()
            terminal_status_line("cmd_u:    0.0 | used:    0.0 | mode: manual | waiting for keyboard input", width=cfg.terminal_status_width)
            print()
        elif quit_watcher is not None:
            quit_watcher.__enter__()

        if calib is not None:
            print(f"[INFO] Loaded calibration json: {args.calibration_json}")
        if radius_measured is not None:
            print(f"[INFO] Loaded radius json: {args.radius_json}")
        if np.isfinite(cfg.cpr):
            print(f"[INFO] CPR from calibration json: {cfg.cpr:.3f} counts/rev")
        print(f"[INFO] Visual link length (--link-length): {cfg.link_L:.6f} m")
        print(f"[INFO] Computation radius (from radius-json): {cfg.radius_m:.6f} m")
        if math.isfinite(run_limit_sec):
            print(f"[INFO] run limit: {run_limit_sec:.1f}s")
        else:
            print("[INFO] run limit: none (quit with q/ESC)")
        if args.real_alpha_source != "omega_diff":
            print("[INFO] --real-alpha-source is deprecated and ignored; using filtered d(omega)/dt.")
        print("[INFO] real alpha source: filtered derivative of omega")

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

                if sim_t < warmup_sec:
                    if host_controller is not None:
                        ros_node.publish_host_cmd(0.0, "warmup")
                    terminal_status_line(
                        f"warmup... {sim_t:4.2f}/{warmup_sec:.2f}s | waiting for stable sensor baseline",
                        width=cfg.terminal_status_width,
                    )
                    if np.isfinite(snap["hw_enc"]):
                        enc_ref = float(snap["hw_enc"])
                        enc_prev = enc_ref
                    if cfg.realtime:
                        time.sleep(min(cfg.step, 0.01))
                    continue

                cmd_u_used = cmd_u_raw

                theta_before = model.get_theta()
                model_out = compute_model_torque_and_electrics(cmd_u_used, theta_before, model.get_omega(), float("nan"), sim_params, cfg)
                model.apply_torque(model_out["tau_net"])
                model.step(cfg.step)

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
                    enc_prev = enc_ref

                theta_real = theta
                omega_real = omega
                alpha_real = alpha
                if snap.get("imu_has_data", False):
                    q = snap["imu_q"]
                    w_imu_raw = snap["imu_w"]
                    R_abs = quat_to_rotmat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    if imu_R0 is None:
                        imu_R0 = R_abs.copy()
                    R_rel = imu_R0.T @ R_abs
                    tip_vec = R_rel @ np.array([0.0, -cfg.radius_m, 0.0], dtype=float)
                    # Physical/logging convention: CCW positive on world XY.
                    # (Viewer-only mirror transforms are handled in imu_viewer, not in logged data.)
                    theta_real = float(math.atan2(float(tip_vec[1]), float(tip_vec[0])))
                    omega_real = float(w_imu_raw[2])

                real_omega_hist.append(float(omega_real))
                real_time_hist.append(float(sim_t))
                alpha_real = float(
                    estimate_filtered_alpha_from_omega(
                        np.asarray(real_omega_hist, dtype=float),
                        t=np.asarray(real_time_hist, dtype=float),
                    )[-1]
                )

                theta_real_prev = theta_real
                e_theta = theta - theta_real
                e_omega = omega - omega_real
                e_alpha = alpha - alpha_real
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
                    mode_name=mode_name,
                    cfg=cfg,
                )
                terminal_status_line(status, width=cfg.terminal_status_width)
                ros_node.publish_sim(theta, omega, alpha, model_out["tau_net"], cmd_u_used, imu_msg, status)

                wr.writerow([
                    wall_now, wall_now - wall_t0, mode_name,
                    cmd_u_raw, cmd_u_used, snap["hw_pwm"], 0.0, model_out["tau_net"],
                    theta, omega, alpha,
                    snap["hw_enc"], snap["hw_arduino_ms"],
                    theta_real, omega_real, alpha_real,
                    0.0,
                    sim_params["l_com"], sim_params["b_eq"], sim_params["tau_eq"], sim_params["K_u"],
                    model.J_rod, model.J_imu, model.J_total,
                    model_out["tau_motor"], model_out["tau_res"], model_out["tau_visc"], model_out["tau_coul"],
                    inst_cost, inst_cost,
                    q_imu[0], q_imu[1], q_imu[2], q_imu[3],
                    w_imu[0], w_imu[1], w_imu[2],
                    a_imu[0], a_imu[1], a_imu[2],
                    "",
                    0,
                    0,
                    "",
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

    meta = {
        "log_csv": log_csv,
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
        "estimated_delay_ms_final": 0.0,
        "cpr_fixed": None if not np.isfinite(cfg.cpr) else float(cfg.cpr),
        "delay_locked": False,
        "best_eval": None,
        "ls_cost": None,
        "fit_done": False,
        "fit_complete": False,
        "fit_complete_wall": None,
        "fit_final_params": None,
    }
    with open(log_meta, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    print(f"saved csv  : {log_csv}")
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
