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


# ============================================================
# utility
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def now_wall():
    return time.time()


def terminal_status_line(msg: str, width: int | None = None):
    term_width = shutil.get_terminal_size((max(width or 120, 40), 24)).columns
    usable_width = max(20, min(width or term_width, term_width) - 1)
    sys.stdout.write("\r\033[2K" + msg[:usable_width].ljust(usable_width))
    sys.stdout.flush()


def sanitize_float(value, default=0.0, limit=np.finfo(np.float32).max * 0.99):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(clamp(value, -limit, limit))


def make_numbered_path(folder: str, prefix: str, ext: str = ".csv") -> str:
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    max_n = 0
    for name in os.listdir(folder):
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return os.path.join(folder, f"{prefix}{max_n + 1}{ext}")


def moving_average(x: np.ndarray, win: int):
    if win <= 1 or len(x) == 0:
        return x.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")
    return y[:len(x)]


def prbs_value(t: float, dt: float = 0.25, seed: int = 12345) -> float:
    if dt <= 1e-12:
        return 1.0
    k = int(t / dt)
    x = (1103515245 * (k + seed) + 12345) & 0x7FFFFFFF
    return 1.0 if (x & 1) else -1.0


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_to_np(q: ch.ChQuaterniond):
    return np.array([q.e0, q.e1, q.e2, q.e3], dtype=float)


# ============================================================
# config
# ============================================================

@dataclass
class BridgeConfig:
    step: float = 0.001
    gravity: float = 9.81

    enable_render: bool = True
    enable_imu_viewer: bool = True
    realtime: bool = True
    win_w: int = 1280
    win_h: int = 900
    window_title: str = "Chrono Pendulum | Online Calibration"
    terminal_status_width: int = 160

    motor_radius: float = 0.020
    motor_length: float = 0.050
    shaft_radius: float = 0.004
    shaft_length: float = 0.015

    link_L: float = 0.285
    radius_m: float = 0.285
    link_W: float = 0.020
    link_T: float = 0.006
    link_mass: float = 0.200
    imu_mass: float = 0.010

    imu_offset_x: float = 0.220
    imu_offset_y: float = 0.000
    imu_offset_z: float = 0.000
    imu_size_x: float = 0.0595
    imu_size_y: float = 0.0460
    imu_size_z: float = 0.0117

    theta0_deg: float = -10.0
    omega0: float = 0.0

    pwm_limit: float = 255.0
    pwm_step: float = 10.0

    # online model parameters (important: J included)
    J_init: float = 0.010
    b_init: float = 0.030
    tau_c_init: float = 0.080
    mgl_init: float = 0.550
    k_t_init: float = 0.250
    i0_init: float = 0.050
    R_init: float = 2.0
    k_e_init: float = 0.020
    tanh_eps: float = 0.05
    j_min: float = 1e-4

    # automatic delay compensation
    auto_delay_comp: bool = True
    delay_init_ms: float = 0.0
    delay_max_ms: float = 120.0
    delay_update_hz: float = 5.0
    delay_buffer_sec: float = 4.0
    delay_smooth_alpha: float = 0.15

    # online EKF-like fitting
    online_fit_enable: bool = True
    q_theta: float = 1e-5
    q_omega: float = 1e-3
    q_J: float = 1e-8
    q_b: float = 1e-7
    q_tauc: float = 1e-7
    q_mgl: float = 1e-7
    q_kt: float = 1e-7
    q_i0: float = 1e-7
    q_delay: float = 1e-6
    r_theta: float = 2e-4
    r_omega: float = 2e-3
    ekf_enable_min_pwm: float = 5.0
    J_max: float = 0.2
    b_max: float = 5.0
    tau_c_max: float = 2.0
    mgl_max: float = 5.0
    k_t_max: float = 5.0
    i0_max: float = 255.0

    # cost weights
    w_theta: float = 5.0
    w_omega: float = 2.5
    w_alpha: float = 0.7
    w_v: float = 0.3
    w_i: float = 0.5
    w_p: float = 0.2

    # control presets
    wave_freq: float = 0.5
    burst_period: float = 2.0
    burst_on_time: float = 0.3
    prbs_dt: float = 0.25
    prbs_seed: int = 12345

    # ros topics
    topic_cmd_u: str = "/cmd/u"
    topic_debug: str = "/cmd/keyboard_state"
    topic_hw_pwm: str = "/hw/pwm_applied"
    topic_hw_enc: str = "/hw/enc"
    topic_hw_arduino_ms: str = "/hw/arduino_ms"
    topic_bus_v: str = "/ina219/bus_voltage_v"
    topic_current_ma: str = "/ina219/current_ma"
    topic_power_mw: str = "/ina219/power_mw"

    topic_sim_theta: str = "/sim/theta"
    topic_sim_omega: str = "/sim/omega"
    topic_sim_alpha: str = "/sim/alpha"
    topic_sim_tau: str = "/sim/tau"
    topic_sim_cmd_used: str = "/sim/cmd_used"
    topic_sim_delay_ms: str = "/sim/delay_ms"
    topic_sim_status: str = "/sim/status"
    topic_imu: str = "/sim/imu/data"
    topic_est_theta: str = "/est/theta"
    topic_est_omega: str = "/est/omega"
    topic_est_alpha: str = "/est/alpha"
    topic_calib_status: str = "/calibration/status"
    calibration_json: str = "./run_logs/calibration_latest.json"

    history_sec: float = 20.0
    log_dir: str = "./run_logs"
    log_prefix: str = "chrono_run_"


# ============================================================
# keyboard controller
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
            self.cfg.pwm_step = max(0.5, self.cfg.pwm_step - 0.5)
        elif key == "]":
            self.cfg.pwm_step = min(100.0, self.cfg.pwm_step + 0.5)
        elif key == "-":
            self.cfg.pwm_limit = max(10.0, self.cfg.pwm_limit - 5.0)
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
            return self.cfg.pwm_limit * 0.60 * math.sin(2.0 * math.pi * self.cfg.wave_freq * t)
        if self.mode == "square":
            return self.cfg.pwm_limit * 0.60 * (1.0 if math.sin(2.0 * math.pi * self.cfg.wave_freq * t) >= 0.0 else -1.0)
        if self.mode == "burst":
            tau = t % self.cfg.burst_period
            return self.cfg.pwm_limit * 0.75 if tau < self.cfg.burst_on_time else 0.0
        if self.mode == "prbs":
            return self.cfg.pwm_limit * 0.45 * prbs_value(t, self.cfg.prbs_dt, self.cfg.prbs_seed)
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
        self.bus_v = float("nan")
        self.current_ma = float("nan")
        self.power_mw = float("nan")
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
                "bus_v": self.bus_v,
                "current_ma": self.current_ma,
                "power_mw": self.power_mw,
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
        self.create_subscription(Float32, cfg.topic_bus_v, self.cb_bus_v, 10)
        self.create_subscription(Float32, cfg.topic_current_ma, self.cb_current_ma, 10)
        self.create_subscription(Float32, cfg.topic_power_mw, self.cb_power_mw, 10)

        self.pub_theta = self.create_publisher(Float32, cfg.topic_sim_theta, 10)
        self.pub_omega = self.create_publisher(Float32, cfg.topic_sim_omega, 10)
        self.pub_alpha = self.create_publisher(Float32, cfg.topic_sim_alpha, 10)
        self.pub_tau = self.create_publisher(Float32, cfg.topic_sim_tau, 10)
        self.pub_cmd_used = self.create_publisher(Float32, cfg.topic_sim_cmd_used, 10)
        self.pub_delay_ms = self.create_publisher(Float32, cfg.topic_sim_delay_ms, 10)
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

    def cb_bus_v(self, msg: Float32):
        with self.shared.lock:
            self.shared.bus_v = float(msg.data)
            self.shared.last_ina_wall = now_wall()

    def cb_current_ma(self, msg: Float32):
        with self.shared.lock:
            self.shared.current_ma = float(msg.data)
            self.shared.last_ina_wall = now_wall()

    def cb_power_mw(self, msg: Float32):
        with self.shared.lock:
            self.shared.power_mw = float(msg.data)
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

    def publish_sim(self, theta, omega, alpha, tau, cmd_used, delay_ms, imu_msg, status_text):
        self._pub_float(self.pub_theta, theta)
        self._pub_float(self.pub_omega, omega)
        self._pub_float(self.pub_alpha, alpha)
        self._pub_float(self.pub_tau, tau)
        self._pub_float(self.pub_cmd_used, cmd_used)
        self._pub_float(self.pub_delay_ms, delay_ms)
        self.pub_imu.publish(imu_msg)
        s = String()
        s.data = status_text
        self.pub_status.publish(s)


# ============================================================
# delay compensator and CPR estimator
# ============================================================

class DelayCompensator:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.delay_sec = cfg.delay_init_ms / 1000.0
        self.cmd_hist = deque()
        self.pwm_hist = deque()
        self.last_update_wall = 0.0

    def push(self, wall_t: float, cmd_u: float, hw_pwm: float):
        self.cmd_hist.append((wall_t, float(cmd_u)))
        self.pwm_hist.append((wall_t, float(hw_pwm)))
        tmin = wall_t - self.cfg.delay_buffer_sec
        while self.cmd_hist and self.cmd_hist[0][0] < tmin:
            self.cmd_hist.popleft()
        while self.pwm_hist and self.pwm_hist[0][0] < tmin:
            self.pwm_hist.popleft()

    def _interp(self, hist, t):
        xs = list(hist)
        if not xs:
            return 0.0
        if t <= xs[0][0]:
            return xs[0][1]
        if t >= xs[-1][0]:
            return xs[-1][1]
        for i in range(1, len(xs)):
            t0, y0 = xs[i - 1]
            t1, y1 = xs[i]
            if t0 <= t <= t1:
                if abs(t1 - t0) < 1e-12:
                    return y0
                a = (t - t0) / (t1 - t0)
                return (1.0 - a) * y0 + a * y1
        return xs[-1][1]

    def estimate_delay(self, wall_now: float):
        if not self.cfg.auto_delay_comp:
            return self.delay_sec
        if wall_now - self.last_update_wall < 1.0 / max(self.cfg.delay_update_hz, 1e-9):
            return self.delay_sec
        self.last_update_wall = wall_now
        if len(self.cmd_hist) < 20 or len(self.pwm_hist) < 20:
            return self.delay_sec
        t0 = max(self.cmd_hist[0][0], self.pwm_hist[0][0])
        t1 = min(self.cmd_hist[-1][0], self.pwm_hist[-1][0])
        if (t1 - t0) < 0.4:
            return self.delay_sec
        dt = 0.01
        ts = np.arange(t0, t1, dt)
        if len(ts) < 20:
            return self.delay_sec
        cmd = np.array([self._interp(self.cmd_hist, t) for t in ts], dtype=float)
        pwm = np.array([self._interp(self.pwm_hist, t) for t in ts], dtype=float)
        cmd -= np.mean(cmd)
        pwm -= np.mean(pwm)
        if np.std(cmd) < 1e-6 or np.std(pwm) < 1e-6:
            return self.delay_sec
        max_lag = int((self.cfg.delay_max_ms / 1000.0) / dt)
        best_lag = 0
        best_score = -1e18
        for lag in range(max_lag + 1):
            if lag >= len(cmd) - 2:
                break
            c = cmd[:-lag] if lag > 0 else cmd
            p = pwm[lag:] if lag > 0 else pwm
            score = float(np.dot(c, p)) / max(len(c), 1)
            if score > best_score:
                best_score = score
                best_lag = lag
        measured = best_lag * dt
        self.delay_sec = (1.0 - self.cfg.delay_smooth_alpha) * self.delay_sec + self.cfg.delay_smooth_alpha * measured
        return self.delay_sec

    def get_delayed_cmd(self, wall_now: float, fallback: float):
        target = wall_now - self.delay_sec
        if len(self.cmd_hist) < 2:
            return fallback
        return self._interp(self.cmd_hist, target)


class CPREstimator:
    def __init__(self):
        self.prev_angle = None
        self.angle_unwrapped = 0.0
        self.angle_travel = 0.0
        self.rev_index = 0
        self.rev_enc_anchor = None
        self.rev_angle_anchor = 0.0
        self.samples = []
        self.last_cpr = np.nan
        self.motion_started = False

    def reset_revolution_window(self, enc_count):
        self.rev_enc_anchor = enc_count
        self.rev_angle_anchor = self.angle_unwrapped
        self.last_cpr = np.nan
        self.motion_started = False

    def update(self, angle_wrapped, enc_count):
        if self.prev_angle is None:
            self.prev_angle = angle_wrapped
            self.reset_revolution_window(enc_count)
            return
        d = angle_wrapped - self.prev_angle
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        self.angle_unwrapped += d
        self.angle_travel += abs(d)
        self.prev_angle = angle_wrapped
        theta_window = self.angle_unwrapped - self.rev_angle_anchor
        if not self.motion_started:
            if abs(theta_window) < math.radians(5.0):
                self.rev_enc_anchor = enc_count
            else:
                self.motion_started = True
        while abs(theta_window) >= (2.0 * math.pi):
            if self.rev_enc_anchor is not None:
                delta = abs(enc_count - self.rev_enc_anchor)
                if delta > 1:
                    self.samples.append(float(delta))
                    self.last_cpr = float(delta)
            self.rev_enc_anchor = enc_count
            self.rev_index += 1
            self.rev_angle_anchor += math.copysign(2.0 * math.pi, theta_window)
            theta_window = self.angle_unwrapped - self.rev_angle_anchor

    @property
    def mean(self):
        if len(self.samples) == 0:
            return np.nan
        return float(np.mean(self.samples))


# ============================================================
# online fitting
# ============================================================

class OnlineParameterEKF:
    """
    augmented state:
    x = [theta, omega, J, b, tau_c, mgl, k_t, i0, delay_sec]
    """
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.x = np.array([
            0.0, 0.0,
            cfg.J_init,
            cfg.b_init,
            cfg.tau_c_init,
            cfg.mgl_init,
            cfg.k_t_init,
            cfg.i0_init,
            cfg.delay_init_ms / 1000.0,
        ], dtype=float)
        self.P = np.diag([1e-3, 1e-2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4])
        self.Q = np.diag([
            cfg.q_theta, cfg.q_omega, cfg.q_J, cfg.q_b, cfg.q_tauc,
            cfg.q_mgl, cfg.q_kt, cfg.q_i0, cfg.q_delay
        ])
        self.R = np.diag([cfg.r_theta, cfg.r_omega])
        self.best_cost = np.inf
        self.best_params = None

    def f_disc(self, x, u, dt):
        th, om, J, b, tau_c, mgl, kt, i0, dly = x
        J = max(J, self.cfg.j_min)
        i_eff = math.copysign(max(abs(u) - i0, 0.0), u)
        alpha = (kt * i_eff - b * om - tau_c * math.tanh(om / max(self.cfg.tanh_eps, 1e-9)) - mgl * math.sin(th)) / J
        xn = np.array([
            th + dt * om,
            om + dt * alpha,
            J,
            max(b, 0.0),
            max(tau_c, 0.0),
            max(mgl, 0.0),
            max(kt, 0.0),
            max(i0, 0.0),
            max(dly, 0.0),
        ], dtype=float)
        xn[2] = max(xn[2], self.cfg.j_min)
        return xn

    def h_meas(self, x):
        return np.array([x[0], x[1]], dtype=float)

    def numeric_jacobian(self, fun, x, eps=1e-6):
        x = np.asarray(x, dtype=float)
        y0 = np.asarray(fun(x), dtype=float)
        m, n = len(y0), len(x)
        J = np.zeros((m, n), dtype=float)
        for i in range(n):
            dx = np.zeros(n, dtype=float)
            step = eps * max(1.0, abs(x[i]))
            dx[i] = step
            y1 = np.asarray(fun(x + dx), dtype=float)
            y2 = np.asarray(fun(x - dx), dtype=float)
            J[:, i] = (y1 - y2) / (2.0 * step)
        return J

    def update(self, theta_meas, omega_meas, u_eff, dt, inst_cost=None):
        f_local = lambda xx: self.f_disc(xx, u_eff, dt)
        F = self.numeric_jacobian(f_local, self.x)
        x_pred = f_local(self.x)
        P_pred = F @ self.P @ F.T + self.Q
        H = self.numeric_jacobian(self.h_meas, x_pred)
        z_pred = self.h_meas(x_pred)
        z = np.array([theta_meas, omega_meas], dtype=float)
        y = z - z_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ P_pred
        self._clip_state()
        if inst_cost is not None and inst_cost < self.best_cost:
            self.best_cost = float(inst_cost)
            self.best_params = self.get_params()

    def _clip_state(self):
        self.x[2] = clamp(self.x[2], self.cfg.j_min, self.cfg.J_max)
        self.x[3] = clamp(self.x[3], 0.0, self.cfg.b_max)
        self.x[4] = clamp(self.x[4], 0.0, self.cfg.tau_c_max)
        self.x[5] = clamp(self.x[5], 0.0, self.cfg.mgl_max)
        self.x[6] = clamp(self.x[6], 0.0, self.cfg.k_t_max)
        self.x[7] = clamp(self.x[7], 0.0, self.cfg.i0_max)
        self.x[8] = clamp(self.x[8], 0.0, self.cfg.delay_max_ms / 1000.0)

    def get_params(self):
        return {
            "theta": float(self.x[0]),
            "omega": float(self.x[1]),
            "J": float(self.x[2]),
            "b": float(self.x[3]),
            "tau_c": float(self.x[4]),
            "mgl": float(self.x[5]),
            "k_t": float(self.x[6]),
            "i0": float(self.x[7]),
            "delay_sec": float(self.x[8]),
            "R": float(self.cfg.R_init),
            "k_e": float(self.cfg.k_e_init),
        }


# ============================================================
# Chrono model
# ============================================================

def add_axes_visual(sys_ch, axis_len=0.12, axis_thk=0.002):
    axes = ch.ChBody()
    axes.SetFixed(True)
    sys_ch.Add(axes)

    x_box = ch.ChVisualShapeBox(axis_len, axis_thk, axis_thk)
    x_box.SetColor(ch.ChColor(1, 0, 0))
    axes.AddVisualShape(x_box, ch.ChFramed(ch.ChVector3d(axis_len / 2.0, 0.0, 0.0), ch.QUNIT))

    y_box = ch.ChVisualShapeBox(axis_thk, axis_len, axis_thk)
    y_box.SetColor(ch.ChColor(0, 1, 0))
    axes.AddVisualShape(y_box, ch.ChFramed(ch.ChVector3d(0.0, axis_len / 2.0, 0.0), ch.QUNIT))

    z_box = ch.ChVisualShapeBox(axis_thk, axis_thk, axis_len)
    z_box.SetColor(ch.ChColor(0, 0, 1))
    axes.AddVisualShape(z_box, ch.ChFramed(ch.ChVector3d(0.0, 0.0, axis_len / 2.0), ch.QUNIT))


class PendulumModel:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.sys = ch.ChSystemNSC()
        self.sys.SetGravitationalAcceleration(ch.ChVector3d(0.0, -cfg.gravity, 0.0))
        add_axes_visual(self.sys)

        self.base = ch.ChBody()
        self.base.SetFixed(True)
        self.base.SetPos(ch.ChVector3d(0.0, 0.0, 0.0))
        self.sys.Add(self.base)

        q_cyl_to_x = ch.QuatFromAngleZ(-math.pi / 2.0)
        motor_cyl = ch.ChVisualShapeCylinder(cfg.motor_radius, cfg.motor_length)
        motor_cyl.SetColor(ch.ChColor(0.05, 0.05, 0.05))
        self.base.AddVisualShape(motor_cyl, ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), q_cyl_to_x))

        shaft = ch.ChVisualShapeCylinder(cfg.shaft_radius, cfg.shaft_length)
        shaft.SetColor(ch.ChColor(0.05, 0.05, 0.05))
        self.base.AddVisualShape(
            shaft,
            ch.ChFramed(
                ch.ChVector3d(cfg.motor_length / 2.0 + cfg.shaft_length / 2.0, 0.0, 0.0),
                q_cyl_to_x,
            ),
        )

        self.link = ch.ChBody()
        self.link.SetMass(cfg.link_mass)
        izz_com = (1.0 / 12.0) * cfg.link_mass * (cfg.link_L ** 2 + cfg.link_W ** 2)
        self.link.SetInertiaXX(ch.ChVector3d(1e-5, 1e-5, izz_com))
        # Body reference frame is at motor pivot; COM is at link center.
        com_frame = ch.ChFramed(ch.ChVector3d(0.0, -cfg.link_L / 2.0, 0.0), ch.QUNIT)
        self.link.SetFrameCOMToRef(com_frame)
        self.link.SetPos(ch.ChVector3d(0.0, 0.0, cfg.motor_length / 2.0))
        self.link.SetRot(ch.QuatFromAngleZ(math.radians(cfg.theta0_deg)))
        self.sys.Add(self.link)
        self.link.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, cfg.omega0))

        vis_link = ch.ChVisualShapeBox(cfg.link_W, cfg.link_L, cfg.link_T)
        vis_link.SetColor(ch.ChColor(0.93, 0.93, 0.93))
        self.link.AddVisualShape(vis_link, ch.ChFramed(ch.ChVector3d(0.0, -cfg.link_L / 2.0, 0.0), ch.QUNIT))

        self.imu = ch.ChBody()
        self.imu.SetMass(cfg.imu_mass)
        self.imu.SetInertiaXX(ch.ChVector3d(1e-6, 1e-6, 1e-6))
        imu_local = ch.ChVector3d(0.0, -cfg.link_L + cfg.imu_size_y / 2.0, 0.0)
        imu_abs = self.link.TransformPointLocalToParent(imu_local)
        self.imu.SetPos(imu_abs)
        self.imu.SetRot(self.link.GetRot())
        self.sys.Add(self.imu)

        vis_imu = ch.ChVisualShapeBox(cfg.imu_size_x, cfg.imu_size_y, cfg.imu_size_z)
        vis_imu.SetColor(ch.ChColor(0.60, 0.60, 0.60))
        self.imu.AddVisualShape(vis_imu)

        fix_frame_abs = ch.ChFramed(imu_abs, self.link.GetRot())
        self.fix_imu = ch.ChLinkLockLock()
        self.fix_imu.Initialize(self.imu, self.link, fix_frame_abs)
        self.sys.Add(self.fix_imu)

        self.motor = ch.ChLinkMotorRotationTorque()
        self.motor.Initialize(self.link, self.base, ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), ch.QUNIT))
        self.tau_fun = ch.ChFunctionConst(0.0)
        self.motor.SetTorqueFunction(self.tau_fun)
        self.sys.Add(self.motor)

        self.prev_sensor_vel = np.zeros(3, dtype=float)
        self.prev_t = None

    def get_theta(self):
        d = self.link.TransformDirectionLocalToParent(ch.ChVector3d(0.0, -1.0, 0.0))
        return math.atan2(float(d.x), -float(d.y))

    def get_omega(self):
        return float(self.link.GetAngVelLocal().z)

    def get_sensor_kinematics(self, cur_t, step):
        pos_w = self.imu.GetPos()
        vel_w = self.imu.GetPosDt()
        p = np.array([float(pos_w.x), float(pos_w.y), float(pos_w.z)], dtype=float)
        v = np.array([float(vel_w.x), float(vel_w.y), float(vel_w.z)], dtype=float)
        if self.prev_t is None:
            a = np.zeros(3, dtype=float)
        else:
            dt = max(cur_t - self.prev_t, step)
            a = (v - self.prev_sensor_vel) / dt
        self.prev_sensor_vel = v.copy()
        self.prev_t = cur_t
        q = quat_to_np(self.imu.GetRot())
        omega = np.array([0.0, 0.0, self.get_omega()], dtype=float)
        return p, v, a, q, omega

    def apply_torque(self, tau_z):
        self.tau_fun.SetConstant(sanitize_float(tau_z))

    def step(self, h):
        self.sys.DoStepDynamics(h)


# ============================================================
# electrical/mechanical model and IMU viewer
# ============================================================

def compute_model_torque_and_electrics(cmd_u, omega, bus_v, p, cfg: BridgeConfig):
    u_eff = math.copysign(max(abs(cmd_u) - p["i0"], 0.0), cmd_u)
    duty = clamp(u_eff / max(cfg.pwm_limit, 1e-9), -1.0, 1.0)
    v_applied = duty * bus_v
    i_pred = (v_applied - p["k_e"] * omega) / max(p["R"], 1e-6)
    tau_motor = p["k_t"] * i_pred
    tau_visc = p["b"] * omega
    tau_coul = p["tau_c"] * math.tanh(omega / max(cfg.tanh_eps, 1e-9))
    tau_net = tau_motor - tau_visc - tau_coul
    p_pred = v_applied * i_pred
    return {
        "u_eff": u_eff,
        "v_pred": v_applied,
        "i_pred": i_pred,
        "p_pred": p_pred,
        "tau_motor": tau_motor,
        "tau_visc": tau_visc,
        "tau_coul": tau_coul,
        "tau_net": tau_net,
    }


def blend_parameters_for_sim(ekf_params: dict, cfg: BridgeConfig):
    return {
        "theta": float(ekf_params["theta"]),
        "omega": float(ekf_params["omega"]),
        "J": float(cfg.J_init),
        "b": float(cfg.b_init),
        "tau_c": float(cfg.tau_c_init),
        "mgl": float(cfg.mgl_init),
        "k_t": float(cfg.k_t_init),
        "i0": float(cfg.i0_init),
        "delay_sec": float(ekf_params["delay_sec"]),
        "R": float(cfg.R_init),
        "k_e": float(cfg.k_e_init),
    }


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


def make_status_line(host_mode: bool, cmd_u_raw: float, cmd_u_used: float, hw_pwm: float, mode_name: str,
                     cfg: BridgeConfig, delay_ms: float, fit_params: dict):
    if host_mode:
        return (
            f"cmd_u: {cmd_u_raw:6.1f} | used: {cmd_u_used:6.1f} | mode: {mode_name:<6} | "
            f"step: {cfg.pwm_step:4.1f} | max: {cfg.pwm_limit:5.1f} | delay: {delay_ms:5.1f} ms | "
            f"J: {fit_params['J']:.5f} | b: {fit_params['b']:.4f} | tc: {fit_params['tau_c']:.4f}"
        )
    return (
        f"cmd_u: {cmd_u_used:6.1f} | hw_pwm: {hw_pwm:6.1f} | mode: external | "
        f"delay: {delay_ms:5.1f} ms"
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


def apply_calibration_json(cfg: BridgeConfig, json_path: str | None):
    if not json_path or not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    model_init = calib.get("model_init", {})
    if not model_init and "best_params" in calib:
        # RL_fitting result schema compatibility
        model_init = calib.get("best_params", {})
        if "Rm" in model_init and "R" not in model_init:
            model_init["R"] = model_init["Rm"]
    delay = calib.get("delay", {})

    cfg.J_init = float(model_init.get("J", cfg.J_init))
    cfg.b_init = float(model_init.get("b", cfg.b_init))
    cfg.tau_c_init = float(model_init.get("tau_c", cfg.tau_c_init))
    cfg.mgl_init = float(model_init.get("mgl", cfg.mgl_init))
    cfg.k_t_init = float(model_init.get("k_t", cfg.k_t_init))
    cfg.i0_init = float(model_init.get("i0", cfg.i0_init))
    cfg.R_init = float(model_init.get("R", cfg.R_init))
    cfg.k_e_init = float(model_init.get("k_e", cfg.k_e_init))
    cfg.delay_init_ms = float(delay.get("effective_control_delay_ms", cfg.delay_init_ms))

    cfg.calibration_json = json_path
    return calib


def extract_radius_from_json(json_path: str | None) -> float | None:
    if not json_path or not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {}) if isinstance(data.get("summary", {}), dict) else {}
    radius_candidates = [
        summary.get("mean_radius_m"),
        summary.get("r_m"),
        data.get("mean_radius_m"),
        data.get("r_from_imu_orientation"),
    ]
    for radius in radius_candidates:
        if radius is None:
            continue
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            continue
        if math.isfinite(radius) and radius > 0.0:
            return radius
    return None


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-imu-viewer", action="store_true")
    ap.add_argument("--duration", type=float, default=20.0)
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
    ap.add_argument("--disable-auto-delay", action="store_true")
    ap.add_argument("--enable-online-fit", action="store_true")
    ap.add_argument("--calibration-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--radius-json", default="./run_logs/calibration_latest.json",
                    help="JSON file containing measured radius (e.g., calibration_latest.json).")
    ap.add_argument("--J", type=float, default=0.010)
    ap.add_argument("--b", type=float, default=0.030)
    ap.add_argument("--tau-c", type=float, default=0.080)
    ap.add_argument("--mgl", type=float, default=0.550)
    ap.add_argument("--k-t", type=float, default=0.250)
    ap.add_argument("--i0", type=float, default=0.050)
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--k-e", type=float, default=0.020)
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
    cfg.link_mass = args.link_mass
    cfg.link_L = args.link_length
    cfg.radius_m = args.link_length
    cfg.J_init = args.J
    cfg.b_init = args.b
    cfg.tau_c_init = args.tau_c
    cfg.mgl_init = args.mgl
    cfg.k_t_init = args.k_t
    cfg.i0_init = args.i0
    cfg.R_init = args.R
    cfg.k_e_init = args.k_e
    cfg.delay_init_ms = args.delay_ms
    cfg.auto_delay_comp = not args.disable_auto_delay
    cfg.online_fit_enable = args.enable_online_fit

    calib = apply_calibration_json(cfg, args.calibration_json)
    radius_measured = extract_radius_from_json(args.radius_json)
    if radius_measured is not None:
        cfg.radius_m = float(radius_measured)

    log_csv = make_numbered_path(cfg.log_dir, cfg.log_prefix, ".csv")
    log_meta = log_csv[:-4] + ".meta.json"

    rclpy.init()
    shared = SharedROSState()
    ros_node = PendulumROSNode(cfg, shared, host_mode=args.host_control)
    ros_stop = threading.Event()
    ros_thr = threading.Thread(target=ros_spin_thread, args=(ros_node, ros_stop), daemon=True)
    ros_thr.start()

    model = PendulumModel(cfg)
    delay_comp = DelayCompensator(cfg)
    cpr_est = CPREstimator()
    ekf = OnlineParameterEKF(cfg)
    best_eval = {"cost": float("inf"), "time": None, "params": None}

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
        wr.writerow([
            "wall_time", "sim_time", "mode",
            "cmd_u_raw", "cmd_u_used", "tau_cmd",
            "theta", "omega", "alpha",
            "hw_pwm", "hw_enc", "hw_arduino_ms",
            "bus_v", "current_A", "power_W",
            "v_pred", "i_pred", "p_pred",
            "delay_ms",
            "J_est", "b_est", "tau_c_est", "mgl_est", "k_t_est", "i0_est",
            "inst_cost", "best_cost_so_far",
            "imu_qw", "imu_qx", "imu_qy", "imu_qz",
            "imu_wx", "imu_wy", "imu_wz",
            "imu_ax", "imu_ay", "imu_az",
            "cpr_last", "cpr_mean"
        ])

        wall_t0 = now_wall()
        alpha_prev = 0.0
        omega_prev = model.get_omega()
        t_prev = 0.0

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
        print(f"[INFO] Visual link length (--link-length): {cfg.link_L:.6f} m")
        print(f"[INFO] Computation radius (from radius-json): {cfg.radius_m:.6f} m")
        print(f"[INFO] online_fit_enable={cfg.online_fit_enable}")

        try:
            while model.sys.GetChTime() < args.duration:
                if host_controller is not None:
                    host_controller.poll()
                    if host_controller.quit_requested:
                        break
                elif quit_watcher is not None:
                    quit_watcher.poll()
                    if quit_watcher.quit_requested:
                        break

                if vis is not None:
                    if not vis.Run():
                        break
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
                delay_comp.push(wall_now, cmd_u_raw, snap["hw_pwm"])
                delay_comp.estimate_delay(wall_now)
                cmd_u_used = delay_comp.get_delayed_cmd(wall_now, cmd_u_raw)

                fit_params = ekf.get_params()
                sim_params = blend_parameters_for_sim(fit_params, cfg)
                bus_v = snap["bus_v"] if np.isfinite(snap["bus_v"]) else 7.4
                current_A = snap["current_ma"] / 1000.0 if np.isfinite(snap["current_ma"]) else 0.0
                power_W = snap["power_mw"] / 1000.0 if np.isfinite(snap["power_mw"]) else 0.0

                model_out = compute_model_torque_and_electrics(cmd_u_used, model.get_omega(), bus_v, sim_params, cfg)
                model.apply_torque(model_out["tau_net"])
                model.step(cfg.step)

                sim_t = model.sys.GetChTime()
                theta = model.get_theta()
                omega = model.get_omega()
                alpha = (omega - omega_prev) / max(sim_t - t_prev, cfg.step) if sim_t > 0 else 0.0
                alpha = 0.8 * alpha_prev + 0.2 * alpha
                alpha = sanitize_float(alpha)
                alpha_prev = alpha
                omega_prev = omega
                t_prev = sim_t

                _, _, a_imu, q_imu, w_imu = model.get_sensor_kinematics(sim_t, cfg.step)
                imu_msg = build_imu_msg(sim_t, q_imu, w_imu, a_imu)

                cpr_est.update(theta, snap["hw_enc"])

                theta_real = theta
                omega_real = omega
                alpha_real = alpha
                e_theta = theta - theta_real
                e_omega = omega - omega_real
                e_alpha = alpha - alpha_real
                e_v = model_out["v_pred"] - bus_v
                e_i = model_out["i_pred"] - current_A
                e_p = model_out["p_pred"] - power_W
                inst_cost = (
                    cfg.w_theta * (e_theta ** 2) +
                    cfg.w_omega * (e_omega ** 2) +
                    cfg.w_alpha * (e_alpha ** 2) +
                    cfg.w_v * (e_v ** 2) +
                    cfg.w_i * (e_i ** 2) +
                    cfg.w_p * (e_p ** 2)
                )

                if cfg.online_fit_enable and abs(cmd_u_used) >= cfg.ekf_enable_min_pwm:
                    ekf.update(theta_real, omega_real, model_out["u_eff"], cfg.step, inst_cost=inst_cost)

                if inst_cost < best_eval["cost"]:
                    best_eval["cost"] = float(inst_cost)
                    best_eval["time"] = float(sim_t)
                    best_eval["params"] = ekf.get_params().copy()

                fit_params = ekf.get_params()
                status = make_status_line(
                    host_mode=(host_controller is not None),
                    cmd_u_raw=cmd_u_raw,
                    cmd_u_used=cmd_u_used,
                    hw_pwm=snap["hw_pwm"],
                    mode_name=mode_name,
                    cfg=cfg,
                    delay_ms=1000.0 * delay_comp.delay_sec,
                    fit_params=fit_params,
                )
                terminal_status_line(status, width=cfg.terminal_status_width)
                ros_node.publish_sim(theta, omega, alpha, model_out["tau_net"], cmd_u_used, 1000.0 * delay_comp.delay_sec, imu_msg, status)

                wr.writerow([
                    wall_now, sim_t, mode_name,
                    cmd_u_raw, cmd_u_used, model_out["tau_net"],
                    theta, omega, alpha,
                    snap["hw_pwm"], snap["hw_enc"], snap["hw_arduino_ms"],
                    bus_v, current_A, power_W,
                    model_out["v_pred"], model_out["i_pred"], model_out["p_pred"],
                    1000.0 * delay_comp.delay_sec,
                    fit_params["J"], fit_params["b"], fit_params["tau_c"], fit_params["mgl"], fit_params["k_t"], fit_params["i0"],
                    inst_cost, best_eval["cost"],
                    q_imu[0], q_imu[1], q_imu[2], q_imu[3],
                    w_imu[0], w_imu[1], w_imu[2],
                    a_imu[0], a_imu[1], a_imu[2],
                    cpr_est.last_cpr if np.isfinite(cpr_est.last_cpr) else "",
                    cpr_est.mean if np.isfinite(cpr_est.mean) else "",
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

    print()
    print("=== best online calibration point ===")
    print(f"time      : {best_eval['time']}")
    print(f"best cost : {best_eval['cost']:.6e}")
    print(json.dumps(best_eval["params"], indent=2))

    meta = {
        "log_csv": log_csv,
        "config": asdict(cfg),
        "calibration_json": cfg.calibration_json if calib is not None else None,
        "radius_json": args.radius_json,
        "estimated_delay_ms_final": 1000.0 * delay_comp.delay_sec,
        "cpr_last": None if not np.isfinite(cpr_est.last_cpr) else float(cpr_est.last_cpr),
        "cpr_mean": None if not np.isfinite(cpr_est.mean) else float(cpr_est.mean),
        "best_eval": best_eval,
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
        ros_node.destroy_node()
    except Exception:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
