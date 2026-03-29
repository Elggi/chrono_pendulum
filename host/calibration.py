#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""수동 회전 기반 캘리브레이션 도구.

- CPR: IMU/엔코더를 구독하며 `imu_viewer.py`의 SharedState 로직을 재사용해 자동 계산
- r  : orientation 기반 tip 좌표에서 실시간 반지름 추정
"""

import argparse
import json
import math
import os
import select
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from statistics import mean
from collections import deque

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32, String

from imu_viewer import SharedState, ViewerNode


def terminal_status_line(msg: str, width: int = 140):
    term_width = shutil.get_terminal_size((max(width, 80), 24)).columns
    usable_width = max(20, min(width, term_width) - 1)
    sys.stdout.write("\r\033[2K" + msg[:usable_width].ljust(usable_width))
    sys.stdout.flush()


class KeyboardReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
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


@dataclass
class ControllerConfig:
    topic_cmd_u: str = "/cmd/u"
    topic_debug: str = "/cmd/keyboard_state"
    loop_hz: float = 20.0
    pwm_step: float = 10.0
    pwm_max: float = 255.0


@dataclass
class DelayConfig:
    topic_cmd_u: str = "/cmd/u"
    topic_hw_pwm: str = "/hw/pwm_applied"
    max_delay_ms: float = 150.0
    update_hz: float = 4.0
    buffer_sec: float = 4.0
    smooth_alpha: float = 0.2
    lock_std_ms: float = 2.0
    lock_hold_updates: int = 12


class CalibrationKeyboardControllerNode(Node):
    def __init__(self, cfg: ControllerConfig):
        super().__init__("calibration_keyboard_controller")
        self.cfg = cfg
        self.pub_cmd = self.create_publisher(Float32, cfg.topic_cmd_u, 10)
        self.pub_debug = self.create_publisher(String, cfg.topic_debug, 10)

        self.current_u = 0.0
        self.preset_mode = "manual"
        self.preset_t0 = time.time()
        self.last_pub_time = 0.0
        self.sin_amp = 60.0
        self.sin_freq = 0.5
        self.square_amp = 60.0
        self.square_freq = 0.5
        self.burst_amp = 60.0
        self.burst_period = 2.0
        self.burst_on_time = 0.30
        self.prbs_amp = 60.0
        self.prbs_dt = 0.25

    def clamp_u(self):
        self.current_u = max(-self.cfg.pwm_max, min(self.cfg.pwm_max, self.current_u))

    def set_manual_mode(self):
        self.preset_mode = "manual"

    def set_preset_mode(self, mode: str):
        self.preset_mode = mode
        self.preset_t0 = time.time()

    def prbs_value(self, t, dt=0.25, seed=12345):
        if dt <= 1e-9:
            return 1.0
        k = int(t / dt)
        x = (1103515245 * (k + seed) + 12345) & 0x7FFFFFFF
        return 1.0 if (x & 1) else -1.0

    def update_auto_signal(self):
        if self.preset_mode == "manual":
            return
        t = time.time() - self.preset_t0
        if self.preset_mode == "sin":
            self.current_u = self.sin_amp * math.sin(2.0 * math.pi * self.sin_freq * t)
        elif self.preset_mode == "square":
            self.current_u = self.square_amp if math.sin(2.0 * math.pi * self.square_freq * t) >= 0.0 else -self.square_amp
        elif self.preset_mode == "burst":
            phase = t % self.burst_period
            self.current_u = self.burst_amp if phase < self.burst_on_time else 0.0
        elif self.preset_mode == "prbs":
            self.current_u = self.prbs_amp * self.prbs_value(t, self.prbs_dt)
        self.clamp_u()

    def publish_state(self, key_name=""):
        msg = Float32()
        msg.data = float(self.current_u)
        self.pub_cmd.publish(msg)
        dbg = String()
        dbg.data = f"key={key_name}, mode={self.preset_mode}, cmd_u={self.current_u:.1f}"
        self.pub_debug.publish(dbg)

    def apply_key(self, key):
        if key is None:
            return False
        changed = False
        if key in ("w", "W", "UP"):
            self.set_manual_mode(); self.current_u += self.cfg.pwm_step; changed = True
        elif key in ("s", "S", "DOWN"):
            self.set_manual_mode(); self.current_u -= self.cfg.pwm_step; changed = True
        elif key in ("d", "D", "RIGHT"):
            self.set_manual_mode(); self.current_u += self.cfg.pwm_step * 0.5; changed = True
        elif key in ("a", "A", "LEFT"):
            self.set_manual_mode(); self.current_u -= self.cfg.pwm_step * 0.5; changed = True
        elif key == " ":
            self.set_manual_mode(); self.current_u = 0.0; changed = True
        elif key == "1":
            self.set_manual_mode(); self.current_u = 60.0; changed = True
        elif key == "2":
            self.set_manual_mode(); self.current_u = -60.0; changed = True
        elif key == "3":
            self.set_manual_mode(); self.current_u = 120.0; changed = True
        elif key == "4":
            self.set_manual_mode(); self.current_u = -120.0; changed = True
        elif key == "5":
            self.set_preset_mode("sin"); changed = True
        elif key == "6":
            self.set_preset_mode("square"); changed = True
        elif key == "7":
            self.set_preset_mode("burst"); changed = True
        elif key == "8":
            self.set_preset_mode("prbs"); changed = True
        elif key == "[":
            self.cfg.pwm_step = max(1.0, self.cfg.pwm_step - 1.0)
        elif key == "]":
            self.cfg.pwm_step = min(100.0, self.cfg.pwm_step + 1.0)
        elif key == "-":
            self.cfg.pwm_max = max(20.0, self.cfg.pwm_max - 5.0); self.clamp_u(); changed = True
        elif key == "=":
            self.cfg.pwm_max = min(255.0, self.cfg.pwm_max + 5.0); self.clamp_u(); changed = True
        elif key in ("x", "X"):
            self.set_manual_mode(); self.current_u = 0.0; changed = True
        self.clamp_u()
        return changed


def print_help():
    print("\n" + "=" * 70)
    print("Calibration Keyboard Controller")
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
    print("  c                 : reset calibration sampling window")
    print("  q                 : finish calibration and compute (after delay lock 확인 권장)")
    print("=" * 70)
    print("Output topic:")
    print("  /cmd/u  (std_msgs/Float32, signed PWM)")
    print("=" * 70 + "\n")


def print_status_line(node: CalibrationKeyboardControllerNode, full_rot: int, cpr_samples_count: int):
    msg = (
        f"cmd_u: {node.current_u:6.1f} | step: {node.cfg.pwm_step:4.1f} | "
        f"max: {node.cfg.pwm_max:5.1f} | mode: {node.preset_mode:<6} | "
        f"rot: {full_rot:4d} | cpr_samples: {cpr_samples_count:4d}"
    )
    terminal_status_line(msg)


def print_status_line_with_delay(node: CalibrationKeyboardControllerNode, full_rot: int, cpr_samples_count: int,
                                 delay_ms: float, delay_locked: bool):
    lock_label = "LOCK" if delay_locked else "RUN"
    msg = (
        f"cmd_u: {node.current_u:6.1f} | step: {node.cfg.pwm_step:4.1f} | "
        f"max: {node.cfg.pwm_max:5.1f} | mode: {node.preset_mode:<6} | "
        f"rot: {full_rot:4d} | cpr_samples: {cpr_samples_count:4d} | "
        f"delay:{delay_ms:5.1f}ms({lock_label})"
    )
    terminal_status_line(msg)


class DelayLockEstimator:
    def __init__(self, cfg: DelayConfig):
        self.cfg = cfg
        self.cmd_hist = deque()
        self.pwm_hist = deque()
        self.last_update_wall = 0.0
        self.delay_sec = 0.0
        self.delay_locked = False
        self.measured_hist = deque(maxlen=max(int(cfg.lock_hold_updates), 1))

    def _trim(self, now_wall: float):
        tmin = now_wall - self.cfg.buffer_sec
        while self.cmd_hist and self.cmd_hist[0][0] < tmin:
            self.cmd_hist.popleft()
        while self.pwm_hist and self.pwm_hist[0][0] < tmin:
            self.pwm_hist.popleft()

    def push_cmd(self, wall_t: float, cmd_u: float):
        self.cmd_hist.append((wall_t, float(cmd_u)))
        self._trim(wall_t)

    def push_pwm(self, wall_t: float, hw_pwm: float):
        self.pwm_hist.append((wall_t, float(hw_pwm)))
        self._trim(wall_t)

    def estimate(self, wall_now: float):
        if self.delay_locked:
            return self.delay_sec
        if wall_now - self.last_update_wall < 1.0 / max(self.cfg.update_hz, 1e-9):
            return self.delay_sec
        self.last_update_wall = wall_now
        if len(self.cmd_hist) < 25 or len(self.pwm_hist) < 25:
            return self.delay_sec

        t0 = max(self.cmd_hist[0][0], self.pwm_hist[0][0])
        t1 = min(self.cmd_hist[-1][0], self.pwm_hist[-1][0])
        if (t1 - t0) < 0.8:
            return self.delay_sec

        dt = 0.01
        ts = np.arange(t0, t1, dt)
        if len(ts) < 30:
            return self.delay_sec

        def interp(hist, t):
            xs = list(hist)
            if t <= xs[0][0]:
                return xs[0][1]
            if t >= xs[-1][0]:
                return xs[-1][1]
            for i in range(len(xs) - 1):
                t0i, y0 = xs[i]
                t1i, y1 = xs[i + 1]
                if t0i <= t <= t1i:
                    a = (t - t0i) / max(t1i - t0i, 1e-9)
                    return (1.0 - a) * y0 + a * y1
            return xs[-1][1]

        cmd = np.array([interp(self.cmd_hist, t) for t in ts], dtype=float)
        pwm = np.array([interp(self.pwm_hist, t) for t in ts], dtype=float)
        cmd -= np.mean(cmd)
        pwm -= np.mean(pwm)
        if np.std(cmd) < 1e-6 or np.std(pwm) < 1e-6:
            return self.delay_sec

        max_lag = int((self.cfg.max_delay_ms / 1000.0) / dt)
        best_lag = 0
        best_score = -1e18
        for lag in range(max_lag + 1):
            if lag >= len(cmd) - 2:
                break
            a = cmd[:-lag] if lag > 0 else cmd
            b = pwm[lag:] if lag > 0 else pwm
            score = float(np.dot(a, b)) / max(len(a), 1)
            if score > best_score:
                best_score = score
                best_lag = lag

        measured = best_lag * dt
        self.measured_hist.append(measured)
        self.delay_sec = (1.0 - self.cfg.smooth_alpha) * self.delay_sec + self.cfg.smooth_alpha * measured
        if len(self.measured_hist) >= max(int(self.cfg.lock_hold_updates), 1):
            hist = np.array(self.measured_hist, dtype=float)
            if float(np.std(hist)) <= (self.cfg.lock_std_ms / 1000.0):
                self.delay_sec = float(np.mean(hist))
                self.delay_locked = True
        return self.delay_sec


class DelayMonitorNode(Node):
    def __init__(self, cfg: DelayConfig):
        super().__init__("calibration_delay_monitor")
        self.cfg = cfg
        self.est = DelayLockEstimator(cfg)
        self.create_subscription(Float32, cfg.topic_cmd_u, self.cb_cmd, 20)
        self.create_subscription(Float32, cfg.topic_hw_pwm, self.cb_pwm, 20)

    def cb_cmd(self, msg: Float32):
        self.est.push_cmd(time.time(), float(msg.data))

    def cb_pwm(self, msg: Float32):
        self.est.push_pwm(time.time(), float(msg.data))


class CprCollector:
    """imu_viewer.py 내부 상태 추적 로직(SharedState)을 그대로 활용한 CPR 수집기."""

    def __init__(self, imu_topic: str, enc_topic: str):
        self.state = SharedState()
        self._node = None
        self._controller_node = None
        self._executor = None
        self._thread = None
        self._running = False
        self.imu_topic = imu_topic
        self.enc_topic = enc_topic
        self.ctrl_cfg = ControllerConfig()
        self.delay_cfg = DelayConfig()
        self._delay_node = None

    def start(self):
        if not rclpy.ok():
            rclpy.init()

        self._node = ViewerNode(self.state, self.imu_topic, self.enc_topic)
        self._controller_node = CalibrationKeyboardControllerNode(self.ctrl_cfg)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor.add_node(self._controller_node)
        self._delay_node = DelayMonitorNode(self.delay_cfg)
        self._executor.add_node(self._delay_node)
        self._running = True

        def _spin_loop():
            while self._running and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.05)

        self._thread = threading.Thread(target=_spin_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._controller_node is not None:
            self._controller_node.current_u = 0.0
            self._controller_node.publish_state("exit")
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._controller_node is not None:
            self._controller_node.destroy_node()
        if self._delay_node is not None:
            self._delay_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def wait_for_imu(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self.state.lock:
                if self.state.has_init and self.state.seq > 0:
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> dict:
        with self.state.lock:
            samples = list(self.state.cpr_samples)
            tip_hist = [tip.tolist() for tip in self.state.tip_hist]
            tip0 = self.state.tip0.tolist()
            return {
                "cpr_samples": samples,
                "last_cpr": self.state.last_cpr,
                "full_rotations": int(self.state.rev_index),
                "angle_unwrapped_rad": float(self.state.angle_unwrapped),
                "angle_travel_rad": float(self.state.angle_travel),
                "tip_hist": tip_hist,
                "tip0": tip0,
                "delay_sec": float(self._delay_node.est.estimate(time.time())) if self._delay_node is not None else 0.0,
                "delay_locked": bool(self._delay_node.est.delay_locked) if self._delay_node is not None else False,
            }

    def reset_revolution_window(self) -> None:
        with self.state.lock:
            if hasattr(self.state, "reset_revolution_window"):
                self.state.reset_revolution_window()
            else:
                self.state.rev_enc_anchor = self.state.enc
                self.state.rev_angle_anchor = self.state.angle_unwrapped
                self.state.last_cpr = None
            if hasattr(self.state, "motion_started"):
                self.state.motion_started = False
            self.state.rev_index = 0
            self.state.cpr_samples = []
            if hasattr(self.state.tip_hist, "clear"):
                self.state.tip_hist.clear()
            tip_now = self.state.last_tip.copy() if hasattr(self.state, "last_tip") else self.state.tip0.copy()
            self.state.tip0 = tip_now.copy()
            self.state.tip_hist.append(tip_now.copy())


def _collect_cpr_and_r_from_imu(args) -> tuple[list[dict], float, list[dict], float, float, bool]:
    collector = CprCollector(imu_topic=args.imu_topic, enc_topic=args.hw_enc_topic)
    viewer_proc = maybe_launch_imu_viewer(args)

    try:
        collector.start()
        print("\n[CPR] IMU/엔코더 기반 자동 CPR 수집")
        if not collector.wait_for_imu(timeout_sec=args.imu_wait_sec):
            raise RuntimeError("IMU 데이터를 받지 못했습니다. 토픽 연결 상태를 확인하세요.")

        print_help()
        collector.reset_revolution_window()
        baseline = collector.snapshot()
        baseline_cpr_idx = len(baseline["cpr_samples"])
        baseline_tip_idx = len(baseline["tip_hist"])
        print("[INFO] 키보드 입력으로 모터를 조작하며 calibration 데이터를 수집합니다.")
        ctrl = collector._controller_node
        period = 1.0 / max(ctrl.cfg.loop_hz, 1e-6)
        with KeyboardReader() as kb:
            while True:
                snap = collector.snapshot()
                key = kb.read_key_nonblocking(timeout=0.05)
                changed = ctrl.apply_key(key)
                if viewer_proc is not None and viewer_proc.poll() is not None:
                    print("\n[INFO] IMU viewer가 종료되어 calibration도 함께 종료합니다.")
                    break
                if key in ("c", "C"):
                    collector.reset_revolution_window()
                    baseline = collector.snapshot()
                    baseline_cpr_idx = len(baseline["cpr_samples"])
                    baseline_tip_idx = len(baseline["tip_hist"])
                elif key in ("q", "Q"):
                    break

                ctrl.update_auto_signal()
                local_cpr_count = max(0, len(snap["cpr_samples"]) - baseline_cpr_idx)
                now = time.time()
                if (now - ctrl.last_pub_time) >= period:
                    ctrl.publish_state(key_name=key or "")
                    ctrl.last_pub_time = now
                    print_status_line_with_delay(
                        ctrl,
                        snap["full_rotations"],
                        local_cpr_count,
                        delay_ms=1000.0 * float(snap.get("delay_sec", 0.0)),
                        delay_locked=bool(snap.get("delay_locked", False)),
                    )
                elif changed:
                    print_status_line_with_delay(
                        ctrl,
                        snap["full_rotations"],
                        local_cpr_count,
                        delay_ms=1000.0 * float(snap.get("delay_sec", 0.0)),
                        delay_locked=bool(snap.get("delay_locked", False)),
                    )
        print()

        snap = collector.snapshot()
        snap["cpr_samples"] = snap["cpr_samples"][baseline_cpr_idx:]
        snap["tip_hist"] = snap["tip_hist"][baseline_tip_idx:]
        cpr_samples = snap["cpr_samples"]
        if len(cpr_samples) < 1:
            raise RuntimeError(
                "CPR 샘플이 부족합니다. 최소 1회전 이상 수행 후 q를 눌러주세요."
            )

        cpr_trials = [
            {
                "trial": idx,
                "cpr": float(cpr),
            }
            for idx, cpr in enumerate(cpr_samples, start=1)
        ]
        mean_cpr = float(mean(cpr_samples))

        r_trials, mean_r = _estimate_r_trials_from_snapshot(snap)
        if not r_trials:
            raise RuntimeError("orientation 데이터로 r 샘플을 만들지 못했습니다.")

        print(f"[INFO] 감지된 full rotation 수: {snap['full_rotations']}")
        print(f"[INFO] CPR 샘플 수: {len(cpr_samples)}")
        print(f"[INFO] r 샘플 수: {len(r_trials)}")
        print(f"[INFO] estimated delay: {1000.0 * float(snap.get('delay_sec', 0.0)):.1f} ms (locked={bool(snap.get('delay_locked', False))})")
        return cpr_trials, mean_cpr, r_trials, mean_r, float(snap.get("delay_sec", 0.0)), bool(snap.get("delay_locked", False))
    finally:
        collector.stop()
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()
            try:
                viewer_proc.wait(timeout=1.5)
            except Exception:
                viewer_proc.kill()


def _estimate_r_trials_from_snapshot(snapshot: dict) -> tuple[list[dict], float | None]:
    tip_hist = snapshot.get("tip_hist", [])
    tip0 = snapshot.get("tip0")
    if not tip_hist or tip0 is None:
        return [], None

    ref = tip0
    ref_norm = (ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2) ** 0.5
    if ref_norm < 1e-9:
        return [], None

    instant_samples = []
    parallel_samples = []

    for idx, tip in enumerate(tip_hist, start=1):
        tx, ty, tz = float(tip[0]), float(tip[1]), float(tip[2])
        tip_norm = (tx * tx + ty * ty + tz * tz) ** 0.5
        if tip_norm < 1e-9:
            continue

        r_instant = tip_norm
        instant_samples.append(
            {
                "trial": idx,
                "method": "tip_norm",
                "radius_m": r_instant,
            }
        )

        dot = tx * ref[0] + ty * ref[1] + tz * ref[2]
        cos_angle = dot / (tip_norm * ref_norm)
        if cos_angle < -0.99:
            dx = tx - ref[0]
            dy = ty - ref[1]
            dz = tz - ref[2]
            chord = (dx * dx + dy * dy + dz * dz) ** 0.5
            r_parallel = chord * 0.5
            parallel_samples.append(
                {
                    "trial": idx,
                    "method": "anti_parallel_chord",
                    "radius_m": r_parallel,
                }
            )

    trials = parallel_samples if parallel_samples else instant_samples
    if not trials:
        return [], None
    mean_r = float(mean(item["radius_m"] for item in trials))
    return trials, mean_r


def run_calibration(args) -> None:
    print("=== Manual Rotation Calibration (CPR/r from IMU+orientation) ===")

    cpr_trials, mean_cpr, r_trials, mean_r, delay_sec, delay_locked = _collect_cpr_and_r_from_imu(args)

    result = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "manual_rotation_with_orientation",
        "summary": {
            "mean_cpr": mean_cpr,
            "mean_radius_m": mean_r,
            "trial_count_cpr": len(cpr_trials),
            "trial_count_r": len(r_trials),
        },
        "delay": {
            "effective_control_delay_ms": 1000.0 * float(delay_sec),
            "locked": bool(delay_locked),
        },
        "cpr_trials": cpr_trials,
        "radius_trials": r_trials,
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== Calibration Result ===")
    print(f"mean CPR      : {mean_cpr:.6f}")
    print(f"mean radius r : {mean_r:.6f} m")
    print(f"delay         : {1000.0 * float(delay_sec):.3f} ms (locked={delay_locked})")
    print(f"JSON saved    : {args.output_json}")


def build_argparser():
    ap = argparse.ArgumentParser(description="CPR/r 캘리브레이션 (IMU CPR + orientation r)")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--hw-enc-topic", default="/hw/enc")
    ap.add_argument("--output-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--imu-wait-sec", type=float, default=5.0)
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
    try:
        run_calibration(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
