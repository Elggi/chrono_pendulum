#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""수동 회전 기반 캘리브레이션 도구.

- CPR: IMU/엔코더를 구독하며 `imu_viewer.py`의 SharedState 로직을 재사용해 자동 계산
- r  : orientation 기반 tip 좌표에서 실시간 반지름 추정
"""

import argparse
import csv
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
from statistics import mean, median, pstdev
from collections import deque

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32, String

from imu_viewer import SharedState, ViewerNode
from chrono_core.signal_filter import CausalIIRFilter


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
    print("  q                 : finish calibration and compute")
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

    def start(self):
        if not rclpy.ok():
            rclpy.init()

        self._node = ViewerNode(self.state, self.imu_topic, self.enc_topic)
        self._controller_node = CalibrationKeyboardControllerNode(self.ctrl_cfg)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor.add_node(self._controller_node)
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
                "enc": float(self.state.enc),
                "angle_travel_rad": float(self.state.angle_travel),
                "tip_hist": tip_hist,
                "tip0": tip0,
                "acc": self.state.acc.tolist(),
                "gyro": self.state.gyro.tolist(),
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

def _collect_cpr_and_r_from_imu(args) -> tuple[list[dict], float, list[dict], float, float | None, int]:
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
        gravity_samples = []
        with KeyboardReader() as kb:
            while True:
                snap = collector.snapshot()
                acc_vec = np.asarray(snap.get("acc", [np.nan, np.nan, np.nan]), dtype=float)
                gyro_vec = np.asarray(snap.get("gyro", [np.nan, np.nan, np.nan]), dtype=float)
                if np.isfinite(acc_vec).all() and np.isfinite(gyro_vec).all():
                    gyro_norm = float(np.linalg.norm(gyro_vec))
                    if gyro_norm <= 0.35:
                        gravity_samples.append(float(np.linalg.norm(acc_vec)))
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
                    print_status_line(ctrl, snap["full_rotations"], local_cpr_count)
                elif changed:
                    print_status_line(ctrl, snap["full_rotations"], local_cpr_count)
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
        g_eff = float(np.median(gravity_samples)) if gravity_samples else None
        return cpr_trials, mean_cpr, r_trials, mean_r, g_eff, len(gravity_samples)
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

    cpr_trials, mean_cpr, r_trials, mean_r, g_eff, gravity_sample_count = _collect_cpr_and_r_from_imu(args)

    current_stats = estimate_current_offset(
        current_topic=args.current_topic,
        pwm_topic=args.hw_pwm_topic,
        sample_sec=args.current_sample_sec,
        pwm_zero_threshold=args.pwm_zero_threshold,
    )

    result = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "manual_rotation_with_orientation",
        "summary": {
            "mean_cpr": mean_cpr,
            "mean_radius_m": mean_r,
            "g_eff_mps2": g_eff,
            "trial_count_cpr": len(cpr_trials),
            "trial_count_r": len(r_trials),
            "trial_count_g": gravity_sample_count,
            "ina_current_offset_mA": current_stats["median"],
        },
        "cpr_trials": cpr_trials,
        "radius_trials": r_trials,
        "ina_current_offset": current_stats,
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== Calibration Result ===")
    print(f"mean CPR      : {mean_cpr:.6f}")
    print(f"mean radius r : {mean_r:.6f} m")
    if g_eff is not None:
        print(f"mean g_eff    : {g_eff:.6f} m/s^2")
    else:
        print("mean g_eff    : n/a (insufficient low-gyro samples)")
    print(f"JSON saved    : {args.output_json}")
    print(
        "ina offset    : "
        f"median={current_stats['median']:.6f} mA, mean={current_stats['mean']:.6f} mA, "
        f"std={current_stats['std']:.6f} mA, n={current_stats['sample_count']}"
    )


def _gyro_norm(snapshot: dict) -> float:
    g = np.asarray(snapshot.get("gyro", [np.nan, np.nan, np.nan]), dtype=float)
    if not np.isfinite(g).all():
        return float("nan")
    return float(np.linalg.norm(g))


def _next_free_decay_csv_path(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    max_idx = 0
    for name in os.listdir(base_dir):
        if not name.startswith("calibration_free_decay_") or not name.endswith(".csv"):
            continue
        stem = name[len("calibration_free_decay_") : -len(".csv")]
        try:
            max_idx = max(max_idx, int(stem))
        except ValueError:
            continue
    return os.path.join(base_dir, f"calibration_free_decay_{max_idx + 1}.csv")


def run_free_decay_collection(args) -> None:
    collector = CprCollector(imu_topic=args.imu_topic, enc_topic=args.hw_enc_topic)
    viewer_proc = maybe_launch_imu_viewer(args)
    ctrl = None

    print("=== Free-Decay Data Collection (Option 2: model calibration) ===")
    print("[INFO] 모터 명령은 0으로 고정합니다. 손으로 로드를 원하는 각도까지 들어 올리세요 (arming...).")
    print("[INFO] 특정 각도에서 2초 이상 정지하면 'armed!'를 출력합니다.")
    print("[INFO] armed 이후 각도가 움직이기 시작하면 자동으로 수집 시작합니다.")
    print("[INFO] 이후 상대각이 0 근처(기본 ±1.9deg)에서 2초간 정지하면 자동 종료합니다.")
    print("[INFO] q 키를 누르면 언제든 수집을 취소할 수 있습니다.")

    try:
        collector.start()
        if not collector.wait_for_imu(timeout_sec=args.imu_wait_sec):
            raise RuntimeError("IMU 데이터를 받지 못했습니다. 토픽 연결 상태를 확인하세요.")

        ctrl = collector._controller_node
        ctrl.set_manual_mode()
        ctrl.current_u = 0.0
        ctrl.publish_state("free_decay_init")

        arm_start_ts = None
        armed = False
        theta_arm = None
        release_ts = None
        stop_hold_start = None
        stop_theta_bin = None
        rows = []
        last_pub_ts = 0.0
        sample_period = 1.0 / max(float(args.free_decay_sample_hz), 1e-6)
        theta_window = deque(maxlen=max(10, int(args.free_decay_sample_hz * max(args.free_decay_steady_window_sec, 0.2))))
        theta_filt = CausalIIRFilter(alpha=float(args.free_decay_theta_filter_alpha))
        omega_filt = CausalIIRFilter(alpha=float(args.free_decay_omega_filter_alpha))
        theta_prev = None
        release_detected = False

        with KeyboardReader() as kb:
            while True:
                if viewer_proc is not None and viewer_proc.poll() is not None:
                    raise RuntimeError("IMU viewer가 종료되어 free decay 수집을 중단합니다.")

                snap = collector.snapshot()
                now = time.monotonic()
                gyro_n = _gyro_norm(snap)
                theta = float(snap.get("angle_unwrapped_rad", 0.0))
                enc = float(snap.get("enc", 0.0))
                omega_z = float(np.asarray(snap.get("gyro", [0.0, 0.0, 0.0]), dtype=float)[2])
                theta_lp = float(theta_filt.update(theta))
                omega_lp = float(omega_filt.update(omega_z))
                theta_window.append(theta_lp)

                key = kb.read_key_nonblocking(timeout=0.01)
                if key in ("q", "Q", "ESC"):
                    raise RuntimeError("사용자 요청으로 free decay 수집을 취소했습니다.")

                # Keep motor command at zero.
                if (now - last_pub_ts) >= 0.05:
                    ctrl.current_u = 0.0
                    ctrl.publish_state("free_decay_zero")
                    last_pub_ts = now

                if not armed:
                    theta_abs_deg = abs(math.degrees(theta_lp))
                    is_lifted = theta_abs_deg >= float(args.free_decay_arm_min_angle_deg)
                    theta_std = float(np.std(np.asarray(theta_window, dtype=float))) if len(theta_window) >= 3 else float("inf")
                    is_steady = (
                        math.isfinite(omega_lp)
                        and abs(omega_lp) <= float(args.free_decay_hold_gyro_threshold)
                        and theta_std <= math.radians(float(args.free_decay_hold_std_deg))
                    )
                    if is_lifted and is_steady:
                        if arm_start_ts is None:
                            arm_start_ts = now
                        arm_elapsed = now - arm_start_ts
                        if arm_elapsed >= float(args.free_decay_hold_min_sec):
                            armed = True
                            theta_arm = float(np.median(np.asarray(theta_window, dtype=float)))
                            print(f"\n[INFO] armed! theta_arm={math.degrees(theta_arm):.3f} deg")
                    else:
                        arm_start_ts = None

                    arm_elapsed = 0.0 if arm_start_ts is None else (now - arm_start_ts)
                    terminal_status_line(
                        f"[arming...] |theta|={theta_abs_deg:6.2f}/{args.free_decay_arm_min_angle_deg:.2f} deg | "
                        f"omega_lp={omega_lp:7.4f} | std_deg={math.degrees(theta_std):6.3f}/{args.free_decay_hold_std_deg:.3f} | "
                        f"steady={arm_elapsed:5.2f}/{args.free_decay_hold_min_sec:.2f}s"
                    )
                    time.sleep(sample_period)
                    continue

                if not release_detected:
                    dtheta = 0.0 if theta_prev is None else (theta_lp - theta_prev)
                    if (
                        math.isfinite(omega_lp)
                        and (
                            abs(dtheta) >= math.radians(float(args.free_decay_release_delta_deg))
                            or abs(omega_lp) >= float(args.free_decay_release_gyro_threshold)
                        )
                    ):
                        release_detected = True
                        release_ts = now
                        print(
                            f"\n[INFO] release 검출: t={release_ts:.6f}, "
                            f"theta={math.degrees(theta_lp):.3f} deg, omega_lp={omega_lp:.6f} rad/s"
                        )
                    else:
                        dtheta_deg = math.degrees(dtheta)
                        terminal_status_line(
                            f"[armed] release 대기 | dtheta={dtheta_deg:7.3f}/{args.free_decay_release_delta_deg:.3f} deg | "
                            f"omega_lp={omega_lp:7.4f}/{args.free_decay_release_gyro_threshold:.4f}"
                        )
                        theta_prev = theta_lp
                        time.sleep(sample_period)
                        continue

                t_rel = now - release_ts
                rows.append(
                    {
                        "t_sec": t_rel,
                        "theta_raw_rad": theta,
                        "theta_filtered_rad": theta_lp,
                        "theta_from_arm_rad": theta_lp - float(theta_arm),
                        "omega_z_rad_s": omega_z,
                        "omega_filtered_rad_s": omega_lp,
                        "gyro_norm_rad_s": gyro_n,
                        "enc_count": enc,
                    }
                )
                terminal_status_line(
                    f"[수집중] t={t_rel:6.3f}s | "
                    f"theta_lp={theta_lp: .5f} rad | omega_lp={omega_lp: .5f} rad/s"
                )

                theta_bin = round(theta_lp, 3)
                if stop_theta_bin is None or theta_bin != stop_theta_bin:
                    stop_theta_bin = theta_bin
                    stop_hold_start = now
                else:
                    if stop_hold_start is not None and (now - stop_hold_start) >= float(args.free_decay_stop_hold_sec):
                        print(
                            f"\n[INFO] stop condition met: rounded theta_lp={theta_bin:.3f} "
                            f"held for {args.free_decay_stop_hold_sec:.2f}s"
                        )
                        break

                theta_prev = theta_lp
                time.sleep(sample_period)
        print()

        if not rows:
            raise RuntimeError("free decay 샘플이 비어 있습니다.")

        out_dir = os.path.abspath(os.path.dirname(args.output_json) or ".")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = _next_free_decay_csv_path(out_dir)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            t_arr = np.asarray([r["t_sec"] for r in rows], dtype=float)
            theta_arr = np.asarray([r["theta_filtered_rad"] for r in rows], dtype=float)
            omega_arr = np.asarray([r["omega_filtered_rad_s"] for r in rows], dtype=float)
            alpha_arr = np.zeros_like(omega_arr)
            if len(omega_arr) >= 2:
                alpha_arr = np.gradient(omega_arr, np.maximum(t_arr, 1e-6), edge_order=1)
            export_rows = []
            for i, r in enumerate(rows):
                export_rows.append(
                    {
                        "wall_elapsed": float(r["t_sec"]),
                        "theta": float(theta_arr[i]),
                        "omega": float(omega_arr[i]),
                        "alpha": float(alpha_arr[i]),
                        "theta_imu": float(r["theta_raw_rad"]),
                        "theta_imu_online": float(theta_arr[i]),
                        "omega_imu": float(r["omega_z_rad_s"]),
                        "omega_imu_online": float(omega_arr[i]),
                        "alpha_linear": float(alpha_arr[i]),
                        "alpha_linear_online": float(alpha_arr[i]),
                        "cmd_u_raw": 0.0,
                        "cmd_u": 0.0,
                        "pwm_hw": 0.0,
                        "hw_pwm": 0.0,
                        "ina_current_corr_mA": 0.0,
                        "ina_current_signed_online_mA": 0.0,
                        "tau_cmd": 0.0,
                        "tau_motor": 0.0,
                        "tau_visc": 0.0,
                        "tau_coul": 0.0,
                        "enc_count": float(r["enc_count"]),
                    }
                )
            writer = csv.DictWriter(f, fieldnames=list(export_rows[0].keys()))
            writer.writeheader()
            writer.writerows(export_rows)

        result = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": "free_decay_auto_arm_release",
            "summary": {
                "sample_count": int(len(export_rows)),
                "theta_arm_deg": float(math.degrees(theta_arm)) if theta_arm is not None else None,
                "stop_rule": "rounded_theta_lp_3dp_hold",
                "stop_hold_sec": float(args.free_decay_stop_hold_sec),
                "theta_unwrapped": True,
            },
            "artifacts": {
                "free_decay_csv": out_csv,
            },
        }
        with open(args.output_json, "w", encoding="utf-8") as jf:
            json.dump(result, jf, indent=2, ensure_ascii=False)

        print("\n=== Free Decay Collection Result ===")
        print(f"theta_arm(deg)    : {float(math.degrees(theta_arm)):.6f}")
        print(f"samples           : {len(rows)}")
        print(f"saved_csv         : {out_csv}")
        print(f"saved_json        : {args.output_json}")

    finally:
        if ctrl is not None:
            ctrl.current_u = 0.0
            ctrl.publish_state("free_decay_exit")
        collector.stop()
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()
            try:
                viewer_proc.wait(timeout=1.5)
            except Exception:
                viewer_proc.kill()


class CurrentOffsetNode(Node):
    def __init__(self, current_topic: str, pwm_topic: str):
        super().__init__("current_offset_collector")
        self.current_samples = []
        self.latest_pwm = 0.0
        self.create_subscription(Float32, current_topic, self.cb_current, 20)
        self.create_subscription(Float32, pwm_topic, self.cb_pwm, 20)

    def cb_current(self, msg: Float32):
        self.current_samples.append(float(msg.data))

    def cb_pwm(self, msg: Float32):
        self.latest_pwm = float(msg.data)


def estimate_current_offset(current_topic: str, pwm_topic: str, sample_sec: float, pwm_zero_threshold: float) -> dict:
    if not rclpy.ok():
        rclpy.init()
    node = CurrentOffsetNode(current_topic=current_topic, pwm_topic=pwm_topic)
    t0 = time.time()
    accepted = []
    try:
        while time.time() - t0 < max(0.5, float(sample_sec)):
            rclpy.spin_once(node, timeout_sec=0.02)
            while node.current_samples:
                cur = node.current_samples.pop(0)
                if abs(node.latest_pwm) <= abs(float(pwm_zero_threshold)):
                    accepted.append(float(cur))
    finally:
        node.destroy_node()
    if not accepted:
        accepted = [0.0]
    return {
        "ina_current_offset_mA": float(median(accepted)),
        "sample_count": int(len(accepted)),
        "mean": float(mean(accepted)),
        "median": float(median(accepted)),
        "std": float(pstdev(accepted)) if len(accepted) > 1 else 0.0,
    }


def build_argparser():
    ap = argparse.ArgumentParser(description="CPR/r 캘리브레이션 (IMU CPR + orientation r)")
    ap.add_argument("--mode", choices=["cpr", "free_decay"], default="cpr")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--hw-enc-topic", default="/hw/enc")
    ap.add_argument("--output-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--imu-wait-sec", type=float, default=5.0)
    ap.add_argument("--no-imu-viewer", action="store_true")
    ap.add_argument("--current-topic", default="/ina219/current_ma")
    ap.add_argument("--hw-pwm-topic", default="/hw/pwm_applied")
    ap.add_argument("--current-sample-sec", type=float, default=3.0)
    ap.add_argument("--pwm-zero-threshold", type=float, default=1.0)
    ap.add_argument("--free-decay-sample-hz", type=float, default=200.0)
    ap.add_argument("--free-decay-arm-min-angle-deg", type=float, default=5.0)
    ap.add_argument("--free-decay-hold-min-sec", type=float, default=2.0)
    ap.add_argument("--free-decay-steady-window-sec", type=float, default=0.6)
    ap.add_argument("--free-decay-hold-std-deg", type=float, default=0.4)
    ap.add_argument("--free-decay-hold-gyro-threshold", type=float, default=0.15)
    ap.add_argument("--free-decay-theta-filter-alpha", type=float, default=0.12)
    ap.add_argument("--free-decay-omega-filter-alpha", type=float, default=0.18)
    ap.add_argument("--free-decay-release-gyro-threshold", type=float, default=0.35)
    ap.add_argument("--free-decay-release-delta-deg", type=float, default=0.25)
    ap.add_argument("--free-decay-stop-hold-sec", type=float, default=2.0)
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
        if args.mode == "free_decay":
            run_free_decay_collection(args)
        else:
            run_calibration(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
