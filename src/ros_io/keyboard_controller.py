"""Keyboard-based /cmd/u controller for host-mode excitation collection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import select
import shutil
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


@dataclass
class ControllerConfig:
    topic_cmd_u: str = "/cmd/u"
    topic_debug: str = "/cmd/keyboard_state"
    loop_hz: float = 20.0
    pwm_step: float = 10.0
    pwm_max: float = 255.0
    auto_zero_on_exit: bool = True


class KeyboardReader:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self) -> "KeyboardReader":
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key_nonblocking(self, timeout: float = 0.0) -> str | None:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if not rlist:
            return None
        ch1 = sys.stdin.read(1)
        if ch1 != "\x1b":
            return ch1
        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
        if not rlist:
            return "ESC"
        ch2 = sys.stdin.read(1)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
        if not rlist:
            return "ESC"
        ch3 = sys.stdin.read(1)
        if ch2 == "[" and ch3 == "A":
            return "UP"
        if ch2 == "[" and ch3 == "B":
            return "DOWN"
        if ch2 == "[" and ch3 == "C":
            return "RIGHT"
        if ch2 == "[" and ch3 == "D":
            return "LEFT"
        return "ESC"


class KeyboardControllerNode(Node):
    def __init__(self, cfg: ControllerConfig):
        super().__init__("host_keyboard_controller")
        self.cfg = cfg
        self.pub_cmd = self.create_publisher(Float32, cfg.topic_cmd_u, 10)
        self.pub_debug = self.create_publisher(String, cfg.topic_debug, 10)
        self.current_u = 0.0
        self.last_pub_time = 0.0
        self.preset_mode = "manual"
        self.preset_t0 = time.time()
        self.sin_amp = 60.0
        self.sin_freq = 0.5
        self.square_amp = 60.0
        self.square_freq = 0.5
        self.burst_amp = 60.0
        self.burst_period = 2.0
        self.burst_on_time = 0.3
        self.prbs_amp = 60.0
        self.prbs_dt = 0.25

    def clamp_u(self) -> None:
        self.current_u = max(-self.cfg.pwm_max, min(self.cfg.pwm_max, self.current_u))

    def set_manual_mode(self) -> None:
        if self.preset_mode != "manual":
            self.preset_mode = "manual"

    def set_preset_mode(self, mode: str) -> None:
        self.preset_mode = mode
        self.preset_t0 = time.time()

    def prbs_value(self, t: float, dt: float = 0.25, seed: int = 12345) -> float:
        if dt <= 1e-9:
            return 1.0
        k = int(t / dt)
        x = (1103515245 * (k + seed) + 12345) & 0x7FFFFFFF
        return 1.0 if (x & 1) else -1.0

    def update_auto_signal(self) -> None:
        if self.preset_mode == "manual":
            return
        t = time.time() - self.preset_t0
        if self.preset_mode == "sin":
            self.current_u = self.sin_amp * math.sin(2.0 * math.pi * self.sin_freq * t)
        elif self.preset_mode == "square":
            self.current_u = self.square_amp if math.sin(2.0 * math.pi * self.square_freq * t) >= 0.0 else -self.square_amp
        elif self.preset_mode == "burst":
            self.current_u = self.burst_amp if (t % self.burst_period) < self.burst_on_time else 0.0
        elif self.preset_mode == "prbs":
            self.current_u = self.prbs_amp * self.prbs_value(t, self.prbs_dt)
        self.clamp_u()

    def publish_state(self, key_name: str = "") -> None:
        self.pub_cmd.publish(Float32(data=float(self.current_u)))
        self.pub_debug.publish(String(data=f"key={key_name}, mode={self.preset_mode}, cmd_u={self.current_u:.1f}"))

    def apply_key(self, key: str | None) -> bool:
        if key is None:
            return False
        changed = False
        if key in ("w", "W", "UP"):
            self.set_manual_mode()
            self.current_u += self.cfg.pwm_step
            changed = True
        elif key in ("s", "S", "DOWN"):
            self.set_manual_mode()
            self.current_u -= self.cfg.pwm_step
            changed = True
        elif key in ("d", "D", "RIGHT"):
            self.set_manual_mode()
            self.current_u += self.cfg.pwm_step * 0.5
            changed = True
        elif key in ("a", "A", "LEFT"):
            self.set_manual_mode()
            self.current_u -= self.cfg.pwm_step * 0.5
            changed = True
        elif key == " " or key in ("x", "X"):
            self.set_manual_mode()
            self.current_u = 0.0
            changed = True
        elif key == "1":
            self.set_manual_mode()
            self.current_u = 60.0
            changed = True
        elif key == "2":
            self.set_manual_mode()
            self.current_u = -60.0
            changed = True
        elif key == "3":
            self.set_manual_mode()
            self.current_u = 120.0
            changed = True
        elif key == "4":
            self.set_manual_mode()
            self.current_u = -120.0
            changed = True
        elif key == "5":
            self.set_preset_mode("sin")
            changed = True
        elif key == "6":
            self.set_preset_mode("square")
            changed = True
        elif key == "7":
            self.set_preset_mode("burst")
            changed = True
        elif key == "8":
            self.set_preset_mode("prbs")
            changed = True
        elif key == "[":
            self.cfg.pwm_step = max(1.0, self.cfg.pwm_step - 1.0)
        elif key == "]":
            self.cfg.pwm_step = min(100.0, self.cfg.pwm_step + 1.0)
        elif key == "-":
            self.cfg.pwm_max = max(20.0, self.cfg.pwm_max - 5.0)
            changed = True
        elif key == "=":
            self.cfg.pwm_max = min(255.0, self.cfg.pwm_max + 5.0)
            changed = True
        self.clamp_u()
        return changed


def print_help() -> None:
    print("\n" + "=" * 70)
    print("Host Keyboard Controller")
    print("=" * 70)
    print("w/s or Up/Down: coarse +/- PWM | a/d or Left/Right: fine +/- PWM")
    print("space/x: stop(0) | 1~4: fixed presets | 5~8: waveform presets")
    print("[/]: step tune | -/=: max tune | q: quit")
    print("=" * 70 + "\n")


def print_status_line(node: KeyboardControllerNode) -> None:
    width = shutil.get_terminal_size((120, 24)).columns
    msg = f"cmd_u:{node.current_u:6.1f} | step:{node.cfg.pwm_step:4.1f} | max:{node.cfg.pwm_max:5.1f} | mode:{node.preset_mode:<6}"
    sys.stdout.write("\r\033[2K" + msg[: max(20, width - 1)].ljust(max(20, width - 1)))
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard /cmd/u publisher")
    parser.add_argument("--topic-cmd-u", default="/cmd/u")
    parser.add_argument("--topic-debug", default="/cmd/keyboard_state")
    parser.add_argument("--loop-hz", type=float, default=20.0)
    parser.add_argument("--pwm-step", type=float, default=10.0)
    parser.add_argument("--pwm-max", type=float, default=255.0)
    args = parser.parse_args()

    cfg = ControllerConfig(
        topic_cmd_u=args.topic_cmd_u,
        topic_debug=args.topic_debug,
        loop_hz=args.loop_hz,
        pwm_step=args.pwm_step,
        pwm_max=args.pwm_max,
    )
    rclpy.init()
    node = KeyboardControllerNode(cfg)
    print_help()
    period = 1.0 / cfg.loop_hz
    try:
        with KeyboardReader() as kb:
            while rclpy.ok():
                key = kb.read_key_nonblocking(timeout=0.01)
                if key in ("q", "Q"):
                    break
                changed = node.apply_key(key)
                node.update_auto_signal()
                now = time.time()
                if (now - node.last_pub_time) >= period:
                    node.publish_state(key_name=key or "")
                    node.last_pub_time = now
                    print_status_line(node)
                elif changed:
                    print_status_line(node)
                rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if cfg.auto_zero_on_exit:
            for _ in range(5):
                node.pub_cmd.publish(Float32(data=0.0))
                node.pub_debug.publish(String(data="key=EXIT, mode=manual, cmd_u=0.0"))
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(0.03)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
