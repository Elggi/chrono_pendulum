#!/usr/bin/env python3
import sys
import termios
import tty
import select
import time
import math
import shutil
from dataclasses import dataclass

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
    timeout_sec: float = 0.5
    auto_zero_on_exit: bool = True


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

        # Arrow keys: ESC [ A/B/C/D
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
                        elif ch3 == "B":
                            return "DOWN"
                        elif ch3 == "C":
                            return "RIGHT"
                        elif ch3 == "D":
                            return "LEFT"
            return "ESC"

        return ch1


class KeyboardControllerNode(Node):
    def __init__(self, cfg: ControllerConfig):
        super().__init__("jetson_keyboard_controller")
        self.cfg = cfg

        self.pub_cmd = self.create_publisher(Float32, cfg.topic_cmd_u, 10)
        self.pub_debug = self.create_publisher(String, cfg.topic_debug, 10)

        self.current_u = 0.0
        self.last_key_time = time.time()
        self.last_pub_time = 0.0

        # auto preset mode
        self.preset_mode = "manual"   # manual / sin / square / burst / prbs
        self.preset_t0 = time.time()

        # preset parameters
        self.sin_amp = 60.0
        self.sin_freq = 0.5

        self.square_amp = 60.0
        self.square_freq = 0.5

        self.burst_amp = 60.0
        self.burst_period = 2.0
        self.burst_on_time = 0.30

        self.prbs_amp = 60.0
        self.prbs_dt = 0.25

        self.get_logger().info("Keyboard controller node started.")
        self.get_logger().info(f"Publishing signed PWM to {cfg.topic_cmd_u}")

    def clamp_u(self):
        self.current_u = max(-self.cfg.pwm_max, min(self.cfg.pwm_max, self.current_u))

    def set_manual_mode(self):
        if self.preset_mode != "manual":
            self.preset_mode = "manual"
            self.get_logger().info("preset_mode -> manual")

    def set_preset_mode(self, mode: str):
        self.preset_mode = mode
        self.preset_t0 = time.time()
        self.get_logger().info(f"preset_mode -> {mode}")

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
            self.current_u = (
                self.square_amp
                if math.sin(2.0 * math.pi * self.square_freq * t) >= 0.0
                else -self.square_amp
            )

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

        # forward / reverse by signed PWM
        if key in ("w", "W", "UP"):
            self.set_manual_mode()
            self.current_u += self.cfg.pwm_step
            changed = True

        elif key in ("s", "S", "DOWN"):
            self.set_manual_mode()
            self.current_u -= self.cfg.pwm_step
            changed = True

        # fine adjustment
        elif key in ("d", "D", "RIGHT"):
            self.set_manual_mode()
            self.current_u += self.cfg.pwm_step * 0.5
            changed = True

        elif key in ("a", "A", "LEFT"):
            self.set_manual_mode()
            self.current_u -= self.cfg.pwm_step * 0.5
            changed = True

        # emergency stop
        elif key == " ":
            self.set_manual_mode()
            self.current_u = 0.0
            changed = True

        # preset keys: fixed PWM
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

        # preset keys: waveform
        elif key == "5":
            self.set_preset_mode("sin")
            changed = True

        elif key == "6":
            self.set_preset_mode("square")
            changed = True

        # recommended extras
        elif key == "7":
            self.set_preset_mode("burst")
            changed = True

        elif key == "8":
            self.set_preset_mode("prbs")
            changed = True

        # tuning
        elif key == "[":
            self.cfg.pwm_step = max(1.0, self.cfg.pwm_step - 1.0)
            self.get_logger().info(f"pwm_step -> {self.cfg.pwm_step:.1f}")

        elif key == "]":
            self.cfg.pwm_step = min(100.0, self.cfg.pwm_step + 1.0)
            self.get_logger().info(f"pwm_step -> {self.cfg.pwm_step:.1f}")

        elif key == "-":
            self.cfg.pwm_max = max(20.0, self.cfg.pwm_max - 5.0)
            self.clamp_u()
            self.get_logger().info(f"pwm_max -> {self.cfg.pwm_max:.1f}")
            changed = True

        elif key == "=":
            self.cfg.pwm_max = min(255.0, self.cfg.pwm_max + 5.0)
            self.clamp_u()
            self.get_logger().info(f"pwm_max -> {self.cfg.pwm_max:.1f}")
            changed = True

        # direct zero
        elif key in ("x", "X"):
            self.set_manual_mode()
            self.current_u = 0.0
            changed = True

        self.clamp_u()

        if changed:
            self.last_key_time = time.time()

        return changed


def print_help():
    print("\n" + "=" * 70)
    print("Jetson Keyboard Controller")
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
    print("  /cmd/u  (std_msgs/Float32, signed PWM)")
    print("=" * 70 + "\n")


def print_status_line(node: KeyboardControllerNode):
    term_width = shutil.get_terminal_size((120, 24)).columns
    msg = (
        f"cmd_u: {node.current_u:6.1f} | step: {node.cfg.pwm_step:4.1f} | "
        f"max: {node.cfg.pwm_max:5.1f} | mode: {node.preset_mode:<6}"
    )
    sys.stdout.write("\r\033[2K" + msg[: max(20, term_width - 1)].ljust(max(20, term_width - 1)))
    sys.stdout.flush()


def main():
    rclpy.init()
    cfg = ControllerConfig()
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

                # auto waveform update
                node.update_auto_signal()

                now = time.time()

                # publish continuously at loop_hz
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
            zero_msg = Float32()
            zero_msg.data = 0.0
            for _ in range(5):
                node.pub_cmd.publish(zero_msg)
                node.pub_debug.publish(String(data="key=EXIT, mode=manual, cmd_u=0.0"))
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(0.03)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
