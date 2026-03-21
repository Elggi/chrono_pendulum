#!/usr/bin/env python3
import math
import sys
import time
import csv
import os
import re
import argparse
import select
import termios
import tty
from dataclasses import dataclass, field

import pychrono as ch
import pychrono.irrlicht as irr
import pychrono.ros as chros

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Imu


# ============================================================
# utility
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q: ch.ChQuaterniond) -> float:
    return math.atan2(
        2.0 * (q.e0 * q.e3 + q.e1 * q.e2),
        1.0 - 2.0 * (q.e2 * q.e2 + q.e3 * q.e3),
    )


def now_wall() -> float:
    return time.time()


def terminal_status_line(msg: str, width: int = 240):
    sys.stdout.write("\r" + msg[:width].ljust(width))
    sys.stdout.flush()


def make_numbered_csv_path(folder: str, prefix: str, ext: str = ".csv") -> str:
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    max_n = 0
    for name in os.listdir(folder):
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return os.path.join(folder, f"{prefix}{max_n + 1}{ext}")


def prbs_value(t: float, dt: float = 0.25, seed: int = 12345) -> float:
    if dt <= 1e-12:
        return 1.0
    k = int(t / dt)
    x = (1103515245 * (k + seed) + 12345) & 0x7FFFFFFF
    return 1.0 if (x & 1) else -1.0


# ============================================================
# config
# ============================================================

@dataclass
class BridgeConfig:
    step: float = 1e-3
    gravity: ch.ChVector3d = field(default_factory=lambda: ch.ChVector3d(0.0, -9.81, 0.0))

    enable_render: bool = True
    win_w: int = 1280
    win_h: int = 900
    window_title: str = "Chrono Pendulum | Unified ROS custom handler bridge"

    motor_radius: float = 0.020
    motor_length: float = 0.050
    shaft_radius: float = 0.003
    shaft_length: float = 0.012

    link_L: float = 0.285
    link_W: float = 0.020
    link_T: float = 0.006
    link_mass: float = 0.022

    imu_mass: float = 0.010
    imu_size_x: float = 0.0595
    imu_size_y: float = 0.0460
    imu_size_z: float = 0.0117

    theta0_deg: float = -10.0
    omega0: float = 0.0

    pwm_limit: float = 60.0
    pwm_step: float = 10.0
    pwm_to_tau_gain: float = 0.35 / 60.0

    cmd_pub_hz: float = 120.0
    ros_rx_hz: float = 200.0
    ros_tx_hz: float = 100.0

    console_hz: float = 120.0
    log_hz: float = 200.0

    wave_freq: float = 0.5
    burst_period: float = 2.0
    burst_on_time: float = 0.3
    prbs_dt: float = 0.25
    prbs_seed: int = 12345

    topic_cmd_u: str = "/cmd/u"
    topic_debug: str = "/cmd/keyboard_state"
    topic_sim_theta: str = "/sim/theta"
    topic_sim_omega: str = "/sim/omega"
    topic_sim_alpha: str = "/sim/alpha"
    topic_sim_cmd_used: str = "/sim/cmd_used"
    topic_sim_tau: str = "/sim/tau"
    topic_sim_status: str = "/sim/status"

    topic_hw_pwm: str = "/hw/pwm_applied"
    topic_hw_enc: str = "/hw/enc"
    topic_hw_arduino_ms: str = "/hw/arduino_ms"
    topic_imu: str = "/imu/data"

    log_dir: str = "./run_logs"
    log_prefix: str = "chrono_run_"


# ============================================================
# keyboard
# ============================================================

class KeyboardReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None
        self.active = False

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        self.active = False

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


# ============================================================
# feedback monitor
# ============================================================

class FeedbackMonitorHandler(chros.ChROSHandler):
    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg.ros_tx_hz)
        self.cfg = cfg

        self.cmd_echo = 0.0
        self.hw_pwm = 0.0
        self.hw_enc = 0.0
        self.hw_arduino_ms = 0.0

        self.imu_wz = 0.0
        self.imu_ax = 0.0
        self.imu_ay = 0.0
        self.imu_az = 0.0

        self.cmd_count = 0
        self.pwm_count = 0
        self.enc_count = 0
        self.imu_count = 0

        self.last_cmd_time = 0.0
        self.last_pwm_time = 0.0
        self.last_enc_time = 0.0
        self.last_imu_time = 0.0

    def Initialize(self, interface: chros.ChROSPythonInterface) -> bool:
        node = interface.GetNode()
        qos = 50
        node.create_subscription(Float32, self.cfg.topic_cmd_u, self.cb_cmd_echo, qos)
        node.create_subscription(Float32, self.cfg.topic_hw_pwm, self.cb_hw_pwm, qos)
        node.create_subscription(Float32, self.cfg.topic_hw_enc, self.cb_hw_enc, qos)
        node.create_subscription(Float32, self.cfg.topic_hw_arduino_ms, self.cb_hw_arduino_ms, qos)
        node.create_subscription(Imu, self.cfg.topic_imu, self.cb_imu, qos)
        return True

    def cb_cmd_echo(self, msg: Float32):
        self.cmd_echo = float(msg.data)
        self.cmd_count += 1
        self.last_cmd_time = now_wall()

    def cb_hw_pwm(self, msg: Float32):
        self.hw_pwm = float(msg.data)
        self.pwm_count += 1
        self.last_pwm_time = now_wall()

    def cb_hw_enc(self, msg: Float32):
        self.hw_enc = float(msg.data)
        self.enc_count += 1
        self.last_enc_time = now_wall()

    def cb_hw_arduino_ms(self, msg: Float32):
        self.hw_arduino_ms = float(msg.data)

    def cb_imu(self, msg: Imu):
        self.imu_wz = float(msg.angular_velocity.z)
        self.imu_ax = float(msg.linear_acceleration.x)
        self.imu_ay = float(msg.linear_acceleration.y)
        self.imu_az = float(msg.linear_acceleration.z)
        self.imu_count += 1
        self.last_imu_time = now_wall()

    def Tick(self, time_now: float):
        return True


# ============================================================
# command source base
# ============================================================

class CommandSourceBase(chros.ChROSHandler):
    def __init__(self, rate_hz: float):
        super().__init__(rate_hz)

    def get_command(self) -> float:
        raise NotImplementedError

    def get_mode_name(self) -> str:
        raise NotImplementedError


# ============================================================
# host publisher mode
# ============================================================

class HostCommandPublisherHandler(CommandSourceBase):
    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg.cmd_pub_hz)
        self.cfg = cfg
        self.pub_cmd = None
        self.pub_debug = None

        self.current_u = 0.0
        self.mode = "manual"
        self.mode_t0 = time.time()

        self.kb = KeyboardReader()
        self.keyboard_active = False
        self.quit_requested = False

    def Initialize(self, interface: chros.ChROSPythonInterface) -> bool:
        node = interface.GetNode()
        self.pub_cmd = node.create_publisher(Float32, self.cfg.topic_cmd_u, 50)
        self.pub_debug = node.create_publisher(String, self.cfg.topic_debug, 10)

        self.kb.__enter__()
        self.keyboard_active = True
        self.mode_t0 = time.time()

        print("\n=== workflow1 : host_pub ===")
        print("Host publishes /cmd/u continuously at high rate")
        print("w/s or ↑/↓ : +/-10, a/d or ←/→ : -/+5, space : 0")
        print("1:+60 2:-60 3:+120 4:-120 5:sin 6:square 7:burst 8:prbs m:manual q:quit\n")
        return True

    def Shutdown(self):
        if self.keyboard_active:
            self.kb.__exit__(None, None, None)
            self.keyboard_active = False
        print()

    def set_manual(self):
        self.mode = "manual"

    def set_mode(self, mode: str):
        self.mode = mode
        self.mode_t0 = time.time()

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
        elif key == " ":
            self.set_manual()
            self.current_u = 0.0
        elif key == "1":
            self.set_manual()
            self.current_u = 60.0
        elif key == "2":
            self.set_manual()
            self.current_u = -60.0
        elif key == "3":
            self.set_manual()
            self.current_u = 120.0
        elif key == "4":
            self.set_manual()
            self.current_u = -120.0
        elif key == "5":
            self.set_mode("sin")
        elif key == "6":
            self.set_mode("square")
        elif key == "7":
            self.set_mode("burst")
        elif key == "8":
            self.set_mode("prbs")
        elif key in ("m", "M"):
            self.set_manual()
        elif key in ("q", "Q", "ESC"):
            self.quit_requested = True

        self.current_u = clamp(self.current_u, -self.cfg.pwm_limit, self.cfg.pwm_limit)

    def update_auto_signal(self):
        if self.mode == "manual":
            return

        t = time.time() - self.mode_t0

        if self.mode == "sin":
            self.current_u = self.cfg.pwm_limit * math.sin(2.0 * math.pi * self.cfg.wave_freq * t)
        elif self.mode == "square":
            s = math.sin(2.0 * math.pi * self.cfg.wave_freq * t)
            self.current_u = self.cfg.pwm_limit if s >= 0 else -self.cfg.pwm_limit
        elif self.mode == "burst":
            self.current_u = self.cfg.pwm_limit if (t % self.cfg.burst_period) < self.cfg.burst_on_time else 0.0
        elif self.mode == "prbs":
            self.current_u = self.cfg.pwm_limit * prbs_value(t, self.cfg.prbs_dt, self.cfg.prbs_seed)

        self.current_u = clamp(self.current_u, -self.cfg.pwm_limit, self.cfg.pwm_limit)

    def Tick(self, time_now: float):
        key = self.kb.read_key_nonblocking(0.0)
        self.apply_key(key)
        self.update_auto_signal()
        self.pub_cmd.publish(Float32(data=float(self.current_u)))
        self.pub_debug.publish(String(data=f"mode={self.mode}, cmd_u={self.current_u:.2f}"))

        return not self.quit_requested

    def get_command(self) -> float:
        return float(self.current_u)

    def get_mode_name(self) -> str:
        return self.mode


# ============================================================
# jetson subscriber mode
# ============================================================

class JetsonCommandSubscriberHandler(CommandSourceBase):
    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg.ros_rx_hz)
        self.cfg = cfg
        self.latest_cmd = 0.0
        self.rx_count = 0
        self.status_pub = None

    def Initialize(self, interface: chros.ChROSPythonInterface) -> bool:
        node = interface.GetNode()
        node.create_subscription(Float32, self.cfg.topic_cmd_u, self.cb_cmd, 50)
        self.status_pub = node.create_publisher(String, self.cfg.topic_sim_status, 10)

        print("\n=== workflow2 : jetson_pub ===")
        print("Host subscribes /cmd/u from Jetson and applies same command to Chrono\n")
        return True

    def cb_cmd(self, msg: Float32):
        self.latest_cmd = float(msg.data)
        self.rx_count += 1

    def Tick(self, time_now: float):
        self.status_pub.publish(String(data=f"rx_count={self.rx_count}, latest_cmd={self.latest_cmd:.2f}"))
        return True

    def get_command(self) -> float:
        return float(self.latest_cmd)

    def get_mode_name(self) -> str:
        return "jetson"


# ============================================================
# sim state publisher
# ============================================================

class SimStatePublisherHandler(chros.ChROSHandler):
    def __init__(self, cfg: BridgeConfig, link: ch.ChBody, cmd_source: CommandSourceBase):
        super().__init__(cfg.ros_tx_hz)
        self.cfg = cfg
        self.link = link
        self.cmd_source = cmd_source

        self.pub_theta = None
        self.pub_omega = None
        self.pub_alpha = None
        self.pub_cmd_used = None
        self.pub_tau = None

        self.theta_unwrap = 0.0
        self.prev_theta_raw = None
        self.prev_omega = 0.0
        self.prev_time = 0.0
        self.last_tau = 0.0

    def Initialize(self, interface: chros.ChROSPythonInterface) -> bool:
        node = interface.GetNode()
        self.pub_theta = node.create_publisher(Float32, self.cfg.topic_sim_theta, 10)
        self.pub_omega = node.create_publisher(Float32, self.cfg.topic_sim_omega, 10)
        self.pub_alpha = node.create_publisher(Float32, self.cfg.topic_sim_alpha, 10)
        self.pub_cmd_used = node.create_publisher(Float32, self.cfg.topic_sim_cmd_used, 10)
        self.pub_tau = node.create_publisher(Float32, self.cfg.topic_sim_tau, 10)

        self.prev_theta_raw = yaw_from_quat(self.link.GetRot())
        self.theta_unwrap = self.prev_theta_raw
        self.prev_omega = self.link.GetAngVelLocal().z
        self.prev_time = 0.0
        return True

    def set_last_tau(self, tau: float):
        self.last_tau = float(tau)

    def Tick(self, time_now: float):
        theta_raw = yaw_from_quat(self.link.GetRot())
        self.theta_unwrap += wrap_to_pi(theta_raw - self.prev_theta_raw)
        self.prev_theta_raw = theta_raw

        omega = self.link.GetAngVelLocal().z
        dt = max(1e-9, time_now - self.prev_time)
        alpha = (omega - self.prev_omega) / dt
        self.prev_omega = omega
        self.prev_time = time_now

        cmd_used = clamp(self.cmd_source.get_command(), -self.cfg.pwm_limit, self.cfg.pwm_limit)

        self.pub_theta.publish(Float32(data=float(self.theta_unwrap)))
        self.pub_omega.publish(Float32(data=float(omega)))
        self.pub_alpha.publish(Float32(data=float(alpha)))
        self.pub_cmd_used.publish(Float32(data=float(cmd_used)))
        self.pub_tau.publish(Float32(data=float(self.last_tau)))
        return True


# ============================================================
# csv logger
# ============================================================

class CsvLogger:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.csv_path = make_numbered_csv_path(cfg.log_dir, cfg.log_prefix)
        self.fp = open(self.csv_path, "w", newline="")
        self.wr = csv.writer(self.fp)
        self.wr.writerow([
            "wall_time",
            "sim_time",
            "mode",
            "cmd_u",
            "tau_cmd",
            "theta",
            "omega",
            "alpha",
            "hw_pwm",
            "hw_enc",
            "hw_arduino_ms",
            "imu_wz",
            "imu_ax",
            "imu_ay",
            "imu_az",
        ])
        self.last_log_t = -1e9
        self.log_dt = 1.0 / max(1e-9, cfg.log_hz)

    def maybe_log(
        self,
        sim_time: float,
        mode: str,
        cmd_u: float,
        tau_cmd: float,
        theta: float,
        omega: float,
        alpha: float,
        feedback: FeedbackMonitorHandler,
    ):
        if sim_time - self.last_log_t < self.log_dt:
            return

        self.last_log_t = sim_time
        self.wr.writerow([
            now_wall(),
            sim_time,
            mode,
            cmd_u,
            tau_cmd,
            theta,
            omega,
            alpha,
            feedback.hw_pwm,
            feedback.hw_enc,
            feedback.hw_arduino_ms,
            feedback.imu_wz,
            feedback.imu_ax,
            feedback.imu_ay,
            feedback.imu_az,
        ])

    def close(self):
        if self.fp:
            self.fp.flush()
            self.fp.close()
            self.fp = None


# ============================================================
# build pendulum
# ============================================================

def add_axes_visual(sys_ch, axis_len=0.12, axis_thk=0.002):
    axes = ch.ChBody()
    axes.SetFixed(True)
    sys_ch.Add(axes)

    x_box = ch.ChVisualShapeBox(axis_len, axis_thk, axis_thk)
    x_box.SetColor(ch.ChColor(1, 0, 0))
    axes.AddVisualShape(x_box, ch.ChFramed(ch.ChVector3d(axis_len / 2.0, 0, 0), ch.QUNIT))

    y_box = ch.ChVisualShapeBox(axis_thk, axis_len, axis_thk)
    y_box.SetColor(ch.ChColor(0, 1, 0))
    axes.AddVisualShape(y_box, ch.ChFramed(ch.ChVector3d(0, axis_len / 2.0, 0), ch.QUNIT))

    z_box = ch.ChVisualShapeBox(axis_thk, axis_thk, axis_len)
    z_box.SetColor(ch.ChColor(0, 0, 1))
    axes.AddVisualShape(z_box, ch.ChFramed(ch.ChVector3d(0, 0, axis_len / 2.0), ch.QUNIT))


def build_pendulum(cfg: BridgeConfig):
    sys_ch = ch.ChSystemNSC()
    sys_ch.SetGravitationalAcceleration(cfg.gravity)
    add_axes_visual(sys_ch)

    motor_body = ch.ChBody()
    motor_body.SetFixed(True)
    sys_ch.Add(motor_body)

    q_cyl_to_x = ch.QuatFromAngleZ(-math.pi / 2.0)

    motor_cyl = ch.ChVisualShapeCylinder(cfg.motor_radius, cfg.motor_length)
    motor_cyl.SetColor(ch.ChColor(0.05, 0.05, 0.05))
    motor_body.AddVisualShape(
        motor_cyl,
        ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), q_cyl_to_x),
    )

    shaft = ch.ChVisualShapeCylinder(cfg.shaft_radius, cfg.shaft_length)
    shaft.SetColor(ch.ChColor(0.05, 0.05, 0.05))
    motor_body.AddVisualShape(
        shaft,
        ch.ChFramed(
            ch.ChVector3d(cfg.motor_length / 2.0 + cfg.shaft_length / 2.0, 0.0, 0.0),
            q_cyl_to_x,
        ),
    )

    link = ch.ChBody()
    link.SetMass(cfg.link_mass)

    izz_com = (1.0 / 12.0) * cfg.link_mass * (cfg.link_L ** 2 + cfg.link_W ** 2)
    izz_pivot = izz_com + cfg.link_mass * (cfg.link_L / 2.0) ** 2
    link.SetInertiaXX(ch.ChVector3d(1e-5, 1e-5, izz_pivot))

    link.SetPos(ch.ChVector3d(0.0, 0.0, cfg.motor_length / 2.0))
    link.SetRot(ch.QuatFromAngleZ(math.radians(cfg.theta0_deg)))
    link.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, cfg.omega0))
    sys_ch.Add(link)

    link_vis = ch.ChVisualShapeBox(cfg.link_W, cfg.link_L, cfg.link_T)
    link_vis.SetColor(ch.ChColor(0.93, 0.93, 0.93))
    link.AddVisualShape(
        link_vis,
        ch.ChFramed(ch.ChVector3d(0.0, -cfg.link_L / 2.0, 0.0), ch.QUNIT),
    )

    imu = ch.ChBody()
    imu.SetMass(cfg.imu_mass)
    imu.SetInertiaXX(ch.ChVector3d(1e-6, 1e-6, 1e-6))

    imu_local = ch.ChVector3d(0.0, -cfg.link_L + cfg.imu_size_y / 2.0, 0.0)
    imu_abs = link.TransformPointLocalToParent(imu_local)
    imu.SetPos(imu_abs)
    imu.SetRot(link.GetRot())
    sys_ch.Add(imu)

    imu_vis = ch.ChVisualShapeBox(cfg.imu_size_x, cfg.imu_size_y, cfg.imu_size_z)
    imu_vis.SetColor(ch.ChColor(0.60, 0.60, 0.60))
    imu.AddVisualShape(imu_vis)

    # 핵심 수정: absolute frame으로 lock
    fix_frame_abs = ch.ChFramed(imu_abs, link.GetRot())
    fix_imu = ch.ChLinkLockLock()
    fix_imu.Initialize(imu, link, fix_frame_abs)
    sys_ch.Add(fix_imu)

    motor = ch.ChLinkMotorRotationTorque()
    motor.Initialize(link, motor_body, ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), ch.QUNIT))

    tau_fun = ch.ChFunctionConst(0.0)
    motor.SetTorqueFunction(tau_fun)
    sys_ch.Add(motor)

    return sys_ch, link, imu, tau_fun


def build_visuals(cfg: BridgeConfig, sys_ch):
    if not cfg.enable_render:
        return None

    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(sys_ch)
    vis.SetWindowSize(cfg.win_w, cfg.win_h)
    vis.SetWindowTitle(cfg.window_title)
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddTypicalLights()
    vis.AddCamera(ch.ChVector3d(0.45, -0.15, 0.65), ch.ChVector3d(0.0, -0.12, 0.0))
    return vis


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["host", "jetson"], default="host")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    cfg = BridgeConfig(enable_render=not args.headless)

    sys_ch, link, imu, tau_fun = build_pendulum(cfg)
    vis = build_visuals(cfg, sys_ch)

    ros_manager = chros.ChROSPythonManager()
    ros_manager.RegisterHandler(chros.ChROSClockHandler())

    if args.mode == "host":
        cmd_source = HostCommandPublisherHandler(cfg)
    else:
        cmd_source = JetsonCommandSubscriberHandler(cfg)

    feedback = FeedbackMonitorHandler(cfg)
    sim_state = SimStatePublisherHandler(cfg, link, cmd_source)

    ros_manager.RegisterPythonHandler(cmd_source)
    ros_manager.RegisterPythonHandler(feedback)
    ros_manager.RegisterPythonHandler(sim_state)
    ros_manager.Initialize()

    logger = CsvLogger(cfg)
    realtime_timer = ch.ChRealtimeStepTimer()

    console_dt = 1.0 / cfg.console_hz
    last_console_wall = 0.0

    prev_omega = link.GetAngVelLocal().z
    prev_t = 0.0

    try:
        while True:
            if vis is not None and not vis.Run():
                break

            cmd_u = clamp(cmd_source.get_command(), -cfg.pwm_limit, cfg.pwm_limit)
            tau_cmd = cfg.pwm_to_tau_gain * cmd_u
            tau_fun.SetConstant(tau_cmd)
            sim_state.set_last_tau(tau_cmd)

            sys_ch.DoStepDynamics(cfg.step)
            sim_t = sys_ch.GetChTime()

            if not ros_manager.Update(sim_t, cfg.step):
                break

            theta = yaw_from_quat(link.GetRot())
            omega = link.GetAngVelLocal().z
            alpha = (omega - prev_omega) / max(1e-9, sim_t - prev_t)
            prev_omega = omega
            prev_t = sim_t

            logger.maybe_log(
                sim_t,
                cmd_source.get_mode_name(),
                cmd_u,
                tau_cmd,
                theta,
                omega,
                alpha,
                feedback,
            )

            now = time.perf_counter()
            if now - last_console_wall >= console_dt:
                terminal_status_line(
                    f"[{args.mode}] "
                    f"src_mode={cmd_source.get_mode_name():<10} "
                    f"cmd_u={cmd_u:+7.2f} "
                    f"tau={tau_cmd:+8.4f} "
                    f"theta={theta:+8.4f} "
                    f"omega={omega:+8.4f} "
                    f"hw_pwm={feedback.hw_pwm:+7.2f} "
                    f"hw_enc={feedback.hw_enc:+10.2f} "
                    f"imu_wz={feedback.imu_wz:+8.4f}"
                )
                last_console_wall = now

            if vis is not None:
                vis.BeginScene()
                vis.Render()
                vis.EndScene()

            realtime_timer.Spin(cfg.step)

    except KeyboardInterrupt:
        pass
    finally:
        if isinstance(cmd_source, HostCommandPublisherHandler):
            try:
                cmd_source.Shutdown()
            except Exception:
                pass

        logger.close()
        print(f"\nSaved log: {logger.csv_path}")


if __name__ == "__main__":
    main()
