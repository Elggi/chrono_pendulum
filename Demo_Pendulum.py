import math
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field

import pychrono.core as chrono
import pychrono.irrlicht as irr


@dataclass
class SimConfig:
    # --- Dynamics ---
    step: float = 1e-3
    t_end: float = 60.0
    gravity: chrono.ChVector3d = field(default_factory=lambda: chrono.ChVector3d(0, -9.81, 0))

    # --- Real-time pacing ---
    # 0.0  : 제한 없음(최대한 빠르게) BUT UI를 위해 frame 당 max_steps_per_frame만 수행
    # 1.0  : real-time 목표 (sim_time ~= wall_time)
    # 2.0  : 최대 2배 real-time까지 허용 (가능하면 더 빨리)
    realtime_factor: float = 2.0

    # UI가 굳는 걸 막기 위해 frame당 step 수행량 상한
    max_steps_per_frame: int = 2000

    # --- Rendering (smoothness) ---
    enable_render: bool = True
    render_fps: float = 60.0
    window_title: str = "Chrono Pendulum | Console HUD"
    win_w: int = 1100
    win_h: int = 800

    # --- Console HUD ---
    console_hz: float = 10.0

    # --- Logs ---
    log_dir: str = r"/home/kyudev/Documents/chrono/chrono_logs"
    matlab_prefix: str = "chrono_log"
    flush_every: int = 1000

    # --- Input torque ---
    input_mode: str = "burst"  # sine / square / burst / prbs
    tau_max: float = 0.35
    deadband: float = 0.02
    sine_amp: float = 0.22
    sine_freq_hz: float = 1.0
    square_amp: float = 0.25
    square_period: float = 0.50
    burst_amp: float = 0.30
    burst_on: float = 0.25
    burst_off: float = 0.25
    prbs_amp: float = 0.20
    prbs_dt: float = 0.08
    prbs_seed: int = 12345

    # --- Geometry ---
    motor_radius: float = 0.020
    motor_length: float = 0.050
    shaft_radius: float = 0.003
    shaft_length: float = 0.012

    link_L: float = 0.200
    link_W: float = 0.020
    link_T: float = 0.006
    pla_density: float = 1250.0

    # Weight + IMU visuals
    weight_radius: float = 0.012
    weight_mass: float = 0.035
    weight_r: float = 0.180

    imu_mass: float = 0.010
    imu_size: chrono.ChVector3d = field(default_factory=lambda: chrono.ChVector3d(0.0595, 0.0460, 0.0117))
    imu_r: float = 0.120

    # Initial state
    theta0_deg: float = 25.0
    omega0: float = 0.0


CFG = SimConfig()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q: chrono.ChQuaterniond) -> float:
    w, x, y, z = q.e0, q.e1, q.e2, q.e3
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def apply_deadband_and_saturation(u, deadband, umax):
    if abs(u) < deadband:
        u = 0.0
    return clamp(u, -umax, umax)


def prbs_value(t, dt, seed=12345):
    if dt <= 1e-9:
        return 1.0
    k = int(t / dt)
    x = (1103515245 * (k + seed) + 12345) & 0x7FFFFFFF
    return 1.0 if (x & 1) else -1.0


def commanded_torque(t, C: SimConfig) -> float:
    m = (C.input_mode or "").strip().lower()
    if m == "sine":
        tau = C.sine_amp * math.sin(2 * math.pi * C.sine_freq_hz * t)
    elif m == "square":
        if C.square_period <= 1e-9:
            tau = 0.0
        else:
            tau = C.square_amp if (t % C.square_period) < (0.5 * C.square_period) else -C.square_amp
    elif m == "burst":
        cyc = C.burst_on + C.burst_off
        tau = C.burst_amp if (cyc > 1e-9 and (t % cyc) < C.burst_on) else 0.0
    elif m == "prbs":
        tau = C.prbs_amp * prbs_value(t, C.prbs_dt, C.prbs_seed)
    else:
        raise ValueError("input_mode must be one of: sine, square, burst, prbs")

    return apply_deadband_and_saturation(tau, C.deadband, C.tau_max)


def make_numbered_csv_path(folder: str, prefix: str, ext: str = ".csv") -> str:
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    max_n = 0
    for name in os.listdir(folder):
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return os.path.join(folder, f"{prefix}{max_n + 1}{ext}")


def console_hud(angle_deg, omega, alpha, sim_time, motor_input, rtf, fps):
    s = (
        f"Angle={angle_deg:+8.3f} deg | "
        f"w={omega:+9.4f} | a={alpha:+10.4f} | "
        f"t={sim_time:7.3f}s | u={motor_input:+7.4f}Nm | "
        f"RTF={rtf:5.2f} | FPS={fps:5.1f}"
    )
    sys.stdout.write("\r" + s + " " * 10)
    sys.stdout.flush()


def add_axes_visual(sys_ch, axis_len=0.12, axis_thk=0.002):
    axes = chrono.ChBody()
    axes.SetFixed(True)
    sys_ch.Add(axes)

    x_box = chrono.ChVisualShapeBox(axis_len, axis_thk, axis_thk)
    x_box.SetColor(chrono.ChColor(1, 0, 0))
    axes.AddVisualShape(x_box, chrono.ChFramed(chrono.ChVector3d(axis_len / 2, 0, 0), chrono.QUNIT))

    y_box = chrono.ChVisualShapeBox(axis_thk, axis_len, axis_thk)
    y_box.SetColor(chrono.ChColor(0, 1, 0))
    axes.AddVisualShape(y_box, chrono.ChFramed(chrono.ChVector3d(0, axis_len / 2, 0), chrono.QUNIT))

    z_box = chrono.ChVisualShapeBox(axis_thk, axis_thk, axis_len)
    z_box.SetColor(chrono.ChColor(0, 0, 1))
    axes.AddVisualShape(z_box, chrono.ChFramed(chrono.ChVector3d(0, 0, axis_len / 2), chrono.QUNIT))


def build_system(C: SimConfig):
    sys_ch = chrono.ChSystemNSC()
    sys_ch.SetGravitationalAcceleration(C.gravity)

    add_axes_visual(sys_ch)

    motor_body = chrono.ChBody()
    motor_body.SetFixed(True)
    motor_body.SetPos(chrono.ChVector3d(0, 0, 0))
    sys_ch.Add(motor_body)

    q_cyl_to_x = chrono.QuatFromAngleZ(-math.pi / 2)

    motor_cyl = chrono.ChVisualShapeCylinder(C.motor_radius, C.motor_length)
    motor_cyl.SetColor(chrono.ChColor(0.05, 0.05, 0.05))
    motor_body.AddVisualShape(motor_cyl, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), q_cyl_to_x))

    shaft = chrono.ChVisualShapeCylinder(C.shaft_radius, C.shaft_length)
    shaft.SetColor(chrono.ChColor(0.05, 0.05, 0.05))
    motor_body.AddVisualShape(
        shaft,
        chrono.ChFramed(chrono.ChVector3d(C.motor_length / 2 + C.shaft_length / 2, 0, 0), q_cyl_to_x)
    )

    link = chrono.ChBody()
    m_link = C.pla_density * (C.link_L * C.link_W * C.link_T)
    link.SetMass(m_link)

    Izz_com = (1 / 12) * m_link * (C.link_L ** 2 + C.link_W ** 2)
    Izz_pivot = Izz_com + m_link * (C.link_L / 2) ** 2
    link.SetInertiaXX(chrono.ChVector3d(1e-4, 1e-4, Izz_pivot))

    theta0 = math.radians(C.theta0_deg)
    link.SetPos(chrono.ChVector3d(0, 0, C.motor_length / 2))
    link.SetRot(chrono.QuatFromAngleZ(theta0))
    link.SetAngVelLocal(chrono.ChVector3d(0, 0, C.omega0))
    sys_ch.Add(link)

    link_vis = chrono.ChVisualShapeBox(C.link_L, C.link_W, C.link_T)
    link_vis.SetColor(chrono.ChColor(1, 1, 1))
    link.AddVisualShape(link_vis, chrono.ChFramed(chrono.ChVector3d(C.link_L / 2, 0, 0), chrono.QUNIT))

    # Weight (visual + fixed to link)
    wr = clamp(C.weight_r, 0.02, C.link_L - 0.02)
    weight = chrono.ChBody()
    weight.SetMass(C.weight_mass)
    Iw = (2/5) * C.weight_mass * (C.weight_radius ** 2)
    weight.SetInertiaXX(chrono.ChVector3d(Iw, Iw, Iw))
    w_local = chrono.ChVector3d(wr, 0, 0)
    weight.SetPos(link.TransformPointLocalToParent(w_local))
    sys_ch.Add(weight)

    w_vis = chrono.ChVisualShapeSphere(C.weight_radius)
    w_vis.SetColor(chrono.ChColor(1, 0, 0))
    weight.AddVisualShape(w_vis)

    fix_w = chrono.ChLinkLockLock()
    fix_w.Initialize(weight, link, chrono.ChFramed(w_local, chrono.QUNIT))
    sys_ch.Add(fix_w)

    # IMU (visual + fixed to link)
    ir = clamp(C.imu_r, 0.02, C.link_L - 0.02)
    imu = chrono.ChBody()
    imu.SetMass(C.imu_mass)
    imu_local = chrono.ChVector3d(ir, 0, 0)
    imu.SetPos(link.TransformPointLocalToParent(imu_local))
    imu.SetRot(link.GetRot())
    sys_ch.Add(imu)

    imu_vis = chrono.ChVisualShapeBox(C.imu_size.x, C.imu_size.y, C.imu_size.z)
    imu_vis.SetColor(chrono.ChColor(0.6, 0.6, 0.6))
    imu.AddVisualShape(imu_vis)

    fix_imu = chrono.ChLinkLockLock()
    fix_imu.Initialize(imu, link, chrono.ChFramed(imu_local, chrono.QUNIT))
    sys_ch.Add(fix_imu)

    # Torque motor joint
    motor = chrono.ChLinkMotorRotationTorque()
    motor.Initialize(link, motor_body, chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT))
    tau_fun = chrono.ChFunctionConst(0.0)
    motor.SetTorqueFunction(tau_fun)
    sys_ch.Add(motor)

    return sys_ch, link, tau_fun


def build_visuals(C: SimConfig, sys_ch):
    if not C.enable_render:
        return None

    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(sys_ch)
    vis.SetWindowSize(C.win_w, C.win_h)
    vis.SetWindowTitle(C.window_title)
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddTypicalLights()
    vis.AddCamera(chrono.ChVector3d(0.35, 0.25, 0.55), chrono.ChVector3d(0.10, 0.00, 0.00))
    return vis


def main(C: SimConfig):
    sys_ch, link, tau_fun = build_system(C)
    vis = build_visuals(C, sys_ch)

    log_path = make_numbered_csv_path(C.log_dir, C.matlab_prefix, ".csv")
    fm = open(log_path, "w", newline="")
    wm = csv.writer(fm)
    wm.writerow(["t", "theta_unwrap_rad", "omega_rad_s", "alpha_rad_s2", "tau_cmd_Nm"])

    console_dt = 1.0 / max(C.console_hz, 1e-9)
    render_dt = 1.0 / max(C.render_fps, 1e-9)
    last_console_wall = -1e9
    last_render_wall = -1e9

    prev_theta_raw = yaw_from_quat(link.GetRot())
    theta_unwrap = prev_theta_raw
    prev_omega = link.GetAngVelLocal().z
    prev_t = sys_ch.GetChTime()

    t0_wall = time.perf_counter()
    frames = 0
    fps = 0.0
    last_fps_wall = t0_wall

    try:
        while True:
            # Always pump window events frequently
            if vis is not None and not vis.Run():
                break

            wall_now = time.perf_counter()
            wall_elapsed = wall_now - t0_wall

            t = sys_ch.GetChTime()
            if t >= C.t_end:
                break

            # Target sim time based on realtime_factor
            if C.realtime_factor and C.realtime_factor > 0:
                target_sim = min(C.t_end, C.realtime_factor * wall_elapsed)
            else:
                # unlimited: still limit steps per frame so UI stays responsive
                target_sim = min(C.t_end, t + C.max_steps_per_frame * C.step)

            steps_done = 0
            while t < target_sim and t < C.t_end and steps_done < C.max_steps_per_frame:
                t = sys_ch.GetChTime()

                tau_cmd = commanded_torque(t, C)
                tau_fun.SetConstant(tau_cmd)

                theta_raw = yaw_from_quat(link.GetRot())
                theta_unwrap += wrap_to_pi(theta_raw - prev_theta_raw)

                omega = link.GetAngVelLocal().z
                dt = max(t - prev_t, 1e-12)
                alpha = (omega - prev_omega) / dt

                wm.writerow([t, theta_unwrap, omega, alpha, tau_cmd])

                prev_theta_raw = theta_raw
                prev_omega = omega
                prev_t = t

                sys_ch.DoStepDynamics(C.step)
                steps_done += 1

            if C.flush_every > 0 and (int(sys_ch.GetChTime() / C.step) % C.flush_every == 0):
                fm.flush()

            # Render at fixed FPS (smooth), independent of step loop
            if vis is not None and (wall_now - last_render_wall) >= render_dt:
                last_render_wall = wall_now
                vis.BeginScene()
                vis.Render()
                vis.EndScene()

                frames += 1
                if (wall_now - last_fps_wall) >= 0.5:
                    fps = frames / (wall_now - last_fps_wall)
                    frames = 0
                    last_fps_wall = wall_now

            # Console HUD (fixed rate)
            if (wall_now - last_console_wall) >= console_dt:
                last_console_wall = wall_now
                t = sys_ch.GetChTime()
                tau_now = commanded_torque(t, C)
                omega_now = link.GetAngVelLocal().z
                # alpha is noisy if sampled sparsely; use last computed alpha if available
                # (here we keep alpha from previous loop if steps were done)
                # If no steps were done this frame, alpha stays from previous values.
                angle_deg = theta_unwrap * 180.0 / math.pi
                rtf = (t / wall_elapsed) if wall_elapsed > 1e-9 else 0.0
                console_hud(angle_deg, omega_now, alpha, t, tau_now, rtf, fps)

            # If real-time and sim is ahead, yield a bit (avoid CPU hog + keep UI smooth)
            if C.realtime_factor and C.realtime_factor > 0:
                t = sys_ch.GetChTime()
                if t > C.realtime_factor * wall_elapsed + 0.002:
                    time.sleep(0.001)
            else:
                # unlimited mode: small yield to keep UI responsive in Spyder
                time.sleep(0.0005)

    finally:
        fm.flush()
        fm.close()
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Ensure window closes cleanly (prevents "응답 없음" lingering)
        if vis is not None:
            dev = vis.GetDevice()
            try:
                dev.closeDevice()
            except Exception:
                pass

    print("Saved Chrono log:", log_path)


if __name__ == "__main__":
    main(CFG)
