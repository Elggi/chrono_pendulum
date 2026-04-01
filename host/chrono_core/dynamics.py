import math

import numpy as np
import pychrono as ch

from .config import BridgeConfig
from .utils import quat_to_np, clamp, sanitize_float


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
        self.link.SetPos(ch.ChVector3d(0.0, 0.0, cfg.motor_length / 2.0))
        self.link.SetRot(ch.QuatFromAngleZ(math.radians(cfg.theta0_deg)))
        self.sys.Add(self.link)
        self.link.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, cfg.omega0))

        self._apply_dynamic_properties()

        vis_link = ch.ChVisualShapeBox(cfg.link_W, cfg.link_L, cfg.link_T)
        vis_link.SetColor(ch.ChColor(0.93, 0.93, 0.93))
        self.link.AddVisualShape(vis_link, ch.ChFramed(ch.ChVector3d(0.0, -cfg.link_L / 2.0, 0.0), ch.QUNIT))

        self.imu = ch.ChBody()
        self.imu.SetMass(1e-6)
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

    @property
    def m_total(self):
        return float(self.cfg.rod_mass + self.cfg.imu_mass)

    @property
    def J_rod(self):
        return float((1.0 / 3.0) * self.cfg.rod_mass * (self.cfg.rod_length ** 2))

    @property
    def J_imu(self):
        return float(self.cfg.imu_mass * (self.cfg.r_imu ** 2))

    @property
    def J_total(self):
        return float(self.J_rod + self.J_imu)

    def _apply_dynamic_properties(self):
        """Apply geometry-based rigid-body properties with fixed inertia."""
        self.link.SetMass(max(self.m_total, 1e-6))
        self.link.SetInertiaXX(ch.ChVector3d(1e-5, 1e-5, max(self.J_total, 1e-8)))

    def update_identified_structure(self, params: dict):
        # l_cg is identified only in gravity torque; inertia remains fixed by geometry.
        self._apply_dynamic_properties()

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


def compute_model_torque_and_electrics(cmd_u, theta, omega, bus_v, p, cfg: BridgeConfig):
    _ = bus_v
    tau_motor = p["K_u"] * cmd_u
    tau_visc = p["b_eq"] * omega
    tau_coul = p["tau_eq"] * math.tanh(omega / max(cfg.tanh_eps, 1e-9))
    tau_res = tau_visc + tau_coul
    tau_gravity = (cfg.rod_mass + cfg.imu_mass) * cfg.gravity * p["l_com"] * math.sin(theta)
    tau_net = tau_motor - tau_res - tau_gravity
    return {
        "duty": clamp(cmd_u / max(cfg.pwm_limit, 1e-9), -1.0, 1.0),
        "tau_motor": tau_motor,
        "tau_visc": tau_visc,
        "tau_coul": tau_coul,
        "tau_res": tau_res,
        "tau_gravity": tau_gravity,
        "tau_net": tau_net,
    }


def blend_parameters_for_sim(ekf_params: dict, cfg: BridgeConfig):
    b_eq = float(ekf_params.get("b_eq", ekf_params.get("b", cfg.b_eq_init)))
    tau_eq = float(ekf_params.get("tau_eq", ekf_params.get("tau_c", cfg.tau_eq_init)))
    return {
        "theta": float(ekf_params["theta"]),
        "omega": float(ekf_params["omega"]),
        "l_com": float(ekf_params.get("l_com", cfg.l_com_init)),
        "b_eq": b_eq,
        "tau_eq": tau_eq,
        "K_u": float(ekf_params.get("K_u", ekf_params.get("k_u", cfg.K_u_init))),
        "delay_sec": float(ekf_params["delay_sec"]),
    }


def enc_to_theta(enc_count: float, enc_ref: float, theta_ref: float, cpr: float | None):
    if cpr is None:
        return None
    try:
        cpr = float(cpr)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cpr) or cpr <= 1.0:
        return None
    return float(theta_ref + (2.0 * math.pi / cpr) * (enc_count - enc_ref))
