import math

import numpy as np
import pychrono as ch

from .config import BridgeConfig
from .utils import quat_to_np, clamp, sanitize_float

EARTH_GRAVITY_MPS2 = 9.81


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
        # Use physical Earth gravity in Chrono dynamics.
        # cfg.gravity can be calibrated for IMU gravity-compensation, but should not alter physics gravity here.
        self.sys.SetGravitationalAcceleration(ch.ChVector3d(0.0, -EARTH_GRAVITY_MPS2, 0.0))
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

        self.link = self._create_easy_box(
            size_x=cfg.link_W,
            size_y=cfg.link_L,
            size_z=cfg.link_T,
            mass=cfg.rod_mass,
            color=ch.ChColor(0.93, 0.93, 0.93),
        )
        theta0 = math.radians(cfg.theta0_deg)
        q_link = ch.QuatFromAngleZ(theta0)
        self.link.SetRot(q_link)
        com_x = -math.sin(theta0) * (cfg.link_L / 2.0)
        com_y = -math.cos(theta0) * (cfg.link_L / 2.0)
        self.link.SetPos(ch.ChVector3d(com_x, com_y, cfg.motor_length / 2.0))
        self.link.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, cfg.omega0))
        self.sys.Add(self.link)

        self.imu = self._create_easy_box(
            size_x=cfg.imu_size_x,
            size_y=cfg.imu_size_y,
            size_z=cfg.imu_size_z,
            mass=cfg.imu_mass,
            color=ch.ChColor(0.60, 0.60, 0.60),
        )
        imu_com_local = self._imu_com_local()
        imu_abs = self.link.TransformPointLocalToParent(imu_com_local)
        self.imu.SetPos(imu_abs)
        self.imu.SetRot(self.link.GetRot())
        self.sys.Add(self.imu)

        fix_frame_abs = ch.ChFramed(imu_abs, self.link.GetRot())
        self.fix_imu = ch.ChLinkLockLock()
        self.fix_imu.Initialize(self.imu, self.link, fix_frame_abs)
        self.sys.Add(self.fix_imu)

        self.motor = ch.ChLinkMotorRotationTorque()
        self.motor.Initialize(self.link, self.base, ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), ch.QUNIT))
        self.tau_fun = ch.ChFunctionConst(0.0)
        self.motor.SetTorqueFunction(self.tau_fun)
        self.sys.Add(self.motor)
        self._torque_model = None

        self.prev_sensor_vel = np.zeros(3, dtype=float)
        self.prev_t = None

    def pivot_pos_world(self):
        return np.array([0.0, 0.0, float(self.cfg.motor_length / 2.0)], dtype=float)

    def rod_com_pos_world(self):
        p = self.link.GetPos()
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=float)

    def imu_com_pos_world(self):
        p = self.imu.GetPos()
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=float)

    def imu_local_on_rod(self):
        p_local = self.link.TransformPointParentToLocal(self.imu.GetPos())
        return np.array([float(p_local.x), float(p_local.y), float(p_local.z)], dtype=float)

    def rod_com_radius_from_pivot(self):
        return float(self.cfg.link_L / 2.0)

    def imu_radius_from_pivot(self):
        return float(np.linalg.norm(self.imu_com_pos_world() - self.pivot_pos_world()))

    def total_com_pos_world(self):
        mr = float(self.link.GetMass())
        mi = float(self.imu.GetMass())
        pr = self.rod_com_pos_world()
        pi = self.imu_com_pos_world()
        return (mr * pr + mi * pi) / max(mr + mi, 1e-12)

    def total_l_com_from_pivot(self):
        return float(np.linalg.norm(self.total_com_pos_world() - self.pivot_pos_world()))

    @property
    def m_total(self):
        return float(self.link.GetMass() + self.imu.GetMass())

    @property
    def J_rod(self):
        return float(self.link.GetInertiaXX().z)

    @property
    def J_imu(self):
        return float(self.imu.GetInertiaXX().z)

    @property
    def J_total(self):
        return float(self.J_rod + self.J_imu)

    def _create_easy_box(self, size_x, size_y, size_z, mass, color):
        vol = max(float(size_x) * float(size_y) * float(size_z), 1e-12)
        density = max(float(mass), 1e-9) / vol
        body = ch.ChBodyEasyBox(float(size_x), float(size_y), float(size_z), density, True, False)
        body.SetMass(max(float(mass), 1e-9))
        if hasattr(body, "GetVisualShape"):
            shape0 = body.GetVisualShape(0)
            if shape0 is not None:
                shape0.SetColor(color)
        return body

    def _imu_com_local(self):
        # r_imu is the calibrated pivot->IMU-center radius.
        return ch.ChVector3d(0.0, -float(self.cfg.r_imu), 0.0)

    def update_identified_structure(self, params: dict):
        _ = params
        # Mass/COM/inertia are derived from geometry+density via ChBodyEasyBox.

    def get_theta(self):
        d = self.link.TransformDirectionLocalToParent(ch.ChVector3d(0.0, -1.0, 0.0))
        return math.atan2(float(d.x), -float(d.y))

    def get_omega(self):
        return float(self.link.GetAngVelLocal().z)

    def set_theta_kinematic(self, theta_rad: float, omega_rad_s: float = 0.0):
        theta = float(theta_rad)
        q_link = ch.QuatFromAngleZ(theta)
        com_x = -math.sin(theta) * (self.cfg.link_L / 2.0)
        com_y = -math.cos(theta) * (self.cfg.link_L / 2.0)
        self.link.SetRot(q_link)
        self.link.SetPos(ch.ChVector3d(com_x, com_y, self.cfg.motor_length / 2.0))
        self.link.SetPosDt(ch.ChVector3d(0.0, 0.0, 0.0))
        self.link.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, float(omega_rad_s)))
        imu_com_local = self._imu_com_local()
        imu_abs = self.link.TransformPointLocalToParent(imu_com_local)
        self.imu.SetPos(imu_abs)
        self.imu.SetRot(q_link)
        self.imu.SetPosDt(ch.ChVector3d(0.0, 0.0, 0.0))
        self.imu.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, float(omega_rad_s)))

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

    def set_torque_model(self, fn):
        """Set callable torque model: fn(theta, omega, context_dict) -> tau_z."""
        self._torque_model = fn

    def apply_modeled_torque(self, theta, omega, context=None):
        if self._torque_model is None:
            return 0.0
        ctx = {} if context is None else dict(context)
        tau = sanitize_float(self._torque_model(float(theta), float(omega), ctx))
        self.apply_torque(tau)
        return float(tau)

    def step(self, h):
        self.sys.DoStepDynamics(h)


def compute_model_torque_and_electrics(motor_input, theta, omega, bus_v, p, cfg: BridgeConfig, cmd_u_for_duty=None):
    _ = bus_v
    motor_gain = float(p.get("K_i", cfg.K_i_init))
    tau_motor = motor_gain * float(motor_input)
    tau_visc = p["b_eq"] * omega
    tau_coul = p["tau_eq"] * math.tanh(omega / max(cfg.tanh_eps, 1e-9))
    tau_res = tau_visc + tau_coul
    tau_gravity = 0.0
    tau_net = tau_motor - tau_res
    duty_input = 0.0 if cmd_u_for_duty is None else float(cmd_u_for_duty)
    return {
        "duty": clamp(duty_input / max(cfg.pwm_limit, 1e-9), -1.0, 1.0),
        "tau_motor": tau_motor,
        "tau_visc": tau_visc,
        "tau_coul": tau_coul,
        "tau_res": tau_res,
        "tau_gravity": tau_gravity,
        "tau_net": tau_net,
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
