from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pychrono as ch


@dataclass
class BodySpec:
    mass: float
    size: tuple[float, float, float]
    inertia: tuple[float, float, float]
    com_local: tuple[float, float, float]


@dataclass
class ConnectorSpec:
    mass: float
    diameter: float = 0.011
    length: float = 0.028
    center_from_base_edge: float = 0.010


@dataclass
class MotorTorqueParams:
    equation: str
    K_I: float
    b: float
    tau_c: float
    tanh_eps: float


@dataclass
class PendulumParams:
    rod: BodySpec
    imu: BodySpec
    connector: ConnectorSpec
    motor: MotorTorqueParams
    gravity: float


def _default_motor_torque_data() -> dict:
    return {
        "units": {"length": "m", "mass": "kg", "inertia": "kg*m^2", "current": "A", "torque": "N*m"},
        "gravity": 9.81,
        "equation": {
            "active": "tau = K_I * I - b * omega - tau_c * tanh(omega/tanh_eps)",
            "residual_ready": "tau = K_I * I + tau_residual(theta, omega, I)",
        },
        "parameters": {"K_I": 0.06, "b": 0.01, "tau_c": 0.0, "tanh_eps": 0.05},
        "rod": {
            "mass": 0.20,
            "length": 0.285,
            "width": 0.020,
            "height": 0.006,
            "inertia": [1.0e-5, 1.0e-3, 1.0e-3],
            "com_local": [0.0, -0.1425, 0.0],
        },
        "imu": {
            "mass": 0.020,
            "length": 0.0595,
            "width": 0.0460,
            "height": 0.0117,
            "inertia": [1.0e-5, 1.0e-5, 1.0e-5],
            "com_local": [0.0, 0.0, 0.0],
        },
        "connector_cyl": {
            "name": "connector_cyl_body",
            "mass": 0.01,
            "diameter": 0.011,
            "length": 0.028,
            "center_from_rod_base_edge": 0.010,
        },
    }


def ensure_motor_torque_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text(json.dumps(_default_motor_torque_data(), indent=2), encoding="utf-8")


def load_pendulum_params(path: Path, imu_radius: float) -> PendulumParams:
    ensure_motor_torque_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    kgm = data.get("known_geometry_mass", {})
    rod = data.get("rod", kgm.get("rod", {}))
    imu = data.get("imu", kgm.get("imu", {}))
    c = data.get("connector_cyl", kgm.get("connector_cyl", {}))
    p = data.get("dynamic_parameters", data.get("parameters", {}))

    imu_local = [0.0, -float(imu_radius), 0.0]
    imu["com_local"] = imu_local

    return PendulumParams(
        rod=BodySpec(
            mass=float(rod["mass"]),
            size=(float(rod["width"]), float(rod["length"]), float(rod["height"])),
            inertia=tuple(float(v) for v in rod["inertia"]),
            com_local=tuple(float(v) for v in rod["com_local"]),
        ),
        imu=BodySpec(
            mass=float(imu["mass"]),
            size=(float(imu["width"]), float(imu["length"]), float(imu["height"])),
            inertia=tuple(float(v) for v in imu["inertia"]),
            com_local=tuple(float(v) for v in imu["com_local"]),
        ),
        connector=ConnectorSpec(
            mass=float(c.get("mass", 0.01)),
            diameter=float(c["diameter"]),
            length=float(c["length"]),
            center_from_base_edge=float(c["center_from_rod_base_edge"]),
        ),
        motor=MotorTorqueParams(
            equation=str(data["equation"]["active"]),
            K_I=float(p.get("K_I", 0.0)),
            b=float(p.get("b_eq", p.get("b", 0.0))),
            tau_c=float(p.get("tau_eq", p.get("tau_c", 0.0))),
            tanh_eps=float(p.get("tanh_eps", 0.05)),
        ),
        gravity=float(data.get("gravity", 9.81)),
    )


class TorqueController:
    def __init__(self, params: MotorTorqueParams, residual: Callable[[float, float, float], float] | None = None):
        self.params = params
        self.residual = residual

    def compute(self, current_a: float, theta: float, omega: float) -> float:
        tau = self.params.K_I * current_a
        tau -= self.params.b * omega
        tau -= self.params.tau_c * math.tanh(omega / max(self.params.tanh_eps, 1e-6))
        if self.residual:
            tau += float(self.residual(theta, omega, current_a))
        return float(tau)


class ChronoRigidPendulum:
    """True Chrono rigid-body pendulum: rod + imu + connector fixed together, motor revolute at pivot."""

    def __init__(self, params: PendulumParams, enable_collision: bool = False):
        self.params = params
        self.sys = ch.ChSystemNSC()
        self.sys.SetGravitationalAcceleration(ch.ChVector3d(0.0, -params.gravity, 0.0))

        self.ground = ch.ChBody()
        self.ground.SetFixed(True)
        self.sys.Add(self.ground)

        self.rod_body = ch.ChBody()
        self.rod_body.SetName("rod_body")
        self.rod_body.SetMass(params.rod.mass)
        self.rod_body.SetInertiaXX(ch.ChVector3d(*params.rod.inertia))
        self.rod_body.SetPos(ch.ChVector3d(0.0, -params.rod.size[1] / 2.0, 0.0))
        self.rod_body.SetCollide(enable_collision)
        self.rod_body.AddVisualShape(ch.ChVisualShapeBox(*params.rod.size), ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), ch.QUNIT))
        self.rod_body.AddCollisionShape(ch.ChCollisionShapeBox(self.sys.GetContactMaterialNSC(), *params.rod.size))
        self.sys.Add(self.rod_body)

        self.imu_body = ch.ChBody()
        self.imu_body.SetName("imu_body")
        self.imu_body.SetMass(params.imu.mass)
        self.imu_body.SetInertiaXX(ch.ChVector3d(*params.imu.inertia))
        self.imu_body.SetPos(ch.ChVector3d(*params.imu.com_local))
        self.imu_body.SetCollide(enable_collision)
        self.imu_body.AddVisualShape(ch.ChVisualShapeBox(*params.imu.size), ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), ch.QUNIT))
        self.imu_body.AddCollisionShape(ch.ChCollisionShapeBox(self.sys.GetContactMaterialNSC(), *params.imu.size))
        self.sys.Add(self.imu_body)

        connector_y = -params.connector.center_from_base_edge
        self.connector_cyl_body = ch.ChBody()
        self.connector_cyl_body.SetName("connector_cyl_body")
        self.connector_cyl_body.SetMass(params.connector.mass)
        self.connector_cyl_body.SetInertiaXX(ch.ChVector3d(1e-6, 1e-6, 1e-6))
        self.connector_cyl_body.SetPos(ch.ChVector3d(0.0, connector_y, 0.0))
        self.connector_cyl_body.SetCollide(enable_collision)
        self.connector_cyl_body.AddVisualShape(ch.ChVisualShapeCylinder(params.connector.diameter / 2.0, params.connector.length))
        self.connector_cyl_body.AddCollisionShape(ch.ChCollisionShapeCylinder(self.sys.GetContactMaterialNSC(), params.connector.diameter / 2.0, params.connector.length))
        self.sys.Add(self.connector_cyl_body)

        for child in (self.imu_body, self.connector_cyl_body):
            fix = ch.ChLinkLockLock()
            fix.Initialize(child, self.rod_body, ch.ChFramed(child.GetPos(), ch.QUNIT))
            self.sys.Add(fix)

        self.motor = ch.ChLinkMotorRotationTorque()
        self.motor.Initialize(self.rod_body, self.ground, ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), ch.QUNIT))
        self.torque_fun = ch.ChFunctionConst(0.0)
        self.motor.SetTorqueFunction(self.torque_fun)
        self.sys.Add(self.motor)

        self._prev_omega = 0.0
        self._prev_t = 0.0
        self.collision_enabled = enable_collision
        self.collision_ball_dropped = False

    def drop_collision_ball(self) -> None:
        if self.collision_ball_dropped:
            return
        ball = ch.ChBodyEasySphere(0.03, 7800, True, True)
        ball.SetName("test_drop_sphere")
        ball.SetPos(ch.ChVector3d(0.0, -0.15, 0.12))
        ball.SetCollide(True)
        self.sys.Add(ball)
        self.collision_ball_dropped = True

    def read_state(self) -> tuple[float, float, float]:
        d = self.rod_body.TransformDirectionLocalToParent(ch.ChVector3d(0.0, -1.0, 0.0))
        theta = math.atan2(float(d.x), -float(d.y))
        omega = float(self.rod_body.GetAngVelLocal().z)
        t = float(self.sys.GetChTime())
        dt = max(t - self._prev_t, 1e-6)
        alpha = (omega - self._prev_omega) / dt
        self._prev_omega = omega
        self._prev_t = t
        return theta, omega, alpha

    def sync_to_theta(self, theta: float) -> None:
        """Hard-sync pendulum pose/velocity to measured angle (used during free-decay warmup)."""
        q = ch.QuatFromAngleZ(float(theta))
        self.rod_body.SetRot(q)
        self.rod_body.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, 0.0))
        self.imu_body.SetRot(q)
        self.imu_body.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, 0.0))
        self.connector_cyl_body.SetRot(q)
        self.connector_cyl_body.SetAngVelLocal(ch.ChVector3d(0.0, 0.0, 0.0))

    def apply_torque(self, tau: float) -> None:
        self.torque_fun.SetConstant(float(tau))

    def step(self, dt: float) -> None:
        self.sys.DoStepDynamics(float(dt))


class CsvLogger:
    def __init__(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        self.raw = outdir / "chrono_run_raw.csv"
        self.filt = outdir / "chrono_run_filtered.csv"
        import time
        ts = int(time.time())
        self.raw_ts = outdir / f"chrono_run_raw_{ts}.csv"
        self.filt_ts = outdir / f"chrono_run_filtered_{ts}.csv"
        self.cols = ["time", "input_current", "commanded_torque", "theta", "omega", "alpha"]

    def _write(self, p: Path, rows: list[dict]) -> None:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def write(self, rows: list[dict], filtered_rows: list[dict]) -> None:
        self._write(self.raw, rows)
        self._write(self.raw_ts, rows)
        self._write(self.filt, filtered_rows)
        self._write(self.filt_ts, filtered_rows)


def ema_filter(rows: list[dict], alpha: float = 0.1) -> list[dict]:
    if not rows:
        return []
    out: list[dict] = []
    prev = {k: float(rows[0][k]) for k in rows[0] if k != "time"}
    out.append(dict(rows[0]))
    for r in rows[1:]:
        cur = {"time": r["time"]}
        for k in ("input_current", "commanded_torque", "theta", "omega", "alpha"):
            prev[k] = (1.0 - alpha) * prev[k] + alpha * float(r[k])
            cur[k] = prev[k]
        out.append(cur)
    return out
