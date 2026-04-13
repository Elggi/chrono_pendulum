"""Chrono pendulum construction with geometry and density-driven bodies."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .config import PendulumPlantConfig


@dataclass
class ChronoPendulumPlant:
    """Physics-grounded pendulum model using Project Chrono rigid bodies.

    Gravity, mass, inertia, and joint dynamics are delegated to Chrono.
    Actuation enters through ``ChLinkMotorRotationTorque``.
    """

    config: PendulumPlantConfig

    def build(self) -> dict[str, Any]:
        """Construct Chrono system and pendulum bodies.

        Returns a dictionary of key Chrono objects to allow downstream modules
        to attach controllers, loggers, and calibration wrappers.
        """
        chrono = import_module("pychrono")
        system = chrono.ChSystemNSC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, -self.config.gravity_m_s2, 0))

        ground = chrono.ChBodyEasyBox(0.05, 0.05, 0.05, 1000, False, False)
        ground.SetFixed(True)
        system.Add(ground)

        rod = chrono.ChBodyEasyCylinder(
            self.config.rod.radius_m,
            self.config.rod.length_m,
            self.config.rod.density_kg_m3,
            True,
            True,
        )
        rod.SetPos(chrono.ChVector3d(0, -self.config.rod.offset_m, 0))
        system.Add(rod)

        imu = chrono.ChBodyEasyBox(
            self.config.imu.length_m,
            self.config.imu.radius_m,
            self.config.imu.radius_m,
            self.config.imu.density_kg_m3,
            True,
            True,
        )
        imu.SetPos(chrono.ChVector3d(0, -self.config.imu.offset_m, 0))
        system.Add(imu)

        hub = chrono.ChBodyEasyCylinder(
            self.config.hub.radius_m,
            self.config.hub.length_m,
            self.config.hub.density_kg_m3,
            True,
            True,
        )
        hub.SetPos(chrono.ChVector3d(0, 0, 0))
        system.Add(hub)

        rod_imu_weld = chrono.ChLinkLockLock()
        rod_imu_weld.Initialize(rod, imu, chrono.ChFrameD(imu.GetPos()))
        system.AddLink(rod_imu_weld)

        rod_hub_weld = chrono.ChLinkLockLock()
        rod_hub_weld.Initialize(rod, hub, chrono.ChFrameD(hub.GetPos()))
        system.AddLink(rod_hub_weld)

        revolute = chrono.ChLinkLockRevolute()
        revolute.Initialize(ground, rod, chrono.ChFrameD(chrono.ChVector3d(0, 0, 0)))
        system.AddLink(revolute)

        motor = chrono.ChLinkMotorRotationTorque()
        motor.Initialize(ground, rod, chrono.ChFrameD(chrono.ChVector3d(0, 0, 0)))
        system.AddLink(motor)

        return {
            "system": system,
            "ground": ground,
            "rod": rod,
            "imu": imu,
            "hub": hub,
            "joint": revolute,
            "motor": motor,
        }
