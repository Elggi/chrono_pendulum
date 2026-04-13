"""Configuration objects for the Chrono pendulum digital twin."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class BodyGeometry:
    """Simple geometric body description for rigid body construction."""

    name: str
    shape: str
    density_kg_m3: float
    length_m: float
    radius_m: float
    offset_m: float


@dataclass
class ContactConfig:
    """Contact and material settings for the Chrono plant."""

    friction: float = 0.5
    restitution: float = 0.1
    young_modulus_pa: float = 2e9


@dataclass
class PendulumPlantConfig:
    """Top-level physics configuration for a 1-DOF rotary pendulum."""

    gravity_m_s2: float = 9.81
    timestep_s: float = 0.002
    rod: BodyGeometry = field(default_factory=lambda: BodyGeometry("rod", "cylinder", 2700.0, 0.22, 0.005, 0.11))
    imu: BodyGeometry = field(default_factory=lambda: BodyGeometry("imu", "box", 1100.0, 0.035, 0.015, 0.17))
    hub: BodyGeometry = field(default_factory=lambda: BodyGeometry("hub", "cylinder", 7800.0, 0.018, 0.012, 0.0))
    contact: ContactConfig = field(default_factory=ContactConfig)

    def to_json(self, path: Path) -> None:
        """Persist config snapshot for reproducibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(asdict(self), fp, indent=2)
