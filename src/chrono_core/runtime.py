"""Chrono runtime entrypoint for sanity checks and short simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from .config import BodyGeometry, ContactConfig, PendulumPlantConfig
from .pendulum_base import ChronoPendulumPlant


def load_config(path: Path) -> PendulumPlantConfig:
    """Load pendulum config JSON into a typed config object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PendulumPlantConfig(
        gravity_m_s2=payload.get("gravity_m_s2", 9.81),
        timestep_s=payload.get("timestep_s", 0.002),
        rod=BodyGeometry(**payload["rod"]),
        imu=BodyGeometry(**payload["imu"]),
        hub=BodyGeometry(**payload["hub"]),
        contact=ContactConfig(**payload.get("contact", {})),
    )


def main() -> None:
    """Run chrono build sanity check or short simulation loop."""
    parser = argparse.ArgumentParser(description="Chrono pendulum runtime")
    parser.add_argument("--mode", choices=["sanity", "simulate"], default="sanity")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    plant = ChronoPendulumPlant(cfg)
    objects = plant.build()
    system = objects["system"]

    print("Chrono plant built successfully.")
    print(f"Mode: {args.mode}")
    if args.mode == "simulate":
        steps = int(args.seconds / cfg.timestep_s)
        for _ in range(steps):
            system.DoStepDynamics(cfg.timestep_s)
        print(f"Simulated {args.seconds:.3f}s with dt={cfg.timestep_s}")


if __name__ == "__main__":
    main()
