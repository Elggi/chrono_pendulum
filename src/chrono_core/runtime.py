"""Chrono runtime entrypoint for sanity checks and short simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import csv
import math

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
    parser.add_argument("--mode", choices=["sanity", "simulate", "collect-excitation"], default="sanity")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--current-file", type=Path, default=Path("/tmp/chrono_current_ma.txt"))
    parser.add_argument("--log-csv", type=Path, default=Path("data/raw/chrono_collect_log.csv"))
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
    elif args.mode == "collect-excitation":
        # placeholder conversion from current to torque
        current_to_torque = 1e-3
        steps = int(args.seconds / cfg.timestep_s)
        torque_fun = None
        if hasattr(objects["motor"], "SetMotorTorque"):
            torque_fun = None
        else:
            torque_fun = objects["motor"].GetTorqueFunction()
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.log_csv.open("w", newline="", encoding="utf-8") as fp:
            wr = csv.writer(fp)
            wr.writerow(["t", "current_ma", "torque_input"])
            for k in range(steps):
                current_ma = 0.0
                if args.current_file.exists():
                    raw = args.current_file.read_text(encoding="utf-8").strip()
                    if raw:
                        try:
                            _, value = raw.split(",", 1)
                            current_ma = float(value)
                        except ValueError:
                            current_ma = 0.0
                torque = current_to_torque * current_ma
                if hasattr(objects["motor"], "SetMotorTorque"):
                    objects["motor"].SetMotorTorque(torque)
                elif torque_fun is not None and hasattr(torque_fun, "SetConstant"):
                    torque_fun.SetConstant(torque)
                system.DoStepDynamics(cfg.timestep_s)
                wr.writerow([k * cfg.timestep_s, current_ma, torque])
        print(f"Collected excitation run for {args.seconds:.3f}s using current file {args.current_file}.")
        print(f"Saved runtime collection log: {args.log_csv}")


if __name__ == "__main__":
    main()
