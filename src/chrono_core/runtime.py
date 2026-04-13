"""Chrono runtime entrypoint for sanity checks and short simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
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
    parser.add_argument("--current-topic", default="/ina219/current_ma")
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
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float32
        except ImportError as exc:
            raise RuntimeError("collect-excitation mode requires ROS2 Python packages (rclpy/std_msgs).") from exc

        class CurrentBuffer(Node):
            def __init__(self, topic: str) -> None:
                super().__init__("chrono_current_ingest")
                self.current_ma = 0.0
                self.create_subscription(Float32, topic, self.on_current, 20)

            def on_current(self, msg: Float32) -> None:
                self.current_ma = float(msg.data)

        rclpy.init()
        node = CurrentBuffer(args.current_topic)
        # placeholder conversion from current to torque
        current_to_torque = 1e-3
        steps = int(args.seconds / cfg.timestep_s)
        torque_fun = None
        if hasattr(objects["motor"], "SetMotorTorque"):
            torque_fun = None
        else:
            torque_fun = objects["motor"].GetTorqueFunction()
        for k in range(steps):
            rclpy.spin_once(node, timeout_sec=0.0)
            t = k * cfg.timestep_s
            torque = current_to_torque * node.current_ma
            if hasattr(objects["motor"], "SetMotorTorque"):
                objects["motor"].SetMotorTorque(torque + 0.0 * math.sin(t))
            elif torque_fun is not None and hasattr(torque_fun, "SetConstant"):
                torque_fun.SetConstant(torque)
            system.DoStepDynamics(cfg.timestep_s)
        node.destroy_node()
        rclpy.shutdown()
        print(f"Collected excitation run for {args.seconds:.3f}s using {args.current_topic}.")


if __name__ == "__main__":
    main()
