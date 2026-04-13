"""Host-side PWM publisher for /cmd/u excitation collection."""

from __future__ import annotations

import argparse
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class CmdUPublisher(Node):
    """Publish a smooth PWM command on /cmd/u for excitation data."""

    def __init__(self, topic: str, hz: float, amplitude: float) -> None:
        super().__init__("host_cmd_u_publisher")
        self.publisher = self.create_publisher(Float32, topic, 20)
        self.amplitude = amplitude
        self.time_s = 0.0
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self.on_timer)
        self.get_logger().info(f"Publishing PWM on {topic} at {hz:.1f} Hz, amplitude={amplitude:.1f}.")

    def on_timer(self) -> None:
        cmd = Float32()
        cmd.data = float(self.amplitude * math.sin(2.0 * math.pi * 0.5 * self.time_s))
        self.publisher.publish(cmd)
        self.time_s += self.dt


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Host-side /cmd/u PWM publisher")
    parser.add_argument("--topic", default="/cmd/u")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--amplitude", type=float, default=120.0)
    args = parser.parse_args()

    rclpy.init()
    node = CmdUPublisher(topic=args.topic, hz=args.hz, amplitude=args.amplitude)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
