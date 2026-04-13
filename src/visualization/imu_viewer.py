"""Realtime IMU topic viewer for pendulum trajectory monitoring."""

from __future__ import annotations

import argparse
from collections import deque
import math
from typing import Deque

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


def quaternion_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert quaternion orientation into pitch angle in radians."""
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


class ImuTrajectoryViewer(Node):
    """ROS2 node that subscribes to IMU and updates a live trajectory plot."""

    def __init__(self, topic: str, window_s: float) -> None:
        super().__init__("imu_trajectory_viewer")
        self.window_s = window_s
        self.t_rel: Deque[float] = deque()
        self.theta: Deque[float] = deque()
        self.omega: Deque[float] = deque()
        self.t0: float | None = None
        self.create_subscription(Imu, topic, self.on_imu, 20)
        self.get_logger().info(f"Subscribed to {topic}. Displaying realtime pendulum trajectory.")

    def on_imu(self, msg: Imu) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        t_rel = t - self.t0
        theta = quaternion_to_pitch(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        omega = msg.angular_velocity.y

        self.t_rel.append(t_rel)
        self.theta.append(theta)
        self.omega.append(omega)
        self._trim_old_samples(t_rel)

    def _trim_old_samples(self, latest_t: float) -> None:
        horizon = latest_t - self.window_s
        while self.t_rel and self.t_rel[0] < horizon:
            self.t_rel.popleft()
            self.theta.popleft()
            self.omega.popleft()


def main() -> None:
    """Run realtime IMU visualization from ROS2 topic."""
    parser = argparse.ArgumentParser(description="Realtime IMU trajectory viewer")
    parser.add_argument("--topic", default="/imu/data", help="IMU topic (sensor_msgs/Imu)")
    parser.add_argument("--window", type=float, default=10.0, help="Visible rolling window [s]")
    args = parser.parse_args()

    rclpy.init()
    node = ImuTrajectoryViewer(topic=args.topic, window_s=args.window)

    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    theta_line, = axes[0].plot([], [], label="theta")
    omega_line, = axes[1].plot([], [], label="omega")
    axes[0].set_ylabel("theta [rad]")
    axes[1].set_ylabel("omega [rad/s]")
    axes[1].set_xlabel("time [s]")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not node.t_rel:
                plt.pause(0.01)
                continue
            x = list(node.t_rel)
            theta_line.set_data(x, list(node.theta))
            omega_line.set_data(x, list(node.omega))
            for ax in axes:
                ax.set_xlim(max(0.0, x[-1] - args.window), x[-1] + 1e-6)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close(fig)


if __name__ == "__main__":
    main()
