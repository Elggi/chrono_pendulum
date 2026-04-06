#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RViz-oriented replacement scaffold for imu_viewer.py.

Publishes:
- visualization_msgs/MarkerArray for pendulum geometry, encoder arm, and current bar

Subscribes:
- /imu/data (sensor_msgs/Imu)
- /hw/enc (std_msgs/Float32)
- /hw/pwm_applied (std_msgs/Float32)
- /ina219/current_ma (std_msgs/Float32)

Optional:
- interactive markers (if interactive_markers is installed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class VizState:
    theta_imu: float = 0.0
    theta_enc: float = 0.0
    pwm_hw: float = 0.0
    current_ma: float = 0.0
    enc0: float | None = None


class RvizPendulumCalibrationNode(Node):
    def __init__(self):
        super().__init__("rviz_pendulum_calibration")
        self.state = VizState()
        self.cpr = float(self.declare_parameter("cpr", 2048.0).value)
        self.radius = float(self.declare_parameter("radius_m", 0.285).value)

        self.create_subscription(Imu, "/imu/data", self.cb_imu, 20)
        self.create_subscription(Float32, "/hw/enc", self.cb_enc, 20)
        self.create_subscription(Float32, "/hw/pwm_applied", self.cb_pwm, 20)
        self.create_subscription(Float32, "/ina219/current_ma", self.cb_current, 20)

        self.pub = self.create_publisher(MarkerArray, "/pendulum/rviz_markers", 10)
        self.create_timer(0.03, self.on_timer)

    def cb_imu(self, msg: Imu):
        qw, qx, qy, qz = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        # yaw extraction (Z axis)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.state.theta_imu = float(math.atan2(siny_cosp, cosy_cosp))

    def cb_enc(self, msg: Float32):
        enc = float(msg.data)
        if self.state.enc0 is None:
            self.state.enc0 = enc
        if self.cpr > 1.0:
            self.state.theta_enc = float(2.0 * math.pi * (enc - self.state.enc0) / self.cpr)

    def cb_pwm(self, msg: Float32):
        self.state.pwm_hw = float(msg.data)

    def cb_current(self, msg: Float32):
        self.state.current_ma = float(msg.data)

    def on_timer(self):
        ma = MarkerArray()
        ma.markers.extend([
            self.make_link_marker(0, self.state.theta_imu, 0.8, 0.2, 0.2),
            self.make_link_marker(1, self.state.theta_enc, 0.2, 0.2, 0.8),
            self.make_current_bar(2, self.state.current_ma),
        ])
        self.pub.publish(ma)

    def make_link_marker(self, mid: int, theta: float, r: float, g: float, b: float) -> Marker:
        mk = Marker()
        mk.header.frame_id = "map"
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "pendulum"
        mk.id = mid
        mk.type = Marker.ARROW
        mk.action = Marker.ADD
        mk.scale.x = 0.01
        mk.scale.y = 0.02
        mk.scale.z = 0.02
        mk.color.a = 1.0
        mk.color.r = float(r)
        mk.color.g = float(g)
        mk.color.b = float(b)
        mk.points = []
        from geometry_msgs.msg import Point

        p0 = Point(); p0.x = 0.0; p0.y = 0.0; p0.z = 0.0
        p1 = Point(); p1.x = float(self.radius * math.sin(theta)); p1.y = float(-self.radius * math.cos(theta)); p1.z = 0.0
        mk.points = [p0, p1]
        return mk

    def make_current_bar(self, mid: int, current_ma: float) -> Marker:
        mk = Marker()
        mk.header.frame_id = "map"
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "current"
        mk.id = mid
        mk.type = Marker.CUBE
        mk.action = Marker.ADD
        mk.pose.position.x = 0.35
        mk.pose.position.y = 0.0
        mk.pose.position.z = 0.05
        mk.scale.x = 0.03
        mk.scale.y = 0.03
        mk.scale.z = max(0.01, min(abs(current_ma) / 2000.0, 0.4))
        mk.color.a = 0.8
        mk.color.r = 0.9
        mk.color.g = 0.3
        mk.color.b = 0.1
        return mk


def main():
    rclpy.init()
    node = RvizPendulumCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
