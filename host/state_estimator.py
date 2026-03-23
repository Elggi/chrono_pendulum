#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import argparse
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String


def quat_to_yaw(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class StateEstimatorNode(Node):
    def __init__(self, args):
        super().__init__("pendulum_state_estimator")
        self.args = args

        self.pub_theta = self.create_publisher(Float32, args.theta_topic, 10)
        self.pub_omega = self.create_publisher(Float32, args.omega_topic, 10)
        self.pub_alpha = self.create_publisher(Float32, args.alpha_topic, 10)
        self.pub_status = self.create_publisher(String, args.status_topic, 10)
        self.pub_imu_yaw = self.create_publisher(Float32, args.imu_yaw_topic, 10)
        self.pub_imu_wz = self.create_publisher(Float32, args.imu_wz_topic, 10)

        self.create_subscription(Imu, args.imu_topic, self.cb_imu, 100)
        self.create_subscription(Float32, args.enc_topic, self.cb_enc, 100)

        self.last_enc = None
        self.last_theta = None
        self.last_omega = 0.0
        self.last_t = None
        self.theta_hist = deque(maxlen=max(args.smooth_window, 3))
        self.omega_hist = deque(maxlen=max(args.smooth_window, 3))
        self.last_imu_stamp = None
        self.last_enc_stamp = None

        self.status_timer = self.create_timer(0.5, self.publish_status)

    def publish_float(self, pub, value):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def cb_imu(self, msg: Imu):
        yaw = quat_to_yaw(
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        )
        self.publish_float(self.pub_imu_yaw, yaw)
        self.publish_float(self.pub_imu_wz, msg.angular_velocity.z)
        self.last_imu_stamp = self.get_clock().now()

    def cb_enc(self, msg: Float32):
        now = self.get_clock().now()
        enc = float(msg.data)
        if self.args.counts_per_revolution <= 0.0:
            self.last_enc_stamp = now
            return

        theta = (
            self.args.theta_sign
            * (2.0 * math.pi / self.args.counts_per_revolution)
            * (enc - (self.last_enc if self.last_enc is not None else enc))
        )
        if self.last_theta is not None:
            theta += self.last_theta
        theta += self.args.theta_offset if self.last_theta is None else 0.0

        if self.last_t is None:
            omega = 0.0
            alpha = 0.0
        else:
            dt = max((now - self.last_t).nanoseconds * 1e-9, 1e-6)
            omega = (theta - self.last_theta) / dt
            alpha = (omega - self.last_omega) / dt

        self.theta_hist.append(theta)
        self.omega_hist.append(omega)
        theta_s = float(np.mean(self.theta_hist))
        omega_s = float(np.mean(self.omega_hist))

        self.publish_float(self.pub_theta, theta_s)
        self.publish_float(self.pub_omega, omega_s)
        self.publish_float(self.pub_alpha, alpha)

        self.last_enc = enc
        self.last_theta = theta_s
        self.last_omega = omega_s
        self.last_t = now
        self.last_enc_stamp = now

    def publish_status(self):
        status = String()
        status.data = (
            f"imu={'ok' if self.last_imu_stamp is not None else 'missing'}, "
            f"enc={'ok' if self.last_enc_stamp is not None else 'missing'}, "
            f"cpr={self.args.counts_per_revolution:.3f}"
        )
        self.pub_status.publish(status)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--enc-topic", default="/hw/enc")
    ap.add_argument("--theta-topic", default="/est/theta")
    ap.add_argument("--omega-topic", default="/est/omega")
    ap.add_argument("--alpha-topic", default="/est/alpha")
    ap.add_argument("--status-topic", default="/est/status")
    ap.add_argument("--imu-yaw-topic", default="/est/imu_yaw")
    ap.add_argument("--imu-wz-topic", default="/est/imu_wz")
    ap.add_argument("--counts-per-revolution", type=float, default=0.0)
    ap.add_argument("--theta-sign", type=float, default=1.0)
    ap.add_argument("--theta-offset", type=float, default=0.0)
    ap.add_argument("--smooth-window", type=int, default=5)
    return ap


def main():
    args = build_argparser().parse_args()
    rclpy.init()
    node = StateEstimatorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
