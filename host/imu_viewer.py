#!/usr/bin/env python3
import math
import threading
import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


def quat_to_rotmat(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)


def rotmat_to_euler_zyx(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        yaw = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return yaw, pitch, roll


class SharedState:
    def __init__(self, link_length=0.25, history_len=4000):
        self.lock = threading.Lock()

        self.q_abs = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)   # w,x,y,z
        self.gyro = np.zeros(3, dtype=float)
        self.acc = np.zeros(3, dtype=float)
        self.enc = 0.0
        self.seq = 0

        self.has_init = False
        self.R0 = np.eye(3)
        self.link_length = link_length

        # 초기 링크 방향을 아래쪽(0, -1)으로 설정
        self.ref_tip_local = np.array([0.0, -link_length, 0.0], dtype=float)
        self.tip0 = self.ref_tip_local.copy()

        self.tip_hist = deque(maxlen=history_len)
        self.angle_unwrapped = 0.0
        self.prev_angle = None

        self.rev_index = 0
        self.rev_enc_anchor = None
        self.last_cpr = None
        self.cpr_samples = []

        self.last_tip = self.ref_tip_local.copy()

    def update_imu(self, q, gyro, acc):
        R_abs = quat_to_rotmat(*q)

        with self.lock:
            self.q_abs[:] = q
            self.gyro[:] = gyro
            self.acc[:] = acc
            self.seq += 1

            if not self.has_init:
                self.R0 = R_abs.copy()
                self.has_init = True
                self.tip0 = self.ref_tip_local.copy()
                self.tip_hist.append(self.tip0.copy())
                self.prev_angle = math.atan2(self.tip0[1], self.tip0[0])
                self.angle_unwrapped = 0.0
                self.rev_index = 0
                self.rev_enc_anchor = self.enc
                self.last_tip = self.tip0.copy()
                return

            R_rel = self.R0.T @ R_abs
            tip = R_rel @ self.ref_tip_local
            self.last_tip = tip.copy()
            self.tip_hist.append(tip.copy())

            angle = math.atan2(tip[1], tip[0])
            if self.prev_angle is None:
                self.prev_angle = angle

            dtheta = angle - self.prev_angle
            while dtheta > math.pi:
                dtheta -= 2.0 * math.pi
            while dtheta < -math.pi:
                dtheta += 2.0 * math.pi

            self.angle_unwrapped += dtheta
            self.prev_angle = angle

            if self.rev_enc_anchor is None:
                self.rev_enc_anchor = self.enc

            new_rev_index = math.floor(abs(self.angle_unwrapped) / (2.0 * math.pi))

            if new_rev_index > self.rev_index:
                delta_counts = abs(self.enc - self.rev_enc_anchor)
                if delta_counts > 0:
                    self.last_cpr = float(delta_counts)
                    self.cpr_samples.append(float(delta_counts))
                self.rev_enc_anchor = self.enc
                self.rev_index = new_rev_index

    def update_enc(self, enc):
        with self.lock:
            self.enc = float(enc)


class ViewerNode(Node):
    def __init__(self, state: SharedState, imu_topic: str, enc_topic: str):
        super().__init__("imu_traj_encoder_viewer")
        self.state = state

        self.sub_imu = self.create_subscription(Imu, imu_topic, self.cb_imu, 100)
        self.sub_enc = self.create_subscription(Float32, enc_topic, self.cb_enc, 100)

    def cb_imu(self, msg: Imu):
        q = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        ], dtype=float)
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=float)
        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=float)

        self.state.update_imu(q, gyro, acc)

    def cb_enc(self, msg: Float32):
        self.state.update_enc(msg.data)


def ros_spin_thread(state, imu_topic, enc_topic):
    rclpy.init()
    node = ViewerNode(state, imu_topic, enc_topic)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def make_box():
    L, W, H = 0.60, 0.18, 0.06
    pts = np.array([
        [-L/2, -W/2, -H/2],
        [ L/2, -W/2, -H/2],
        [ L/2,  W/2, -H/2],
        [-L/2,  W/2, -H/2],
        [-L/2, -W/2,  H/2],
        [ L/2, -W/2,  H/2],
        [ L/2,  W/2,  H/2],
        [-L/2,  W/2,  H/2],
    ], dtype=float)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    return pts, edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu_topic", default="/imu/data")
    parser.add_argument("--enc_topic", default="/hw/enc")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--link_length", type=float, default=0.25)
    parser.add_argument("--history_len", type=int, default=4000)
    args = parser.parse_args()

    state = SharedState(link_length=args.link_length, history_len=args.history_len)
    t = threading.Thread(
        target=ros_spin_thread,
        args=(state, args.imu_topic, args.enc_topic),
        daemon=True
    )
    t.start()

    fig = plt.figure(figsize=(15, 7.8))
    fig.canvas.manager.set_window_title("IMU + Trajectory + Encoder Viewer")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.90, wspace=0.18)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    box_pts, box_edges = make_box()

    def update(_frame):
        with state.lock:
            q = state.q_abs.copy()
            gyro = state.gyro.copy()
            acc = state.acc.copy()
            enc = state.enc
            seq = state.seq
            has_init = state.has_init
            R0 = state.R0.copy()
            tip0 = state.tip0.copy()
            tip_hist = np.array(state.tip_hist) if len(state.tip_hist) > 0 else np.zeros((0, 3))
            angle_unwrapped = state.angle_unwrapped
            rev_index = state.rev_index
            last_cpr = state.last_cpr
            cpr_samples = list(state.cpr_samples)
            link_length = state.link_length

        ax3d.cla()
        ax2d.cla()

        fig.suptitle("IMU Orientation + Live Tip Trajectory + Encoder Counts", fontsize=14, y=0.965)

        # ---------- 3D Orientation ----------
        ax3d.set_title("Orientation Viewer", pad=18)
        ax3d.set_xlim(-1.0, 1.0)
        ax3d.set_ylim(-1.0, 1.0)
        ax3d.set_zlim(-1.0, 1.0)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.view_init(elev=22, azim=45)

        ax3d.plot([0, 0.8], [0, 0], [0, 0], linewidth=2)
        ax3d.plot([0, 0], [0, 0.8], [0, 0], linewidth=2)
        ax3d.plot([0, 0], [0, 0], [0, 0.8], linewidth=2)
        ax3d.text(0.86, 0, 0, "World X")
        ax3d.text(0, 0.86, 0, "World Y")
        ax3d.text(0, 0, 0.86, "World Z")

        if has_init:
            R_abs = quat_to_rotmat(*q)
            R_rel = R0.T @ R_abs

            yaw, pitch, roll = rotmat_to_euler_zyx(R_rel)
            pts = (R_rel @ box_pts.T).T

            for i, j in box_edges:
                ax3d.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    [pts[i, 2], pts[j, 2]],
                    linewidth=2
                )

            bx = R_rel @ np.array([0.7, 0, 0], dtype=float)
            by = R_rel @ np.array([0, 0.5, 0], dtype=float)
            bz = R_rel @ np.array([0, 0, 0.4], dtype=float)

            ax3d.plot([0, bx[0]], [0, bx[1]], [0, bx[2]], linewidth=3)
            ax3d.plot([0, by[0]], [0, by[1]], [0, by[2]], linewidth=3)
            ax3d.plot([0, bz[0]], [0, bz[1]], [0, bz[2]], linewidth=3)

            ax3d.text(bx[0], bx[1], bx[2], "Body X")
            ax3d.text(by[0], by[1], by[2], "Body Y")
            ax3d.text(bz[0], bz[1], bz[2], "Body Z")

            if np.linalg.norm(acc) > 1e-9:
                avec = acc / np.linalg.norm(acc) * 0.8
                ax3d.plot([0, avec[0]], [0, avec[1]], [0, avec[2]], "--", linewidth=2)
                ax3d.text(avec[0], avec[1], avec[2], "Acc")

            info = (
                f"seq: {seq}\n"
                f"quat [w x y z]: {q[0]: .3f}, {q[1]: .3f}, {q[2]: .3f}, {q[3]: .3f}\n"
                f"yaw/pitch/roll [deg]: "
                f"{math.degrees(yaw): .1f}, {math.degrees(pitch): .1f}, {math.degrees(roll): .1f}\n"
                f"gyro [rad/s]: {gyro[0]: .3f}, {gyro[1]: .3f}, {gyro[2]: .3f}\n"
                f"acc [m/s^2]: {acc[0]: .3f}, {acc[1]: .3f}, {acc[2]: .3f}"
            )
            ax3d.text2D(0.02, 0.97, info, transform=ax3d.transAxes, va="top", family="monospace")

        # ---------- 2D Trajectory ----------
        ax2d.set_title("Live Tip Trajectory (XY plane)", pad=18)
        lim = max(1.2 * link_length, 0.1)
        ax2d.set_xlim(-lim, lim)
        ax2d.set_ylim(-lim, lim)
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.set_xlabel("X [m]")
        ax2d.set_ylabel("Y [m]")
        ax2d.grid(True, alpha=0.35)

        ax2d.axhline(0.0, linewidth=1)
        ax2d.axvline(0.0, linewidth=1)

        # initial position -> 아래 방향 (0, -1)
        ax2d.plot([0, tip0[0]], [0, tip0[1]], linestyle="--", linewidth=2, label="Initial link")
        ax2d.scatter([tip0[0]], [tip0[1]], marker="*", s=180, label="Initial tip")

        if len(tip_hist) > 1:
            ax2d.plot(tip_hist[:, 0], tip_hist[:, 1], linewidth=2, label="Tip trajectory")
            ax2d.plot([0, tip_hist[-1, 0]], [0, tip_hist[-1, 1]], linewidth=3, label="Current link")
            ax2d.scatter([tip_hist[-1, 0]], [tip_hist[-1, 1]], s=60, label="Current tip")

        mean_cpr = float(np.mean(cpr_samples)) if len(cpr_samples) > 0 else None
        angle_deg = math.degrees(angle_unwrapped)

        traj_info = [
            f"encoder count: {enc:.1f}",
            f"relative angle unwrapped [deg]: {angle_deg:.1f}",
            f"full rotations detected: {rev_index}",
            f"last counts / rotation: {last_cpr:.1f}" if last_cpr is not None else "last counts / rotation: n/a",
            f"mean counts / rotation: {mean_cpr:.1f}" if mean_cpr is not None else "mean counts / rotation: n/a",
        ]
        ax2d.text(
            0.02, 0.98,
            "\n".join(traj_info),
            transform=ax2d.transAxes,
            va="top",
            family="monospace"
        )

        ax2d.legend(loc="lower left", fontsize=9)

    ani = FuncAnimation(fig, update, interval=int(1000 / max(args.fps, 1)), cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
