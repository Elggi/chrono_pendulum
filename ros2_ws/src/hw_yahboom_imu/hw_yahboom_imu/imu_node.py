import math
import struct
import time
import serial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


def s16(x: int) -> int:
    return x - 65536 if x > 32767 else x


def euler_to_quat(roll, pitch, yaw):
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5);   sy = math.sin(yaw * 0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw


class YahboomImu(Node):
    """
    WT/JY901 style 0x55 11-byte frames.
    Publishes /imu/data (sensor_msgs/Imu).
    """

    def __init__(self):
        super().__init__("hw_yahboom_imu")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("frame_id", "imu_link")

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.pub = self.create_publisher(Imu, "/imu/data", 50)

        self.ser = None
        self.buf = bytearray()

        self.ax = self.ay = self.az = 0.0
        self.gx = self.gy = self.gz = 0.0
        self.roll = self.pitch = self.yaw = 0.0

        self.try_open_serial()

        self.read_timer = self.create_timer(0.005, self.read_tick)   # 200 Hz
        self.pub_timer = self.create_timer(0.02, self.publish_tick)  # 50 Hz

    def try_open_serial(self):
        if self.ser:
            return
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
            self.get_logger().info(f"Opened IMU: {self.port} @ {self.baud}")
        except Exception as e:
            self.get_logger().warn(f"IMU open failed ({self.port}): {e}")
            self.ser = None

    def parse_frame(self, fr: bytes):
        if fr[0] != 0x55:
            return
        ck = sum(fr[0:10]) & 0xFF
        if ck != fr[10]:
            return

        t = fr[1]
        v = struct.unpack("<hhhh", fr[2:10])
        v = [s16(x) for x in v]

        if t == 0x51:
            ax, ay, az, _ = v
            self.ax = ax / 32768.0 * 16.0
            self.ay = ay / 32768.0 * 16.0
            self.az = az / 32768.0 * 16.0
        elif t == 0x52:
            gx, gy, gz, _ = v
            self.gx = gx / 32768.0 * 2000.0
            self.gy = gy / 32768.0 * 2000.0
            self.gz = gz / 32768.0 * 2000.0
        elif t == 0x53:
            r, p, y, _ = v
            self.roll = r / 32768.0 * 180.0
            self.pitch = p / 32768.0 * 180.0
            self.yaw = y / 32768.0 * 180.0

    def read_tick(self):
        # reconnect if needed
        if self.ser is None:
            self.try_open_serial()
            return

        try:
            data = self.ser.read(512)
            if data:
                self.buf += data
        except Exception as e:
            self.get_logger().warn(f"IMU serial read error: {e}")
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            self.buf.clear()
            time.sleep(0.05)
            return

        # Extract 11-byte frames starting with 0x55
        while True:
            idx = self.buf.find(b"\x55")
            if idx < 0:
                self.buf.clear()
                return
            if len(self.buf) < idx + 11:
                if idx > 0:
                    del self.buf[:idx]
                return
            fr = bytes(self.buf[idx:idx+11])
            del self.buf[:idx+11]
            self.parse_frame(fr)

    def publish_tick(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        rr = math.radians(self.roll)
        pp = math.radians(self.pitch)
        yy = math.radians(self.yaw)
        qx, qy, qz, qw = euler_to_quat(rr, pp, yy)

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        msg.angular_velocity.x = math.radians(self.gx)
        msg.angular_velocity.y = math.radians(self.gy)
        msg.angular_velocity.z = math.radians(self.gz)

        g0 = 9.80665
        msg.linear_acceleration.x = self.ax * g0
        msg.linear_acceleration.y = self.ay * g0
        msg.linear_acceleration.z = self.az * g0

        self.pub.publish(msg)

    def destroy_node(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = YahboomImu()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
