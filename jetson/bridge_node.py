#!/usr/bin/env python3
import time
import threading
import serial

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class ArduinoBridge(Node):
    """
    Sub: /cmd/u (Float32, signed PWM)

    Pub:
      /hw/enc               (Float32)
      /hw/pwm_applied       (Float32)
      /hw/arduino_ms        (Float32)
      /ina219/bus_voltage_v (Float32)
      /ina219/current_ma    (Float32)
      /ina219/power_mw      (Float32)

    Serial:
      TX: U,<pwm>\n
      RX: S,<enc>,<pwm>,<ms>[,<bus_v>,<current_ma>,<power_mw>]\n
    """

    def __init__(self):
        super().__init__("hw_arduino_bridge")

        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("send_rate_hz", 50.0)
        self.declare_parameter("quiet_rx", True)
        self.declare_parameter("pwm_limit", 60.0)

        port = str(self.get_parameter("port").value)
        baud = int(self.get_parameter("baud").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout_sec").value)
        self.send_rate = float(self.get_parameter("send_rate_hz").value)
        self.quiet_rx = bool(self.get_parameter("quiet_rx").value)
        self.pwm_limit = float(self.get_parameter("pwm_limit").value)

        self.get_logger().info(f"Opening Arduino serial: {port} @ {baud}")
        self.ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(0.2)

        self.get_logger().info(f"Opened Arduino serial: {port} @ {baud}")
        self.get_logger().info(
            f"Expecting /cmd/u as signed PWM in [-{self.pwm_limit:.0f}, {self.pwm_limit:.0f}]"
        )

        self.u_latest = 0.0
        self.last_cmd_time = time.time()

        self.sub = self.create_subscription(Float32, "/cmd/u", self.on_cmd, 10)

        self.pub_enc = self.create_publisher(Float32, "/hw/enc", 10)
        self.pub_pwm = self.create_publisher(Float32, "/hw/pwm_applied", 10)
        self.pub_pwm_tx = self.create_publisher(Float32, "/hw/pwm_tx", 10)
        self.pub_ms = self.create_publisher(Float32, "/hw/arduino_ms", 10)

        self.pub_bus_v = self.create_publisher(Float32, "/ina219/bus_voltage_v", 10)
        self.pub_current_ma = self.create_publisher(Float32, "/ina219/current_ma", 10)
        self.pub_power_mw = self.create_publisher(Float32, "/ina219/power_mw", 10)

        self.stop_flag = False
        self.last_rx_time = time.time()
        self.last_tx_warn_time = 0.0
        self.rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
        self.rx_thread.start()

        period = 1.0 / max(1.0, self.send_rate)
        self.tx_timer = self.create_timer(period, self.tx_tick)

    def on_cmd(self, msg: Float32):
        self.u_latest = float(msg.data)
        self.last_cmd_time = time.time()

    def tx_tick(self):
        if (time.time() - self.last_cmd_time) > self.cmd_timeout:
            pwm = 0
        else:
            pwm = int(round(clamp(self.u_latest, -self.pwm_limit, self.pwm_limit)))

        try:
            self.ser.write(f"U,{pwm}\n".encode("ascii"))
            self.publish_float(self.pub_pwm_tx, pwm)
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {repr(e)}")
            return

        if abs(pwm) > 0 and (time.time() - self.last_rx_time) > 0.5:
            if (time.time() - self.last_tx_warn_time) > 1.0:
                self.last_tx_warn_time = time.time()
                self.get_logger().warn(
                    f"TX pwm={pwm} is being sent but no fresh Arduino RX status has arrived for "
                    f"{time.time() - self.last_rx_time:.2f}s"
                )

    def publish_float(self, pub, value):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def handle_line(self, line: str):
        if not line:
            return

        if not self.quiet_rx:
            self.get_logger().info(f"RX: {line}")

        if not line.startswith("S,"):
            return

        parts = line.split(",")

        try:
            if len(parts) < 4:
                return

            enc = float(int(parts[1]))
            pwm = float(int(parts[2]))
            ms = float(int(parts[3]))
            self.last_rx_time = time.time()

            self.publish_float(self.pub_enc, enc)
            self.publish_float(self.pub_pwm, pwm)
            self.publish_float(self.pub_ms, ms)

            if len(parts) >= 7:
                bus_v = float(parts[4])
                current_ma = float(parts[5])
                power_mw = float(parts[6])

                self.publish_float(self.pub_bus_v, bus_v)
                self.publish_float(self.pub_current_ma, current_ma)
                self.publish_float(self.pub_power_mw, power_mw)

        except Exception as e:
            if not self.quiet_rx:
                self.get_logger().warn(f"RX parse failed: {line} / {repr(e)}")

    def rx_loop(self):
        buf = b""
        while rclpy.ok() and not self.stop_flag:
            try:
                chunk = self.ser.read(512)
                if not chunk:
                    continue

                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("ascii", errors="ignore").strip()
                    self.handle_line(line)

            except Exception as e:
                self.get_logger().error(f"Serial read failed: {repr(e)}")
                time.sleep(0.2)

    def destroy_node(self):
        self.stop_flag = True
        try:
            self.ser.write(b"U,0\n")
            time.sleep(0.05)
        except Exception:
            pass

        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = ArduinoBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
