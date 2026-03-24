#!/usr/bin/env python3
import math
import shutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class LiveMonitor(Node):
    def __init__(self):
        super().__init__('jetson_live_monitor')
        self.values = {
            'cmd_u': 0.0,
            'hw_pwm': 0.0,
            'enc': 0.0,
            'bus_v': float('nan'),
            'current_ma': float('nan'),
            'power_mw': float('nan'),
        }
        self.create_subscription(Float32, '/cmd/u', self._cb('cmd_u'), 100)
        self.create_subscription(Float32, '/hw/pwm_applied', self._cb('hw_pwm'), 100)
        self.create_subscription(Float32, '/hw/enc', self._cb('enc'), 100)
        self.create_subscription(Float32, '/ina219/bus_voltage_v', self._cb('bus_v'), 100)
        self.create_subscription(Float32, '/ina219/current_ma', self._cb('current_ma'), 100)
        self.create_subscription(Float32, '/ina219/power_mw', self._cb('power_mw'), 100)
        self.create_timer(0.1, self.render)

    def _cb(self, key):
        def inner(msg):
            self.values[key] = float(msg.data)
        return inner

    def render(self):
        width = shutil.get_terminal_size((180, 24)).columns - 1
        def fmt(v, spec):
            if isinstance(v, float) and math.isnan(v):
                return 'nan'
            return format(v, spec)
        msg = (
            f"cmd_u={fmt(self.values['cmd_u'], '7.1f')} | "
            f"hw_pwm={fmt(self.values['hw_pwm'], '7.1f')} | "
            f"enc={fmt(self.values['enc'], '11.1f')} | "
            f"bus_v={fmt(self.values['bus_v'], '6.2f')} | "
            f"current_mA={fmt(self.values['current_ma'], '7.1f')} | "
            f"power_mW={fmt(self.values['power_mw'], '7.1f')}"
        )
        print('\r\033[2K' + msg[:width].ljust(width), end='', flush=True)


def main():
    rclpy.init()
    node = LiveMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
