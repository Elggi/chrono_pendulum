"""Shared CSV schema for pendulum runtime/replay exports.

This keeps the exact column order used by ``chrono_pendulum.py`` so offline replay
outputs can be consumed directly by ``plot_pendulum.py``.
"""

PENDULUM_LOG_COLUMNS = [
    "wall_time", "wall_elapsed", "mode",
    "cmd_u_raw", "cmd_u_delayed", "hw_pwm", "tau_cmd",
    "theta", "omega", "alpha",
    "hw_enc", "hw_arduino_ms",
    "theta_real", "omega_real", "alpha_real",
    "J_rod", "J_imu", "J_total",
    "tau_motor", "tau_res", "tau_visc", "tau_coul",
    "inst_cost", "best_cost_so_far",
    "imu_qw", "imu_qx", "imu_qy", "imu_qz",
    "imu_wx", "imu_wy", "imu_wz",
    "imu_ax", "imu_ay", "imu_az",
]
