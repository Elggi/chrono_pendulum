"""Shared CSV schema for pendulum runtime/replay exports.

This keeps the exact column order used by ``chrono_pendulum.py`` so offline replay
outputs can be consumed directly by ``plot_pendulum.py``.
"""

PENDULUM_LOG_COLUMNS = [
    "wall_time", "wall_elapsed", "mode",
    "cmd_u_raw", "cmd_u_delayed", "hw_pwm", "tau_cmd",
    "ina_current_raw_mA", "ina_bus_voltage_v", "ina_power_mw",
    "ina_current_offset_mA", "ina_current_corr_mA", "ina_current_signed_mA",
    "pwm_hw",
    "theta_imu", "theta_encoder",
    "omega_imu", "omega_encoder",
    "alpha_imu", "alpha_linear", "alpha_encoder",
    "theta_imu_online", "theta_encoder_online",
    "omega_imu_online", "omega_encoder_online",
    "alpha_imu_online", "alpha_linear_online", "alpha_encoder_online",
    "ina_current_signed_online_mA",
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
