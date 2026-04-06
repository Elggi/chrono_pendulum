from dataclasses import dataclass


@dataclass
class BridgeConfig:
    step: float = 0.001
    gravity: float = 9.81

    enable_render: bool = True
    enable_imu_viewer: bool = True
    realtime: bool = True
    win_w: int = 1280
    win_h: int = 900
    window_title: str = "Chrono Pendulum | Online Calibration"
    terminal_status_width: int = 180

    motor_radius: float = 0.020
    motor_length: float = 0.050
    shaft_radius: float = 0.004
    shaft_length: float = 0.015

    # Visual geometry (rendering / IMU widget placement)
    link_L: float = 0.285
    link_W: float = 0.020
    link_T: float = 0.006

    # Dynamic geometry and encoder scaling
    radius_m: float = 0.285
    cpr: float = float("nan")

    # Fixed masses (physical model)
    rod_mass: float = 0.200
    imu_mass: float = 0.020
    rod_length: float = 0.285
    # IMU radius from pivot (loaded from calibration when available)
    r_imu: float = 0.285
    l_com_init: float = 0.1425  # default midpoint for 0.285 m link
    # Effective input-to-torque gain in surrogate dynamics: tau_motor = K_u * u
    K_u_init: float = 1.0e-5
    # Legacy aliases kept to avoid breaking old scripts/options.
    link_mass: float = 0.200
    J_cm_base: float = 0.0

    imu_offset_x: float = 0.220
    imu_offset_y: float = 0.000
    imu_offset_z: float = 0.000
    imu_size_x: float = 0.0595
    imu_size_y: float = 0.0460
    imu_size_z: float = 0.0117

    theta0_deg: float = 0.0
    omega0: float = 0.0

    pwm_limit: float = 255.0
    pwm_step: float = 10.0

    # Surrogate damping/friction parameters
    b_eq_init: float = 0.0
    tau_eq_init: float = 0.0
    tanh_eps: float = 0.05

    l_com_min: float = 0.03
    l_com_max: float = 0.45
    b_eq_max: float = 5.0
    tau_eq_max: float = 2.0

    # cost weights
    w_theta: float = 5.0
    w_omega: float = 2.5
    w_alpha: float = 0.7
    w_du: float = 0.01
    w_d2u: float = 0.005

    # control presets
    wave_freq: float = 0.5
    burst_period: float = 2.0
    burst_on_time: float = 0.3
    prbs_dt: float = 0.25
    prbs_seed: int = 12345

    # ros topics
    topic_cmd_u: str = "/cmd/u"
    topic_debug: str = "/cmd/keyboard_state"
    topic_hw_pwm: str = "/hw/pwm_applied"
    topic_hw_enc: str = "/hw/enc"
    topic_hw_imu: str = "/imu/data"
    topic_hw_arduino_ms: str = "/hw/arduino_ms"
    topic_hw_current_ma: str = "/ina219/current_ma"
    topic_hw_bus_voltage_v: str = "/ina219/bus_voltage_v"
    topic_hw_power_mw: str = "/ina219/power_mw"
    topic_sim_theta: str = "/sim/theta"
    topic_sim_omega: str = "/sim/omega"
    topic_sim_alpha: str = "/sim/alpha"
    topic_sim_tau: str = "/sim/tau_applied"
    topic_sim_cmd_used: str = "/sim/cmd_u_used"
    topic_sim_delay_ms: str = "/sim/delay_ms"
    topic_sim_status: str = "/sim/status"

    # Keep both names for backward compatibility with pre-modularized code paths.
    topic_imu: str = "/sim/imu/data"
    topic_imu_data: str = "/sim/imu/data"

    log_dir: str = "./run_logs"
    log_prefix: str = "chrono_run_"

    calibration_json: str = ""
