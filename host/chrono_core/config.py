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
    terminal_status_width: int = 160

    motor_radius: float = 0.020
    motor_length: float = 0.050
    shaft_radius: float = 0.004
    shaft_length: float = 0.015

    link_L: float = 0.285
    radius_m: float = 0.285
    cpr: float = float("nan")
    link_W: float = 0.020
    link_T: float = 0.006
    link_mass: float = 0.200
    imu_mass: float = 0.010

    imu_offset_x: float = 0.220
    imu_offset_y: float = 0.000
    imu_offset_z: float = 0.000
    imu_size_x: float = 0.0595
    imu_size_y: float = 0.0460
    imu_size_z: float = 0.0117

    theta0_deg: float = -10.0
    omega0: float = 0.0

    pwm_limit: float = 255.0
    pwm_step: float = 10.0

    # online model parameters (important: J included)
    J_init: float = 0.010
    b_init: float = 0.030
    tau_c_init: float = 0.080
    mgl_init: float = 0.550
    k_t_init: float = 0.250
    i0_init: float = 0.050
    R_init: float = 2.0
    k_e_init: float = 0.020
    tanh_eps: float = 0.05
    j_min: float = 1e-4

    # automatic delay compensation
    auto_delay_comp: bool = True
    delay_init_ms: float = 0.0
    delay_max_ms: float = 120.0
    delay_update_hz: float = 5.0
    delay_buffer_sec: float = 4.0
    delay_smooth_alpha: float = 0.15

    # online EKF-like fitting
    online_fit_enable: bool = True
    self_fit_mode: str = "on"
    q_theta: float = 1e-5
    q_omega: float = 1e-3
    q_J: float = 1e-8
    q_b: float = 1e-7
    q_tauc: float = 1e-7
    q_mgl: float = 1e-7
    q_kt: float = 1e-7
    q_i0: float = 1e-7
    q_delay: float = 1e-6
    r_theta: float = 2e-4
    r_omega: float = 2e-3
    ekf_enable_min_pwm: float = 5.0
    obs_lpf_tau_sec: float = 0.030
    fit_conv_window_sec: float = 1.5
    fit_conv_hold_sec: float = 2.0
    fit_conv_rms_theta: float = 0.060
    fit_conv_rms_omega: float = 0.450
    fit_conv_rms_alpha: float = 1.600
    J_max: float = 0.2
    b_max: float = 5.0
    tau_c_max: float = 2.0
    mgl_max: float = 5.0
    k_t_max: float = 5.0
    i0_max: float = 255.0

    # cost weights
    w_theta: float = 5.0
    w_omega: float = 2.5
    w_alpha: float = 0.7
    w_v: float = 0.3
    w_i: float = 0.5
    w_p: float = 0.2
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
    topic_hw_arduino_ms: str = "/hw/arduino_ms"
    topic_bus_v: str = "/ina219/bus_voltage_v"
    topic_current_ma: str = "/ina219/current_ma"
    topic_power_mw: str = "/ina219/power_mw"
    topic_est_theta: str = "/est/theta"
    topic_est_omega: str = "/est/omega"
    topic_est_alpha: str = "/est/alpha"

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
