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
    link_mass: float = 0.200
    imu_mass: float = 0.020

    # Effective COM model: J_pivot = J_cm_base + m_total * l_com**2
    J_cm_base: float = 0.0020
    l_com_init: float = 0.1425

    # Deprecated compatibility field (loaded but ignored in model/fit)
    mgl_init: float = 0.550

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

    # Unified resistance/electrical model parameters
    b_eq_init: float = 0.030
    tau_eq_init: float = 0.080
    k_t_init: float = 0.250
    i0_init: float = 0.050
    R_init: float = 2.0
    k_e_init: float = 0.020
    tanh_eps: float = 0.05

    # Electrical options
    electrical_use_ina_bus_voltage: bool = True
    nominal_bus_voltage: float = 7.4
    current_clip_enable: bool = True
    current_clip_A: float = 3.0

    # INA219 robust filtering
    ina_enable_bus_filter: bool = True
    ina_bus_median_window: int = 7
    ina_bus_hampel_k: int = 7
    ina_bus_hampel_sigma: float = 3.0
    ina_bus_lpf_tau_sec: float = 0.20
    ina_enable_current_filter: bool = True
    ina_current_median_window: int = 5
    ina_current_hampel_k: int = 5
    ina_current_hampel_sigma: float = 3.5
    ina_current_lpf_tau_sec: float = 0.15

    # fixed control delay (set from CLI/calibration/parameter JSON)
    delay_init_ms: float = 0.0
    delay_max_ms: float = 120.0

    # online EKF-like fitting
    online_fit_enable: bool = True
    self_fit_mode: str = "on"
    q_theta: float = 1e-5
    q_omega: float = 1e-3
    q_l_com: float = 1e-7
    q_b_eq: float = 1e-7
    q_tau_eq: float = 1e-7
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

    l_com_min: float = 0.03
    l_com_max: float = 0.45
    b_eq_max: float = 5.0
    tau_eq_max: float = 2.0

    # cost weights
    w_theta: float = 5.0
    w_omega: float = 2.5
    w_alpha: float = 0.7
    w_v: float = 0.3
    w_i: float = 0.5
    w_p: float = 0.2
    w_du: float = 0.01
    w_d2u: float = 0.005
    fit_use_electrical_cost: bool = False

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
