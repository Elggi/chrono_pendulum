"""Shared pendulum CSV schema for runtime and offline replay export."""

PENDULUM_CSV_HEADER = [
    "wall_time", "wall_elapsed", "sim_time", "mode",
    "cmd_u_raw", "cmd_u_delayed", "hw_pwm", "delay_sec_est", "tau_cmd",
    "theta", "omega", "alpha",
    "hw_enc", "hw_arduino_ms",
    "theta_real", "omega_real", "alpha_real",
    "delay_ms",
    "l_com_est", "b_eq_est", "tau_eq_est", "k_t_est", "i0_est", "R_est", "k_e_est",
    "bus_v_raw", "bus_v_filtered", "current_raw_A", "current_filtered_A", "power_raw_W",
    "tau_motor", "tau_res", "tau_visc", "tau_coul", "i_pred", "v_applied",
    "inst_cost", "best_cost_so_far",
    "imu_qw", "imu_qx", "imu_qy", "imu_qz",
    "imu_wx", "imu_wy", "imu_wz",
    "imu_ax", "imu_ay", "imu_az",
    "ls_cost", "fit_done", "fit_complete", "fit_final_params",
]
