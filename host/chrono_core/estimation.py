import math
from collections import deque

import numpy as np

from .config import BridgeConfig
from .utils import clamp


class RobustSignalFilter:
    def __init__(self, median_window: int, hampel_k: int, hampel_sigma: float, lpf_tau_sec: float):
        self.median_window = max(int(median_window), 1)
        self.hampel_k = max(int(hampel_k), 1)
        self.hampel_sigma = max(float(hampel_sigma), 0.1)
        self.lpf_tau_sec = max(float(lpf_tau_sec), 1e-4)
        self.raw = deque(maxlen=max(self.median_window, self.hampel_k))
        self.lpf_state = None

    def update(self, value: float, dt: float) -> float:
        val = float(value)
        self.raw.append(val)
        med_src = list(self.raw)[-self.median_window:]
        med = float(np.median(np.array(med_src, dtype=float)))

        hampel_src = np.array(list(self.raw)[-self.hampel_k:], dtype=float)
        center = float(np.median(hampel_src))
        mad = float(np.median(np.abs(hampel_src - center)))
        scale = 1.4826 * mad
        cleaned = med
        if scale > 1e-9 and abs(val - center) <= self.hampel_sigma * scale:
            cleaned = val

        if self.lpf_state is None:
            self.lpf_state = cleaned
        else:
            dt = max(float(dt), 1e-6)
            alpha = dt / (self.lpf_tau_sec + dt)
            self.lpf_state = (1.0 - alpha) * self.lpf_state + alpha * cleaned
        return float(self.lpf_state)


class OnlineParameterEKF:
    """Augmented state: [theta, omega, l_com, b_eq, tau_eq, delay_sec]."""

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.x = np.array([
            0.0,
            0.0,
            cfg.l_com_init,
            cfg.b_eq_init,
            cfg.tau_eq_init,
            cfg.delay_init_ms / 1000.0,
        ], dtype=float)
        self.P = np.diag([1e-3, 1e-2, 1e-3, 1e-3, 1e-3, 1e-4])
        self.Q = np.diag([cfg.q_theta, cfg.q_omega, cfg.q_l_com, cfg.q_b_eq, cfg.q_tau_eq, cfg.q_delay])
        self.R = np.diag([cfg.r_theta, cfg.r_omega])
        self.best_cost = np.inf
        self.best_params = None

    def _calc_alpha(self, th, om, l_com, b_eq, tau_eq, u_cmd):
        l_com = max(float(l_com), self.cfg.l_com_min)
        b_eq = max(float(b_eq), 0.0)
        tau_eq = max(float(tau_eq), 0.0)
        j_pivot = self.cfg.J_cm_base + (self.cfg.link_mass + self.cfg.imu_mass) * (l_com ** 2)
        j_pivot = max(j_pivot, 1e-6)

        duty = clamp(u_cmd / max(self.cfg.pwm_limit, 1e-9), -1.0, 1.0)
        v_applied = duty * self.cfg.nominal_bus_voltage
        i_raw = (v_applied - self.cfg.k_e_init * om) / max(self.cfg.R_init, 1e-6)
        i_eff = math.copysign(max(abs(i_raw) - self.cfg.i0_init, 0.0), i_raw)
        if self.cfg.current_clip_enable:
            i_eff = clamp(i_eff, -self.cfg.current_clip_A, self.cfg.current_clip_A)

        tau_motor = self.cfg.k_t_init * i_eff
        tau_res = b_eq * om + tau_eq * math.tanh(om / max(self.cfg.tanh_eps, 1e-9))
        tau_gravity = (self.cfg.link_mass + self.cfg.imu_mass) * self.cfg.gravity * l_com * math.sin(th)
        return (tau_motor - tau_res - tau_gravity) / j_pivot

    def f_disc(self, x, u, dt):
        th, om, l_com, b_eq, tau_eq, dly = x
        alpha = self._calc_alpha(th, om, l_com, b_eq, tau_eq, u)
        xn = np.array([
            th + dt * om,
            om + dt * alpha,
            l_com,
            max(b_eq, 0.0),
            max(tau_eq, 0.0),
            max(dly, 0.0),
        ], dtype=float)
        return xn

    def h_meas(self, x):
        return np.array([x[0], x[1]], dtype=float)

    @staticmethod
    def numeric_jacobian(fun, x, eps=1e-6):
        x = np.asarray(x, dtype=float)
        y0 = np.asarray(fun(x), dtype=float)
        m, n = len(y0), len(x)
        jac = np.zeros((m, n), dtype=float)
        for i in range(n):
            dx = np.zeros(n, dtype=float)
            step = eps * max(1.0, abs(x[i]))
            dx[i] = step
            y1 = np.asarray(fun(x + dx), dtype=float)
            y2 = np.asarray(fun(x - dx), dtype=float)
            jac[:, i] = (y1 - y2) / (2.0 * step)
        return jac

    def update(self, theta_meas, omega_meas, u_eff, dt, inst_cost=None):
        f_local = lambda xx: self.f_disc(xx, u_eff, dt)
        F = self.numeric_jacobian(f_local, self.x)
        x_pred = f_local(self.x)
        P_pred = F @ self.P @ F.T + self.Q
        H = self.numeric_jacobian(self.h_meas, x_pred)
        z_pred = self.h_meas(x_pred)
        z = np.array([theta_meas, omega_meas], dtype=float)
        y = z - z_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ P_pred
        self._clip_state()

        if inst_cost is not None and inst_cost < self.best_cost:
            self.best_cost = float(inst_cost)
            self.best_params = self.get_params()

    def _clip_state(self):
        self.x[2] = clamp(self.x[2], self.cfg.l_com_min, self.cfg.l_com_max)
        self.x[3] = clamp(self.x[3], 0.0, self.cfg.b_eq_max)
        self.x[4] = clamp(self.x[4], 0.0, self.cfg.tau_eq_max)
        self.x[5] = clamp(self.x[5], 0.0, self.cfg.delay_max_ms / 1000.0)

    def get_params(self):
        return {
            "theta": float(self.x[0]),
            "omega": float(self.x[1]),
            "l_com": float(self.x[2]),
            "b_eq": float(self.x[3]),
            "tau_eq": float(self.x[4]),
            # compatibility aliases
            "b": float(self.x[3]),
            "tau_c": float(self.x[4]),
            "k_t": float(self.cfg.k_t_init),
            "i0": float(self.cfg.i0_init),
            "delay_sec": float(self.x[5]),
            "R": float(self.cfg.R_init),
            "k_e": float(self.cfg.k_e_init),
        }


class ObservationLPF:
    def __init__(self, tau_sec: float):
        self.tau_sec = max(float(tau_sec), 1e-5)
        self.state = {}

    def update(self, key: str, value: float, dt: float) -> float:
        value = float(value)
        if key not in self.state:
            self.state[key] = value
            return value
        dt = max(float(dt), 1e-6)
        alpha = dt / (self.tau_sec + dt)
        self.state[key] = (1.0 - alpha) * self.state[key] + alpha * value
        return float(self.state[key])


class FitConvergenceMonitor:
    def __init__(self, cfg: BridgeConfig):
        self.window_sec = max(float(cfg.fit_conv_window_sec), 0.2)
        self.hold_sec = max(float(cfg.fit_conv_hold_sec), 0.1)
        self.th_theta = float(cfg.fit_conv_rms_theta)
        self.th_omega = float(cfg.fit_conv_rms_omega)
        self.th_alpha = float(cfg.fit_conv_rms_alpha)
        self.samples = deque()
        self.fit_done = False
        self.fit_complete = False
        self.fit_complete_wall = None
        self.fit_final_params = None
        self.hold_start = None

    def update(self, wall_t: float, e_theta: float, e_omega: float, e_alpha: float):
        self.samples.append((float(wall_t), float(e_theta), float(e_omega), float(e_alpha)))
        tmin = float(wall_t) - self.window_sec
        while self.samples and self.samples[0][0] < tmin:
            self.samples.popleft()
        if len(self.samples) < 5:
            return None
        arr = np.array([[s[1], s[2], s[3]] for s in self.samples], dtype=float)
        rms = np.sqrt(np.mean(arr * arr, axis=0))
        is_good = bool(rms[0] <= self.th_theta and rms[1] <= self.th_omega and rms[2] <= self.th_alpha)
        if self.fit_done:
            return rms
        if is_good:
            if self.hold_start is None:
                self.hold_start = float(wall_t)
            elif (float(wall_t) - self.hold_start) >= self.hold_sec:
                self.fit_done = True
                self.fit_complete = True
                self.fit_complete_wall = float(wall_t)
        else:
            self.hold_start = None
        return rms
