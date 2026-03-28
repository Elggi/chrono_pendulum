import math
from collections import deque

import numpy as np

from .config import BridgeConfig
from .utils import clamp


class DelayCompensator:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.delay_sec = cfg.delay_init_ms / 1000.0
        self.cmd_hist = deque()
        self.pwm_hist = deque()
        self.last_update_wall = 0.0

    def push(self, wall_t: float, cmd_u: float, hw_pwm: float):
        self.cmd_hist.append((wall_t, float(cmd_u)))
        self.pwm_hist.append((wall_t, float(hw_pwm)))
        tmin = wall_t - self.cfg.delay_buffer_sec
        while self.cmd_hist and self.cmd_hist[0][0] < tmin:
            self.cmd_hist.popleft()
        while self.pwm_hist and self.pwm_hist[0][0] < tmin:
            self.pwm_hist.popleft()

    def _interp(self, hist, t):
        xs = list(hist)
        if not xs:
            return 0.0
        if t <= xs[0][0]:
            return xs[0][1]
        if t >= xs[-1][0]:
            return xs[-1][1]
        for i in range(1, len(xs)):
            t0, y0 = xs[i - 1]
            t1, y1 = xs[i]
            if t0 <= t <= t1:
                if abs(t1 - t0) < 1e-12:
                    return y0
                a = (t - t0) / (t1 - t0)
                return (1.0 - a) * y0 + a * y1
        return xs[-1][1]

    def estimate_delay(self, wall_now: float):
        if not self.cfg.auto_delay_comp:
            return self.delay_sec
        if wall_now - self.last_update_wall < 1.0 / max(self.cfg.delay_update_hz, 1e-9):
            return self.delay_sec
        self.last_update_wall = wall_now
        if len(self.cmd_hist) < 20 or len(self.pwm_hist) < 20:
            return self.delay_sec
        t0 = max(self.cmd_hist[0][0], self.pwm_hist[0][0])
        t1 = min(self.cmd_hist[-1][0], self.pwm_hist[-1][0])
        if (t1 - t0) < 0.4:
            return self.delay_sec
        dt = 0.01
        ts = np.arange(t0, t1, dt)
        if len(ts) < 20:
            return self.delay_sec
        cmd = np.array([self._interp(self.cmd_hist, t) for t in ts], dtype=float)
        pwm = np.array([self._interp(self.pwm_hist, t) for t in ts], dtype=float)
        cmd -= np.mean(cmd)
        pwm -= np.mean(pwm)
        if np.std(cmd) < 1e-6 or np.std(pwm) < 1e-6:
            return self.delay_sec
        max_lag = int((self.cfg.delay_max_ms / 1000.0) / dt)
        best_lag = 0
        best_score = -1e18
        for lag in range(max_lag + 1):
            if lag >= len(cmd) - 2:
                break
            c = cmd[:-lag] if lag > 0 else cmd
            p = pwm[lag:] if lag > 0 else pwm
            score = float(np.dot(c, p)) / max(len(c), 1)
            if score > best_score:
                best_score = score
                best_lag = lag
        measured = best_lag * dt
        self.delay_sec = (1.0 - self.cfg.delay_smooth_alpha) * self.delay_sec + self.cfg.delay_smooth_alpha * measured
        return self.delay_sec

    def get_delayed_cmd(self, wall_now: float, fallback: float):
        target = wall_now - self.delay_sec
        if len(self.cmd_hist) < 2:
            return fallback
        return self._interp(self.cmd_hist, target)


class CPREstimator:
    def __init__(self):
        self.prev_angle = None
        self.angle_unwrapped = 0.0
        self.angle_travel = 0.0
        self.rev_index = 0
        self.rev_enc_anchor = None
        self.rev_angle_anchor = 0.0
        self.samples = []
        self.last_cpr = np.nan
        self.motion_started = False

    def reset_revolution_window(self, enc_count):
        self.rev_enc_anchor = enc_count
        self.rev_angle_anchor = self.angle_unwrapped
        self.last_cpr = np.nan
        self.motion_started = False

    def update(self, angle_wrapped, enc_count):
        if self.prev_angle is None:
            self.prev_angle = angle_wrapped
            self.reset_revolution_window(enc_count)
            return
        d = angle_wrapped - self.prev_angle
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        self.angle_unwrapped += d
        self.angle_travel += abs(d)
        self.prev_angle = angle_wrapped
        theta_window = self.angle_unwrapped - self.rev_angle_anchor
        if not self.motion_started:
            if abs(theta_window) < math.radians(5.0):
                self.rev_enc_anchor = enc_count
            else:
                self.motion_started = True
        while abs(theta_window) >= (2.0 * math.pi):
            if self.rev_enc_anchor is not None:
                delta = abs(enc_count - self.rev_enc_anchor)
                if delta > 1:
                    self.samples.append(float(delta))
                    self.last_cpr = float(delta)
            self.rev_enc_anchor = enc_count
            self.rev_index += 1
            self.rev_angle_anchor += math.copysign(2.0 * math.pi, theta_window)
            theta_window = self.angle_unwrapped - self.rev_angle_anchor

    @property
    def mean(self):
        if len(self.samples) == 0:
            return np.nan
        return float(np.mean(self.samples))


class OnlineParameterEKF:
    """
    augmented state:
    x = [theta, omega, J, b, tau_c, mgl, k_t, i0, delay_sec]
    """
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.x = np.array([
            0.0, 0.0,
            cfg.J_init,
            cfg.b_init,
            cfg.tau_c_init,
            cfg.mgl_init,
            cfg.k_t_init,
            cfg.i0_init,
            cfg.delay_init_ms / 1000.0,
        ], dtype=float)
        self.P = np.diag([1e-3, 1e-2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4])
        self.Q = np.diag([
            cfg.q_theta, cfg.q_omega, cfg.q_J, cfg.q_b, cfg.q_tauc,
            cfg.q_mgl, cfg.q_kt, cfg.q_i0, cfg.q_delay
        ])
        self.R = np.diag([cfg.r_theta, cfg.r_omega])
        self.best_cost = np.inf
        self.best_params = None

    def f_disc(self, x, u, dt):
        th, om, J, b, tau_c, mgl, kt, i0, dly = x
        J = max(J, self.cfg.j_min)
        i_eff = math.copysign(max(abs(u) - i0, 0.0), u)
        alpha = (kt * i_eff - b * om - tau_c * math.tanh(om / max(self.cfg.tanh_eps, 1e-9)) - mgl * math.sin(th)) / J
        xn = np.array([
            th + dt * om,
            om + dt * alpha,
            J,
            max(b, 0.0),
            max(tau_c, 0.0),
            max(mgl, 0.0),
            max(kt, 0.0),
            max(i0, 0.0),
            max(dly, 0.0),
        ], dtype=float)
        xn[2] = max(xn[2], self.cfg.j_min)
        return xn

    def h_meas(self, x):
        return np.array([x[0], x[1]], dtype=float)

    def numeric_jacobian(self, fun, x, eps=1e-6):
        x = np.asarray(x, dtype=float)
        y0 = np.asarray(fun(x), dtype=float)
        m, n = len(y0), len(x)
        J = np.zeros((m, n), dtype=float)
        for i in range(n):
            dx = np.zeros(n, dtype=float)
            step = eps * max(1.0, abs(x[i]))
            dx[i] = step
            y1 = np.asarray(fun(x + dx), dtype=float)
            y2 = np.asarray(fun(x - dx), dtype=float)
            J[:, i] = (y1 - y2) / (2.0 * step)
        return J

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
        self.x[2] = clamp(self.x[2], self.cfg.j_min, self.cfg.J_max)
        self.x[3] = clamp(self.x[3], 0.0, self.cfg.b_max)
        self.x[4] = clamp(self.x[4], 0.0, self.cfg.tau_c_max)
        self.x[5] = clamp(self.x[5], 0.0, self.cfg.mgl_max)
        self.x[6] = clamp(self.x[6], 0.0, self.cfg.k_t_max)
        self.x[7] = clamp(self.x[7], 0.0, self.cfg.i0_max)
        self.x[8] = clamp(self.x[8], 0.0, self.cfg.delay_max_ms / 1000.0)

    def get_params(self):
        return {
            "theta": float(self.x[0]),
            "omega": float(self.x[1]),
            "J": float(self.x[2]),
            "b": float(self.x[3]),
            "tau_c": float(self.x[4]),
            "mgl": float(self.x[5]),
            "k_t": float(self.x[6]),
            "i0": float(self.x[7]),
            "delay_sec": float(self.x[8]),
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
