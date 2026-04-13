from __future__ import annotations

import numpy as np


def euler_step(theta: float, omega: float, alpha: float, dt: float) -> tuple[float, float]:
    omega_n = omega + dt * alpha
    theta_n = theta + dt * omega
    return float(theta_n), float(omega_n)


def rollout(theta0: float, omega0: float, alpha_series: np.ndarray, dt_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(alpha_series)
    th = np.zeros(n)
    om = np.zeros(n)
    th[0], om[0] = theta0, omega0
    for k in range(n - 1):
        th[k + 1], om[k + 1] = euler_step(th[k], om[k], float(alpha_series[k]), float(max(dt_series[k], 1e-4)))
    return th, om
