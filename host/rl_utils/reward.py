from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    w_theta: float = 1.0
    w_omega: float = 1.0
    w_alpha: float = 0.5
    w_param: float = 0.01
    invalid_penalty: float = 5.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_reward(
    theta_sim: np.ndarray,
    theta_real: np.ndarray,
    omega_sim: np.ndarray,
    omega_real: np.ndarray,
    alpha_sim: np.ndarray,
    alpha_real: np.ndarray,
    param_vec: np.ndarray,
    weights: RewardWeights,
    valid: bool,
) -> tuple[float, dict[str, float]]:
    rmse_theta = rmse(theta_sim, theta_real)
    rmse_omega = rmse(omega_sim, omega_real)
    rmse_alpha = rmse(alpha_sim, alpha_real)
    param_pen = float(np.linalg.norm(param_vec))
    base = (
        weights.w_theta * rmse_theta
        + weights.w_omega * rmse_omega
        + weights.w_alpha * rmse_alpha
        + weights.w_param * param_pen
    )
    if not valid:
        base += weights.invalid_penalty
    reward = -float(base)
    return reward, {
        "rmse_theta": rmse_theta,
        "rmse_omega": rmse_omega,
        "rmse_alpha": rmse_alpha,
        "param_penalty": param_pen,
    }
