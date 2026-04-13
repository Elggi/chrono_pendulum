"""Pipeline C-1 environment: RL for simulator calibration parameters."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class CalibrationTargets:
    """Reference trajectory snippets from real hardware."""

    theta: np.ndarray
    omega: np.ndarray
    u: np.ndarray


class CalibrationEnv(gym.Env):
    """Agent adjusts calibration parameters to minimize rollout mismatch."""

    metadata = {"render_modes": []}

    def __init__(self, targets: CalibrationTargets) -> None:
        super().__init__()
        self.targets = targets
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32)
        self.params = np.zeros(5, dtype=np.float32)
        self.step_idx = 0

    def _simulate_step(self, theta: float, omega: float, u: float) -> tuple[float, float]:
        gain, visc, coul, delay, residual = self.params
        tau = (1.0 + gain) * u - visc * omega - coul * np.sign(omega) + residual * np.sin(theta)
        omega_next = omega + 0.002 * (tau - np.sin(theta))
        theta_next = theta + 0.002 * omega_next
        if delay > 0:
            theta_next -= 0.0005 * delay
        return theta_next, omega_next

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.params[:] = 0.0
        self.step_idx = 0
        obs = np.array([self.targets.theta[0], self.targets.omega[0], self.targets.u[0], *self.params], dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        self.params = np.clip(self.params + action, -1.0, 1.0)
        i = self.step_idx
        theta_hat, omega_hat = self._simulate_step(self.targets.theta[i], self.targets.omega[i], self.targets.u[i])
        err = np.array([theta_hat - self.targets.theta[i + 1], omega_hat - self.targets.omega[i + 1]])
        reward = -float(np.dot(err, err))
        self.step_idx += 1
        done = self.step_idx >= len(self.targets.theta) - 2
        obs = np.array([theta_hat, omega_hat, self.targets.u[self.step_idx], *self.params], dtype=np.float32)
        return obs, reward, done, False, {"rmse": float(np.sqrt(np.mean(err**2)))}
