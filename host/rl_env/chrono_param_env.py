from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chrono_core.chrono_rigid_pendulum import ChronoRigidPendulum, TorqueController, load_pendulum_params
from rl_utils.data_loader import RolloutData
from rl_utils.reward import RewardWeights, compute_reward


class ChronoParamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        trajectories: list[RolloutData],
        motor_torque_json: str,
        calibration_json: str,
        action_scale: np.ndarray | None = None,
        reward_weights: RewardWeights | None = None,
    ):
        super().__init__()
        if not trajectories:
            raise ValueError("ChronoParamEnv requires at least one trajectory")
        self.trajectories = trajectories
        self.motor_torque_json = Path(motor_torque_json)
        self.calibration_json = Path(calibration_json)
        self.weights = reward_weights or RewardWeights()

        motor = json.loads(self.motor_torque_json.read_text(encoding="utf-8"))
        dyn = motor.get("dynamic_parameters", {})
        self.base_params = np.array([
            float(dyn.get("J", 0.02)),
            float(dyn.get("b_eq", 0.01)),
            float(dyn.get("tau_eq", 0.0)),
            float(dyn.get("K_I", 0.06)),
        ])
        self.bounds_low = np.array(motor.get("parameter_bounds", {}).get("J", [0.001, 0.2])[0:1] +
                                   motor.get("parameter_bounds", {}).get("b_eq", [0.0, 2.0])[0:1] +
                                   motor.get("parameter_bounds", {}).get("tau_eq", [0.0, 1.0])[0:1] +
                                   motor.get("parameter_bounds", {}).get("K_I", [0.0, 2.0])[0:1], dtype=float)
        self.bounds_high = np.array(motor.get("parameter_bounds", {}).get("J", [0.001, 0.2])[1:2] +
                                    motor.get("parameter_bounds", {}).get("b_eq", [0.0, 2.0])[1:2] +
                                    motor.get("parameter_bounds", {}).get("tau_eq", [0.0, 1.0])[1:2] +
                                    motor.get("parameter_bounds", {}).get("K_I", [0.0, 2.0])[1:2], dtype=float)

        self.action_scale = action_scale if action_scale is not None else np.array([0.01, 0.005, 0.005, 0.01])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32)

        self.current_params = self.base_params.copy()
        self.best_reward = -np.inf
        self.last_metrics = {"rmse_theta": 0.0, "rmse_omega": 0.0, "rmse_alpha": 0.0}

        calib = json.loads(self.calibration_json.read_text(encoding="utf-8"))
        summary = calib.get("summary", {}) if isinstance(calib.get("summary"), dict) else {}
        self.imu_radius = float(summary.get("mean_radius_m", calib.get("mean_radius_m", 0.22)))

    def _simulate_with_params(self, tr: RolloutData, params: np.ndarray):
        p = load_pendulum_params(self.motor_torque_json, imu_radius=self.imu_radius)
        p.motor.K_I = float(params[3])
        p.motor.b = float(params[1])
        p.motor.tau_c = float(params[2])
        model = ChronoRigidPendulum(p, enable_collision=False)
        ctrl = TorqueController(p.motor)

        # set initial condition to real start
        model.sync_to_theta(float(tr.theta[0]))

        n = len(tr.t)
        th = np.zeros(n)
        om = np.zeros(n)
        al = np.zeros(n)
        for k in range(n):
            theta, omega, alpha = model.read_state()
            th[k], om[k], al[k] = theta, omega, alpha
            tau = ctrl.compute(float(tr.current[k]), theta, omega)
            model.apply_torque(tau)
            if k < n - 1:
                dt = max(float(tr.t[k + 1] - tr.t[k]), 1e-4)
                model.step(dt)
        return th, om, al

    def _evaluate(self, params: np.ndarray):
        rmses = []
        valid = np.all(params >= self.bounds_low) and np.all(params <= self.bounds_high)
        for tr in self.trajectories:
            th, om, al = self._simulate_with_params(tr, params)
            reward, met = compute_reward(th, tr.theta, om, tr.omega, al, tr.alpha, params, self.weights, valid)
            rmses.append((reward, met))
        avg_reward = float(np.mean([r for r, _ in rmses]))
        mean_metrics = {
            "rmse_theta": float(np.mean([m["rmse_theta"] for _, m in rmses])),
            "rmse_omega": float(np.mean([m["rmse_omega"] for _, m in rmses])),
            "rmse_alpha": float(np.mean([m["rmse_alpha"] for _, m in rmses])),
        }
        return avg_reward, mean_metrics, valid

    def _obs(self):
        return np.array([
            self.current_params[0],
            self.current_params[1],
            self.current_params[2],
            self.current_params[3],
            self.last_metrics["rmse_theta"],
            self.last_metrics["rmse_omega"],
            self.last_metrics["rmse_alpha"],
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_params = self.base_params.copy()
        reward, met, _ = self._evaluate(self.current_params)
        self.last_metrics = met
        self.best_reward = max(self.best_reward, reward)
        return self._obs(), {"reward": reward, "params": self.current_params.tolist(), **met}

    def step(self, action):
        delta = np.asarray(action, dtype=float) * self.action_scale
        self.current_params = np.clip(self.current_params + delta, self.bounds_low, self.bounds_high)
        reward, met, valid = self._evaluate(self.current_params)
        self.last_metrics = met
        self.best_reward = max(self.best_reward, reward)
        info = {
            "reward": reward,
            "best_reward": self.best_reward,
            "params": {
                "J": float(self.current_params[0]),
                "b_eq": float(self.current_params[1]),
                "tau_eq": float(self.current_params[2]),
                "K_I": float(self.current_params[3]),
            },
            "valid": bool(valid),
            **met,
        }
        done = True  # one parameter proposal == one episode
        return self._obs(), reward, done, False, info
