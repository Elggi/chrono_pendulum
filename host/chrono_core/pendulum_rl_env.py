from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .chrono_rigid_pendulum import ChronoRigidPendulum, TorqueController, load_pendulum_params

PARAM_KEYS = ["K_I", "b", "tau_c"]


@dataclass
class ReplayTrajectory:
    t: np.ndarray
    i: np.ndarray
    theta: np.ndarray
    omega: np.ndarray


def load_replay_csv(path: str | Path) -> ReplayTrajectory:
    df = pd.read_csv(path)
    t = pd.to_numeric(df.get("time", df.get("wall_elapsed")), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    i = pd.to_numeric(df.get("input_current", df.get("ina_current_signed_mA", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "ina_current_signed_mA" in df.columns:
        i = i * 0.001
    theta = pd.to_numeric(df["theta"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    omega = pd.to_numeric(df["omega"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return ReplayTrajectory(t=t, i=i, theta=theta, omega=omega)


def chrono_rollout(traj: ReplayTrajectory, motor_json: Path, calib_json: Path, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    import json

    calib = json.loads(calib_json.read_text(encoding="utf-8"))
    summary = calib.get("summary", {}) if isinstance(calib.get("summary"), dict) else {}
    r = float(summary.get("mean_radius_m", calib.get("mean_radius_m", 0.22)))
    pend_params = load_pendulum_params(motor_json, imu_radius=r)
    pend_params.motor.K_I = float(params["K_I"])
    pend_params.motor.b = float(params["b"])
    pend_params.motor.tau_c = float(params["tau_c"])
    model = ChronoRigidPendulum(pend_params, enable_collision=False)
    ctrl = TorqueController(pend_params.motor)

    n = len(traj.t)
    th = np.zeros(n)
    om = np.zeros(n)
    for k in range(n):
        theta, omega, _ = model.read_state()
        th[k] = theta
        om[k] = omega
        tau = ctrl.compute(float(traj.i[k]), theta, omega)
        model.apply_torque(tau)
        dt = 0.001 if k == n - 1 else max(float(traj.t[k + 1] - traj.t[k]), 1e-4)
        model.step(dt)
    return th, om


class ChronoParamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, traj: ReplayTrajectory, motor_json: Path, calib_json: Path, center: dict[str, float]):
        super().__init__()
        self.traj = traj
        self.motor_json = motor_json
        self.calib_json = calib_json
        self.center = center
        self.param_keys = PARAM_KEYS
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        self.best = float("inf")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cur = dict(self.center)
        return np.array([self.cur[k] for k in self.param_keys] + [0, 0, 0], dtype=np.float32), {}

    def step(self, action):
        for i, k in enumerate(self.param_keys):
            self.cur[k] = max(0.0, float(self.cur[k] * (1.0 + 0.1 * float(action[i]))))
        th, om = chrono_rollout(self.traj, self.motor_json, self.calib_json, self.cur)
        e_th = float(np.sqrt(np.mean((th - self.traj.theta) ** 2)))
        e_om = float(np.sqrt(np.mean((om - self.traj.omega) ** 2)))
        loss = e_th + e_om
        self.best = min(self.best, loss)
        obs = np.array([self.cur[k] for k in self.param_keys] + [e_th, e_om, loss], dtype=np.float32)
        info = {"params": dict(self.cur), "loss": loss, "best_loss": self.best}
        return obs, -loss, True, False, info


def build_init_params() -> dict[str, float]:
    return {"K_I": 0.06, "b": 0.01, "tau_c": 0.0}
