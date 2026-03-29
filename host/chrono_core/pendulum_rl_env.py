from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .config import BridgeConfig

RL_PARAM_KEYS = ["l_com", "J_cm_base", "b_eq", "tau_eq", "k_t", "i0", "R", "k_e"]


@dataclass
class ReplayTrajectory:
    name: str
    df: pd.DataFrame
    delay_sec_est: float


@dataclass
class ReplaySampleResult:
    loss: float
    metrics: dict[str, float]


def _median_dt(t: np.ndarray) -> float:
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 0.001


def estimate_delay_from_csv(df: pd.DataFrame, max_delay_sec: float = 0.2) -> float:
    if "cmd_u_raw" not in df or "hw_pwm" not in df or "sim_time" not in df:
        return 0.0
    cmd = df["cmd_u_raw"].to_numpy(dtype=float)
    pwm = df["hw_pwm"].to_numpy(dtype=float)
    t = df["sim_time"].to_numpy(dtype=float)
    if len(cmd) < 8:
        return 0.0
    dt = _median_dt(t)
    max_lag = max(1, int(max_delay_sec / max(dt, 1e-4)))
    cmd = cmd - np.nanmean(cmd)
    pwm = pwm - np.nanmean(pwm)
    best_lag = 0
    best_score = -np.inf
    for lag in range(0, max_lag + 1):
        if lag >= len(cmd) - 3:
            break
        c = cmd[:-lag or None]
        p = pwm[lag:]
        score = float(np.dot(c, p) / (np.linalg.norm(c) * np.linalg.norm(p) + 1e-9))
        if score > best_score:
            best_score, best_lag = score, lag
    return float(best_lag * dt)


def shift_signal_by_delay(t: np.ndarray, u: np.ndarray, delay_sec: float) -> np.ndarray:
    return np.interp(t - delay_sec, t, u, left=u[0], right=u[-1])


def params_from_cfg_and_json(cfg: BridgeConfig, parameter_json: str | None) -> dict[str, float]:
    p = {
        "l_com": cfg.l_com_init,
        "J_cm_base": cfg.J_cm_base,
        "b_eq": cfg.b_eq_init,
        "tau_eq": cfg.tau_eq_init,
        "k_t": cfg.k_t_init,
        "i0": cfg.i0_init,
        "R": cfg.R_init,
        "k_e": cfg.k_e_init,
    }
    if parameter_json and Path(parameter_json).exists():
        data = json.loads(Path(parameter_json).read_text())
        for k in RL_PARAM_KEYS + ["delay_sec"]:
            if k in data:
                p[k] = float(data[k])
    return p


def simulate_replay(df: pd.DataFrame, params: dict[str, float], delay_sec: float) -> dict[str, np.ndarray]:
    t = df["sim_time"].to_numpy(dtype=float)
    theta_real = df["theta_real"].to_numpy(dtype=float)
    omega_real = df["omega_real"].to_numpy(dtype=float)
    bus_v = df.get("bus_v_filtered", pd.Series(np.full(len(df), 7.4))).to_numpy(dtype=float)
    cmd = df["cmd_u_raw"].to_numpy(dtype=float)
    pwm_eff = shift_signal_by_delay(t, cmd, delay_sec)

    th = np.zeros_like(t)
    om = np.zeros_like(t)
    al = np.zeros_like(t)
    i_pred = np.zeros_like(t)
    v_applied = np.zeros_like(t)
    tau_motor = np.zeros_like(t)
    tau_res = np.zeros_like(t)

    th[0] = theta_real[0]
    om[0] = omega_real[0]
    m_total = 0.220
    g = 9.81
    pwm_limit = 255.0
    eps = 0.05

    for k in range(len(t) - 1):
        dt = max(float(t[k + 1] - t[k]), 1e-4)
        duty = np.clip(pwm_eff[k] / pwm_limit, -1.0, 1.0)
        v = duty * (bus_v[k] if np.isfinite(bus_v[k]) else 7.4)
        i_raw = (v - params["k_e"] * om[k]) / max(params["R"], 1e-6)
        i_eff = math.copysign(max(abs(i_raw) - max(params["i0"], 0.0), 0.0), i_raw)
        tau_m = params["k_t"] * i_eff
        tau_r = params["b_eq"] * om[k] + params["tau_eq"] * math.tanh(om[k] / eps)
        tau_g = m_total * g * params["l_com"] * math.sin(th[k])
        j_pivot = max(params["J_cm_base"] + m_total * params["l_com"] ** 2, 1e-6)
        alpha = (tau_m - tau_r - tau_g) / j_pivot
        om[k + 1] = om[k] + dt * alpha
        th[k + 1] = th[k] + dt * om[k + 1]
        al[k] = alpha
        i_pred[k] = i_eff
        v_applied[k] = v
        tau_motor[k] = tau_m
        tau_res[k] = tau_r
    al[-1] = al[-2] if len(al) > 1 else 0.0
    i_pred[-1] = i_pred[-2] if len(i_pred) > 1 else 0.0
    v_applied[-1] = v_applied[-2] if len(v_applied) > 1 else 0.0
    tau_motor[-1] = tau_motor[-2] if len(tau_motor) > 1 else 0.0
    tau_res[-1] = tau_res[-2] if len(tau_res) > 1 else 0.0
    return {
        "theta": th,
        "omega": om,
        "alpha": al,
        "cmd_u_delayed": pwm_eff,
        "i_pred": i_pred,
        "v_applied": v_applied,
        "tau_motor": tau_motor,
        "tau_res": tau_res,
    }


def replay_loss(df: pd.DataFrame, sim: dict[str, np.ndarray]) -> ReplaySampleResult:
    theta = df["theta_real"].to_numpy(dtype=float)
    omega = df["omega_real"].to_numpy(dtype=float)
    alpha = df["alpha_real"].to_numpy(dtype=float)
    pwm_hw = df.get("hw_pwm", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    theta_rmse = float(np.sqrt(np.mean((sim["theta"] - theta) ** 2)))
    omega_rmse = float(np.sqrt(np.mean((sim["omega"] - omega) ** 2)))
    alpha_rmse = float(np.sqrt(np.mean((sim["alpha"] - alpha) ** 2)))
    pwm_rmse = float(np.sqrt(np.mean((sim["cmd_u_delayed"] - pwm_hw) ** 2)))
    theta_bias = float(np.mean(sim["theta"] - theta))
    omega_bias = float(np.mean(sim["omega"] - omega))
    loss = 4.0 * theta_rmse + 2.0 * omega_rmse + 1.0 * alpha_rmse + 0.05 * pwm_rmse
    return ReplaySampleResult(loss=loss, metrics={
        "rmse_theta": theta_rmse,
        "rmse_omega": omega_rmse,
        "rmse_alpha": alpha_rmse,
        "rmse_pwm": pwm_rmse,
        "bias_theta": theta_bias,
        "bias_omega": omega_bias,
    })


def split_trajectories(files: list[Path], seed: int = 0) -> tuple[list[Path], list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    if len(files) == 1:
        return files, files, files
    n = len(files)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:] or files[-1:]
    return train, val, test


class PendulumReplayCalibrationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, trajectories: list[ReplayTrajectory], init_params: dict[str, float],
                 learn_delay: bool = False, delay_jitter_ms: float = 0.0,
                 domain_randomization: bool = True, max_refine_steps: int = 12, seed: int = 0):
        super().__init__()
        self.trajectories = trajectories
        self.learn_delay = learn_delay
        self.delay_jitter_ms = delay_jitter_ms
        self.domain_randomization = domain_randomization
        self.max_refine_steps = max_refine_steps
        self.rng = np.random.default_rng(seed)
        self.param_keys = RL_PARAM_KEYS + (["delay_sec"] if learn_delay else [])
        base = np.array([init_params[k] for k in RL_PARAM_KEYS], dtype=np.float32)
        if learn_delay:
            base = np.concatenate([base, np.array([init_params.get("delay_sec", 0.12)], dtype=np.float32)])
        self.base = base
        self.params = self.base.copy()
        self.delta_scale = np.array([0.005, 0.0008, 0.01, 0.01, 0.01, 0.005, 0.05, 0.005] + ([0.003] if learn_delay else []), dtype=np.float32)
        self.low = np.array([0.03, 1e-4, 0.0, 0.0, 0.05, 0.0, 0.2, 0.0] + ([0.04] if learn_delay else []), dtype=np.float32)
        self.high = np.array([0.45, 0.02, 2.0, 1.5, 1.5, 0.6, 20.0, 0.8] + ([0.2] if learn_delay else []), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.param_keys),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(len(self.param_keys) + 10,), dtype=np.float32)
        self.last_loss = 0.0
        self.last_metrics: dict[str, float] = {}
        self.step_idx = 0

    def _params_dict(self) -> dict[str, float]:
        out = {k: float(v) for k, v in zip(self.param_keys, self.params)}
        return out

    def _evaluate(self) -> tuple[float, dict[str, float]]:
        metrics_all = []
        total = 0.0
        p = self._params_dict()
        for tr in self.trajectories:
            d = tr.delay_sec_est
            if self.learn_delay:
                d = p["delay_sec"]
            elif self.domain_randomization and self.delay_jitter_ms > 0.0:
                d = d + self.rng.uniform(-self.delay_jitter_ms, self.delay_jitter_ms) * 1e-3
            sim = simulate_replay(tr.df, p, d)
            r = replay_loss(tr.df, sim)
            total += r.loss
            metrics_all.append(r.metrics)
        avg = total / max(len(metrics_all), 1)
        m = {k: float(np.mean([mm[k] for mm in metrics_all])) for k in metrics_all[0]}
        return float(avg), m

    def _obs(self) -> np.ndarray:
        norm = (self.params - self.base) / (np.abs(self.base) + 1e-6)
        feats = np.array([
            self.last_metrics.get("rmse_theta", 0.0),
            self.last_metrics.get("rmse_omega", 0.0),
            self.last_metrics.get("rmse_alpha", 0.0),
            self.last_metrics.get("rmse_pwm", 0.0),
            self.last_metrics.get("bias_theta", 0.0),
            self.last_metrics.get("bias_omega", 0.0),
            self.last_loss,
            float(self.step_idx) / max(self.max_refine_steps, 1),
            float(np.linalg.norm(norm)),
            1.0,
        ], dtype=np.float32)
        return np.concatenate([norm.astype(np.float32), feats])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.params = self.base.copy()
        if self.domain_randomization:
            self.params *= self.rng.uniform(0.95, 1.05, size=self.params.shape)
        self.params = np.clip(self.params, self.low, self.high)
        self.step_idx = 0
        self.last_loss, self.last_metrics = self._evaluate()
        return self._obs(), {"loss": self.last_loss, **self.last_metrics}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        self.params = np.clip(self.params + self.delta_scale * action, self.low, self.high)
        prev_loss = self.last_loss
        self.last_loss, self.last_metrics = self._evaluate()
        improvement = prev_loss - self.last_loss
        action_penalty = 0.01 * float(np.mean(action ** 2))
        reward = float(improvement - 0.1 * self.last_loss - action_penalty)
        self.step_idx += 1
        done = self.step_idx >= self.max_refine_steps
        if done and self.last_loss < 0.2:
            reward += 0.5
        return self._obs(), reward, done, False, {"loss": self.last_loss, **self.last_metrics}


def load_replay_trajectories(csv_paths: list[Path], delay_override: float | None = None) -> tuple[list[ReplayTrajectory], dict[str, float]]:
    out: list[ReplayTrajectory] = []
    delays: dict[str, float] = {}
    for p in csv_paths:
        df = pd.read_csv(p)
        delay = delay_override if delay_override is not None else estimate_delay_from_csv(df)
        out.append(ReplayTrajectory(name=p.name, df=df, delay_sec_est=float(delay)))
        delays[p.name] = float(delay)
    return out, delays


def deterministic_prefit(trajectories: list[ReplayTrajectory], init_params: dict[str, float],
                         iters: int = 80, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    best = dict(init_params)

    def eval_p(pp: dict[str, float]) -> float:
        total = 0.0
        for tr in trajectories:
            sim = simulate_replay(tr.df, pp, tr.delay_sec_est)
            total += replay_loss(tr.df, sim).loss
        return total / max(1, len(trajectories))

    best_loss = eval_p(best)
    for i in range(iters):
        cand = dict(best)
        scale = max(0.01, 0.2 * (1.0 - i / max(iters, 1)))
        for k in RL_PARAM_KEYS:
            cand[k] = max(1e-6, cand[k] * (1.0 + rng.uniform(-scale, scale)))
        loss = eval_p(cand)
        if loss < best_loss:
            best, best_loss = cand, loss
    return best
