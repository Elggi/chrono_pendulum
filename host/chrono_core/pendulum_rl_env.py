#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BridgeConfig

PARAM_KEYS = ["l_com", "J_cm_base", "b_eq", "tau_eq", "k_t", "i0", "R", "k_e"]


@dataclass
class ReplayTrajectory:
    name: str
    t: np.ndarray
    dt: np.ndarray
    cmd_u: np.ndarray
    hw_pwm: np.ndarray
    theta_real: np.ndarray
    omega_real: np.ndarray
    alpha_real: np.ndarray
    bus_v: np.ndarray
    current_a: np.ndarray
    power_w: np.ndarray
    delay_sec_est: float


def _safe_col(df: pd.DataFrame, col: str, fallback: float = 0.0):
    if col not in df.columns:
        return np.full(len(df), float(fallback), dtype=float)
    out = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    out[~np.isfinite(out)] = fallback
    return out


def _gradient(x: np.ndarray, dt: np.ndarray):
    if len(x) < 2:
        return np.zeros_like(x)
    h = np.maximum(np.asarray(dt, dtype=float), 1e-6)
    t = np.cumsum(h)
    t -= t[0]
    # np.gradient with time-axis coordinates is numerically safer than passing raw dt.
    return np.gradient(x, t, edge_order=1)


def _sanitize_timeseries(arr: np.ndarray):
    out = np.asarray(arr, dtype=float).copy()
    n = len(out)
    if n == 0:
        return out
    good = np.isfinite(out)
    if np.all(good):
        return out
    if not np.any(good):
        out[:] = 0.0
        return out
    idx = np.arange(n, dtype=float)
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _winsorize_abs(x: np.ndarray, q: float = 99.5):
    y = np.asarray(x, dtype=float).copy()
    if len(y) == 0:
        return y
    lim = float(np.nanpercentile(np.abs(y), q))
    if np.isfinite(lim) and lim > 0.0:
        y = np.clip(y, -lim, lim)
    return y


def estimate_delay_from_signals(t: np.ndarray, cmd_u: np.ndarray, hw_pwm: np.ndarray, max_delay_sec: float = 0.25):
    if len(t) < 3:
        return 0.0
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.001
    dt = max(dt, 1e-4)
    max_lag = int(max_delay_sec / dt)
    cmd = cmd_u - np.mean(cmd_u)
    pwm = hw_pwm - np.mean(hw_pwm)
    best_lag = 0
    best_score = -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            a = cmd
            b = pwm
        else:
            a = cmd[:-lag]
            b = pwm[lag:]
        if len(a) < 4:
            continue
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return float(best_lag * dt)


def shifted_signal(t: np.ndarray, signal: np.ndarray, delay_sec: float):
    tgt = np.clip(t - float(delay_sec), t[0], t[-1])
    return np.interp(tgt, t, signal)


def simulate_trajectory(traj: ReplayTrajectory, params: dict[str, float], cfg: BridgeConfig, delay_sec: float):
    t, dt = traj.t, traj.dt
    n = len(t)
    theta = np.zeros(n, dtype=float)
    omega = np.zeros(n, dtype=float)
    alpha = np.zeros(n, dtype=float)
    i_pred = np.zeros(n, dtype=float)
    v_applied = np.zeros(n, dtype=float)
    tau_motor = np.zeros(n, dtype=float)
    tau_visc = np.zeros(n, dtype=float)
    tau_coul = np.zeros(n, dtype=float)
    tau_res = np.zeros(n, dtype=float)
    cmd_delayed = shifted_signal(t, traj.cmd_u, delay_sec)

    theta[0] = float(traj.theta_real[0])
    omega[0] = float(traj.omega_real[0])

    m_total = cfg.link_mass + cfg.imu_mass
    J_pivot = max(float(params["J_cm_base"] + m_total * (params["l_com"] ** 2)), 1e-6)

    for k in range(n - 1):
        h = max(float(dt[k]), 1e-6)
        u = cmd_delayed[k]
        duty = np.clip(u / max(cfg.pwm_limit, 1e-9), -1.0, 1.0)
        vb = traj.bus_v[k] if np.isfinite(traj.bus_v[k]) and traj.bus_v[k] > 0.0 else cfg.nominal_bus_voltage
        v = duty * vb
        i_raw = (v - params["k_e"] * omega[k]) / max(params["R"], 1e-6)
        i_eff = math.copysign(max(abs(i_raw) - max(params["i0"], 0.0), 0.0), i_raw)
        tm = params["k_t"] * i_eff
        tv = params["b_eq"] * omega[k]
        tc = params["tau_eq"] * math.tanh(omega[k] / max(cfg.tanh_eps, 1e-6))
        tg = m_total * cfg.gravity * params["l_com"] * math.sin(theta[k])
        domega = (tm - tv - tc - tg) / J_pivot

        theta[k + 1] = theta[k] + h * omega[k]
        omega[k + 1] = omega[k] + h * domega

        i_pred[k] = i_eff
        v_applied[k] = v
        tau_motor[k] = tm
        tau_visc[k] = tv
        tau_coul[k] = tc
        tau_res[k] = tv + tc

    if n > 1:
        alpha[:] = _gradient(omega, dt)
        i_pred[-1] = i_pred[-2]
        v_applied[-1] = v_applied[-2]
        tau_motor[-1] = tau_motor[-2]
        tau_visc[-1] = tau_visc[-2]
        tau_coul[-1] = tau_coul[-2]
        tau_res[-1] = tau_res[-2]

    return {
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "cmd_delayed": cmd_delayed,
        "i_pred": i_pred,
        "v_applied": v_applied,
        "tau_motor": tau_motor,
        "tau_visc": tau_visc,
        "tau_coul": tau_coul,
        "tau_res": tau_res,
    }


def compute_error_features(traj: ReplayTrajectory, sim: dict[str, np.ndarray], delay_quality: float = 1.0):
    e_th = _winsorize_abs(sim["theta"] - traj.theta_real, q=99.5)
    e_om = _winsorize_abs(sim["omega"] - traj.omega_real, q=99.5)
    e_al = _winsorize_abs(sim["alpha"] - traj.alpha_real, q=99.5)

    rmse_theta = float(np.sqrt(np.mean(e_th ** 2)))
    rmse_omega = float(np.sqrt(np.mean(e_om ** 2)))
    rmse_alpha = float(np.sqrt(np.mean(e_al ** 2)))

    bias_theta = float(np.mean(e_th))
    bias_omega = float(np.mean(e_om))
    peak_amp_mismatch = float(np.max(np.abs(sim["theta"])) - np.max(np.abs(traj.theta_real)))

    return {
        "rmse_theta": rmse_theta,
        "rmse_omega": rmse_omega,
        "rmse_alpha": rmse_alpha,
        "bias_theta": bias_theta,
        "bias_omega": bias_omega,
        "peak_amp_mismatch": peak_amp_mismatch,
        "delay_quality": float(delay_quality),
    }


def weighted_loss(feat: dict[str, float], weights: dict[str, float]):
    return float(
        weights.get("theta", 5.0) * feat["rmse_theta"]
        + weights.get("omega", 2.5) * feat["rmse_omega"]
        + weights.get("alpha", 0.8) * feat["rmse_alpha"]
        + weights.get("bias_theta", 0.4) * abs(feat["bias_theta"])
        + weights.get("bias_omega", 0.3) * abs(feat["bias_omega"])
        + weights.get("peak", 0.2) * abs(feat["peak_amp_mismatch"])
        + weights.get("delay_quality", 0.2) * (1.0 - max(0.0, min(1.0, feat["delay_quality"])))
    )


def load_replay_csv(path: str | Path, cfg: BridgeConfig, delay_override: float | None = None):
    p = Path(path)
    df = pd.read_csv(p)
    if "wall_elapsed" in df.columns:
        t = _safe_col(df, "wall_elapsed")
    elif "wall_time" in df.columns:
        wt = _safe_col(df, "wall_time")
        t = wt - wt[0] if len(wt) > 0 else wt
    else:
        t = np.arange(len(df), dtype=float) * cfg.step
    if len(t) < 2:
        t = np.arange(len(df), dtype=float) * cfg.step
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.maximum(dt, 1e-6)

    cmd_u = _safe_col(df, "cmd_u_raw")
    hw_pwm = _safe_col(df, "hw_pwm")
    theta_real = _safe_col(df, "theta_real", np.nan)
    omega_real = _safe_col(df, "omega_real", np.nan)
    alpha_real = _safe_col(df, "alpha_real", np.nan)

    # fallback to sim columns when real estimates are absent in old logs
    if not np.isfinite(theta_real).any():
        theta_real = _safe_col(df, "theta")
    if not np.isfinite(omega_real).any():
        omega_real = _safe_col(df, "omega")
    if not np.isfinite(alpha_real).any():
        alpha_real = _safe_col(df, "alpha")
    theta_real = _sanitize_timeseries(theta_real)
    omega_real = _sanitize_timeseries(omega_real)
    alpha_real = _sanitize_timeseries(alpha_real)
    omega_from_theta = _gradient(theta_real, dt)
    alpha_from_omega = _gradient(omega_from_theta, dt)

    # If runtime logging briefly broke real-state channels, repair them from theta.
    if float(np.nanpercentile(np.abs(omega_real), 99.5)) > 80.0 and float(np.nanpercentile(np.abs(omega_from_theta), 99.5)) < 50.0:
        omega_real = omega_from_theta
    if float(np.nanpercentile(np.abs(alpha_real), 99.5)) > 800.0 and float(np.nanpercentile(np.abs(alpha_from_omega), 99.5)) < 300.0:
        alpha_real = alpha_from_omega

    omega_real = _winsorize_abs(omega_real, q=99.5)
    alpha_real = _winsorize_abs(alpha_real, q=99.5)

    bus_v = _safe_col(df, "bus_v_filtered", cfg.nominal_bus_voltage)
    current_a = _safe_col(df, "current_filtered_A", 0.0)
    power_w = _safe_col(df, "power_raw_W", 0.0)

    if delay_override is not None:
        delay_sec = float(delay_override)
    elif "delay_sec_est" in df.columns:
        delay_arr = _safe_col(df, "delay_sec_est", cfg.delay_init_ms / 1000.0)
        delay_sec = float(np.median(delay_arr)) if len(delay_arr) > 0 else (cfg.delay_init_ms / 1000.0)
    elif "delay_ms" in df.columns:
        delay_ms_arr = _safe_col(df, "delay_ms", cfg.delay_init_ms)
        delay_sec = 1e-3 * float(np.median(delay_ms_arr)) if len(delay_ms_arr) > 0 else (cfg.delay_init_ms / 1000.0)
    else:
        delay_sec = cfg.delay_init_ms / 1000.0

    return ReplayTrajectory(
        name=p.name,
        t=t,
        dt=dt,
        cmd_u=cmd_u,
        hw_pwm=hw_pwm,
        theta_real=theta_real,
        omega_real=omega_real,
        alpha_real=alpha_real,
        bus_v=bus_v,
        current_a=current_a,
        power_w=power_w,
        delay_sec_est=delay_sec,
    )


def default_param_bounds(center: dict[str, float], learn_delay: bool = False):
    bounds = {
        "l_com": (0.03, 0.45),
        "J_cm_base": (1e-4, 0.05),
        "b_eq": (0.0, 2.0),
        "tau_eq": (0.0, 1.5),
        "k_t": (0.01, 1.0),
        "i0": (0.0, 1.0),
        "R": (0.1, 30.0),
        "k_e": (0.001, 1.0),
    }
    if learn_delay:
        bounds["delay_sec"] = (0.0, 0.25)
    # tighten around center if too wide.
    for k in list(bounds.keys()):
        c = float(center.get(k, (bounds[k][0] + bounds[k][1]) * 0.5))
        lo, hi = bounds[k]
        r = max(abs(c) * 0.7, (hi - lo) * 0.08)
        bounds[k] = (max(lo, c - r), min(hi, c + r))
    return bounds


class PendulumRLEnv:
    """Offline episodic replay environment.

    Action is a normalized parameter delta. The environment replays one or more
    trajectories, computes mismatch statistics, and returns reward based on
    improvement and regularization.
    """

    def __init__(
        self,
        trajectories: list[ReplayTrajectory],
        cfg: BridgeConfig,
        init_params: dict[str, float],
        learn_delay: bool = False,
        delay_jitter_ms: float = 0.0,
        domain_randomization: bool = True,
        seed: int = 0,
        max_refine_steps: int = 12,
        reward_weights: dict[str, float] | None = None,
    ):
        self.trajectories = trajectories
        self.cfg = cfg
        self.learn_delay = bool(learn_delay)
        self.delay_jitter_ms = float(delay_jitter_ms)
        self.domain_randomization = bool(domain_randomization)
        self.rng = np.random.default_rng(seed)
        self.max_refine_steps = int(max_refine_steps)
        self.reward_weights = reward_weights or {}

        self.param_keys = list(PARAM_KEYS)
        if self.learn_delay:
            self.param_keys.append("delay_sec")
        self.center = {k: float(init_params[k]) for k in self.param_keys if k in init_params}
        if self.learn_delay and "delay_sec" not in self.center:
            self.center["delay_sec"] = float(np.mean([t.delay_sec_est for t in trajectories]))

        self.bounds = default_param_bounds(self.center, learn_delay=self.learn_delay)
        self.state_dim = len(self.param_keys) + 8
        self.action_dim = len(self.param_keys)
        self.reset()

    def _pack_params(self):
        return {k: float(v) for k, v in zip(self.param_keys, self.param_vec)}

    def _normalize_params(self):
        out = []
        for i, k in enumerate(self.param_keys):
            lo, hi = self.bounds[k]
            out.append(2.0 * (self.param_vec[i] - lo) / max(hi - lo, 1e-9) - 1.0)
        return np.asarray(out, dtype=float)

    def _apply_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        for i, k in enumerate(self.param_keys):
            lo, hi = self.bounds[k]
            step = 0.08 * (hi - lo)
            self.param_vec[i] = np.clip(self.param_vec[i] + step * action[i], lo, hi)

    def _rollout_loss(self, params: dict[str, float]):
        feats = []
        losses = []
        jitter = self.rng.uniform(-self.delay_jitter_ms, self.delay_jitter_ms) * 1e-3 if self.domain_randomization else 0.0
        for traj in self.trajectories:
            d = float(params.get("delay_sec", traj.delay_sec_est + jitter))
            sim = simulate_trajectory(traj, params, self.cfg, delay_sec=max(0.0, d))
            f = compute_error_features(traj, sim, delay_quality=1.0)
            feats.append(f)
            losses.append(weighted_loss(f, self.reward_weights))
        loss = float(np.mean(losses)) if losses else 0.0
        feat_mean = {k: float(np.mean([f[k] for f in feats])) for k in feats[0]} if feats else {
            "rmse_theta": 0.0,
            "rmse_omega": 0.0,
            "rmse_alpha": 0.0,
            "bias_theta": 0.0,
            "bias_omega": 0.0,
            "peak_amp_mismatch": 0.0,
            "delay_quality": 1.0,
        }
        return loss, feat_mean

    def _state(self):
        p = self._normalize_params()
        f = self.last_feat
        extras = np.array([
            f["rmse_theta"], f["rmse_omega"], f["rmse_alpha"],
            f["bias_theta"], f["bias_omega"], f["peak_amp_mismatch"],
            f["delay_quality"], self.last_loss,
        ], dtype=float)
        extras = np.tanh(extras)
        return np.concatenate([p, extras]).astype(np.float32)

    def reset(self):
        self.step_idx = 0
        self.param_vec = np.asarray([self.center[k] for k in self.param_keys], dtype=float)
        if self.domain_randomization:
            for i, k in enumerate(self.param_keys):
                lo, hi = self.bounds[k]
                span = (hi - lo)
                self.param_vec[i] = np.clip(self.param_vec[i] + self.rng.normal(0.0, 0.07 * span), lo, hi)
        self.last_loss, self.last_feat = self._rollout_loss(self._pack_params())
        self.best_loss = self.last_loss
        self.best_params = self._pack_params()
        return self._state()

    def step(self, action: np.ndarray):
        prev_loss = self.last_loss
        self._apply_action(action)
        params = self._pack_params()
        loss, feat = self._rollout_loss(params)
        improvement = prev_loss - loss
        action_penalty = 0.01 * float(np.mean(np.square(action)))
        reward = improvement - 0.2 * loss - action_penalty

        self.last_loss = loss
        self.last_feat = feat
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params

        self.step_idx += 1
        done = self.step_idx >= self.max_refine_steps
        if done and loss < 0.1:
            reward += 0.25
        info = {
            "loss": loss,
            "improvement": improvement,
            **feat,
            "params": params,
        }
        return self._state(), float(reward), done, info


def split_trajectories(paths: list[Path], seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    if len(paths) <= 1:
        return paths, paths, paths
    n_train = max(1, int(0.7 * len(paths)))
    n_val = max(1, int(0.15 * len(paths)))
    tr = [paths[i] for i in idx[:n_train]]
    va = [paths[i] for i in idx[n_train:n_train + n_val]] or tr
    te = [paths[i] for i in idx[n_train + n_val:]] or va
    return tr, va, te


def build_init_params(cfg: BridgeConfig, calibration: dict[str, Any] | None = None, parameter_json: dict[str, Any] | None = None):
    out = {
        "l_com": float(cfg.l_com_init),
        "J_cm_base": float(cfg.J_cm_base),
        "b_eq": float(cfg.b_eq_init),
        "tau_eq": float(cfg.tau_eq_init),
        "k_t": float(cfg.k_t_init),
        "i0": float(cfg.i0_init),
        "R": float(cfg.R_init),
        "k_e": float(cfg.k_e_init),
        "delay_sec": float(cfg.delay_init_ms) / 1000.0,
    }

    def _merge(src):
        if not isinstance(src, dict):
            return
        for k in list(out.keys()) + ["delay_sec"]:
            if k in src:
                out[k] = float(src[k])

    if calibration:
        _merge(calibration.get("model_init", calibration.get("best_params", {})))
        if isinstance(calibration.get("delay"), dict) and "effective_control_delay_ms" in calibration["delay"]:
            out["delay_sec"] = float(calibration["delay"]["effective_control_delay_ms"]) / 1000.0
    if parameter_json:
        _merge(parameter_json.get("model_init", parameter_json.get("best_params", parameter_json)))
        if isinstance(parameter_json.get("delay"), dict) and "effective_control_delay_ms" in parameter_json["delay"]:
            out["delay_sec"] = float(parameter_json["delay"]["effective_control_delay_ms"]) / 1000.0
    return out
