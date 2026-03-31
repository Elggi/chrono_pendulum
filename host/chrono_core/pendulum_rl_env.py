#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BridgeConfig
from .signal_filter import estimate_filtered_alpha_from_omega

PARAM_KEYS = ["K_u", "b_eq", "tau_eq", "l_com"]


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
    delay_sec_est: float
    alpha_real_raw: np.ndarray | None = None
    theta_source: str = "theta_real"
    omega_source: str = "omega_real"
    alpha_source: str = "real_alpha_filtered"
    input_source: str = "cmd_u_raw"
    target_source: str = "J*real_alpha_filtered"


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


def _unwrap_angle_series(theta: np.ndarray):
    out = np.asarray(theta, dtype=float).copy()
    if len(out) == 0:
        return out
    out = _sanitize_timeseries(out)
    finite = np.isfinite(out)
    if not np.any(finite):
        out[:] = 0.0
        return out
    idx = np.where(finite)[0]
    unwrapped = np.unwrap(out[idx])
    out[idx] = unwrapped
    # Keep leading/trailing values stable if finite window does not span all rows.
    out[: idx[0]] = out[idx[0]]
    out[idx[-1] + 1 :] = out[idx[-1]]
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
    tau_motor = np.zeros(n, dtype=float)
    tau_visc = np.zeros(n, dtype=float)
    tau_coul = np.zeros(n, dtype=float)
    tau_res = np.zeros(n, dtype=float)
    cmd_delayed = shifted_signal(t, traj.cmd_u, delay_sec)

    theta[0] = float(traj.theta_real[0]) if np.isfinite(traj.theta_real[0]) else 0.0
    omega[0] = float(traj.omega_real[0]) if np.isfinite(traj.omega_real[0]) else 0.0

    m_total = cfg.rod_mass + cfg.imu_mass
    J_rod = (1.0 / 3.0) * cfg.rod_mass * (cfg.rod_length ** 2)
    J_imu = cfg.imu_mass * (cfg.r_imu ** 2)
    J_pivot = max(float(J_rod + J_imu), 1e-6)
    max_theta = 6.0 * math.pi
    max_omega = 2.0e3

    for k in range(n - 1):
        h = max(float(dt[k]), 1e-6)
        if not np.isfinite(theta[k]) or not np.isfinite(omega[k]):
            theta[k] = 0.0
            omega[k] = 0.0
        u = cmd_delayed[k]
        tm = params["K_u"] * u
        tv = params["b_eq"] * omega[k]
        tc = params["tau_eq"] * math.tanh(omega[k] / max(cfg.tanh_eps, 1e-6))
        th_k = float(np.clip(theta[k], -max_theta, max_theta))
        tg = m_total * cfg.gravity * params["l_com"] * math.sin(th_k)
        domega = (tm - tv - tc - tg) / J_pivot
        domega = float(np.clip(domega, -1.0e6, 1.0e6))

        theta[k + 1] = float(np.clip(theta[k] + h * omega[k], -max_theta, max_theta))
        omega[k + 1] = float(np.clip(omega[k] + h * domega, -max_omega, max_omega))

        tau_motor[k] = tm
        tau_visc[k] = tv
        tau_coul[k] = tc
        tau_res[k] = tv + tc

    if n > 1:
        alpha[:] = _gradient(omega, dt)
        tau_motor[-1] = tau_motor[-2]
        tau_visc[-1] = tau_visc[-2]
        tau_coul[-1] = tau_coul[-2]
        tau_res[-1] = tau_res[-2]

    return {
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "cmd_delayed": cmd_delayed,
        "tau_motor": tau_motor,
        "tau_visc": tau_visc,
        "tau_coul": tau_coul,
        "tau_res": tau_res,
    }


def compute_error_features(
    traj: ReplayTrajectory,
    sim: dict[str, np.ndarray],
    delay_quality: float = 1.0,
    align_shift_sec: float = 0.0,
):
    if abs(float(align_shift_sec)) > 1e-9:
        th_sim = shifted_signal(traj.t, sim["theta"], float(align_shift_sec))
        om_sim = shifted_signal(traj.t, sim["omega"], float(align_shift_sec))
        al_sim = shifted_signal(traj.t, sim["alpha"], float(align_shift_sec))
    else:
        th_sim = sim["theta"]
        om_sim = sim["omega"]
        al_sim = sim["alpha"]

    e_th = _winsorize_abs(th_sim - traj.theta_real, q=99.5)
    e_om = _winsorize_abs(om_sim - traj.omega_real, q=99.5)
    e_al = _winsorize_abs(al_sim - traj.alpha_real, q=99.5)

    rmse_theta = float(np.sqrt(np.mean(e_th ** 2)))
    rmse_omega = float(np.sqrt(np.mean(e_om ** 2)))
    rmse_alpha = float(np.sqrt(np.mean(e_al ** 2)))

    bias_theta = float(np.mean(e_th))
    bias_omega = float(np.mean(e_om))
    peak_amp_mismatch = float(np.max(np.abs(th_sim)) - np.max(np.abs(traj.theta_real)))

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


def simplified_loss(feat: dict[str, float], weights: dict[str, float] | None = None):
    w = weights or {}
    return float(
        w.get("alpha", 1.0) * feat["rmse_alpha"]
        + w.get("omega", 1.0) * feat["rmse_omega"]
        + w.get("theta", 1.0) * feat["rmse_theta"]
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

    if "cmd_u_raw" in df.columns:
        cmd_u = _safe_col(df, "cmd_u_raw")
        input_source = "cmd_u_raw"
    elif "cmd_u" in df.columns:
        cmd_u = _safe_col(df, "cmd_u")
        input_source = "cmd_u"
    else:
        cmd_u = _safe_col(df, "cmd_u_raw")
        input_source = "cmd_u_raw_missing_fallback_zero"
    hw_pwm = _safe_col(df, "hw_pwm")
    theta_real = _safe_col(df, "theta_real", np.nan)
    omega_real = _safe_col(df, "omega_real", np.nan)
    alpha_real_raw = _safe_col(df, "alpha_real", np.nan)
    theta_source = "theta_real"
    omega_source = "omega_real"

    # fallback to sim columns when real estimates are absent in old logs
    if not np.isfinite(theta_real).any():
        theta_real = _safe_col(df, "theta")
        theta_source = "theta(sim_fallback)"
    if not np.isfinite(omega_real).any():
        omega_real = _safe_col(df, "omega")
        omega_source = "omega(sim_fallback)"
    if not np.isfinite(alpha_real_raw).any():
        alpha_real_raw = _safe_col(df, "alpha")
    theta_real = _unwrap_angle_series(theta_real)
    omega_real = _sanitize_timeseries(omega_real)
    alpha_real_raw = _sanitize_timeseries(alpha_real_raw)
    omega_from_theta = _gradient(theta_real, dt)
    # If runtime logging briefly broke real-state channels, repair them from theta.
    if float(np.nanpercentile(np.abs(omega_real), 99.5)) > 80.0 and float(np.nanpercentile(np.abs(omega_from_theta), 99.5)) < 50.0:
        omega_real = omega_from_theta
    # Effective alpha target is always filtered d(omega)/dt.
    alpha_real = estimate_filtered_alpha_from_omega(omega_real, t=t)

    omega_real = _winsorize_abs(omega_real, q=99.5)
    alpha_real = _winsorize_abs(alpha_real, q=99.5)

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
        alpha_real_raw=alpha_real_raw,
        delay_sec_est=delay_sec,
        theta_source=theta_source,
        omega_source=omega_source,
        alpha_source="real_alpha_filtered",
        input_source=input_source,
        target_source="J*real_alpha_filtered",
    )


def default_param_bounds(center: dict[str, float], learn_delay: bool = False):
    bounds = {
        "K_u": (1e-5, 0.1),
        "l_com": (0.03, 0.45),
        "b_eq": (0.0, 2.0),
        "tau_eq": (0.0, 1.5),
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
        action_step_frac: float = 0.08,
        init_noise_frac: float = 0.07,
        param_keys_override: list[str] | None = None,
        bounds_override: dict[str, tuple[float, float]] | None = None,
        loss_mode: str = "full",
    ):
        self.trajectories = trajectories
        self.cfg = cfg
        self.learn_delay = bool(learn_delay)
        self.delay_jitter_ms = float(delay_jitter_ms)
        self.domain_randomization = bool(domain_randomization)
        self.rng = np.random.default_rng(seed)
        self.max_refine_steps = int(max_refine_steps)
        self.reward_weights = reward_weights or {}
        self.action_step_frac = float(max(1e-4, action_step_frac))
        self.init_noise_frac = float(max(0.0, init_noise_frac))

        self.param_keys = list(param_keys_override) if param_keys_override else list(PARAM_KEYS)
        if self.learn_delay:
            self.param_keys.append("delay_sec")
        # Keep full simulation parameter context even when only a subset is optimized.
        self.base_params = {
            "K_u": float(init_params.get("K_u", cfg.K_u_init)),
            "l_com": float(init_params.get("l_com", cfg.l_com_init)),
            "b_eq": float(init_params.get("b_eq", cfg.b_eq_init)),
            "tau_eq": float(init_params.get("tau_eq", cfg.tau_eq_init)),
            "delay_sec": float(init_params.get("delay_sec", cfg.delay_init_ms / 1000.0)),
        }
        self.center = {k: float(init_params[k]) for k in self.param_keys if k in init_params}
        if self.learn_delay and "delay_sec" not in self.center:
            self.center["delay_sec"] = float(np.mean([t.delay_sec_est for t in trajectories]))

        self.bounds = default_param_bounds(self.center, learn_delay=self.learn_delay)
        if bounds_override:
            for k, v in bounds_override.items():
                if k in self.bounds:
                    self.bounds[k] = (float(v[0]), float(v[1]))
        self.loss_mode = str(loss_mode).strip().lower()
        self.state_dim = len(self.param_keys) + 8
        self.action_dim = len(self.param_keys)
        self.reset()

    def _pack_params(self):
        out = dict(self.base_params)
        for k, v in zip(self.param_keys, self.param_vec):
            out[k] = float(v)
        return out

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
            step = self.action_step_frac * (hi - lo)
            self.param_vec[i] = np.clip(self.param_vec[i] + step * action[i], lo, hi)

    def _rollout_loss(self, params: dict[str, float]):
        feats = []
        losses = []
        jitter = self.rng.uniform(-self.delay_jitter_ms, self.delay_jitter_ms) * 1e-3 if self.domain_randomization else 0.0
        for traj in self.trajectories:
            d = float(params.get("delay_sec", traj.delay_sec_est + jitter))
            sim = simulate_trajectory(traj, params, self.cfg, delay_sec=max(0.0, d))
            # simulate_trajectory already applies delay via delayed command input.
            # Do not apply an additional alignment shift here.
            f = compute_error_features(traj, sim, delay_quality=1.0, align_shift_sec=0.0)
            feats.append(f)
            if self.loss_mode == "simplified":
                losses.append(simplified_loss(f, self.reward_weights))
            else:
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
                self.param_vec[i] = np.clip(self.param_vec[i] + self.rng.normal(0.0, self.init_noise_frac * span), lo, hi)
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
    n_paths = len(paths)
    if n_paths < 3:
        raise ValueError(
            "Need at least 3 CSV trajectories for strict train/val/test split without overlap."
        )

    n_train = max(1, int(0.7 * n_paths))
    n_val = max(1, int(0.15 * n_paths))
    if n_train + n_val > n_paths - 1:
        n_train = max(1, n_paths - 1 - n_val)
    if n_train + n_val > n_paths - 1:
        n_val = max(1, n_paths - 1 - n_train)
    n_test = n_paths - n_train - n_val
    if n_test < 1:
        raise ValueError("Split failed to allocate test set. Provide additional CSV trajectories.")

    tr = [paths[i] for i in idx[:n_train]]
    va = [paths[i] for i in idx[n_train:n_train + n_val]]
    te = [paths[i] for i in idx[n_train + n_val:n_train + n_val + n_test]]
    return tr, va, te


def build_init_params(cfg: BridgeConfig, calibration: dict[str, Any] | None = None, parameter_json: dict[str, Any] | None = None):
    out = {
        "K_u": float(cfg.K_u_init),
        "l_com": float(cfg.l_com_init),
        "b_eq": float(cfg.b_eq_init),
        "tau_eq": float(cfg.tau_eq_init),
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
