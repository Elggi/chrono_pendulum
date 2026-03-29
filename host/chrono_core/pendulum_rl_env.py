#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Offline replay RL environment for pendulum parameter calibration.

This module is intentionally separate from runtime `chrono_pendulum.py`.
It calibrates parameters by replaying logged trajectories offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BridgeConfig

PARAM_NAMES_BASE = ["l_com", "J_cm_base", "b_eq", "tau_eq", "k_t", "i0", "R", "k_e"]


@dataclass
class RLWeights:
    theta: float = 5.0
    omega: float = 2.5
    alpha: float = 0.7
    pwm: float = 0.2
    current: float = 0.1
    power: float = 0.05
    param_reg: float = 0.02
    action_penalty: float = 0.01


@dataclass
class TrajectoryData:
    path: str
    t: np.ndarray
    cmd_u: np.ndarray
    hw_pwm: np.ndarray
    theta_real: np.ndarray
    omega_real: np.ndarray
    alpha_real: np.ndarray
    bus_v: np.ndarray
    current_a: np.ndarray
    power_w: np.ndarray
    delay_est: float
    delay_quality_corr: float


def _choose_col(df: pd.DataFrame, names: list[str], default=None):
    for name in names:
        if name in df.columns:
            return df[name].to_numpy(dtype=float)
    if default is None:
        raise KeyError(f"None of columns {names} were found")
    return np.full(len(df), float(default), dtype=float)


def _dt_from_t(t: np.ndarray):
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    pos = dt[dt > 1e-6]
    fallback = float(np.median(pos)) if len(pos) else 1e-3
    dt[dt <= 1e-6] = fallback
    return dt


def _rmse(x: np.ndarray):
    return float(np.sqrt(np.mean(np.square(x))))


def estimate_delay_from_csv(df: pd.DataFrame, max_delay_sec: float = 0.20, dt_grid: float = 0.002):
    """Estimate cmd->pwm lag via coarse cross-correlation on uniform resampling."""
    t = _choose_col(df, ["sim_time", "wall_elapsed", "wall_time"])
    t = t - t[0]
    cmd = _choose_col(df, ["cmd_u_raw", "cmd_u_delayed", "cmd_u"], default=0.0)
    pwm = _choose_col(df, ["hw_pwm", "cmd_u_delayed"], default=0.0)
    if len(t) < 20:
        return 0.0, 0.0

    tu = np.arange(0.0, t[-1], dt_grid)
    if len(tu) < 20:
        return 0.0, 0.0
    cmd_u = np.interp(tu, t, cmd)
    pwm_u = np.interp(tu, t, pwm)
    cmd_u -= np.mean(cmd_u)
    pwm_u -= np.mean(pwm_u)
    stdc = float(np.std(cmd_u))
    stdp = float(np.std(pwm_u))
    if stdc < 1e-6 or stdp < 1e-6:
        return 0.0, 0.0

    max_lag = int(max_delay_sec / dt_grid)
    best_lag = 0
    best = -1e18
    for lag in range(max_lag + 1):
        c = cmd_u[:-lag] if lag > 0 else cmd_u
        p = pwm_u[lag:] if lag > 0 else pwm_u
        if len(c) < 10:
            break
        score = float(np.dot(c, p) / len(c))
        if score > best:
            best = score
            best_lag = lag
    delay = best_lag * dt_grid
    denom = max(stdc * stdp, 1e-9)
    quality = float(best / denom)
    return delay, quality


def shift_by_delay(t: np.ndarray, u: np.ndarray, delay_sec: float):
    shifted = np.zeros_like(u)
    for i in range(len(t)):
        td = t[i] - delay_sec
        if td <= t[0]:
            shifted[i] = u[0]
        elif td >= t[-1]:
            shifted[i] = u[-1]
        else:
            shifted[i] = np.interp(td, t, u)
    return shifted


def load_trajectories(csv_paths: list[str], delay_override: float | None = None, max_delay_sec: float = 0.2):
    trajectories: list[TrajectoryData] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        t = _choose_col(df, ["sim_time", "wall_elapsed", "wall_time"])
        t = t - t[0]
        cmd_u = _choose_col(df, ["cmd_u_raw", "cmd_u_delayed", "cmd_u"], default=0.0)
        hw_pwm = _choose_col(df, ["hw_pwm", "cmd_u_delayed"], default=0.0)

        theta_real = _choose_col(df, ["theta_real", "est_theta", "theta"], default=0.0)
        omega_real = _choose_col(df, ["omega_real", "est_omega", "omega"], default=0.0)
        if "alpha_real" in df.columns or "est_alpha" in df.columns or "alpha" in df.columns:
            alpha_real = _choose_col(df, ["alpha_real", "est_alpha", "alpha"], default=0.0)
        else:
            dt = _dt_from_t(t)
            alpha_real = np.gradient(omega_real, dt)

        bus_v = _choose_col(df, ["bus_v_filtered", "bus_v_raw"], default=7.4)
        current_a = _choose_col(df, ["current_filtered_A", "current_raw_A"], default=np.nan)
        power_w = _choose_col(df, ["power_raw_W"], default=np.nan)

        delay_est, quality = estimate_delay_from_csv(df, max_delay_sec=max_delay_sec)
        if delay_override is not None:
            delay_est = float(delay_override)

        trajectories.append(
            TrajectoryData(
                path=path,
                t=t,
                cmd_u=cmd_u,
                hw_pwm=hw_pwm,
                theta_real=theta_real,
                omega_real=omega_real,
                alpha_real=alpha_real,
                bus_v=bus_v,
                current_a=current_a,
                power_w=power_w,
                delay_est=delay_est,
                delay_quality_corr=quality,
            )
        )
    return trajectories


def initial_params_from_files(calibration_json: str, parameter_json: str | None):
    cfg = BridgeConfig()
    # calibration + defaults first
    if calibration_json:
        with open(calibration_json, "r", encoding="utf-8") as f:
            calib = json.load(f)
        model = calib.get("model_init") or calib.get("best_params") or {}
        cfg.l_com_init = float(model.get("l_com", cfg.l_com_init))
        cfg.J_cm_base = float(model.get("J_cm_base", model.get("J", cfg.J_cm_base)))
        cfg.b_eq_init = float(model.get("b_eq", model.get("b", cfg.b_eq_init)))
        cfg.tau_eq_init = float(model.get("tau_eq", model.get("tau_c", cfg.tau_eq_init)))
        cfg.k_t_init = float(model.get("k_t", cfg.k_t_init))
        cfg.i0_init = float(model.get("i0", cfg.i0_init))
        cfg.R_init = float(model.get("R", model.get("Rm", cfg.R_init)))
        cfg.k_e_init = float(model.get("k_e", cfg.k_e_init))
    if parameter_json:
        with open(parameter_json, "r", encoding="utf-8") as f:
            pobj = json.load(f)
        prm = pobj.get("model_init") or pobj.get("best_params") or pobj
        cfg.l_com_init = float(prm.get("l_com", cfg.l_com_init))
        cfg.J_cm_base = float(prm.get("J_cm_base", prm.get("J", cfg.J_cm_base)))
        cfg.b_eq_init = float(prm.get("b_eq", prm.get("b", cfg.b_eq_init)))
        cfg.tau_eq_init = float(prm.get("tau_eq", prm.get("tau_c", cfg.tau_eq_init)))
        cfg.k_t_init = float(prm.get("k_t", cfg.k_t_init))
        cfg.i0_init = float(prm.get("i0", cfg.i0_init))
        cfg.R_init = float(prm.get("R", prm.get("Rm", cfg.R_init)))
        cfg.k_e_init = float(prm.get("k_e", cfg.k_e_init))

    return {
        "l_com": float(cfg.l_com_init),
        "J_cm_base": float(cfg.J_cm_base),
        "b_eq": float(cfg.b_eq_init),
        "tau_eq": float(cfg.tau_eq_init),
        "k_t": float(cfg.k_t_init),
        "i0": float(cfg.i0_init),
        "R": float(cfg.R_init),
        "k_e": float(cfg.k_e_init),
    }, cfg


def simulate_trajectory(tr: TrajectoryData, params: dict[str, float], cfg: BridgeConfig, delay_used: float,
                        randomization: dict[str, float] | None = None):
    """Simple ODE replay (offline) for trajectory fit evaluation."""
    m_total = cfg.link_mass + cfg.imu_mass
    l_com = max(float(params["l_com"]), 1e-4)
    j_pivot = max(float(params["J_cm_base"] + m_total * l_com * l_com), 1e-7)
    b_eq = max(float(params["b_eq"]), 0.0)
    tau_eq = max(float(params["tau_eq"]), 0.0)
    k_t = max(float(params["k_t"]), 1e-6)
    i0 = max(float(params["i0"]), 0.0)
    R = max(float(params["R"]), 1e-6)
    k_e = max(float(params["k_e"]), 0.0)

    rand = randomization or {}
    delay_jitter = float(rand.get("delay_jitter", 0.0))
    pwm_scale = float(rand.get("pwm_scale", 1.0))
    bus_scale = float(rand.get("bus_scale", 1.0))
    friction_scale = float(rand.get("friction_scale", 1.0))
    dtheta0 = float(rand.get("theta0_offset", 0.0))
    domega0 = float(rand.get("omega0_offset", 0.0))

    u_aligned = shift_by_delay(tr.t, tr.cmd_u, delay_used + delay_jitter)
    u_aligned = pwm_scale * u_aligned
    bus = bus_scale * tr.bus_v

    n = len(tr.t)
    dt = _dt_from_t(tr.t)
    theta = np.zeros(n, dtype=float)
    omega = np.zeros(n, dtype=float)
    alpha = np.zeros(n, dtype=float)
    i_pred = np.zeros(n, dtype=float)
    v_applied = np.zeros(n, dtype=float)
    tau_motor = np.zeros(n, dtype=float)
    tau_res = np.zeros(n, dtype=float)

    theta[0] = float(tr.theta_real[0] + dtheta0)
    omega[0] = float(tr.omega_real[0] + domega0)

    def dynamics(th, om, u, bus_v):
        duty = np.clip(u / max(cfg.pwm_limit, 1e-9), -1.0, 1.0)
        v = duty * bus_v
        i_raw = (v - k_e * om) / R
        i_eff = np.sign(i_raw) * max(abs(i_raw) - i0, 0.0)
        tau_m = k_t * i_eff
        tau_v = friction_scale * b_eq * om
        tau_c = friction_scale * tau_eq * np.tanh(om / max(cfg.tanh_eps, 1e-9))
        tau_r = tau_v + tau_c
        tau_g = m_total * cfg.gravity * l_com * np.sin(th)
        a = (tau_m - tau_r - tau_g) / j_pivot
        return om, a, i_eff, v, tau_m, tau_r

    for k in range(n - 1):
        h = float(dt[k])
        u = float(u_aligned[k])
        vbus = float(bus[k])
        th = theta[k]
        om = omega[k]

        k1_th, k1_om, i1, v1, tm1, tr1 = dynamics(th, om, u, vbus)
        k2_th, k2_om, *_ = dynamics(th + 0.5 * h * k1_th, om + 0.5 * h * k1_om, u, vbus)
        k3_th, k3_om, *_ = dynamics(th + 0.5 * h * k2_th, om + 0.5 * h * k2_om, u, vbus)
        k4_th, k4_om, *_ = dynamics(th + h * k3_th, om + h * k3_om, u, vbus)

        theta[k + 1] = th + (h / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[k + 1] = om + (h / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)
        i_pred[k] = i1
        v_applied[k] = v1
        tau_motor[k] = tm1
        tau_res[k] = tr1

    i_pred[-1] = i_pred[-2] if n > 1 else 0.0
    v_applied[-1] = v_applied[-2] if n > 1 else 0.0
    tau_motor[-1] = tau_motor[-2] if n > 1 else 0.0
    tau_res[-1] = tau_res[-2] if n > 1 else 0.0
    alpha[:] = np.gradient(omega, dt)

    return {
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "i_pred": i_pred,
        "v_applied": v_applied,
        "tau_motor": tau_motor,
        "tau_res": tau_res,
        "u_aligned": u_aligned,
        "delay_used": delay_used + delay_jitter,
    }


def compute_metrics(tr: TrajectoryData, sim: dict[str, np.ndarray], w: RLWeights):
    e_theta = sim["theta"] - tr.theta_real
    e_omega = sim["omega"] - tr.omega_real
    e_alpha = sim["alpha"] - tr.alpha_real
    e_pwm = sim["u_aligned"] - tr.hw_pwm

    valid_i = np.isfinite(tr.current_a)
    valid_p = np.isfinite(tr.power_w)
    e_i = sim["i_pred"][valid_i] - tr.current_a[valid_i] if np.any(valid_i) else np.array([], dtype=float)
    p_pred = sim["i_pred"] * sim["v_applied"]
    e_p = p_pred[valid_p] - tr.power_w[valid_p] if np.any(valid_p) else np.array([], dtype=float)

    weighted_loss = (
        w.theta * np.mean(e_theta ** 2)
        + w.omega * np.mean(e_omega ** 2)
        + w.alpha * np.mean(e_alpha ** 2)
        + w.pwm * np.mean(e_pwm ** 2)
    )
    if e_i.size:
        weighted_loss += w.current * np.mean(e_i ** 2)
    if e_p.size:
        weighted_loss += w.power * np.mean(e_p ** 2)

    return {
        "weighted_loss": float(weighted_loss),
        "rmse_theta": _rmse(e_theta),
        "rmse_omega": _rmse(e_omega),
        "rmse_alpha": _rmse(e_alpha),
        "bias_theta": float(np.mean(e_theta)),
        "bias_omega": float(np.mean(e_omega)),
        "pwm_rmse": _rmse(e_pwm),
        "delay_quality_corr": float(tr.delay_quality_corr),
        "peak_theta_abs": float(np.max(np.abs(sim["theta"]))),
        "peak_real_theta_abs": float(np.max(np.abs(tr.theta_real))),
    }


class PendulumRLEnv:
    """Episodic replay env.

    Action = parameter delta (not motor command).
    State = normalized params + compact replay error statistics.
    Reward = improvement - weighted_loss - regularization.
    """

    def __init__(
        self,
        train_trajectories: list[TrajectoryData],
        val_trajectories: list[TrajectoryData],
        base_params: dict[str, float],
        cfg: BridgeConfig,
        learn_delay: bool = False,
        delay_jitter_ms: float = 0.0,
        domain_randomization: bool = True,
        seed: int = 0,
        max_refine_steps: int = 8,
    ):
        self.train_trajectories = train_trajectories
        self.val_trajectories = val_trajectories
        self.base_params = dict(base_params)
        self.cfg = cfg
        self.learn_delay = bool(learn_delay)
        self.delay_jitter_ms = float(delay_jitter_ms)
        self.domain_randomization = bool(domain_randomization)
        self.max_refine_steps = int(max_refine_steps)
        self.rng = np.random.default_rng(seed)
        self.weights = RLWeights()

        self.param_names = list(PARAM_NAMES_BASE)
        self.param_bounds = {
            "l_com": (0.03, 0.45),
            "J_cm_base": (1e-5, 0.02),
            "b_eq": (0.0, 5.0),
            "tau_eq": (0.0, 2.0),
            "k_t": (0.01, 2.0),
            "i0": (0.0, 1.5),
            "R": (0.05, 40.0),
            "k_e": (0.0, 2.0),
        }
        if self.learn_delay:
            self.param_names.append("delay_offset_sec")
            self.param_bounds["delay_offset_sec"] = (-0.03, 0.03)

        self.param_center = np.array([self.base_params.get(k, 0.0) for k in PARAM_NAMES_BASE], dtype=float)
        self.param_scale = np.maximum(np.abs(self.param_center) * 0.25, 1e-3)
        if self.learn_delay:
            self.param_center = np.append(self.param_center, 0.0)
            self.param_scale = np.append(self.param_scale, 0.01)

        self.params = dict(base_params)
        self.params["delay_offset_sec"] = 0.0
        self.step_count = 0
        self.last_loss = None
        self.last_metrics = None

    @property
    def state_dim(self):
        return len(self.param_names) + 10

    @property
    def action_dim(self):
        return len(self.param_names)

    def _clip_params(self):
        for k in self.param_names:
            lo, hi = self.param_bounds[k]
            self.params[k] = float(np.clip(self.params[k], lo, hi))

    def _param_vec(self):
        return np.array([self.params[k] for k in self.param_names], dtype=float)

    def _norm_param_vec(self):
        v = self._param_vec()
        return (v - self.param_center[: len(v)]) / self.param_scale[: len(v)]

    def _randomization_dict(self, progress: float):
        if not self.domain_randomization:
            return {}
        # curriculum: larger early, tighter later
        s = max(0.1, 1.0 - progress)
        return {
            "theta0_offset": self.rng.uniform(-0.08, 0.08) * s,
            "omega0_offset": self.rng.uniform(-0.4, 0.4) * s,
            "pwm_scale": 1.0 + self.rng.uniform(-0.06, 0.06) * s,
            "bus_scale": 1.0 + self.rng.uniform(-0.03, 0.03) * s,
            "friction_scale": 1.0 + self.rng.uniform(-0.08, 0.08) * s,
            "delay_jitter": self.rng.uniform(-1.0, 1.0) * (self.delay_jitter_ms * 1e-3),
        }

    def _eval_set(self, trajs: list[TrajectoryData], progress: float = 0.0):
        metric_list = []
        for tr in trajs:
            delay = tr.delay_est
            if self.learn_delay:
                delay = max(delay + self.params.get("delay_offset_sec", 0.0), 0.0)
            sim = simulate_trajectory(tr, self.params, self.cfg, delay, self._randomization_dict(progress))
            metric_list.append(compute_metrics(tr, sim, self.weights))
        agg = {}
        for key in metric_list[0].keys():
            agg[key] = float(np.mean([m[key] for m in metric_list]))
        return agg

    def _state_from_metrics(self, m: dict[str, float]):
        p = self._norm_param_vec()
        peak_ratio = m["peak_theta_abs"] / max(m["peak_real_theta_abs"], 1e-5)
        feat = np.array([
            m["weighted_loss"],
            m["rmse_theta"],
            m["rmse_omega"],
            m["rmse_alpha"],
            m["bias_theta"],
            m["bias_omega"],
            m["pwm_rmse"],
            m["delay_quality_corr"],
            peak_ratio,
            float(self.step_count) / max(self.max_refine_steps, 1),
        ], dtype=float)
        return np.concatenate([p, feat]).astype(np.float32)

    def reset(self):
        self.step_count = 0
        self.params = dict(self.base_params)
        self.params["delay_offset_sec"] = 0.0
        # randomize around center for sample efficiency
        for k in PARAM_NAMES_BASE:
            self.params[k] = float(self.base_params[k] * (1.0 + self.rng.uniform(-0.12, 0.12)))
        self._clip_params()
        self.last_metrics = self._eval_set(self.train_trajectories, progress=0.0)
        self.last_loss = self.last_metrics["weighted_loss"]
        return self._state_from_metrics(self.last_metrics)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim={self.action_dim}, got {action.shape}")
        delta = 0.10 * action * self.param_scale[: self.action_dim]
        for i, k in enumerate(self.param_names):
            self.params[k] += float(delta[i])
        self._clip_params()

        self.step_count += 1
        progress = self.step_count / max(self.max_refine_steps, 1)
        cur = self._eval_set(self.train_trajectories, progress=progress)
        reg = self.weights.param_reg * float(np.mean(np.square(self._norm_param_vec())))
        action_pen = self.weights.action_penalty * float(np.mean(np.square(action)))

        improvement = float(self.last_loss - cur["weighted_loss"])
        reward = improvement - cur["weighted_loss"] - reg - action_pen

        self.last_loss = cur["weighted_loss"]
        self.last_metrics = cur
        done = self.step_count >= self.max_refine_steps

        info = {
            "metrics": cur,
            "reward_terms": {
                "improvement": improvement,
                "weighted_loss": cur["weighted_loss"],
                "param_reg": reg,
                "action_penalty": action_pen,
            },
        }
        return self._state_from_metrics(cur), float(reward), done, info


def split_trajectories(all_paths: list[str], seed: int = 0, val_ratio: float = 0.2, test_ratio: float = 0.1):
    rng = np.random.default_rng(seed)
    items = list(all_paths)
    rng.shuffle(items)
    n = len(items)
    n_test = int(round(n * test_ratio)) if n > 2 else 0
    n_val = int(round(n * val_ratio)) if n > 2 else 0
    n_train = max(n - n_val - n_test, 1)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def chunk_single_csv(csv_path: str, out_dir: str, chunk_sec: float = 8.0):
    """Split a single CSV into trajectory chunks by time-window."""
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    t = _choose_col(df, ["sim_time", "wall_elapsed", "wall_time"])
    t = t - t[0]
    n_chunks = max(int(math.ceil(t[-1] / max(chunk_sec, 0.5))), 1)
    paths = []
    for idx in range(n_chunks):
        t0 = idx * chunk_sec
        t1 = (idx + 1) * chunk_sec
        sub = df[(t >= t0) & (t < t1)].copy()
        if len(sub) < 10:
            continue
        p = outp / f"chunk_{idx:03d}.csv"
        sub.to_csv(p, index=False)
        paths.append(str(p))
    return paths
