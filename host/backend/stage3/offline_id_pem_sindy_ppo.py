#!/usr/bin/env python3
"""Three-stage offline identification benchmark for Chrono pendulum logs.

Stages:
1) Nominal PEM fit using ODE simulation + nonlinear least squares.
2) Residual dynamics discovery using PySINDy SINDy-PI on omega-dot residuals.
3) PPO parameter proposal optimization (SB3) using rollout-level rewards.

Artifacts are written under reports/PEM_SINDy_PPO.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import sys

HOST_DIR = Path(__file__).resolve().parents[2]
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from typing import Callable

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - dependency gate
    raise SystemExit(
        "This benchmark requires numpy/scipy/matplotlib (+ optional pysindy/sb3). "
        "Install dependencies first (e.g., pip install numpy scipy matplotlib pysindy stable-baselines3 gymnasium)."
    ) from exc


EPS = 1e-9
PARAM_KEYS = ("K_i", "b_eq", "tau_eq", "l_com")


@dataclass
class PhysicalConfig:
    m: float
    g: float
    eps: float
    J: float


@dataclass
class RunData:
    name: str
    t: np.ndarray
    u: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray | None


@dataclass
class Stage1Result:
    params: dict[str, float]
    method: str
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    one_step_metrics: dict[str, float]
    stability: dict[str, float]


@dataclass
class Stage2Result:
    active_terms: list[str]
    coefficients: dict[str, float]
    val_metrics: dict[str, float]
    delta_vs_stage1: dict[str, float]
    robustness: list[dict[str, float]]


@dataclass
class Stage3Result:
    best_params: dict[str, float]
    best_cost: float
    mean_params: dict[str, float]
    std_params: dict[str, float]
    seed_scores: list[dict[str, float]]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names or ()}


def load_run(csv_path: Path) -> RunData:
    cols = _read_csv_columns(csv_path)
    required = [
        "wall_elapsed",
        "I_filtered_mA",
        "theta_imu_filtered_unwrapped",
        "omega_imu_filtered",
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        raise KeyError(f"{csv_path}: missing required columns {missing}")

    alpha = cols.get("alpha_from_linear_accel_filtered")
    t = np.asarray(cols["wall_elapsed"], dtype=float)
    t = t - t[0]

    good = np.isfinite(t)
    for key in ["I_filtered_mA", "theta_imu_filtered_unwrapped", "omega_imu_filtered"]:
        good &= np.isfinite(cols[key])
    if alpha is not None:
        good &= np.isfinite(alpha)
    if np.count_nonzero(good) < 4:
        raise ValueError(f"{csv_path}: insufficient valid samples after filtering")

    t = t[good]
    u = cols["I_filtered_mA"][good]
    theta = np.unwrap(cols["theta_imu_filtered_unwrapped"][good])
    omega = cols["omega_imu_filtered"][good]
    alpha_clean = alpha[good] if alpha is not None else None

    sort_idx = np.argsort(t)
    return RunData(
        name=csv_path.stem,
        t=t[sort_idx],
        u=u[sort_idx],
        theta=theta[sort_idx],
        omega=omega[sort_idx],
        alpha=alpha_clean[sort_idx] if alpha_clean is not None else None,
    )


def infer_physical_config(meta_path: Path) -> PhysicalConfig:
    meta = _load_json(meta_path)
    cfg = meta.get("config", {})
    inertia = meta.get("inertia", {})
    m = float(cfg.get("rod_mass", 0.2) + cfg.get("imu_mass", 0.02))
    g = float(cfg.get("gravity", 9.81))
    eps = float(cfg.get("tanh_eps", 0.05))
    j = float(inertia.get("J_total", 0.0065))
    return PhysicalConfig(m=m, g=g, eps=max(eps, 1e-6), J=max(j, 1e-8))


def nrmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((np.asarray(y_hat) - np.asarray(y)) ** 2)))
    return rmse / (float(np.std(y)) + EPS)


def _rhs(x: np.ndarray, u: float, p: dict[str, float], cfg: PhysicalConfig) -> np.ndarray:
    theta, omega = float(x[0]), float(x[1])
    domega = (
        p["K_i"] * u
        - p["b_eq"] * omega
        - p["tau_eq"] * math.tanh(omega / cfg.eps)
        - cfg.m * cfg.g * p["l_com"] * math.sin(theta)
    ) / cfg.J
    return np.array([omega, domega], dtype=float)


def _rk4_step(x: np.ndarray, u: float, h: float, p: dict[str, float], cfg: PhysicalConfig) -> np.ndarray:
    k1 = _rhs(x, u, p, cfg)
    k2 = _rhs(x + 0.5 * h * k1, u, p, cfg)
    k3 = _rhs(x + 0.5 * h * k2, u, p, cfg)
    k4 = _rhs(x + h * k3, u, p, cfg)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_irregular_rk45(run: RunData, p: dict[str, float], cfg: PhysicalConfig) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:
        raise RuntimeError("scipy is required for RK45 integration") from exc

    x0 = np.array([run.theta[0], run.omega[0]], dtype=float)

    def ode(ti: float, xi: np.ndarray) -> np.ndarray:
        u = float(np.interp(ti, run.t, run.u))
        return _rhs(xi, u, p, cfg)

    sol = solve_ivp(ode, (float(run.t[0]), float(run.t[-1])), x0=x0, t_eval=run.t, method="RK45")
    if not sol.success:
        raise RuntimeError(f"RK45 integration failed: {sol.message}")
    return np.asarray(sol.y[0], dtype=float), np.asarray(sol.y[1], dtype=float)


def resample_uniform(run: RunData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    dt_mean = float(np.mean(np.diff(run.t)))
    t_u = np.arange(run.t[0], run.t[-1] + dt_mean * 0.5, dt_mean)
    u_u = np.interp(t_u, run.t, run.u)
    th_u = np.interp(t_u, run.t, run.theta)
    om_u = np.interp(t_u, run.t, run.omega)
    al_u = np.interp(t_u, run.t, run.alpha) if run.alpha is not None else None
    return t_u, u_u, th_u, om_u, al_u


def simulate_uniform_rk4(run: RunData, p: dict[str, float], cfg: PhysicalConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_u, u_u, th_u, om_u, _ = resample_uniform(run)
    x = np.array([th_u[0], om_u[0]], dtype=float)
    x_hist = np.zeros((len(t_u), 2), dtype=float)
    x_hist[0] = x
    h = float(t_u[1] - t_u[0]) if len(t_u) > 1 else 1e-3
    for i in range(len(t_u) - 1):
        x = _rk4_step(x, float(u_u[i]), h, p, cfg)
        x_hist[i + 1] = x
    return t_u, x_hist[:, 0], x_hist[:, 1]


def _weighted_objective(theta_hat: np.ndarray, omega_hat: np.ndarray, run: RunData, w: dict[str, float]) -> float:
    cost = w["theta"] * nrmse(theta_hat, run.theta) + w["omega"] * nrmse(omega_hat, run.omega)
    if run.alpha is not None:
        if len(omega_hat) == len(run.t):
            alpha_hat = np.gradient(omega_hat, run.t)
        else:
            t_u, _, _, _, al_u = resample_uniform(run)
            alpha_hat = np.gradient(np.interp(t_u, run.t, omega_hat), t_u)
            alpha_ref = al_u if al_u is not None else np.interp(t_u, run.t, run.alpha)
            return cost + w["alpha"] * nrmse(alpha_hat, alpha_ref)
        cost += w["alpha"] * nrmse(alpha_hat, run.alpha)
    return float(cost)


def _metrics(theta_hat: np.ndarray, omega_hat: np.ndarray, run: RunData, weights: dict[str, float]) -> dict[str, float]:
    met = {
        "nrmse_theta": nrmse(theta_hat, run.theta),
        "nrmse_omega": nrmse(omega_hat, run.omega),
    }
    if run.alpha is not None:
        alpha_hat = np.gradient(omega_hat, run.t)
        met["nrmse_alpha"] = nrmse(alpha_hat, run.alpha)
    else:
        met["nrmse_alpha"] = float("nan")
    met["loss"] = (
        weights["theta"] * met["nrmse_theta"] + weights["omega"] * met["nrmse_omega"] + weights["alpha"] * met["nrmse_alpha"]
    )
    return met


def one_step_metrics(run: RunData, p: dict[str, float], cfg: PhysicalConfig, weights: dict[str, float]) -> dict[str, float]:
    th_pred = np.zeros_like(run.theta)
    om_pred = np.zeros_like(run.omega)
    th_pred[0] = run.theta[0]
    om_pred[0] = run.omega[0]
    for i in range(len(run.t) - 1):
        h = float(max(run.t[i + 1] - run.t[i], 1e-6))
        x_next = _rk4_step(np.array([run.theta[i], run.omega[i]]), float(run.u[i]), h, p, cfg)
        th_pred[i + 1] = x_next[0]
        om_pred[i + 1] = x_next[1]
    return _metrics(th_pred, om_pred, run, weights)


def residual_autocorr(signal: np.ndarray, max_lag: int = 50) -> np.ndarray:
    x = np.asarray(signal, dtype=float) - float(np.mean(signal))
    den = float(np.dot(x, x)) + EPS
    ac = [1.0]
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            ac.append(np.nan)
        else:
            ac.append(float(np.dot(x[:-lag], x[lag:]) / den))
    return np.asarray(ac, dtype=float)


def _param_from_vec(v: np.ndarray, fit_lcom: bool, lcom_fixed: float) -> dict[str, float]:
    out = {"K_i": float(v[0]), "b_eq": float(v[1]), "tau_eq": float(v[2])}
    out["l_com"] = float(v[3]) if fit_lcom else float(lcom_fixed)
    return out


def _vec_from_param(p: dict[str, float], fit_lcom: bool) -> np.ndarray:
    base = [p["K_i"], p["b_eq"], p["tau_eq"]]
    if fit_lcom:
        base.append(p["l_com"])
    return np.asarray(base, dtype=float)


def fit_stage1_pem(
    run_train: RunData,
    run_val: RunData,
    cfg: PhysicalConfig,
    init_params: dict[str, float],
    bounds: tuple[np.ndarray, np.ndarray],
    weights: dict[str, float],
    fit_lcom: bool,
    outdir: Path,
) -> Stage1Result:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for Stage 1 least_squares") from exc

    x0 = _vec_from_param(init_params, fit_lcom)

    def residual_fn(v: np.ndarray) -> np.ndarray:
        p = _param_from_vec(v, fit_lcom=fit_lcom, lcom_fixed=init_params["l_com"])
        th_hat, om_hat = simulate_irregular_rk45(run_train, p, cfg)
        r = [
            math.sqrt(weights["theta"]) * (th_hat - run_train.theta) / (np.std(run_train.theta) + EPS),
            math.sqrt(weights["omega"]) * (om_hat - run_train.omega) / (np.std(run_train.omega) + EPS),
        ]
        if run_train.alpha is not None:
            al_hat = np.gradient(om_hat, run_train.t)
            r.append(math.sqrt(weights["alpha"]) * (al_hat - run_train.alpha) / (np.std(run_train.alpha) + EPS))
        return np.concatenate(r)

    res = least_squares(residual_fn, x0=x0, bounds=bounds, method="trf")
    p_fit = _param_from_vec(res.x, fit_lcom=fit_lcom, lcom_fixed=init_params["l_com"])

    th_train, om_train = simulate_irregular_rk45(run_train, p_fit, cfg)
    th_val, om_val = simulate_irregular_rk45(run_val, p_fit, cfg)
    one_step = one_step_metrics(run_val, p_fit, cfg, weights)

    train_metrics = _metrics(th_train, om_train, run_train, weights)
    val_metrics = _metrics(th_val, om_val, run_val, weights)

    # RK4 baseline on uniform resampling (comparison required by spec).
    t_u, th_u, om_u = simulate_uniform_rk4(run_val, p_fit, cfg)
    _, _, th_ref_u, om_ref_u, _ = resample_uniform(run_val)
    val_metrics["uniform_rk4_nrmse_theta"] = nrmse(th_u, th_ref_u)
    val_metrics["uniform_rk4_nrmse_omega"] = nrmse(om_u, om_ref_u)
    val_metrics["uniform_dt_mean"] = float(np.mean(np.diff(t_u))) if len(t_u) > 1 else float("nan")

    stable = np.isfinite(th_val).all() and np.isfinite(om_val).all()
    max_abs_omega = float(np.max(np.abs(om_val)))
    stability = {"is_finite": float(stable), "max_abs_omega": max_abs_omega}

    err_omega = om_val - run_val.omega
    ac = residual_autocorr(err_omega, max_lag=80)
    _save_stage1_plots(run_val, th_val, om_val, err_omega, ac, outdir)

    return Stage1Result(
        params=p_fit,
        method="least_squares(trf)+RK45",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        one_step_metrics=one_step,
        stability=stability,
    )


def _save_stage1_plots(run: RunData, th_hat: np.ndarray, om_hat: np.ndarray, err_om: np.ndarray, ac: np.ndarray, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    ensure_dir(outdir)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(run.t, run.theta, label="theta real")
    axes[0].plot(run.t, th_hat, "--", label="theta sim")
    axes[0].legend(loc="best")
    axes[0].set_ylabel("theta [rad]")
    axes[1].plot(run.t, run.omega, label="omega real")
    axes[1].plot(run.t, om_hat, "--", label="omega sim")
    axes[1].legend(loc="best")
    axes[1].set_ylabel("omega [rad/s]")
    axes[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(outdir / "stage1_trajectories.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(run.t, err_om)
    ax.set_title("Omega residual vs time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("omega error [rad/s]")
    fig.tight_layout()
    fig.savefig(outdir / "stage1_error_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.stem(np.arange(len(ac)), ac, basefmt=" ")
    ax.set_title("Residual autocorrelation (omega)")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorr")
    fig.tight_layout()
    fig.savefig(outdir / "stage1_residual_autocorr.png", dpi=150)
    plt.close(fig)


def _compute_omega_dot_measured(run: RunData, method: str = "smooth_grad") -> np.ndarray:
    if method == "alpha" and run.alpha is not None:
        return np.asarray(run.alpha, dtype=float)

    if method == "smooth_grad":
        try:
            from scipy.signal import savgol_filter

            dt_mean = float(np.mean(np.diff(run.t)))
            win = max(7, int(round(0.35 / max(dt_mean, 1e-6))) | 1)
            poly = 3 if win > 5 else 2
            omega_smooth = savgol_filter(run.omega, win, poly, mode="interp")
            return np.gradient(omega_smooth, run.t)
        except Exception:
            return np.gradient(run.omega, run.t)

    return np.gradient(run.omega, run.t)


def fit_stage2_sindy(
    run_train: RunData,
    run_val: RunData,
    cfg: PhysicalConfig,
    stage1: Stage1Result,
    weights: dict[str, float],
    seeds: list[int],
    outdir: Path,
) -> tuple[Stage2Result, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    try:
        import pysindy as ps
    except Exception as exc:
        raise RuntimeError("pysindy is required for Stage 2 SINDy-PI") from exc

    p = stage1.params
    ode_nom = (
        p["K_i"] * run_train.u
        - p["b_eq"] * run_train.omega
        - p["tau_eq"] * np.tanh(run_train.omega / cfg.eps)
        - cfg.m * cfg.g * p["l_com"] * np.sin(run_train.theta)
    ) / cfg.J
    omega_dot_meas = _compute_omega_dot_measured(run_train, method="smooth_grad")
    residual = omega_dot_meas - ode_nom

    def feature_matrix(theta: np.ndarray, omega: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.ones_like(theta),
                omega,
                np.abs(omega),
                omega ** 2,
                omega * np.abs(omega),
                np.sin(theta),
                np.cos(theta),
                u,
                u * omega,
                np.tanh(omega / cfg.eps),
            ]
        )

    feature_names = [
        "1",
        "omega",
        "abs(omega)",
        "omega^2",
        "omega*abs(omega)",
        "sin(theta)",
        "cos(theta)",
        "u",
        "u*omega",
        "tanh(omega/eps)",
    ]

    x_train = feature_matrix(run_train.theta, run_train.omega, run_train.u)
    optimizer = ps.SINDyPI(threshold=0.02, tol=1e-5, thresholder="l1")
    model = ps.SINDy(feature_library=ps.IdentityLibrary(), optimizer=optimizer, feature_names=feature_names)
    model.fit(x_train, t=run_train.t, x_dot=residual.reshape(-1, 1))

    coef = np.asarray(model.coefficients()).reshape(-1)
    coef_dict = {name: float(c) for name, c in zip(feature_names, coef)}
    active_terms = [k for k, v in coef_dict.items() if abs(v) > 1e-10]

    def r_sindy(theta: np.ndarray, omega: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = feature_matrix(theta, omega, u)
        return x @ coef

    th2, om2 = rollout_with_residual(run_val, stage1.params, cfg, r_sindy)
    val_metrics = _metrics(th2, om2, run_val, weights)
    delta = {
        "delta_nrmse_theta": val_metrics["nrmse_theta"] - stage1.val_metrics["nrmse_theta"],
        "delta_nrmse_omega": val_metrics["nrmse_omega"] - stage1.val_metrics["nrmse_omega"],
        "delta_loss": val_metrics["loss"] - stage1.val_metrics["loss"],
    }

    robustness = []
    for sd in seeds:
        np.random.seed(sd)
        jitter = residual + 0.01 * np.std(residual) * np.random.randn(*residual.shape)
        m_sd = ps.SINDy(feature_library=ps.IdentityLibrary(), optimizer=optimizer, feature_names=feature_names)
        m_sd.fit(x_train, t=run_train.t, x_dot=jitter.reshape(-1, 1))
        c_sd = np.asarray(m_sd.coefficients()).reshape(-1)
        nz = float(np.count_nonzero(np.abs(c_sd) > 1e-10))
        robustness.append({"seed": float(sd), "active_terms": nz, "l1_norm": float(np.sum(np.abs(c_sd)))})

    ensure_dir(outdir)
    with (outdir / "stage2_sindy_equation.txt").open("w", encoding="utf-8") as f:
        f.write("residual(theta, omega, u) = ")
        terms = [f"({coef_dict[k]:+.6g})*{k}" for k in active_terms]
        f.write(" ".join(terms) if terms else "0")
        f.write("\n")

    return (
        Stage2Result(
            active_terms=active_terms,
            coefficients=coef_dict,
            val_metrics=val_metrics,
            delta_vs_stage1=delta,
            robustness=robustness,
        ),
        r_sindy,
    )


def rollout_with_residual(
    run: RunData,
    p: dict[str, float],
    cfg: PhysicalConfig,
    residual_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.zeros_like(run.theta)
    omega = np.zeros_like(run.omega)
    theta[0] = run.theta[0]
    omega[0] = run.omega[0]
    for i in range(len(run.t) - 1):
        h = float(max(run.t[i + 1] - run.t[i], 1e-6))
        nom = (
            p["K_i"] * run.u[i]
            - p["b_eq"] * omega[i]
            - p["tau_eq"] * math.tanh(omega[i] / cfg.eps)
            - cfg.m * cfg.g * p["l_com"] * math.sin(theta[i])
        ) / cfg.J
        res = float(residual_fn(np.array([theta[i]]), np.array([omega[i]]), np.array([run.u[i]]))[0]) if residual_fn else 0.0
        domega = nom + res
        theta[i + 1] = theta[i] + h * omega[i]
        omega[i + 1] = omega[i] + h * domega
    return theta, omega


def fit_stage3_ppo(
    runs: list[RunData],
    cfg: PhysicalConfig,
    nominal_params: dict[str, float],
    residual_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None,
    weights: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    seed_list: list[int],
    total_timesteps: int,
    outdir: Path,
) -> Stage3Result:
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError("gymnasium and stable_baselines3 are required for Stage 3") from exc

    class ParamProposalEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, seed: int):
            super().__init__()
            self.nom = nominal_params
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
            self.rng = np.random.default_rng(seed)
            self.stats = np.array([
                np.mean(np.concatenate([r.u for r in runs])),
                np.std(np.concatenate([r.u for r in runs])) + EPS,
                np.mean(np.concatenate([r.theta for r in runs])),
                np.std(np.concatenate([r.theta for r in runs])) + EPS,
            ])

        def _action_to_params(self, action: np.ndarray) -> dict[str, float]:
            span = {
                "K_i": 0.5 * abs(self.nom["K_i"] + 1e-6),
                "b_eq": 0.5 * max(abs(self.nom["b_eq"]), 1e-4),
                "tau_eq": 0.5 * max(abs(self.nom["tau_eq"]), 1e-4),
                "l_com": 0.2 * max(abs(self.nom["l_com"]), 1e-3),
            }
            p = {}
            for i, k in enumerate(PARAM_KEYS):
                p[k] = float(self.nom[k] + float(action[i]) * span[k])
            return p

        def _penalty_bounds(self, p: dict[str, float]) -> float:
            pen = 0.0
            for k, (lo, hi) in bounds.items():
                if p[k] < lo:
                    pen += (lo - p[k]) ** 2
                if p[k] > hi:
                    pen += (p[k] - hi) ** 2
            if p["K_i"] <= 0.0:
                pen += 100.0
            if p["b_eq"] < 0.0 or p["tau_eq"] < 0.0:
                pen += 100.0
            return float(pen)

        def _run_cost(self, run: RunData, p: dict[str, float]) -> tuple[float, dict[str, float]]:
            th, om = rollout_with_residual(run, p, cfg, residual_fn)
            met = _metrics(th, om, run, weights)
            reg = float(sum((p[k] - self.nom[k]) ** 2 for k in PARAM_KEYS))
            bound_pen = self._penalty_bounds(p)
            unstable = float((not np.isfinite(th).all()) or (not np.isfinite(om).all()) or (np.max(np.abs(om)) > 5e3))
            j = (
                weights["theta"] * met["nrmse_theta"]
                + weights["omega"] * met["nrmse_omega"]
                + weights["alpha"] * met["nrmse_alpha"]
                + 0.1 * reg
                + 1.0 * bound_pen
                + 50.0 * unstable
            )
            return float(j), {
                "j_track": float(met["loss"]),
                "j_reg": reg,
                "j_bound": bound_pen,
                "j_stab": unstable,
            }

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            run0 = runs[0]
            obs = np.array(
                [run0.theta[0], run0.omega[0], self.nom["K_i"], self.nom["b_eq"], self.nom["tau_eq"], self.nom["l_com"], *self.stats],
                dtype=np.float32,
            )
            return obs, {}

        def step(self, action: np.ndarray):
            p = self._action_to_params(np.asarray(action, dtype=float))
            run_costs = []
            breakdown = []
            for run in runs:
                c, b = self._run_cost(run, p)
                run_costs.append(c)
                breakdown.append(b)
            j_total = 0.7 * float(np.mean(run_costs)) + 0.3 * float(np.max(run_costs))
            reward = -j_total
            obs, _ = self.reset()
            info = {
                "params": p,
                "cost": j_total,
                "cost_runs": run_costs,
                "cost_breakdown_mean": {
                    k: float(np.mean([b[k] for b in breakdown])) for k in breakdown[0].keys()
                },
            }
            return obs, float(reward), True, False, info

    histories = []
    seed_scores = []
    best_global = {"cost": float("inf"), "params": nominal_params.copy()}

    for seed in seed_list:
        env = ParamProposalEnv(seed=seed)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=32, batch_size=32, gamma=0.99)
        model.learn(total_timesteps=total_timesteps)

        obs, _ = env.reset(seed=seed + 99)
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        score = -float(reward)
        seed_scores.append({"seed": float(seed), "cost": score, **{f"param_{k}": float(info["params"][k]) for k in PARAM_KEYS}})
        histories.append((seed, score, info["params"]))

        if score < best_global["cost"]:
            best_global = {"cost": score, "params": info["params"]}

        ensure_dir(outdir / "checkpoints")
        model.save(str(outdir / "checkpoints" / f"ppo_seed_{seed}"))

    arr = np.array([[h[2][k] for k in PARAM_KEYS] for h in histories], dtype=float)
    mean_params = {k: float(np.mean(arr[:, i])) for i, k in enumerate(PARAM_KEYS)}
    std_params = {k: float(np.std(arr[:, i])) for i, k in enumerate(PARAM_KEYS)}

    _save_stage3_plots(seed_scores, outdir)
    return Stage3Result(
        best_params={k: float(v) for k, v in best_global["params"].items()},
        best_cost=float(best_global["cost"]),
        mean_params=mean_params,
        std_params=std_params,
        seed_scores=seed_scores,
    )


def _save_stage3_plots(seed_scores: list[dict[str, float]], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    ensure_dir(outdir)
    seeds = [d["seed"] for d in seed_scores]
    costs = [d["cost"] for d in seed_scores]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(seeds, costs, marker="o")
    ax.set_xlabel("seed")
    ax.set_ylabel("best rollout cost")
    ax.set_title("PPO convergence summary across seeds")
    fig.tight_layout()
    fig.savefig(outdir / "stage3_seed_costs.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.ravel()
    for i, k in enumerate(PARAM_KEYS):
        vals = [d[f"param_{k}"] for d in seed_scores]
        axs[i].plot(seeds, vals, marker="o")
        axs[i].set_title(k)
    fig.tight_layout()
    fig.savefig(outdir / "stage3_parameter_evolution.png", dpi=150)
    plt.close(fig)


def split_run(run: RunData, train_ratio: float) -> tuple[RunData, RunData]:
    n = len(run.t)
    idx = max(4, min(n - 4, int(n * train_ratio)))

    def part(name: str, s: slice) -> RunData:
        t = run.t[s]
        t = t - t[0]
        return RunData(
            name=f"{run.name}_{name}",
            t=t,
            u=run.u[s],
            theta=run.theta[s],
            omega=run.omega[s],
            alpha=(run.alpha[s] if run.alpha is not None else None),
        )

    return part("train", slice(0, idx)), part("val", slice(idx, n))


def write_summary_report(
    outdir: Path,
    run_name: str,
    stage1: Stage1Result,
    stage2: Stage2Result | None,
    stage3: Stage3Result | None,
    config_dump: dict,
) -> None:
    lines = [
        "# PEM + SINDy-PI + PPO Offline Identification Report",
        "",
        f"Run: `{run_name}`",
        "",
        "## Stage role decomposition",
        "- **PEM** identifies physically interpretable nominal dynamics parameters.",
        "- **SINDy-PI** identifies only the residual acceleration dynamics.",
        "- **PPO** proposes parameter vectors evaluated at rollout level (not control).",
        "",
        "## Stage 1 (Nominal PEM)",
        f"- Method: `{stage1.method}`",
        f"- Parameters: `{json.dumps(stage1.params, indent=2)}`",
        f"- Train metrics: `{json.dumps(stage1.train_metrics, indent=2)}`",
        f"- Validation metrics: `{json.dumps(stage1.val_metrics, indent=2)}`",
        f"- One-step metrics: `{json.dumps(stage1.one_step_metrics, indent=2)}`",
        f"- Stability: `{json.dumps(stage1.stability, indent=2)}`",
        "",
    ]
    if stage2 is not None:
        lines += [
            "## Stage 2 (Residual SINDy-PI)",
            f"- Active terms: `{stage2.active_terms}`",
            f"- Coefficients: `{json.dumps(stage2.coefficients, indent=2)}`",
            f"- Validation metrics: `{json.dumps(stage2.val_metrics, indent=2)}`",
            f"- Improvement delta vs Stage 1: `{json.dumps(stage2.delta_vs_stage1, indent=2)}`",
            f"- Robustness (seed sweep): `{json.dumps(stage2.robustness, indent=2)}`",
            "",
        ]
    if stage3 is not None:
        lines += [
            "## Stage 3 (PPO parameter proposals)",
            f"- Best params: `{json.dumps(stage3.best_params, indent=2)}`",
            f"- Best cost: `{stage3.best_cost}`",
            f"- Params mean/std over seeds: mean=`{json.dumps(stage3.mean_params)}`, std=`{json.dumps(stage3.std_params)}`",
            f"- Seed costs: `{json.dumps(stage3.seed_scores, indent=2)}`",
            "",
        ]
    lines += [
        "## Reproducibility",
        "- All configs are saved in `config_used.json`.",
        "- Model outputs are stored under this report folder.",
        "",
        "## Config used",
        "```json",
        json.dumps(config_dump, indent=2),
        "```",
        "",
    ]
    (outdir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline PEM + SINDy-PI + PPO benchmark")
    p.add_argument("--csv", type=Path, default=Path("host/run_logs/chrono_run_1.finalized.csv"))
    p.add_argument("--meta", type=Path, default=Path("host/run_logs/chrono_run_1.meta.json"))
    p.add_argument("--extra-csv", type=Path, nargs="*", default=[])
    p.add_argument("--outdir", type=Path, default=Path("reports/PEM_SINDy_PPO"))
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--fit-lcom", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--ppo-steps", type=int, default=12000)
    p.add_argument("--ppo-seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--w-theta", type=float, default=5.0)
    p.add_argument("--w-omega", type=float, default=2.5)
    p.add_argument("--w-alpha", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    runs = [load_run(args.csv)] + [load_run(p) for p in args.extra_csv]

    phy_cfg = infer_physical_config(args.meta)
    train_run, val_run = split_run(runs[0], train_ratio=args.train_ratio)

    meta = _load_json(args.meta)
    cfg_meta = meta.get("config", {})
    init_params = {
        "K_i": float(cfg_meta.get("K_i_init", 1e-5)),
        "b_eq": float(cfg_meta.get("b_eq_init", 0.02)),
        "tau_eq": float(cfg_meta.get("tau_eq_init", 0.01)),
        "l_com": float(cfg_meta.get("l_com_init", 0.1425)),
    }
    weights = {"theta": args.w_theta, "omega": args.w_omega, "alpha": args.w_alpha}

    lo = np.array([1e-9, 0.0, 0.0, 0.03] if args.fit_lcom else [1e-9, 0.0, 0.0], dtype=float)
    hi = np.array([1.0, 10.0, 10.0, 0.45] if args.fit_lcom else [1.0, 10.0, 10.0], dtype=float)

    stage1 = fit_stage1_pem(
        run_train=train_run,
        run_val=val_run,
        cfg=phy_cfg,
        init_params=init_params,
        bounds=(lo, hi),
        weights=weights,
        fit_lcom=args.fit_lcom,
        outdir=args.outdir,
    )

    stage2 = None
    residual_fn = None
    if not args.skip_stage2:
        stage2, residual_fn = fit_stage2_sindy(
            run_train=train_run,
            run_val=val_run,
            cfg=phy_cfg,
            stage1=stage1,
            weights=weights,
            seeds=[11, 22, 33],
            outdir=args.outdir,
        )

    stage3 = None
    if not args.skip_stage3:
        stage3 = fit_stage3_ppo(
            runs=runs,
            cfg=phy_cfg,
            nominal_params=stage1.params,
            residual_fn=residual_fn,
            weights=weights,
            bounds={"K_i": (1e-9, 1.0), "b_eq": (0.0, 10.0), "tau_eq": (0.0, 10.0), "l_com": (0.03, 0.45)},
            seed_list=list(args.ppo_seeds),
            total_timesteps=args.ppo_steps,
            outdir=args.outdir,
        )

    config_dump = {
        "args": vars(args),
        "physical": asdict(phy_cfg),
        "init_params": init_params,
        "weights": weights,
        "run_count": len(runs),
        "notes": {
            "irregular_sampling_strategy": "primary RK45 with native timestamps",
            "uniform_strategy": "resample dt_mean + RK4 baseline metrics",
            "ppo_role": "parameter proposal agent (single proposal per episode)",
        },
    }
    (args.outdir / "config_used.json").write_text(json.dumps(config_dump, indent=2, default=str), encoding="utf-8")

    (args.outdir / "stage1_result.json").write_text(json.dumps(asdict(stage1), indent=2), encoding="utf-8")
    if stage2 is not None:
        (args.outdir / "stage2_result.json").write_text(json.dumps(asdict(stage2), indent=2), encoding="utf-8")
    if stage3 is not None:
        (args.outdir / "stage3_result.json").write_text(json.dumps(asdict(stage3), indent=2), encoding="utf-8")

    write_summary_report(
        outdir=args.outdir,
        run_name=runs[0].name,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        config_dump=config_dump,
    )
    print(f"[DONE] artifacts written to {args.outdir}")


if __name__ == "__main__":
    main()
