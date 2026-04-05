#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-wise trajectory-level system identification (physical parameters only).

This pipeline optimizes only physical parameters:
    p = [K_u, l_com, b_eq, tau_eq]

Key rules:
- real-data-only fitting: theta_real, omega_real, hw_pwm
- free-running rollout (teacher forcing disabled)
- Chrono simulation loop in loss evaluation
- stage-wise optimization:
    Stage1(sin):   optimize [K_u, l_com]
    Stage2(square): optimize [b_eq]      with Stage1 params frozen
    Stage3(burst): optimize [tau_eq]     with others frozen
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chrono_core.config import BridgeConfig
from chrono_core.dynamics import PendulumModel, compute_model_torque_and_electrics


@dataclass
class PreprocessConfig:
    theta_sign: float = 1.0
    theta_offset: float = 0.0
    omega_smooth_window: int = 5
    omega_outlier_sigma: float = 4.0
    pwm_clip: float = 255.0


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 5.0e-3
    weight_decay: float = 0.0
    fd_eps_ratio: float = 2.0e-4
    w_theta: float = 1.0
    w_omega: float = 0.3
    optimizer_name: str = "adamw"
    grad_activation: str = "none"
    stabilization_sec: float = 1.0


@dataclass
class StageSpec:
    stage: int
    excitation_type: str
    csv_path: Path
    optimize_keys: tuple[str, ...]


@dataclass
class StageResult:
    stage: int
    csv_path: str
    optimize_keys: list[str]
    final_loss: float
    best_loss: float
    params: dict[str, float]
    fit_summary_plot: str
    overlay_plot: str
    loss_curve_plot: str
    param_trajectory_csv: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input(prompt: str, default: str | None = None) -> str:
    raw = input(prompt).strip()
    if raw == "" and default is not None:
        return default
    return raw


def list_csv_logs(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("*.csv"))


def choose_one_csv(items: list[Path], title: str) -> Path:
    if not items:
        raise ValueError("No CSV files found.")
    print(title)
    for i, item in enumerate(items, start=1):
        print(f"[{i}] {item.name}")
    while True:
        raw = _input("Select index: ")
        try:
            idx = int(raw)
            chosen = items[idx - 1]
            print(f"[INFO] selected_csv: {chosen}")
            return chosen
        except (ValueError, IndexError):
            print("Invalid selection. Please choose a valid index.")


def finite_interp(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    n = len(y)
    if n == 0:
        return y
    good = np.isfinite(y)
    if np.all(good):
        return y
    if not np.any(good):
        return np.zeros_like(y)
    idx = np.arange(n, dtype=float)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def robust_clip_sigma(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad + 1e-9
    lo = med - sigma * scale
    hi = med + sigma * scale
    return np.clip(x, lo, hi)


def smooth_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")


def preprocess_real_timeseries(df: pd.DataFrame, cfg: PreprocessConfig) -> dict[str, np.ndarray]:
    required = ["theta_real", "omega_real", "hw_pwm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    theta = pd.to_numeric(df["theta_real"], errors="coerce").to_numpy(dtype=float)
    omega = pd.to_numeric(df["omega_real"], errors="coerce").to_numpy(dtype=float)
    pwm = pd.to_numeric(df["hw_pwm"], errors="coerce").to_numpy(dtype=float)

    theta = finite_interp(theta)
    omega = finite_interp(omega)
    theta = np.unwrap(theta)
    theta = cfg.theta_sign * theta + cfg.theta_offset
    omega = robust_clip_sigma(omega, sigma=cfg.omega_outlier_sigma)
    omega = smooth_moving_average(omega, window=cfg.omega_smooth_window)
    # Keep measured/applied hw_pwm as-is except physical clipping.
    pwm = np.clip(pwm, -abs(cfg.pwm_clip), abs(cfg.pwm_clip))

    mask = np.isfinite(theta) & np.isfinite(omega) & np.isfinite(pwm)
    if int(mask.sum()) < 64:
        raise ValueError("Not enough valid samples after preprocessing.")

    return {
        "theta": theta[mask],
        "omega": omega[mask],
        "u": pwm[mask],
    }


def extract_current_ma(df: pd.DataFrame, n: int) -> np.ndarray:
    for col in ("current_mA", "ina219_current_ma", "ina_current_ma"):
        if col in df.columns:
            cur = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            cur = finite_interp(cur)
            if len(cur) >= n:
                return cur[:n]
            out = np.zeros(n, dtype=float)
            out[: len(cur)] = cur
            return out
    return np.zeros(n, dtype=float)


def parameter_bounds(cfg: BridgeConfig) -> dict[str, tuple[float, float]]:
    return {
        "K_u": (1.0e-6, 2.0e-1),
        "l_com": (cfg.l_com_min, cfg.l_com_max),
        "b_eq": (0.0, cfg.b_eq_max),
        "tau_eq": (0.0, cfg.tau_eq_max),
    }


def build_param_tensors(cfg: BridgeConfig) -> dict[str, torch.nn.Parameter]:
    return {
        "K_u": torch.nn.Parameter(torch.tensor(float(cfg.K_u_init), dtype=torch.float64)),
        "l_com": torch.nn.Parameter(torch.tensor(float(cfg.l_com_init), dtype=torch.float64)),
        "b_eq": torch.nn.Parameter(torch.tensor(float(cfg.b_eq_init), dtype=torch.float64)),
        "tau_eq": torch.nn.Parameter(torch.tensor(float(cfg.tau_eq_init), dtype=torch.float64)),
    }


def tensors_to_float_dict(params: dict[str, torch.nn.Parameter]) -> dict[str, float]:
    return {k: float(v.detach().cpu().item()) for k, v in params.items()}


def clamp_in_place(params: dict[str, torch.nn.Parameter], bounds: dict[str, tuple[float, float]]):
    with torch.no_grad():
        for k, p in params.items():
            lo, hi = bounds[k]
            p.data.clamp_(min=float(lo), max=float(hi))


def chrono_rollout(
    theta0: float,
    omega0: float,
    u: np.ndarray,
    dt: np.ndarray,
    cfg: BridgeConfig,
    p: dict[str, float],
):
    cfg_local = copy.deepcopy(cfg)
    cfg_local.theta0_deg = float(math.degrees(theta0))
    cfg_local.omega0 = float(omega0)

    model = PendulumModel(cfg_local)
    n = len(u)
    theta_sim = np.zeros(n, dtype=float)
    omega_sim = np.zeros(n, dtype=float)
    theta_sim[0] = float(theta0)
    omega_sim[0] = float(omega0)

    for k in range(n - 1):
        theta_k = model.get_theta()
        omega_k = model.get_omega()
        model_out = compute_model_torque_and_electrics(float(u[k]), theta_k, omega_k, float("nan"), p, cfg_local)
        model.apply_torque(model_out["tau_net"])
        model.step(max(float(dt[k]), 1e-6))
        theta_sim[k + 1] = model.get_theta()
        omega_sim[k + 1] = model.get_omega()

    # PendulumModel.get_theta() is wrapped to [-pi, pi]. For trajectory-level
    # matching against unwrapped theta_real, unwrap simulation output as well.
    theta_sim = np.unwrap(theta_sim)
    theta_sim = theta_sim + (float(theta0) - float(theta_sim[0]))
    return theta_sim, omega_sim


def trajectory_loss(theta_real: np.ndarray, omega_real: np.ndarray, theta_sim: np.ndarray, omega_sim: np.ndarray, cfg: TrainConfig) -> float:
    e_th = theta_sim - theta_real
    e_om = omega_sim - omega_real
    return float(np.mean(cfg.w_theta * (e_th ** 2) + cfg.w_omega * (e_om ** 2)))


def evaluate_loss(params: dict[str, torch.nn.Parameter], theta: np.ndarray, omega: np.ndarray, u: np.ndarray, dt: np.ndarray, cfg: BridgeConfig, train_cfg: TrainConfig):
    p = tensors_to_float_dict(params)
    theta_sim, omega_sim = chrono_rollout(
        theta[0],
        omega[0],
        u,
        dt,
        cfg,
        p,
    )
    loss = trajectory_loss(theta, omega, theta_sim, omega_sim, train_cfg)
    return loss, theta_sim, omega_sim


def apply_initial_stabilization(theta: np.ndarray, omega: np.ndarray, u: np.ndarray, dt: np.ndarray, hold_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hold = max(float(hold_sec), 0.0)
    if hold <= 0.0:
        return theta, omega, u, dt
    dt_nominal = float(np.median(dt)) if len(dt) > 0 else 0.01
    dt_nominal = min(max(dt_nominal, 1e-4), 0.2)
    n_hold = max(1, int(round(hold / dt_nominal)))
    theta_pad = np.zeros(n_hold, dtype=float)
    omega_pad = np.zeros(n_hold, dtype=float)
    u_pad = np.zeros(n_hold, dtype=float)
    dt_pad = np.full(n_hold, dt_nominal, dtype=float)
    theta_aug = np.concatenate([theta_pad, theta], axis=0)
    omega_aug = np.concatenate([omega_pad, omega], axis=0)
    u_aug = np.concatenate([u_pad, u], axis=0)
    dt_aug = np.concatenate([dt_pad, dt], axis=0)
    return theta_aug, omega_aug, u_aug, dt_aug


def apply_gradient_activation(grads: dict[str, float], active_keys: tuple[str, ...], activation: str) -> dict[str, float]:
    mode = activation.lower()
    if mode == "none":
        return grads
    arr = np.array([grads[k] for k in active_keys], dtype=float)
    if mode == "relu":
        arr_out = np.maximum(arr, 0.0)
    elif mode == "sigmoid":
        arr_out = 2.0 / (1.0 + np.exp(-arr)) - 1.0
    elif mode == "tanh":
        arr_out = np.tanh(arr)
    elif mode == "softmax":
        z = arr - np.max(arr)
        ex = np.exp(z)
        arr_out = ex / max(np.sum(ex), 1e-12)
    else:
        arr_out = arr
    out = dict(grads)
    for i, k in enumerate(active_keys):
        out[k] = float(arr_out[i])
    return out


def build_optimizer(params: dict[str, torch.nn.Parameter], keys: tuple[str, ...], train_cfg: TrainConfig):
    target = [params[k] for k in keys]
    name = train_cfg.optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(target, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    return torch.optim.AdamW(target, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)


def maybe_interactive_training_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.interactive:
        return args
    print("\n=== System Identification Training Config (Interactive) ===")
    raw_w_theta = _input(f"w_theta (current: {args.w_theta}): ", str(args.w_theta))
    raw_w_omega = _input(f"w_omega (current: {args.w_omega}): ", str(args.w_omega))
    raw_opt = _input(f"optimizer [adam/adamw] (current: {args.optimizer}): ", args.optimizer)
    raw_act = _input(
        f"gradient activation [none/relu/sigmoid/softmax/tanh] (current: {args.grad_activation}): ",
        args.grad_activation,
    )
    raw_stab = _input(f"stabilization_sec (current: {args.stabilization_sec}): ", str(args.stabilization_sec))

    try:
        args.w_theta = float(raw_w_theta)
        args.w_omega = float(raw_w_omega)
        args.stabilization_sec = float(raw_stab)
    except ValueError:
        print("[WARN] Invalid numeric input detected. Keeping previous numeric values.")
    raw_opt = raw_opt.strip().lower()
    raw_act = raw_act.strip().lower()
    if raw_opt in ("adam", "adamw"):
        args.optimizer = raw_opt
    else:
        print(f"[WARN] Unsupported optimizer '{raw_opt}'. Keeping {args.optimizer}.")
    if raw_act in ("none", "relu", "sigmoid", "softmax", "tanh"):
        args.grad_activation = raw_act
    else:
        print(f"[WARN] Unsupported gradient activation '{raw_act}'. Keeping {args.grad_activation}.")

    print("\n[Terminal Monitor] Active Training Setup")
    print("┌──────────────────────────────────────────────────────────────┐")
    print(f"│ w_theta={args.w_theta:<10.6f}  w_omega={args.w_omega:<10.6f}            │")
    print(f"│ optimizer={args.optimizer:<8}  grad_activation={args.grad_activation:<8}      │")
    print(f"│ stabilization_sec={args.stabilization_sec:<8.3f}                               │")
    print("└──────────────────────────────────────────────────────────────┘")
    return args


def finite_difference_grad(
    params: dict[str, torch.nn.Parameter],
    active_keys: tuple[str, ...],
    theta: np.ndarray,
    omega: np.ndarray,
    u: np.ndarray,
    dt: np.ndarray,
    cfg: BridgeConfig,
    train_cfg: TrainConfig,
    bounds: dict[str, tuple[float, float]],
):
    base_loss, _, _ = evaluate_loss(params, theta, omega, u, dt, cfg, train_cfg)
    grads: dict[str, float] = {}

    for k in active_keys:
        cur = float(params[k].detach().cpu().item())
        lo, hi = bounds[k]
        eps = max(train_cfg.fd_eps_ratio * max(abs(cur), 1.0), 1e-7)

        with torch.no_grad():
            params[k].fill_(min(max(cur + eps, lo), hi))
        lp, _, _ = evaluate_loss(params, theta, omega, u, dt, cfg, train_cfg)

        with torch.no_grad():
            params[k].fill_(min(max(cur - eps, lo), hi))
        lm, _, _ = evaluate_loss(params, theta, omega, u, dt, cfg, train_cfg)

        with torch.no_grad():
            params[k].fill_(cur)

        grads[k] = float((lp - lm) / max(2.0 * eps, 1e-12))

    return base_loss, grads


def save_fit_summary_plot(out_path: Path, theta_real: np.ndarray, omega_real: np.ndarray, theta_sim: np.ndarray, omega_sim: np.ndarray):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(theta_real, label="theta_real", lw=1.2)
    axes[0, 0].plot(theta_sim, label="theta_sim", lw=1.2)
    axes[0, 0].set_title("Theta trajectory overlay")
    axes[0, 0].legend()

    axes[0, 1].plot(omega_real, label="omega_real", lw=1.2)
    axes[0, 1].plot(omega_sim, label="omega_sim", lw=1.2)
    axes[0, 1].set_title("Omega trajectory overlay")
    axes[0, 1].legend()

    axes[1, 0].scatter(theta_real, theta_sim, s=6, alpha=0.4)
    axes[1, 0].set_xlabel("theta_real")
    axes[1, 0].set_ylabel("theta_sim")

    axes[1, 1].scatter(omega_real, omega_sim, s=6, alpha=0.4)
    axes[1, 1].set_xlabel("omega_real")
    axes[1, 1].set_ylabel("omega_sim")

    fig.suptitle("Trajectory-level system identification summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_overlay_plot(
    out_path: Path,
    u: np.ndarray,
    current_ma: np.ndarray,
    theta_real: np.ndarray,
    omega_real: np.ndarray,
    theta_sim: np.ndarray,
    omega_sim: np.ndarray,
):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)

    axes[0].plot(u, label="hw_pwm", lw=1.2, color="tab:purple")
    ax0r = axes[0].twinx()
    ax0r.plot(current_ma, label="current_mA", lw=1.1, color="tab:red", alpha=0.9)
    axes[0].set_ylabel("pwm")
    ax0r.set_ylabel("current [mA]")
    l1, t1 = axes[0].get_legend_handles_labels()
    l2, t2 = ax0r.get_legend_handles_labels()
    axes[0].legend(l1 + l2, t1 + t2, loc="upper right")

    axes[1].plot(theta_real, label="theta_real", lw=1.2)
    axes[1].plot(theta_sim, label="theta_sim", lw=1.2)
    axes[1].set_ylabel("theta")
    axes[1].legend(loc="upper right")

    axes[2].plot(omega_real, label="omega_real", lw=1.2)
    axes[2].plot(omega_sim, label="omega_sim", lw=1.2)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("omega")
    axes[2].legend(loc="upper right")

    fig.suptitle("PWM/Theta/Omega overlay (real vs Chrono sim)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_stage123_overlay_plot(out_path: Path, stage_series: list[dict[str, np.ndarray]]):
    import matplotlib.pyplot as plt

    if not stage_series:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    u_all = np.concatenate([s["u"] for s in stage_series], axis=0)
    cur_all = np.concatenate([s["current_ma"] for s in stage_series], axis=0)
    theta_real_all = np.concatenate([s["theta_real"] for s in stage_series], axis=0)
    theta_sim_all = np.concatenate([s["theta_sim"] for s in stage_series], axis=0)
    omega_real_all = np.concatenate([s["omega_real"] for s in stage_series], axis=0)
    omega_sim_all = np.concatenate([s["omega_sim"] for s in stage_series], axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9.0), sharex=True)
    axes[0].plot(u_all, label="hw_pwm", lw=1.1, color="tab:purple")
    ax0r = axes[0].twinx()
    ax0r.plot(cur_all, label="current_mA", lw=1.0, color="tab:red", alpha=0.9)
    axes[0].set_ylabel("pwm")
    ax0r.set_ylabel("current [mA]")
    l1, t1 = axes[0].get_legend_handles_labels()
    l2, t2 = ax0r.get_legend_handles_labels()
    axes[0].legend(l1 + l2, t1 + t2, loc="upper right")

    axes[1].plot(theta_real_all, label="theta_real", lw=1.0)
    axes[1].plot(theta_sim_all, label="theta_sim", lw=1.0)
    axes[1].set_ylabel("theta")
    axes[1].legend(loc="upper right")

    axes[2].plot(omega_real_all, label="omega_real", lw=1.0)
    axes[2].plot(omega_sim_all, label="omega_sim", lw=1.0)
    axes[2].set_ylabel("omega")
    axes[2].set_xlabel("concatenated step")
    axes[2].legend(loc="upper right")

    x_offset = 0
    for i, s in enumerate(stage_series, start=1):
        x_offset += len(s["u"])
        if i < len(stage_series):
            for ax in axes:
                ax.axvline(x_offset, linestyle="--", linewidth=1.0, alpha=0.35, color="gray")
        axes[0].text(
            max(0.0, x_offset - len(s["u"]) * 0.5),
            float(np.nanmax(u_all)) if np.isfinite(u_all).any() else 0.0,
            f"S{i}",
            fontsize=9,
            ha="center",
            va="bottom",
        )

    fig.suptitle("Stage1~3 concatenated overlay with PWM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_parameter_trajectory_plots(outdir: Path, rows: list[dict[str, float]]) -> list[str]:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    x = np.arange(1, len(df) + 1, dtype=float)
    out_paths: list[str] = []
    for key in ["K_u", "l_com", "b_eq", "tau_eq"]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(x, df[key].to_numpy(dtype=float), lw=1.2)
        ax.set_xlabel("global update step")
        ax.set_ylabel(key)
        ax.set_title(f"Parameter fitting trajectory: {key}")
        ax.grid(True, alpha=0.25)
        path = outdir / f"parameter_trajectory_{key}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        out_paths.append(str(path))
    return out_paths


def save_loss_convergence_plot(out_path: Path, losses: list[float]):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(losses) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(epochs, losses, label="trajectory_loss", lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Trajectory loss convergence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_params_json(path: Path, params: dict[str, torch.nn.Parameter]):
    payload = {k: float(v.detach().cpu().item()) for k, v in params.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_on_stage(
    stage_spec: StageSpec,
    params: dict[str, torch.nn.Parameter],
    bounds: dict[str, tuple[float, float]],
    cfg: BridgeConfig,
    pre_cfg: PreprocessConfig,
    train_cfg: TrainConfig,
    outdir: Path,
) -> StageResult:
    print(f"[INFO] Stage {stage_spec.stage} start")
    print(f"  - dataset_path: {stage_spec.csv_path}")
    print(f"  - excitation_type: {stage_spec.excitation_type}")
    print(f"  - optimize_keys: {list(stage_spec.optimize_keys)}")
    print("  - source_policy: theta=theta_real, omega=omega_real, input=hw_pwm")
    print("  - simulation_backend: in-process PendulumModel (chrono_core.dynamics), no external chrono_pendulum.py process during fitting")

    df = pd.read_csv(stage_spec.csv_path)
    proc = preprocess_real_timeseries(df, pre_cfg)
    theta, omega, u = proc["theta"], proc["omega"], proc["u"]
    current_ma = extract_current_ma(df, len(theta))
    current_ma = np.sign(u) * current_ma

    dt = np.full(len(theta), float(cfg.step), dtype=float)
    if "wall_elapsed" in df.columns:
        t_raw = pd.to_numeric(df["wall_elapsed"], errors="coerce").to_numpy(dtype=float)
        t_raw = finite_interp(t_raw)
        if len(t_raw) == len(theta):
            dt = np.diff(t_raw, prepend=t_raw[0])
            if len(dt) > 1:
                dt[0] = dt[1]
            dt = np.clip(dt, 1e-6, 0.2)
    n_before = len(theta)
    theta, omega, u, dt = apply_initial_stabilization(theta, omega, u, dt, train_cfg.stabilization_sec)
    n_hold = len(theta) - n_before
    if n_hold > 0:
        current_ma = np.concatenate([np.zeros(n_hold, dtype=float), current_ma], axis=0)

    optimizer = build_optimizer(params, stage_spec.optimize_keys, train_cfg)

    best_loss = float("inf")
    best_snapshot = tensors_to_float_dict(params)
    history: list[float] = []
    param_history_rows: list[dict[str, float]] = []

    for ep in range(1, train_cfg.epochs + 1):
        base_loss, grads_raw = finite_difference_grad(
            params, stage_spec.optimize_keys, theta, omega, u, dt, cfg, train_cfg, bounds
        )
        grads = apply_gradient_activation(grads_raw, stage_spec.optimize_keys, train_cfg.grad_activation)

        optimizer.zero_grad(set_to_none=True)
        for key in stage_spec.optimize_keys:
            params[key].grad = torch.tensor(grads[key], dtype=params[key].dtype)
        optimizer.step()
        clamp_in_place(params, bounds)

        cur_loss, _, _ = evaluate_loss(params, theta, omega, u, dt, cfg, train_cfg)
        history.append(cur_loss)
        p_row = tensors_to_float_dict(params)
        p_row["epoch"] = float(ep)
        p_row["loss"] = float(cur_loss)
        param_history_rows.append(p_row)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_snapshot = tensors_to_float_dict(params)

        if ep == 1 or ep % 10 == 0 or ep == train_cfg.epochs:
            p = tensors_to_float_dict(params)
            print(
                f"  [Stage {stage_spec.stage}] epoch {ep}/{train_cfg.epochs} "
                f"loss(before={base_loss:.6f}, after={cur_loss:.6f}) "
                f"K_u={p['K_u']:.6g}, l_com={p['l_com']:.6g}, b_eq={p['b_eq']:.6g}, tau_eq={p['tau_eq']:.6g}"
            )

    final_params = tensors_to_float_dict(params)
    final_loss, theta_sim, omega_sim = evaluate_loss(params, theta, omega, u, dt, cfg, train_cfg)

    stage_dir = outdir / f"stage{stage_spec.stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    fit_plot = stage_dir / f"stage{stage_spec.stage}_fit_summary.png"
    overlay_plot = stage_dir / f"stage{stage_spec.stage}_theta_omega_overlay.png"
    loss_curve_plot = stage_dir / f"stage{stage_spec.stage}_loss_convergence.png"
    loss_curve_csv = stage_dir / f"stage{stage_spec.stage}_loss_history.csv"
    param_traj_csv = stage_dir / f"stage{stage_spec.stage}_param_trajectory.csv"
    stage_param_json = stage_dir / "trajectory_model_params.json"

    save_fit_summary_plot(fit_plot, theta, omega, theta_sim, omega_sim)
    save_overlay_plot(overlay_plot, u, current_ma, theta, omega, theta_sim, omega_sim)
    save_loss_convergence_plot(loss_curve_plot, history)
    pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "trajectory_loss": history}).to_csv(loss_curve_csv, index=False)
    pd.DataFrame(param_history_rows).to_csv(param_traj_csv, index=False)
    save_params_json(stage_param_json, params)

    metadata = {
        "timestamp": utc_now(),
        "stage": stage_spec.stage,
        "csv_path": str(stage_spec.csv_path),
        "excitation_type": stage_spec.excitation_type,
        "pipeline": "trajectory_level_physical_parameter_identification",
        "model_type": "physical_parameter_only",
        "optimize_keys": list(stage_spec.optimize_keys),
        "loss": {
            "type": "trajectory_mse",
            "w_theta": train_cfg.w_theta,
            "w_omega": train_cfg.w_omega,
        },
        "source_policy": {
            "real_data_only": True,
            "theta_source": "theta_real",
            "omega_source": "omega_real",
            "input_source": "hw_pwm",
            "cmd_u_raw_used_for_fitting": False,
            "sim_data_used_for_fitting": False,
            "teacher_forcing": False,
        },
        "best_loss": float(best_loss),
        "best_params": best_snapshot,
        "final_loss": float(final_loss),
        "final_params": final_params,
        "plot_paths": {
            "fit_summary": str(fit_plot),
            "overlay": str(overlay_plot),
            "loss_curve": str(loss_curve_plot),
        },
    }
    with (stage_dir / f"stage{stage_spec.stage}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Stage {stage_spec.stage} best params: {best_snapshot}")
    print(f"[INFO] Stage {stage_spec.stage} final params: {final_params}")

    return StageResult(
        stage=stage_spec.stage,
        csv_path=str(stage_spec.csv_path),
        optimize_keys=list(stage_spec.optimize_keys),
        final_loss=float(final_loss),
        best_loss=float(best_loss),
        params=final_params,
        fit_summary_plot=str(fit_plot),
        overlay_plot=str(overlay_plot),
        loss_curve_plot=str(loss_curve_plot),
        param_trajectory_csv=str(param_traj_csv),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-wise trajectory-level system identification (physical params only)")
    ap.add_argument("--run-logs", default="run_logs", help="Directory containing chrono_run_*.csv")
    ap.add_argument("--stage1-csv", default="", help="Stage1 (sin) CSV path")
    ap.add_argument("--stage2-csv", default="", help="Stage2 (square) CSV path")
    ap.add_argument("--stage3-csv", default="", help="Stage3 (burst) CSV path")
    ap.add_argument("--mode", choices=["stage1", "stage12", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5.0e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--fd-eps-ratio", type=float, default=2.0e-4)
    ap.add_argument("--w-theta", type=float, default=1.0)
    ap.add_argument("--w-omega", type=float, default=0.3)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--grad-activation", choices=["none", "relu", "sigmoid", "softmax", "tanh"], default="none")
    ap.add_argument("--stabilization-sec", type=float, default=1.0)
    ap.add_argument("--theta-sign", type=float, default=1.0)
    ap.add_argument("--theta-offset", type=float, default=0.0)
    ap.add_argument("--omega-smooth-window", type=int, default=5)
    ap.add_argument("--omega-outlier-sigma", type=float, default=4.0)
    ap.add_argument("--pwm-clip", type=float, default=255.0)
    ap.add_argument("--interactive", action="store_true")
    return ap.parse_args()


def maybe_resolve_stage_csvs(args: argparse.Namespace, run_logs: Path):
    s1 = Path(args.stage1_csv).expanduser().resolve() if args.stage1_csv else None
    s2 = Path(args.stage2_csv).expanduser().resolve() if args.stage2_csv else None
    s3 = Path(args.stage3_csv).expanduser().resolve() if args.stage3_csv else None

    if args.interactive and (s1 is None or (args.mode in ("stage12", "full") and s2 is None) or (args.mode == "full" and s3 is None)):
        csvs = list_csv_logs(run_logs)
        if s1 is None:
            s1 = choose_one_csv(csvs, "Select Stage1 (sin) CSV")
        if args.mode in ("stage12", "full") and s2 is None:
            s2 = choose_one_csv(csvs, "Select Stage2 (square) CSV")
        if args.mode == "full" and s3 is None:
            s3 = choose_one_csv(csvs, "Select Stage3 (burst) CSV")

    if s1 is None:
        raise ValueError("--stage1-csv is required (or use --interactive)")
    if args.mode in ("stage12", "full") and s2 is None:
        raise ValueError("--stage2-csv is required for mode stage12/full")
    if args.mode == "full" and s3 is None:
        raise ValueError("--stage3-csv is required for mode full")

    return s1, s2, s3


def run_pipeline(args: argparse.Namespace):
    args = maybe_interactive_training_config(args)
    run_logs = Path(args.run_logs).expanduser().resolve()
    outdir = run_logs
    outdir.mkdir(parents=True, exist_ok=True)

    s1, s2, s3 = maybe_resolve_stage_csvs(args, run_logs)

    pre_cfg = PreprocessConfig(
        theta_sign=args.theta_sign,
        theta_offset=args.theta_offset,
        omega_smooth_window=args.omega_smooth_window,
        omega_outlier_sigma=args.omega_outlier_sigma,
        pwm_clip=args.pwm_clip,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fd_eps_ratio=args.fd_eps_ratio,
        w_theta=args.w_theta,
        w_omega=args.w_omega,
        optimizer_name=args.optimizer,
        grad_activation=args.grad_activation,
        stabilization_sec=args.stabilization_sec,
    )

    cfg = BridgeConfig()
    params = build_param_tensors(cfg)
    bounds = parameter_bounds(cfg)

    stage_specs = [
        StageSpec(1, "sin", s1, ("K_u", "l_com")),
        StageSpec(2, "square", s2, ("b_eq",)),
        StageSpec(3, "burst", s3, ("tau_eq",)),
    ]
    max_stage = {"stage1": 1, "stage12": 2, "full": 3}[args.mode]

    results: list[StageResult] = []
    for spec in stage_specs[:max_stage]:
        if spec.csv_path is None:
            continue
        res = train_on_stage(spec, params, bounds, cfg, pre_cfg, train_cfg, outdir)
        results.append(res)

    # Consolidated system-identification visual summary (8 plots total).
    summary_dir = outdir / "system_identification_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    stage_overlay_paths: list[str] = []
    stage_series: list[dict[str, np.ndarray]] = []
    global_param_rows: list[dict[str, float]] = []
    p_final = tensors_to_float_dict(params)

    for res in results:
        df = pd.read_csv(res.csv_path)
        proc = preprocess_real_timeseries(df, pre_cfg)
        theta = proc["theta"]
        omega = proc["omega"]
        u = proc["u"]
        current_ma = extract_current_ma(df, len(theta))
        current_ma = np.sign(u) * current_ma
        dt = np.full(len(theta), float(cfg.step), dtype=float)
        if "wall_elapsed" in df.columns:
            t_raw = pd.to_numeric(df["wall_elapsed"], errors="coerce").to_numpy(dtype=float)
            t_raw = finite_interp(t_raw)
            if len(t_raw) == len(theta):
                dt = np.diff(t_raw, prepend=t_raw[0])
                if len(dt) > 1:
                    dt[0] = dt[1]
                dt = np.clip(dt, 1e-6, 0.2)
        n_before = len(theta)
        theta, omega, u, dt = apply_initial_stabilization(theta, omega, u, dt, train_cfg.stabilization_sec)
        n_hold = len(theta) - n_before
        if n_hold > 0:
            current_ma = np.concatenate([np.zeros(n_hold, dtype=float), current_ma], axis=0)
        theta_sim, omega_sim = chrono_rollout(0.0, 0.0, u, dt, cfg, p_final)
        stage_out = summary_dir / f"stage{res.stage}_overlay_with_pwm.png"
        save_overlay_plot(stage_out, u, current_ma, theta, omega, theta_sim, omega_sim)
        stage_overlay_paths.append(str(stage_out))
        stage_series.append(
            {
                "u": u,
                "current_ma": current_ma,
                "theta_real": theta,
                "theta_sim": theta_sim,
                "omega_real": omega,
                "omega_sim": omega_sim,
            }
        )

        traj_df = pd.read_csv(res.param_trajectory_csv)
        for _, row in traj_df.iterrows():
            global_param_rows.append(
                {
                    "K_u": float(row["K_u"]),
                    "l_com": float(row["l_com"]),
                    "b_eq": float(row["b_eq"]),
                    "tau_eq": float(row["tau_eq"]),
                }
            )

    stage123_overlay_path = summary_dir / "stage123_overlay_with_pwm.png"
    save_stage123_overlay_plot(stage123_overlay_path, stage_series)
    param_plot_paths = save_parameter_trajectory_plots(summary_dir, global_param_rows)

    summary = {
        "timestamp": utc_now(),
        "pipeline": "stage_wise_trajectory_level_system_identification",
        "model_type": "physical_parameter_only",
        "optimization_backend": "torch_adamw_with_finite_difference_gradients",
        "train_config": asdict(train_cfg),
        "preprocess_config": asdict(pre_cfg),
        "stages": [asdict(r) for r in results],
        "final_params": tensors_to_float_dict(params),
        "system_identification_summary": {
            "output_dir": str(summary_dir),
            "plots": stage_overlay_paths + [str(stage123_overlay_path)] + param_plot_paths,
        },
    }
    summary_path = outdir / "trajectory_fit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    final_param_path = outdir / "trajectory_model_params.json"
    save_params_json(final_param_path, params)

    print(f"[INFO] pipeline_done: mode={args.mode}")
    print(f"[INFO] summary_json: {summary_path}")
    print(f"[INFO] trajectory_model_params: {final_param_path}")
    print(f"[INFO] final_params: {tensors_to_float_dict(params)}")


if __name__ == "__main__":
    run_pipeline(parse_args())
