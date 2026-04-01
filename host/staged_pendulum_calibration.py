#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-wise discrete-time trajectory fitting pipeline (PyTorch GRU).

This module keeps the Stage 1/2/3 user workflow but replaces the previous
closed-form J*alpha regression with a direct black-box dynamics learner:
    (theta, omega, hw_pwm, history) -> (theta_next, omega_next)

Design constraints:
- real-data-only fitting
- hw_pwm as the fitting input source (cmd_u_raw not used for training)
- discrete-time trajectory matching as the optimization target
- stage-wise refinement: Stage1(sin) -> Stage2(square) -> Stage3(burst)
- PyTorch-only training backend (no TensorFlow)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chrono_core.config import BridgeConfig


# -----------------------------
# Config and data containers
# -----------------------------


@dataclass
class PreprocessConfig:
    theta_sign: float = 1.0
    theta_offset: float = 0.0
    omega_smooth_window: int = 7
    omega_outlier_sigma: float = 4.0
    pwm_clip: float = 255.0


@dataclass
class TrainConfig:
    sequence_length: int = 32
    rollout_horizon: int = 5
    batch_size: int = 128
    epochs: int = 120
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-7
    grad_clip: float = 1.0
    val_split: float = 0.2
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    w_theta: float = 1.0
    w_omega: float = 1.0
    rollout_weight: float = 0.3
    seed: int = 7


@dataclass
class StageSpec:
    stage: int
    excitation_type: str
    csv_path: Path


@dataclass
class FeatureStats:
    mean: list[float]
    std: list[float]


@dataclass
class StageResult:
    stage: int
    csv_path: str
    checkpoint_path: str
    train_loss: float
    val_loss: float
    fit_summary_plot: str
    overlay_plot: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Utility / preprocessing
# -----------------------------


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
    pwm = finite_interp(pwm)

    # theta_real preprocessing
    theta = np.unwrap(theta)
    theta = cfg.theta_sign * theta + cfg.theta_offset

    # omega_real preprocessing
    omega = robust_clip_sigma(omega, sigma=cfg.omega_outlier_sigma)
    omega = smooth_moving_average(omega, window=cfg.omega_smooth_window)

    # hw_pwm preprocessing
    pwm = np.clip(pwm, -abs(cfg.pwm_clip), abs(cfg.pwm_clip))

    mask = np.isfinite(theta) & np.isfinite(omega) & np.isfinite(pwm)
    if int(mask.sum()) < 64:
        raise ValueError("Not enough valid samples after preprocessing.")

    theta = theta[mask]
    omega = omega[mask]
    pwm = pwm[mask]

    return {
        "theta": theta,
        "omega": omega,
        "u": pwm,

     
    }
def gradient_with_unit_dt(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x)

  
# -----------------------------
# Dataset / model
# -----------------------------


class SequenceDataset(Dataset):
    def __init__(
        self,
        x_hist: np.ndarray,
        y_next: np.ndarray,
        u_future: np.ndarray,
        y_future: np.ndarray,
    ):
        self.x_hist = torch.tensor(x_hist, dtype=torch.float32)
        self.y_next = torch.tensor(y_next, dtype=torch.float32)
        self.u_future = torch.tensor(u_future, dtype=torch.float32)
        self.y_future = torch.tensor(y_future, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_hist.shape[0]

    def __getitem__(self, idx: int):
        return self.x_hist[idx], self.y_next[idx], self.u_future[idx], self.y_future[idx]


class GRUDynamics(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x_hist)
        return self.head(out[:, -1, :])


def make_windows(theta: np.ndarray, omega: np.ndarray, u: np.ndarray, L: int, H: int) -> tuple[np.ndarray, ...]:
    n = len(theta)
    x_hist, y_next, u_future, y_future = [], [], [], []
    feats = np.column_stack([theta, omega, u])
    states = np.column_stack([theta, omega])

    # t: last index of history window
    for t in range(L - 1, n - H - 1):
        x_hist.append(feats[t - L + 1 : t + 1])
        y_next.append(states[t + 1])
        u_future.append(u[t + 1 : t + H + 1])
        y_future.append(states[t + 1 : t + H + 1])

    if not x_hist:
        raise ValueError("Dataset too short for selected sequence_length/rollout_horizon.")

    return (
        np.asarray(x_hist, dtype=float),
        np.asarray(y_next, dtype=float),
        np.asarray(u_future, dtype=float),
        np.asarray(y_future, dtype=float),
    )


def compute_feature_stats(x_hist: np.ndarray) -> FeatureStats:
    flat = x_hist.reshape(-1, x_hist.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return FeatureStats(mean=mean.tolist(), std=std.tolist())


def scale_features(x: np.ndarray, stats: FeatureStats) -> np.ndarray:
    mean = np.asarray(stats.mean, dtype=float)
    std = np.asarray(stats.std, dtype=float)
    return (x - mean) / std


def split_train_val(n: int, val_split: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    cut = int(n * (1.0 - val_split))
    cut = max(1, min(cut, n - 1))
    return idx[:cut], idx[cut:]


def autoregressive_rollout(model: GRUDynamics, hist: torch.Tensor, u_future: torch.Tensor, horizon: int) -> torch.Tensor:
    # hist: [B, L, 3], u_future: [B, H]
    cur_hist = hist
    preds = []
    for k in range(horizon):
        state_pred = model(cur_hist)
        preds.append(state_pred)
        next_in = torch.cat([state_pred, u_future[:, k : k + 1]], dim=1).unsqueeze(1)
        cur_hist = torch.cat([cur_hist[:, 1:, :], next_in], dim=1)
    return torch.stack(preds, dim=1)


def compute_loss(
    model: GRUDynamics,
    batch: tuple[torch.Tensor, ...],
    train_cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    x_hist, y_next, u_future, y_future = batch
    pred_next = model(x_hist)

    theta_loss = torch.mean((pred_next[:, 0] - y_next[:, 0]) ** 2)
    omega_loss = torch.mean((pred_next[:, 1] - y_next[:, 1]) ** 2)
    one_step = train_cfg.w_theta * theta_loss + train_cfg.w_omega * omega_loss

    total = one_step
    rollout = torch.tensor(0.0, device=x_hist.device)
    if train_cfg.rollout_weight > 0 and train_cfg.rollout_horizon > 0:
        pred_roll = autoregressive_rollout(model, x_hist, u_future, train_cfg.rollout_horizon)
        rollout_theta = torch.mean((pred_roll[:, :, 0] - y_future[:, :, 0]) ** 2)
        rollout_omega = torch.mean((pred_roll[:, :, 1] - y_future[:, :, 1]) ** 2)
        rollout = train_cfg.w_theta * rollout_theta + train_cfg.w_omega * rollout_omega
        total = total + train_cfg.rollout_weight * rollout

    return total, {
        "one_step": float(one_step.detach().cpu().item()),
        "rollout": float(rollout.detach().cpu().item()),
    }


  def run_epoch(model: GRUDynamics, loader: DataLoader, optimizer: torch.optim.Optimizer | None, cfg: TrainConfig, device: torch.device) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    for batch in loader:
        batch = tuple(x.to(device) for x in batch)
        loss, _ = compute_loss(model, batch, cfg)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


# -----------------------------
# Plotting and reporting
# -----------------------------


def generate_predictions(model: GRUDynamics, theta: np.ndarray, omega: np.ndarray, u: np.ndarray, stats: FeatureStats, seq_len: int, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    features = np.column_stack([theta, omega, u])
    features = scale_features(features, stats)
    theta_pred = np.full_like(theta, np.nan)
    omega_pred = np.full_like(omega, np.nan)
    with torch.no_grad():
        for t in range(seq_len - 1, len(theta) - 1):
            h = torch.tensor(features[t - seq_len + 1 : t + 1], dtype=torch.float32, device=device).unsqueeze(0)
            y = model(h).squeeze(0).cpu().numpy()
            theta_pred[t + 1] = y[0] * stats.std[0] + stats.mean[0]
            omega_pred[t + 1] = y[1] * stats.std[1] + stats.mean[1]

    alpha_real = gradient_with_unit_dt(omega)
    alpha_sim = gradient_with_unit_dt(np.nan_to_num(omega_pred, nan=0.0))
    return {
        "theta_pred": theta_pred,
        "omega_pred": omega_pred,
        "alpha_pred": alpha_sim,
        "alpha_real": alpha_real,
    }


def save_fit_summary_plot(out_path: Path, theta_real: np.ndarray, omega_real: np.ndarray, theta_pred: np.ndarray, omega_pred: np.ndarray):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(theta_pred) & np.isfinite(omega_pred)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(theta_real, label="theta_real", lw=1.2)
    ax.plot(theta_pred, label="theta_pred", lw=1.2)
    ax.set_title("Theta trajectory overlay")
    ax.set_xlabel("step")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(omega_real, label="omega_real", lw=1.2)
    ax.plot(omega_pred, label="omega_pred", lw=1.2)
    ax.set_title("Omega trajectory overlay")
    ax.set_xlabel("step")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(theta_real[valid], theta_pred[valid], s=6, alpha=0.4)
    ax.set_title("Theta prediction vs target")
    ax.set_xlabel("theta_real")
    ax.set_ylabel("theta_pred")

    ax = axes[1, 1]
    ax.scatter(omega_real[valid], omega_pred[valid], s=6, alpha=0.4)
    ax.set_title("Omega prediction vs target")
    ax.set_xlabel("omega_real")
    ax.set_ylabel("omega_pred")

    fig.suptitle("Discrete-time trajectory fitting summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_overlay_plot(out_path: Path, theta_real: np.ndarray, omega_real: np.ndarray, alpha_real: np.ndarray, theta_pred: np.ndarray, omega_pred: np.ndarray, alpha_pred: np.ndarray):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(theta_real, label="theta_real", lw=1.2)
    axes[0].plot(theta_pred, label="theta_sim(pred)", lw=1.2)
    axes[0].set_ylabel("theta")
    axes[0].legend(loc="upper right")

    axes[1].plot(omega_real, label="omega_real", lw=1.2)
    axes[1].plot(omega_pred, label="omega_sim(pred)", lw=1.2)
    axes[1].set_ylabel("omega")
    axes[1].legend(loc="upper right")

    axes[2].plot(alpha_real, label="alpha_real(post)", lw=1.2)
    axes[2].plot(alpha_pred, label="alpha_sim(post)", lw=1.2)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("alpha")
    axes[2].legend(loc="upper right")

    fig.suptitle("Theta/Omega/Alpha overlay (real vs sim-pred)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_loss_convergence_plot(out_path: Path, train_loss: list[float], val_loss: list[float]):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(epochs, train_loss, label="train_loss", lw=1.4)
    ax.plot(epochs, val_loss, label="val_loss", lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss convergence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# -----------------------------
# Stage execution
# -----------------------------


def build_dataloaders(theta: np.ndarray, omega: np.ndarray, u: np.ndarray, train_cfg: TrainConfig) -> tuple[DataLoader, DataLoader, FeatureStats]:
    x_hist, y_next, u_future, y_future = make_windows(theta, omega, u, train_cfg.sequence_length, train_cfg.rollout_horizon)
    stats = compute_feature_stats(x_hist)

    x_hist_s = scale_features(x_hist, stats)
    y_next_s = scale_features(y_next, FeatureStats(mean=stats.mean[:2], std=stats.std[:2]))
    u_future_s = (u_future - stats.mean[2]) / stats.std[2]
    y_future_s = scale_features(y_future, FeatureStats(mean=stats.mean[:2], std=stats.std[:2]))

    tr_idx, va_idx = split_train_val(len(x_hist_s), train_cfg.val_split)
    tr_ds = SequenceDataset(x_hist_s[tr_idx], y_next_s[tr_idx], u_future_s[tr_idx], y_future_s[tr_idx])
    va_ds = SequenceDataset(x_hist_s[va_idx], y_next_s[va_idx], u_future_s[va_idx], y_future_s[va_idx])

    tr_loader = DataLoader(tr_ds, batch_size=train_cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=train_cfg.batch_size, shuffle=False)
    return tr_loader, va_loader, stats


def train_on_stage(
    model: GRUDynamics,
    csv_path: Path,
    outdir: Path,
    stage_spec: StageSpec,
    pre_cfg: PreprocessConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> tuple[StageResult, FeatureStats]:
    print(f"[INFO] Stage {stage_spec.stage} start")
    print(f"  - dataset_path: {csv_path}")
    print(f"  - excitation_type: {stage_spec.excitation_type}")
    print("  - source_policy: theta=theta_real, omega=omega_real, input=hw_pwm")
    print("  - model_type: PyTorch GRU black-box learner")
    print("  - target: theta_next_omega_next (discrete-time)")

    df = pd.read_csv(csv_path)
    proc = preprocess_real_timeseries(df, pre_cfg)
    theta, omega, u = proc["theta"], proc["omega"], proc["u"]

    tr_loader, va_loader, stats = build_dataloaders(theta, omega, u, train_cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    hist_train: list[float] = []
    hist_val: list[float] = []

    for ep in range(1, train_cfg.epochs + 1):
        tr_loss = run_epoch(model, tr_loader, optimizer, train_cfg, device)
        va_loss = run_epoch(model, va_loader, None, train_cfg, device)
        hist_train.append(tr_loss)
        hist_val.append(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == train_cfg.epochs:
            print(f"  [Stage {stage_spec.stage}] epoch {ep}/{train_cfg.epochs} train={tr_loss:.6f} val={va_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    stage_dir = outdir / f"stage{stage_spec.stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = stage_dir / "gru_dynamics.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_stats": asdict(stats),
        "stage": stage_spec.stage,
        "sequence_length": train_cfg.sequence_length,
    }, ckpt_path)

    pred = generate_predictions(model, theta, omega, u, stats, train_cfg.sequence_length, device)

    fit_plot = stage_dir / f"stage{stage_spec.stage}_fit_summary.png"
    overlay_plot = stage_dir / f"stage{stage_spec.stage}_theta_omega_alpha_overlay.png"
    save_fit_summary_plot(fit_plot, theta, omega, pred["theta_pred"], pred["omega_pred"])
    save_overlay_plot(
        overlay_plot,
        theta,
        omega,
        pred["alpha_real"],
        pred["theta_pred"],
        pred["omega_pred"],
        pred["alpha_pred"],
    )
    loss_curve_plot = stage_dir / f"stage{stage_spec.stage}_loss_convergence.png"
    loss_curve_csv = stage_dir / f"stage{stage_spec.stage}_loss_history.csv"
    pd.DataFrame({
        "epoch": np.arange(1, len(hist_train) + 1, dtype=int),
        "train_loss": hist_train,
        "val_loss": hist_val,
    }).to_csv(loss_curve_csv, index=False)
    save_loss_convergence_plot(loss_curve_plot, hist_train, hist_val)

    metadata = {
        "timestamp": utc_now(),
        "stage": stage_spec.stage,
        "csv_path": str(csv_path),
        "excitation_type": stage_spec.excitation_type,
        "model_type": "pytorch_gru_black_box_dynamics",
        "theta_source": "theta_real",
        "omega_source": "omega_real",
        "input_source": "hw_pwm",
        "target_type": "theta_next_omega_next",
        "sequence_length": train_cfg.sequence_length,
        "rollout_horizon": train_cfg.rollout_horizon,
        "normalization_method": "zscore_per_feature",
        "training_loss": float(hist_train[-1]),
        "validation_loss": float(hist_val[-1]),
        "best_validation_loss": float(best_val),
        "checkpoint_path": str(ckpt_path),
        "plot_paths": {
            "fit_summary": str(fit_plot),
            "overlay": str(overlay_plot),
            "loss_convergence": str(loss_curve_plot),
        },
        "loss_history_csv": str(loss_curve_csv),
        "epoch_logs": [
            {
                "epoch": int(i + 1),
                "train_loss": float(hist_train[i]),
                "val_loss": float(hist_val[i]),
            }
            for i in range(len(hist_train))
        ],
        "source_policy": {
            "sim_data_used_for_fitting": False,
            "cmd_u_raw_used_for_fitting": False,
            "real_data_only": True,
        },
    }
    with (stage_dir / f"stage{stage_spec.stage}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return StageResult(
        stage=stage_spec.stage,
        csv_path=str(csv_path),
        checkpoint_path=str(ckpt_path),
        train_loss=float(hist_train[-1]),
        val_loss=float(best_val),
        fit_summary_plot=str(fit_plot),
        overlay_plot=str(overlay_plot),
    ), stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-wise discrete-time trajectory fitting (PyTorch GRU)")
    p.add_argument("--run_logs", type=Path, default=Path(__file__).resolve().parent / "run_logs")
    p.add_argument("--mode", choices=["interactive", "stage1", "stage12", "full", "eval"], default="interactive")
    p.add_argument("--stage1_csv", type=Path, default=None)
    p.add_argument("--stage2_csv", type=Path, default=None)
    p.add_argument("--stage3_csv", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--sequence_length", type=int, default=32)
    p.add_argument("--rollout_horizon", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--w_theta", type=float, default=1.0)
    p.add_argument("--w_omega", type=float, default=1.0)
    p.add_argument("--rollout_weight", type=float, default=0.3)
    p.add_argument("--save_checkpoint", action="store_true")
    p.add_argument("--skip_plots", action="store_true")
    return p.parse_args()


def interactive_menu(args: argparse.Namespace) -> tuple[str, Path | None, Path | None, Path | None]:
    logs = list_csv_logs(args.run_logs)
    print("\n=== Chrono Pendulum GRU Trajectory Fitting ===")
    print("1) Run Stage 1 only")
    print("2) Run Stage 1 -> 2")
    print("3) Run Full Pipeline (1 -> 2 -> 3)")
    print("4) Evaluate trained model (latest metadata)")
    print("5) Exit")
    choice = _input("Select menu [1-5]: ", "3")
    if choice == "5":
        return "exit", None, None, None
    if choice == "4":
        return "eval", None, None, None

    s1 = choose_one_csv(logs, "Select Stage 1 CSV (sin):") if args.stage1_csv is None else args.stage1_csv
    s2 = choose_one_csv(logs, "Select Stage 2 CSV (square):") if args.stage2_csv is None else args.stage2_csv
    s3 = choose_one_csv(logs, "Select Stage 3 CSV (burst):") if args.stage3_csv is None else args.stage3_csv

    mode = {"1": "stage1", "2": "stage12", "3": "full"}.get(choice, "full")
    return mode, s1, s2, s3


def run_pipeline(args: argparse.Namespace):
    torch.manual_seed(7)
    np.random.seed(7)

    outdir = args.output_dir or args.run_logs
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "interactive":
        mode, s1, s2, s3 = interactive_menu(args)
        if mode == "exit":
            print("[INFO] Exit requested.")
            return
        args.mode = mode
        args.stage1_csv, args.stage2_csv, args.stage3_csv = s1, s2, s3

    if args.mode == "eval":
        meta_path = outdir / "trajectory_fit_summary.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No summary metadata found: {meta_path}")
        print(meta_path.read_text(encoding="utf-8"))
        return

    cfg = BridgeConfig()  # keep linkage with project config / conventions

    train_cfg = TrainConfig(
        sequence_length=args.sequence_length,
        rollout_horizon=args.rollout_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        w_theta=args.w_theta,
        w_omega=args.w_omega,
        rollout_weight=args.rollout_weight,
    )
    pre_cfg = PreprocessConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUDynamics(
        input_size=3,
        hidden_size=train_cfg.hidden_size,
        num_layers=train_cfg.num_layers,
        dropout=train_cfg.dropout,
    ).to(device)

    stage_specs: list[StageSpec] = [
        StageSpec(1, "sin", args.stage1_csv),
        StageSpec(2, "square", args.stage2_csv),
        StageSpec(3, "burst", args.stage3_csv),
    ]

    max_stage = {"stage1": 1, "stage12": 2, "full": 3}[args.mode]
    results: list[StageResult] = []
    final_stats: FeatureStats | None = None

    for spec in stage_specs[:max_stage]:
        if spec.csv_path is None:
            raise ValueError(f"Missing CSV path for Stage {spec.stage}")
        stage_result, final_stats = train_on_stage(
            model=model,
            csv_path=spec.csv_path,
            outdir=outdir,
            stage_spec=spec,
            pre_cfg=pre_cfg,
            train_cfg=train_cfg,
            device=device,
        )
        results.append(stage_result)

    summary = {
        "timestamp": utc_now(),
        "pipeline": "stage_wise_discrete_time_trajectory_fitting",
        "model_type": "PyTorch GRU black-box dynamics learner",
        "input_features": ["theta_real", "omega_real", "hw_pwm"],
        "target": ["theta_next", "omega_next"],
        "real_data_only": True,
        "sim_data_used_for_fitting": False,
        "cmd_u_raw_used_for_fitting": False,
        "train_config": asdict(train_cfg),
        "preprocess_config": asdict(pre_cfg),
        "stages": [asdict(r) for r in results],
        "feature_stats_last_stage": asdict(final_stats) if final_stats is not None else None,
    }

    summary_path = outdir / "trajectory_fit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Chrono-compatible parameter JSON (can be passed to --parameter-json).
    # model_init/best_params are included for backward compatibility with existing loaders.
    chrono_param_json_path = outdir / "trajectory_model_params.json"
    compatible_params = {
        "timestamp": utc_now(),
        "model_type": "pytorch_gru_black_box_dynamics",
        "model_init": {
            "K_u": float(cfg.K_u_init),
            "l_com": float(cfg.l_com_init),
            "b_eq": float(cfg.b_eq_init),
            "tau_eq": float(cfg.tau_eq_init),
        },
        "best_params": {
            "K_u": float(cfg.K_u_init),
            "l_com": float(cfg.l_com_init),
            "b_eq": float(cfg.b_eq_init),
            "tau_eq": float(cfg.tau_eq_init),
        },
        "nn_dynamics": {
            "checkpoint_path": results[-1].checkpoint_path if results else None,
            "sequence_length": int(train_cfg.sequence_length),
            "input_features": ["theta_real", "omega_real", "hw_pwm"],
            "target_features": ["theta_next", "omega_next"],
            "feature_stats": asdict(final_stats) if final_stats is not None else None,
        },
    }
    with chrono_param_json_path.open("w", encoding="utf-8") as f:
        json.dump(compatible_params, f, indent=2)

    if args.save_checkpoint and results:
        final_ckpt = outdir / "final_gru_dynamics.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "feature_stats": asdict(final_stats) if final_stats is not None else None,
            "sequence_length": train_cfg.sequence_length,
            "timestamp": utc_now(),
        }, final_ckpt)
        print(f"[INFO] saved_final_checkpoint: {final_ckpt}")

    print(f"[INFO] pipeline_done: mode={args.mode}")
    print(f"[INFO] summary_json: {summary_path}")
    print(f"[INFO] chrono_parameter_json: {chrono_param_json_path}")
    for r in results:
        print(f"[INFO] stage={r.stage} checkpoint={r.checkpoint_path}")
        if not args.skip_plots:
            print(f"[INFO] stage={r.stage} fit_summary_plot={r.fit_summary_plot}")
            print(f"[INFO] stage={r.stage} overlay_plot={r.overlay_plot}")


if __name__ == "__main__":
    run_pipeline(parse_args())