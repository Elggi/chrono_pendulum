#!/usr/bin/env python3
"""Offline identification benchmark: LSTM nominal + residual SINDy-PI + PPO parameter proposal.

This script implements a three-stage pipeline for pendulum-like system identification:
1) Nominal sequence model (PyTorch LSTM)
2) Residual discovery with PySINDy SINDy-PI on model residuals only
3) PPO parameter proposal optimization over rollout-level trajectory loss

Outputs are written under reports/LSTM_SINDy_PPO/.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Optional dependencies are imported lazily so --help works even on slim environments.


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required for Stage 1. Install torch first.")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    return torch, nn, Dataset, DataLoader


def _require_matplotlib():
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is required to emit plots. Install matplotlib.")
    import matplotlib.pyplot as plt
    return plt


def _require_pysindy():
    if importlib.util.find_spec("pysindy") is None:
        raise RuntimeError("PySINDy is required for Stage 2 (SINDy-PI). Install pysindy.")
    import pysindy as ps
    return ps


def _require_sb3_and_gym():
    if importlib.util.find_spec("gymnasium") is None or importlib.util.find_spec("stable_baselines3") is None:
        raise RuntimeError("stable-baselines3 + gymnasium are required for Stage 3 PPO. Install both dependencies.")
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    return gym, spaces, PPO


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_json_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    t_last: np.ndarray
    x0_state: np.ndarray


@dataclass
class BenchmarkConfig:
    data_csv: str = "host/run_logs/chrono_run_1.finalized.csv"
    data_meta: str = "host/run_logs/chrono_run_1.meta.json"
    out_dir: str = "reports/LSTM_SINDy_PPO"
    seed: int = 42
    sequence_len: int = 32
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 80
    use_cuda: bool = False
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    w_theta: float = 1.0
    w_omega: float = 1.0
    rollout_clip: float = 25.0
    ppo_steps: int = 3000
    ppo_lr: float = 3e-4
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_lam: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_action_scale: float = 0.2
    lambda_mean: float = 0.7
    lambda_max: float = 0.3
    w_alpha: float = 0.1
    w_reg: float = 0.15
    w_bound: float = 10.0
    w_stab: float = 10.0
    dt_min_guard: float = 1e-4
    resample_points_min: int = 256
    stage_seeds: tuple[int, ...] = (7, 42, 1337)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _best_signal(rows: list[dict[str, str]], names: list[str], fallback: float = 0.0) -> np.ndarray:
    cols = rows[0].keys()
    selected = next((n for n in names if n in cols), None)
    if selected is None:
        return np.full(len(rows), fallback, dtype=float)

    out = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        try:
            out[i] = float(r[selected])
        except Exception:
            out[i] = fallback
    finite = np.isfinite(out)
    if not np.any(finite):
        return np.full(len(rows), fallback, dtype=float)
    if not np.all(finite):
        idx = np.arange(len(out), dtype=float)
        out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def unwrap(theta: np.ndarray) -> np.ndarray:
    return np.unwrap(theta)


def load_dataset(data_csv: Path, data_meta: Path) -> dict[str, np.ndarray]:
    if not data_csv.exists() and data_csv.name.endswith(".finalized.csv"):
        fallback = data_csv.with_name(data_csv.name.replace(".finalized", ""))
        if fallback.exists():
            data_csv = fallback
    rows = _load_csv_dicts(data_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {data_csv}")

    # Canonical aliases from prompt + existing repository schema.
    t = _best_signal(rows, ["wall_elapsed", "t", "time_sec", "wall_time"], fallback=0.0)
    if np.allclose(t, t[0]):
        t = t - t[0]
    if np.any(np.diff(t) <= 0.0):
        t = np.maximum.accumulate(t)
        eps = 1e-6
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + eps

    u = _best_signal(rows, ["I_filtered_mA", "cmd_u_delayed", "cmd_u_raw", "tau_cmd", "hw_pwm"], fallback=0.0)
    theta = _best_signal(
        rows,
        ["theta_imu_filtered_unwrapped", "theta_imu_filtered", "theta_real", "theta"],
        fallback=0.0,
    )
    omega = _best_signal(rows, ["omega_imu_filtered", "omega_real", "omega"], fallback=0.0)
    alpha = _best_signal(rows, ["alpha_from_linear_accel_filtered", "alpha_real", "alpha"], fallback=0.0)

    theta = unwrap(theta)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.clip(dt, 1e-6, np.inf)

    meta: dict[str, Any] = {}
    if data_meta.exists():
        meta = json.loads(data_meta.read_text())

    return {"t": t, "dt": dt, "u": u, "theta": theta, "omega": omega, "alpha": alpha, "meta": meta}


def resample_uniform(data: dict[str, np.ndarray], dt_guard: float, min_points: int) -> dict[str, np.ndarray]:
    t = data["t"]
    dt_mean = float(np.mean(np.diff(t)))
    dt_mean = max(dt_mean, dt_guard)
    t_uniform = np.arange(t[0], t[-1] + 0.5 * dt_mean, dt_mean)
    if len(t_uniform) < min_points:
        t_uniform = np.linspace(t[0], t[-1], min_points)
        dt_mean = float(np.mean(np.diff(t_uniform)))
    out = {"t": t_uniform}
    for k in ["u", "theta", "omega", "alpha"]:
        out[k] = np.interp(t_uniform, t, data[k])
    out["dt"] = np.full(len(t_uniform), dt_mean, dtype=float)
    out["dt_mean"] = np.array([dt_mean], dtype=float)
    return out


def as_irregular(data: dict[str, np.ndarray], dt_guard: float) -> dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in data.items() if k in {"t", "u", "theta", "omega", "alpha", "dt"}}
    out["dt"] = np.clip(out["dt"], dt_guard, np.inf)
    out["dt_mean"] = np.array([float(np.mean(out["dt"]))], dtype=float)
    return out


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.std(y_true) + eps)
    return rmse / denom


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def lag_estimate(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int = 200) -> int:
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    max_lag = min(max_lag, len(yt) - 2)
    if max_lag <= 1:
        return 0
    best_lag = 0
    best_corr = -1e18
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = yt[-lag:], yp[: lag + len(yp)]
        elif lag > 0:
            a, b = yt[:-lag], yp[lag:]
        else:
            a, b = yt, yp
        if len(a) < 8:
            continue
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return int(best_lag)


class SequenceDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return int(len(self.x))

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_supervised(
    data: dict[str, np.ndarray],
    seq_len: int,
    target_mode: str,
    include_dt: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = data["theta"]
    omega = data["omega"]
    u = data["u"]
    dt = data["dt"]
    n = len(theta)

    features = [theta, omega, u]
    if include_dt:
        features.append(dt)

    feat = np.stack(features, axis=1)
    states = np.stack([theta, omega], axis=1)

    xs, ys, t_last, x0_state = [], [], [], []
    for end in range(seq_len, n - 1):
        start = end - seq_len
        window = feat[start:end]
        xk = states[end - 1]
        xkp1 = states[end]
        if target_mode == "delta":
            y = xkp1 - xk
        elif target_mode == "state":
            y = xkp1
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")
        xs.append(window)
        ys.append(y)
        t_last.append(data["t"][end - 1])
        x0_state.append(xk)

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(t_last, dtype=float),
        np.asarray(x0_state, dtype=float),
    )


def time_split(x: np.ndarray, y: np.ndarray, t_last: np.ndarray, x0: np.ndarray, val_ratio: float, test_ratio: float):
    n = len(x)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 32:
        raise RuntimeError("Train split too small; reduce val/test ratios or sequence length.")

    return {
        "train": SplitData(x[:n_train], y[:n_train], t_last[:n_train], x0[:n_train]),
        "val": SplitData(x[n_train : n_train + n_val], y[n_train : n_train + n_val], t_last[n_train : n_train + n_val], x0[n_train : n_train + n_val]),
        "test": SplitData(x[n_train + n_val :], y[n_train + n_val :], t_last[n_train + n_val :], x0[n_train + n_val :]),
    }


def fit_scaler(x: np.ndarray) -> Scaler:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return Scaler(mean=mean, std=std)


def fit_scaler_3d(x: np.ndarray) -> Scaler:
    flat = x.reshape(-1, x.shape[-1])
    return fit_scaler(flat)


def normalize_dataset(splits: dict[str, SplitData]):
    x_scaler = fit_scaler_3d(splits["train"].x)
    y_scaler = fit_scaler(splits["train"].y)

    norm: dict[str, SplitData] = {}
    for name, d in splits.items():
        xn = x_scaler.transform(d.x)
        yn = y_scaler.transform(d.y)
        norm[name] = SplitData(x=xn, y=yn, t_last=d.t_last, x0_state=d.x0_state)
    return norm, x_scaler, y_scaler


class LSTMDynamics:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int):
        torch, nn, _, _ = _require_torch()

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                do = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=do,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, x):
                y, _ = self.lstm(x)
                h = y[:, -1, :]
                return self.head(h)

        self.torch = torch
        self.nn = nn
        self.model = _Net()

    def to(self, device: str):
        self.model.to(device)
        return self


class ResidualSindyModel:
    def __init__(self, feature_order: list[str], target_dim: int):
        self.feature_order = feature_order
        self.target_dim = target_dim
        self.models: list[Any] = []
        self.coefficients_: list[np.ndarray] = []
        self.equations_: list[str] = []

    @staticmethod
    def _library_matrix(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta = x[:, 0]
        omega = x[:, 1]
        terms = np.stack(
            [
                np.ones_like(theta),
                theta,
                omega,
                np.abs(omega),
                omega**2,
                omega * np.abs(omega),
                np.sin(theta),
                np.cos(theta),
                u,
                u * omega,
            ],
            axis=1,
        )
        return terms

    def fit(self, x_state: np.ndarray, u: np.ndarray, residual: np.ndarray) -> None:
        ps = _require_pysindy()
        feat_names = self.feature_order

        # Build custom feature library with mandatory terms.
        library_functions = [
            lambda z: np.ones(z.shape[0]),
            lambda z: z[:, 0],
            lambda z: z[:, 1],
            lambda z: np.abs(z[:, 1]),
            lambda z: z[:, 1] ** 2,
            lambda z: z[:, 1] * np.abs(z[:, 1]),
            lambda z: np.sin(z[:, 0]),
            lambda z: np.cos(z[:, 0]),
            lambda z: z[:, 2],
            lambda z: z[:, 2] * z[:, 1],
        ]
        function_names = [
            lambda _: "1",
            lambda _: "theta",
            lambda _: "omega",
            lambda _: "abs(omega)",
            lambda _: "omega^2",
            lambda _: "omega*abs(omega)",
            lambda _: "sin(theta)",
            lambda _: "cos(theta)",
            lambda _: "u",
            lambda _: "u*omega",
        ]

        z = np.column_stack([x_state, u])
        self.models.clear()
        self.coefficients_.clear()
        self.equations_.clear()
        for d in range(self.target_dim):
            model = ps.SINDy(
                feature_library=ps.CustomLibrary(library_functions=library_functions, function_names=function_names),
                optimizer=ps.SINDyPI(threshold=0.02, tol=1e-5, thresholder="l1"),
                feature_names=feat_names,
            )
            # residual is treated as discrete-time map output.
            model.fit(z, x_dot=residual[:, d], t=1.0, quiet=True)
            self.models.append(model)
            self.coefficients_.append(model.coefficients().copy())
            self.equations_.append(model.equations()[0])

    def predict(self, x_state: np.ndarray, u: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros((len(x_state), self.target_dim), dtype=float)
        z = np.column_stack([x_state, u])
        out = []
        for model in self.models:
            y = model.predict(z)
            y = np.asarray(y).reshape(-1)
            out.append(y)
        return np.stack(out, axis=1)


class ParameterProposalEnvBase:
    def __init__(
        self,
        rollout_inputs: dict[str, np.ndarray],
        lstm_runner,
        sindy_model: ResidualSindyModel,
        base_params: np.ndarray,
        cfg: BenchmarkConfig,
    ):
        self.rollout_inputs = rollout_inputs
        self.lstm_runner = lstm_runner
        self.sindy_model = sindy_model
        self.base_params = np.asarray(base_params, dtype=float)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n_res = len(self.base_params) - 2

    def run_rollout_cost(self, params: np.ndarray) -> tuple[float, dict[str, float]]:
        scales = params[:2]
        coeff_adjust = params[2:]

        pred = self.lstm_runner(scales=scales, residual_adjust=coeff_adjust)
        theta_p, omega_p, alpha_p = pred["theta"], pred["omega"], pred["alpha"]
        theta_m = self.rollout_inputs["theta"]
        omega_m = self.rollout_inputs["omega"]
        alpha_m = self.rollout_inputs["alpha"]

        j_run = (
            self.cfg.w_theta * nrmse(theta_m, theta_p)
            + self.cfg.w_omega * nrmse(omega_m, omega_p)
            + self.cfg.w_alpha * nrmse(alpha_m, alpha_p)
        )

        param_dev = params - self.base_params
        reg = self.cfg.w_reg * float(np.dot(param_dev, param_dev))
        out_of_bounds = np.maximum(np.abs(params) - 3.0, 0.0)
        bound_pen = self.cfg.w_bound * float(np.sum(out_of_bounds**2))

        unstable = 1.0 if (np.any(~np.isfinite(theta_p)) or np.any(np.abs(theta_p) > self.cfg.rollout_clip * math.pi)) else 0.0
        stab_pen = self.cfg.w_stab * unstable

        total = float(j_run + reg + bound_pen + stab_pen)
        return total, {
            "j_run": float(j_run),
            "reg": float(reg),
            "bound": float(bound_pen),
            "stab": float(stab_pen),
        }



def train_stage1(
    cfg: BenchmarkConfig,
    splits: dict[str, SplitData],
    x_scaler: Scaler,
    y_scaler: Scaler,
    target_mode: str,
    feature_names: list[str],
    out_dir: Path,
    tag: str,
):
    torch, _, Dataset, DataLoader = _require_torch()
    device = "cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu"
    net = LSTMDynamics(
        input_size=len(feature_names),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        output_size=2,
    ).to(device)
    model = net.model

    train_ds = Dataset.__new__(Dataset)  # type: ignore[misc]
    train_ds = SequenceDataset(splits["train"].x, splits["train"].y)
    val_ds = SequenceDataset(splits["val"].x, splits["val"].y)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_loss": []}

    def weighted_nrmse_loss(y_true, y_pred):
        eps = 1e-6
        err = y_pred - y_true
        rmse_v = torch.sqrt(torch.mean(err**2, dim=0) + eps)
        std_v = torch.std(y_true, dim=0) + eps
        nrmse_v = rmse_v / std_v
        return cfg.w_theta * nrmse_v[0] + cfg.w_omega * nrmse_v[1]

    best_state = None
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        tl = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            yp = model(xb)
            loss = weighted_nrmse_loss(yb, yp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl.append(float(loss.detach().cpu().item()))

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yp = model(xb)
                loss = weighted_nrmse_loss(yb, yp)
                vl.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(tl)) if tl else float("nan")
        val_loss = float(np.mean(vl)) if vl else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "input_size": len(feature_names),
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "sequence_len": cfg.sequence_len,
                "target_mode": target_mode,
                "feature_names": feature_names,
            },
            "x_scaler": x_scaler.to_json_dict(),
            "y_scaler": y_scaler.to_json_dict(),
            "history": history,
        },
        out_dir / f"lstm_{tag}.pt",
    )

    return model, history, device


def evaluate_one_step(
    model,
    split: SplitData,
    y_scaler: Scaler,
    target_mode: str,
    x0_state: np.ndarray,
    device: str,
):
    torch, _, _, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(split.x, dtype=torch.float32, device=device)
        yp_n = model(xb).detach().cpu().numpy()

    yp = y_scaler.inverse(yp_n)
    y_true = y_scaler.inverse(split.y)

    if target_mode == "delta":
        pred_state = x0_state + yp
        true_state = x0_state + y_true
    else:
        pred_state = yp
        true_state = y_true

    metrics = {
        "rmse_theta": rmse(true_state[:, 0], pred_state[:, 0]),
        "rmse_omega": rmse(true_state[:, 1], pred_state[:, 1]),
        "nrmse_theta": nrmse(true_state[:, 0], pred_state[:, 0]),
        "nrmse_omega": nrmse(true_state[:, 1], pred_state[:, 1]),
    }
    return true_state, pred_state, metrics


def autoregressive_rollout(
    model,
    data: dict[str, np.ndarray],
    seq_len: int,
    include_dt: bool,
    x_scaler: Scaler,
    y_scaler: Scaler,
    target_mode: str,
    device: str,
    residual_model: ResidualSindyModel | None = None,
    residual_adjust: np.ndarray | None = None,
    lstm_scales: np.ndarray | None = None,
):
    torch, _, _, _ = _require_torch()
    theta = data["theta"].copy()
    omega = data["omega"].copy()
    u = data["u"]
    dt = data["dt"]
    n = len(theta)

    pred_theta = np.zeros(n, dtype=float)
    pred_omega = np.zeros(n, dtype=float)
    pred_theta[:seq_len] = theta[:seq_len]
    pred_omega[:seq_len] = omega[:seq_len]

    scales = np.array([1.0, 1.0]) if lstm_scales is None else np.asarray(lstm_scales, dtype=float)

    for k in range(seq_len, n - 1):
        feat_cols = [pred_theta[k - seq_len : k], pred_omega[k - seq_len : k], u[k - seq_len : k]]
        if include_dt:
            feat_cols.append(dt[k - seq_len : k])
        xw = np.stack(feat_cols, axis=1)
        xw_n = x_scaler.transform(xw)

        with torch.no_grad():
            xb = torch.tensor(xw_n[None, :, :], dtype=torch.float32, device=device)
            ypn = model(xb).detach().cpu().numpy()[0]
        yp = y_scaler.inverse(ypn)

        xk = np.array([pred_theta[k - 1], pred_omega[k - 1]], dtype=float)
        if target_mode == "delta":
            xkp1 = xk + yp
        else:
            xkp1 = yp
        xkp1 = xk + scales * (xkp1 - xk)

        if residual_model is not None:
            r = residual_model.predict(xk[None, :], np.array([u[k - 1]], dtype=float))[0]
            if residual_adjust is not None and len(residual_adjust) > 0:
                # residual_adjust are simple gain controls on each state residual channel.
                gain = np.clip(1.0 + residual_adjust[:2], 0.0, 2.5)
                r = gain * r
            xkp1 = xkp1 + r

        pred_theta[k] = xkp1[0]
        pred_omega[k] = xkp1[1]

    pred_theta[-1] = pred_theta[-2]
    pred_omega[-1] = pred_omega[-2]
    pred_alpha = np.gradient(pred_omega, data["t"], edge_order=1)

    return {"theta": pred_theta, "omega": pred_omega, "alpha": pred_alpha}


def rollout_metrics(meas: dict[str, np.ndarray], pred: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for horizon in [10, 50, len(meas["theta"]) - 1]:
        h = min(horizon, len(meas["theta"]) - 1)
        out[f"k{h}_nrmse_theta"] = nrmse(meas["theta"][:h], pred["theta"][:h])
        out[f"k{h}_nrmse_omega"] = nrmse(meas["omega"][:h], pred["omega"][:h])
    out["full_nrmse_theta"] = nrmse(meas["theta"], pred["theta"])
    out["full_nrmse_omega"] = nrmse(meas["omega"], pred["omega"])
    out["drift_theta_end"] = float(pred["theta"][-1] - meas["theta"][-1])
    out["drift_omega_end"] = float(pred["omega"][-1] - meas["omega"][-1])
    out["phase_lag_samples_theta"] = lag_estimate(meas["theta"], pred["theta"])
    out["phase_lag_samples_omega"] = lag_estimate(meas["omega"], pred["omega"])
    out["bias_theta"] = float(np.mean(pred["theta"] - meas["theta"]))
    out["bias_omega"] = float(np.mean(pred["omega"] - meas["omega"]))
    out["unstable"] = bool(np.any(~np.isfinite(pred["theta"])) or np.max(np.abs(pred["theta"])) > 1e3)
    return out


def plot_stage_outputs(
    out_dir: Path,
    tag: str,
    t: np.ndarray,
    meas: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
):
    plt = _require_matplotlib()

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, meas["theta"], label="theta_meas", linewidth=1.2)
    ax[0].plot(t, pred["theta"], label="theta_pred", linewidth=1.1)
    ax[0].set_ylabel("theta [rad]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(t, meas["omega"], label="omega_meas", linewidth=1.2)
    ax[1].plot(t, pred["omega"], label="omega_pred", linewidth=1.1)
    ax[1].set_ylabel("omega [rad/s]")
    ax[1].set_xlabel("t [s]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_measured_vs_predicted.png", dpi=180)
    plt.close(fig)

    e_theta = np.abs(pred["theta"] - meas["theta"])
    e_omega = np.abs(pred["omega"] - meas["omega"])
    horizons = np.arange(1, len(t) + 1)
    err_growth_theta = np.array([np.sqrt(np.mean((pred["theta"][:h] - meas["theta"][:h]) ** 2)) for h in horizons])
    err_growth_omega = np.array([np.sqrt(np.mean((pred["omega"][:h] - meas["omega"][:h]) ** 2)) for h in horizons])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(horizons, err_growth_theta, label="RMSE theta")
    ax[0].plot(horizons, err_growth_omega, label="RMSE omega")
    ax[0].set_title("Error growth vs horizon")
    ax[0].set_xlabel("horizon")
    ax[0].set_ylabel("RMSE")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(t, e_theta, label="|e_theta|")
    ax[1].plot(t, e_omega, label="|e_omega|")
    ax[1].set_title("Rollout divergence case")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("absolute error")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_rollout_error_growth.png", dpi=180)
    plt.close(fig)


def write_report(
    out_dir: Path,
    cfg: BenchmarkConfig,
    summary: dict[str, Any],
):
    lines = []
    lines.append("# LSTM + SINDy-PI + PPO Offline Identification Benchmark")
    lines.append("")
    lines.append("## Configuration")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Results Summary")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Required Comparisons")
    lines.append("- Uniform resampling vs Δt-aware: included under `sampling_comparison`.")
    lines.append("- Delta prediction vs state prediction: included under `target_comparison`.")
    lines.append("- Ablation: LSTM only vs LSTM+SINDy vs LSTM+SINDy+PPO.")
    (out_dir / "benchmark_report.md").write_text("\n".join(lines))


def run(cfg: BenchmarkConfig) -> None:
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(Path(cfg.data_csv), Path(cfg.data_meta))
    uniform = resample_uniform(raw, cfg.dt_min_guard, cfg.resample_points_min)
    irregular = as_irregular(raw, cfg.dt_min_guard)

    data_variants = {
        "uniform": uniform,
        "irregular_dtaware": irregular,
    }

    summary: dict[str, Any] = {
        "data": {
            "raw_samples": int(len(raw["t"])),
            "uniform_samples": int(len(uniform["t"])),
            "raw_dt_mean": float(np.mean(raw["dt"])),
            "uniform_dt": float(uniform["dt_mean"][0]),
        },
        "stage1": {},
        "stage2": {},
        "stage3": {},
        "sampling_comparison": {},
        "target_comparison": {},
        "ablation": {},
    }

    stage1_cache: dict[tuple[str, str], Any] = {}

    # Stage 1 comparisons (uniform vs dt-aware) x (state vs delta target)
    for variant_name, data in data_variants.items():
        for target_mode in ["state", "delta"]:
            include_dt = variant_name == "irregular_dtaware"
            x, y, t_last, x0 = build_supervised(
                data=data,
                seq_len=cfg.sequence_len,
                target_mode=target_mode,
                include_dt=include_dt,
            )
            splits = time_split(x, y, t_last, x0, cfg.val_ratio, cfg.test_ratio)
            splits_n, x_scaler, y_scaler = normalize_dataset(splits)
            feat_names = ["theta", "omega", "u"] + (["dt"] if include_dt else [])
            tag = f"{variant_name}_{target_mode}"

            model, history, device = train_stage1(
                cfg=cfg,
                splits=splits_n,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                target_mode=target_mode,
                feature_names=feat_names,
                out_dir=out_dir,
                tag=tag,
            )

            y_true, y_pred, one_step = evaluate_one_step(
                model=model,
                split=splits_n["test"],
                y_scaler=y_scaler,
                target_mode=target_mode,
                x0_state=splits["test"].x0_state,
                device=device,
            )

            rollout_pred = autoregressive_rollout(
                model=model,
                data=data,
                seq_len=cfg.sequence_len,
                include_dt=include_dt,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                target_mode=target_mode,
                device=device,
            )
            rollout_meas = {k: data[k] for k in ["theta", "omega", "alpha"]}
            roll_m = rollout_metrics(rollout_meas, rollout_pred)

            plot_stage_outputs(out_dir, f"stage1_{tag}", data["t"], rollout_meas, rollout_pred)

            key = f"{variant_name}:{target_mode}"
            summary["stage1"][key] = {
                "one_step": one_step,
                "rollout": roll_m,
                "history": history,
            }

            stage1_cache[(variant_name, target_mode)] = {
                "model": model,
                "device": device,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "data": data,
                "include_dt": include_dt,
                "target_mode": target_mode,
                "rollout": rollout_pred,
                "splits_raw": splits,
                "splits_norm": splits_n,
            }

    # Required comparisons for Stage 1
    u_delta = summary["stage1"]["uniform:delta"]["rollout"]["full_nrmse_theta"] + summary["stage1"]["uniform:delta"]["rollout"]["full_nrmse_omega"]
    dta_delta = summary["stage1"]["irregular_dtaware:delta"]["rollout"]["full_nrmse_theta"] + summary["stage1"]["irregular_dtaware:delta"]["rollout"]["full_nrmse_omega"]
    summary["sampling_comparison"] = {
        "uniform_delta_full_nrmse_sum": float(u_delta),
        "dtaware_delta_full_nrmse_sum": float(dta_delta),
        "prediction_accuracy_difference_dtaware_minus_uniform": float(dta_delta - u_delta),
        "rollout_stability_uniform": bool(summary["stage1"]["uniform:delta"]["rollout"]["unstable"]),
        "rollout_stability_dtaware": bool(summary["stage1"]["irregular_dtaware:delta"]["rollout"]["unstable"]),
        "long_horizon_drift_uniform": {
            "theta": float(summary["stage1"]["uniform:delta"]["rollout"]["drift_theta_end"]),
            "omega": float(summary["stage1"]["uniform:delta"]["rollout"]["drift_omega_end"]),
        },
        "long_horizon_drift_dtaware": {
            "theta": float(summary["stage1"]["irregular_dtaware:delta"]["rollout"]["drift_theta_end"]),
            "omega": float(summary["stage1"]["irregular_dtaware:delta"]["rollout"]["drift_omega_end"]),
        },
    }

    state_uniform = summary["stage1"]["uniform:state"]["rollout"]["full_nrmse_theta"] + summary["stage1"]["uniform:state"]["rollout"]["full_nrmse_omega"]
    delta_uniform = summary["stage1"]["uniform:delta"]["rollout"]["full_nrmse_theta"] + summary["stage1"]["uniform:delta"]["rollout"]["full_nrmse_omega"]
    summary["target_comparison"] = {
        "uniform_state_full_nrmse_sum": float(state_uniform),
        "uniform_delta_full_nrmse_sum": float(delta_uniform),
        "delta_minus_state": float(delta_uniform - state_uniform),
        "preferred_target": "delta" if delta_uniform <= state_uniform else "state",
    }

    # Select best stage-1 nominal model as base for Stage 2+3.
    best_key = min(summary["stage1"].keys(), key=lambda k: summary["stage1"][k]["rollout"]["full_nrmse_theta"] + summary["stage1"][k]["rollout"]["full_nrmse_omega"])
    variant, target_mode = best_key.split(":")
    base = stage1_cache[(variant, target_mode)]
    base_model = base["model"]
    base_data = base["data"]

    # Stage 2: residual modeling only (measured x_{k+1} - LSTM x_{k+1}).
    n = len(base_data["theta"])
    xk = np.stack([base_data["theta"][:-1], base_data["omega"][:-1]], axis=1)
    xkp1_meas = np.stack([base_data["theta"][1:], base_data["omega"][1:]], axis=1)

    base_roll = autoregressive_rollout(
        model=base_model,
        data=base_data,
        seq_len=cfg.sequence_len,
        include_dt=base["include_dt"],
        x_scaler=base["x_scaler"],
        y_scaler=base["y_scaler"],
        target_mode=base["target_mode"],
        device=base["device"],
    )
    xkp1_lstm = np.stack([base_roll["theta"][1:], base_roll["omega"][1:]], axis=1)
    residual = xkp1_meas - xkp1_lstm

    rs = ResidualSindyModel(feature_order=["theta", "omega", "u"], target_dim=2)
    rs.fit(x_state=xk, u=base_data["u"][:-1], residual=residual)

    (out_dir / "sindy_equations.json").write_text(json.dumps({"equations": rs.equations_}, indent=2))

    roll_stage2 = autoregressive_rollout(
        model=base_model,
        data=base_data,
        seq_len=cfg.sequence_len,
        include_dt=base["include_dt"],
        x_scaler=base["x_scaler"],
        y_scaler=base["y_scaler"],
        target_mode=base["target_mode"],
        device=base["device"],
        residual_model=rs,
    )
    meas = {k: base_data[k] for k in ["theta", "omega", "alpha"]}
    stage1_roll_m = rollout_metrics(meas, base_roll)
    stage2_roll_m = rollout_metrics(meas, roll_stage2)
    plot_stage_outputs(out_dir, "stage2_composite", base_data["t"], meas, roll_stage2)

    summary["stage2"] = {
        "base_stage1_key": best_key,
        "sindy_equations": rs.equations_,
        "rollout": stage2_roll_m,
        "improvement_delta": {
            "full_nrmse_theta": float(stage2_roll_m["full_nrmse_theta"] - stage1_roll_m["full_nrmse_theta"]),
            "full_nrmse_omega": float(stage2_roll_m["full_nrmse_omega"] - stage1_roll_m["full_nrmse_omega"]),
        },
        "stability_check": {
            "stage1_unstable": bool(stage1_roll_m["unstable"]),
            "stage2_unstable": bool(stage2_roll_m["unstable"]),
        },
    }

    # Stage 2 robustness across seeds: retrain stage1 with different seeds and re-fit residual.
    seed_results = {}
    for s in cfg.stage_seeds:
        set_seed(s)
        # Reuse selected variant+target preprocessing.
        x, y, t_last, x0 = build_supervised(
            data=base_data,
            seq_len=cfg.sequence_len,
            target_mode=base["target_mode"],
            include_dt=base["include_dt"],
        )
        splits = time_split(x, y, t_last, x0, cfg.val_ratio, cfg.test_ratio)
        splits_n, x_sc, y_sc = normalize_dataset(splits)
        model_s, _, device_s = train_stage1(
            cfg=cfg,
            splits=splits_n,
            x_scaler=x_sc,
            y_scaler=y_sc,
            target_mode=base["target_mode"],
            feature_names=["theta", "omega", "u"] + (["dt"] if base["include_dt"] else []),
            out_dir=out_dir,
            tag=f"seed{s}_{variant}_{target_mode}",
        )
        roll_s = autoregressive_rollout(
            model=model_s,
            data=base_data,
            seq_len=cfg.sequence_len,
            include_dt=base["include_dt"],
            x_scaler=x_sc,
            y_scaler=y_sc,
            target_mode=base["target_mode"],
            device=device_s,
        )
        xkp1_l = np.stack([roll_s["theta"][1:], roll_s["omega"][1:]], axis=1)
        res_s = xkp1_meas - xkp1_l
        rs_s = ResidualSindyModel(feature_order=["theta", "omega", "u"], target_dim=2)
        rs_s.fit(x_state=xk, u=base_data["u"][:-1], residual=res_s)
        comp_s = autoregressive_rollout(
            model=model_s,
            data=base_data,
            seq_len=cfg.sequence_len,
            include_dt=base["include_dt"],
            x_scaler=x_sc,
            y_scaler=y_sc,
            target_mode=base["target_mode"],
            device=device_s,
            residual_model=rs_s,
        )
        seed_results[str(s)] = rollout_metrics(meas, comp_s)
    summary["stage2"]["robustness_across_seeds"] = seed_results

    # Stage 3 PPO: parameter proposal only.
    gym, spaces, PPO = _require_sb3_and_gym()

    # Build a rollout function closure to evaluate candidate params.
    def _run_composite(scales: np.ndarray, residual_adjust: np.ndarray):
        return autoregressive_rollout(
            model=base_model,
            data=base_data,
            seq_len=cfg.sequence_len,
            include_dt=base["include_dt"],
            x_scaler=base["x_scaler"],
            y_scaler=base["y_scaler"],
            target_mode=base["target_mode"],
            device=base["device"],
            residual_model=rs,
            residual_adjust=residual_adjust,
            lstm_scales=scales,
        )

    residual_params_dim = 2
    base_params = np.zeros(2 + residual_params_dim, dtype=float)

    class ParameterProposalEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(base_params),), dtype=np.float32)
            self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)
            self.current_obs = np.zeros(6, dtype=np.float32)
            self.last_info: dict[str, Any] = {}

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            self.current_obs = np.array(
                [
                    float(nrmse(meas["theta"], base_roll["theta"])),
                    float(nrmse(meas["omega"], base_roll["omega"])),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            self.last_info = {}
            return self.current_obs.copy(), {}

        def step(self, action):
            action = np.asarray(action, dtype=float)
            params = base_params + cfg.ppo_action_scale * action
            scales = params[:2]
            residual_adjust = params[2:]

            rollout = _run_composite(scales=scales, residual_adjust=residual_adjust)
            theta_cost = nrmse(meas["theta"], rollout["theta"])
            omega_cost = nrmse(meas["omega"], rollout["omega"])
            alpha_cost = nrmse(meas["alpha"], rollout["alpha"])

            j_run = cfg.w_theta * theta_cost + cfg.w_omega * omega_cost + cfg.w_alpha * alpha_cost
            reg = cfg.w_reg * float(np.dot(params - base_params, params - base_params))
            bound = cfg.w_bound * float(np.sum(np.maximum(np.abs(params) - 3.0, 0.0) ** 2))
            unstable = 1.0 if (np.any(~np.isfinite(rollout["theta"])) or np.max(np.abs(rollout["theta"])) > cfg.rollout_clip * math.pi) else 0.0
            stab = cfg.w_stab * unstable
            j_total = cfg.lambda_mean * j_run + cfg.lambda_max * j_run + reg + bound + stab
            reward = -float(j_total)

            self.current_obs = np.array([theta_cost, omega_cost, alpha_cost, reg, bound, stab], dtype=np.float32)
            terminated = True  # one proposal per episode
            truncated = False
            self.last_info = {
                "params": params.tolist(),
                "j_total": float(j_total),
                "j_run": float(j_run),
                "components": {
                    "theta": float(theta_cost),
                    "omega": float(omega_cost),
                    "alpha": float(alpha_cost),
                    "reg": float(reg),
                    "bound": float(bound),
                    "stab": float(stab),
                },
            }
            return self.current_obs.copy(), reward, terminated, truncated, self.last_info

    env = ParameterProposalEnv()
    ppo = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=cfg.ppo_lr,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lam,
        clip_range=cfg.ppo_clip_range,
        seed=cfg.seed,
    )
    ppo.learn(total_timesteps=cfg.ppo_steps, progress_bar=False)
    ppo.save(str(out_dir / "ppo_parameter_proposal.zip"))

    obs, _ = env.reset()
    action, _ = ppo.predict(obs, deterministic=True)
    _, reward, _, _, info = env.step(action)

    params_star = np.asarray(info["params"], dtype=float)
    roll_stage3 = _run_composite(scales=params_star[:2], residual_adjust=params_star[2:])
    stage3_roll_m = rollout_metrics(meas, roll_stage3)
    plot_stage_outputs(out_dir, "stage3_ppo_refined", base_data["t"], meas, roll_stage3)

    summary["stage3"] = {
        "reward": float(reward),
        "best_params": info["params"],
        "cost_breakdown": info["components"],
        "rollout": stage3_roll_m,
    }

    summary["ablation"] = {
        "lstm_only": stage1_roll_m,
        "lstm_plus_sindy": stage2_roll_m,
        "lstm_plus_sindy_plus_ppo": stage3_roll_m,
    }

    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "benchmark_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    write_report(out_dir, cfg, summary)


def parse_args() -> BenchmarkConfig:
    p = argparse.ArgumentParser(description="Offline identification benchmark: LSTM + SINDy-PI + PPO parameter proposal")
    p.add_argument("--data-csv", default="host/run_logs/chrono_run_1.finalized.csv")
    p.add_argument("--data-meta", default="host/run_logs/chrono_run_1.meta.json")
    p.add_argument("--out-dir", default="reports/LSTM_SINDy_PPO")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sequence-len", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=96)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--ppo-steps", type=int, default=3000)
    args = p.parse_args()
    cfg = BenchmarkConfig(
        data_csv=args.data_csv,
        data_meta=args.data_meta,
        out_dir=args.out_dir,
        seed=args.seed,
        sequence_len=args.sequence_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        use_cuda=args.use_cuda,
        ppo_steps=args.ppo_steps,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
