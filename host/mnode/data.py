from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fill(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float).copy()
    if len(x) == 0:
        return x
    m = np.isfinite(x)
    if not m.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[~m] = np.interp(idx[~m], idx[m], x[m])
    return x


def _load_one_csv(path: str) -> dict:
    df = pd.read_csv(path)
    t = _fill(pd.to_numeric(df["time"] if "time" in df.columns else df["wall_elapsed"], errors="coerce").to_numpy(dtype=float))
    theta = _fill(pd.to_numeric(df.get("theta", 0.0), errors="coerce").to_numpy(dtype=float))
    omega = _fill(pd.to_numeric(df.get("omega", 0.0), errors="coerce").to_numpy(dtype=float))
    if "alpha" in df.columns:
        alpha = _fill(pd.to_numeric(df["alpha"], errors="coerce").to_numpy(dtype=float))
    else:
        alpha = np.gradient(omega, t, edge_order=1) if len(t) > 1 else np.zeros_like(omega)
    if "input_current" in df.columns:
        current = _fill(pd.to_numeric(df["input_current"], errors="coerce").to_numpy(dtype=float))
    elif "ina_current_signed_mA" in df.columns:
        current = _fill(pd.to_numeric(df["ina_current_signed_mA"], errors="coerce").to_numpy(dtype=float)) * 0.001
    else:
        current = np.zeros(len(df), dtype=float)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    return {"t": t, "dt": _fill(dt), "theta": theta, "omega": omega, "alpha": alpha, "current": current, "source": str(Path(path).resolve())}


def load_dataset(csvs: list[str], npz: str | None = None) -> list[dict]:
    if npz:
        n = np.load(npz, allow_pickle=True)
        t_list = list(n["t"])
        X = list(n["X"])
        U = list(n["U"])
        A = list(n["full_alpha"])
        out = []
        for i in range(len(t_list)):
            x = np.asarray(X[i], dtype=float)
            t = np.asarray(t_list[i], dtype=float)
            dt = np.diff(t, prepend=t[0]);
            if len(dt) > 1: dt[0] = dt[1]
            out.append({"t": t, "dt": dt, "theta": x[:, 0], "omega": x[:, 1], "alpha": np.asarray(A[i], dtype=float), "current": np.asarray(U[i], dtype=float).reshape(-1), "source": str(i)})
        return out
    return [_load_one_csv(c) for c in csvs]


def build_training_tensors(dataset: list[dict], motor_json: str) -> tuple[np.ndarray, np.ndarray, dict]:
    motor = json.loads(Path(motor_json).read_text(encoding="utf-8"))
    dyn = motor.get("dynamic_parameters", {})
    J = float(dyn.get("J", 0.02)); b = float(dyn.get("b_eq", 0.01)); tau = float(dyn.get("tau_eq", 0.0)); KI = float(dyn.get("K_I", 0.06)); l = float(dyn.get("l_com", 0.14))
    m = float(motor["rod"]["mass"] + motor["imu"]["mass"] + motor.get("connector_cyl", {}).get("mass", 0.0)); g = float(motor.get("gravity", 9.81))
    eps = float(dyn.get("tanh_eps", 0.05))

    X_parts, y_parts = [], []
    for tr in dataset:
        theta, omega, alpha, current = tr["theta"], tr["omega"], tr["alpha"], tr["current"]
        tau_nom = KI * current - b * omega - tau * np.tanh(omega / max(eps, 1e-6)) - m * g * l * np.sin(theta)
        tau_target = J * alpha - tau_nom  # learned residual torque target
        X_parts.append(np.column_stack([theta, omega, current]))
        y_parts.append(tau_target[:, None])

    X = np.vstack(X_parts)
    y = np.vstack(y_parts)
    mu = X.mean(axis=0); sigma = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma
    scaler = {"mean": mu.tolist(), "std": sigma.tolist(), "features": ["theta", "omega", "input_current"], "target": "tau_learned"}
    return Xn.astype(np.float32), y.astype(np.float32), scaler
