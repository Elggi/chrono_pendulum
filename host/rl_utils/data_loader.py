from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RolloutData:
    trajectory_id: str
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    current: np.ndarray
    source: str


def _to_num(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


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


def load_from_csvs(csv_paths: list[str]) -> list[RolloutData]:
    out: list[RolloutData] = []
    for i, p in enumerate(csv_paths):
        df = pd.read_csv(p)
        t = _fill(_to_num(df["time"] if "time" in df.columns else df["wall_elapsed"]))
        theta = _fill(_to_num(df.get("theta", 0.0)))
        omega = _fill(_to_num(df.get("omega", 0.0)))
        if "alpha" in df.columns:
            alpha = _fill(_to_num(df["alpha"]))
        else:
            alpha = np.gradient(omega, t, edge_order=1) if len(t) > 1 else np.zeros_like(omega)
        if "input_current" in df.columns:
            current = _fill(_to_num(df["input_current"]))
        elif "ina_current_signed_mA" in df.columns:
            current = _fill(_to_num(df["ina_current_signed_mA"])) * 0.001
        else:
            current = np.zeros(len(df), dtype=float)
        out.append(RolloutData(f"traj_{i}", t, theta, omega, alpha, current, str(Path(p).resolve())))
    return out


def load_from_npz(npz_path: str) -> list[RolloutData]:
    npz = np.load(npz_path, allow_pickle=True)
    t_list = list(npz["t"])
    x_list = list(npz["X"])
    u_list = list(npz["U"])
    alpha_list = list(npz["full_alpha"])
    ids = list(npz["trajectory_id"]) if "trajectory_id" in npz else [f"traj_{i}" for i in range(len(t_list))]
    srcs = list(npz["source_files"]) if "source_files" in npz else [""] * len(t_list)

    out: list[RolloutData] = []
    for i in range(len(t_list)):
        x = np.asarray(x_list[i], dtype=float)
        out.append(
            RolloutData(
                trajectory_id=str(ids[i]),
                t=np.asarray(t_list[i], dtype=float),
                theta=x[:, 0],
                omega=x[:, 1],
                alpha=np.asarray(alpha_list[i], dtype=float),
                current=np.asarray(u_list[i], dtype=float).reshape(-1),
                source=str(srcs[i]),
            )
        )
    return out
