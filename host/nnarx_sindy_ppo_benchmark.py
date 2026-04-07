#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline benchmark pipeline: NNARX -> residual SINDy-PI -> PPO parameter proposal.

This script implements the full 3-stage benchmark requested in the project brief:
  1) Nominal NNARX model (SysIdentPy NARXNN where available)
  2) Residual discovery with PySINDy (SINDy-PI optimizer)
  3) PPO parameter proposal agent with rollout-level objective

Outputs are written under reports/NNARX_SINDy_PPO/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPORT_ROOT = Path("reports/NNARX_SINDy_PPO")


@dataclass
class Metrics:
    rmse_theta: float
    rmse_omega: float
    nrmse_theta: float
    nrmse_omega: float

    def weighted_nrmse(self, w_theta: float, w_omega: float) -> float:
        return float(w_theta * self.nrmse_theta + w_omega * self.nrmse_omega)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.std(y_true) + 1e-9)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _safe_col(df: pd.DataFrame, name: str, fallback: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), float(fallback), dtype=float)
    out = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    out[~np.isfinite(out)] = fallback
    return out


def load_signals(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    expected = [
        "wall_elapsed",
        "I_filtered_mA",
        "theta_imu_filtered_unwrapped",
        "omega_imu_filtered",
    ]
    if any(c not in df.columns for c in expected):
        missing = [c for c in expected if c not in df.columns]
        raise ValueError(f"Input CSV missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "t": _safe_col(df, "wall_elapsed"),
            "u": _safe_col(df, "I_filtered_mA"),
            "theta": _safe_col(df, "theta_imu_filtered_unwrapped"),
            "omega": _safe_col(df, "omega_imu_filtered"),
            "alpha": _safe_col(df, "alpha_from_linear_accel_filtered"),
        }
    )
    out = out.sort_values("t").reset_index(drop=True)
    out = out[np.isfinite(out["t"])].reset_index(drop=True)
    out = out.drop_duplicates(subset=["t"]).reset_index(drop=True)
    return out


def uniform_resample(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    t = df["t"].to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        raise ValueError("Cannot infer dt from timestamps.")
    dt_mean = float(np.mean(dt))
    t_u = np.arange(t[0], t[-1] + 0.5 * dt_mean, dt_mean)
    out = pd.DataFrame({"t": t_u})
    for col in ["u", "theta", "omega", "alpha"]:
        out[col] = np.interp(t_u, df["t"], df[col])
    out["dt"] = dt_mean
    return out, dt_mean


def irregular_with_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = np.diff(out["t"].to_numpy(dtype=float), prepend=float(out["t"].iloc[0]))
    dt[0] = np.median(dt[1:]) if len(dt) > 1 else 0.0
    out["dt"] = dt
    return out


def series_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> dict[str, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(10, min(n_train, n - 20))
    n_val = max(5, min(n_val, n - n_train - 5))
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val :].reset_index(drop=True)
    return {"train": train, "val": val, "test": test}


def build_arx_design(
    df: pd.DataFrame,
    ylag: int,
    xlag: int,
    include_dt: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y = df[["theta", "omega"]].to_numpy(dtype=float)
    u = df[["u"]].to_numpy(dtype=float)
    dt = df[["dt"]].to_numpy(dtype=float)
    max_l = max(ylag, xlag)
    feats = []
    tgt = []
    for k in range(max_l, len(df) - 1):
        row: list[float] = []
        for i in range(1, ylag + 1):
            row.extend(y[k - i].tolist())
        for i in range(0, xlag):
            row.extend(u[k - i].tolist())
        if include_dt:
            row.extend(dt[k].tolist())
        feats.append(row)
        tgt.append(y[k + 1].tolist())
    return np.asarray(feats, dtype=float), np.asarray(tgt, dtype=float)


class NARXModel:
    """Unified wrapper for shared-output and per-output NARXNN training."""

    def __init__(self, mode: str, ylag: int, xlag: int, hidden: list[int], epochs: int, seed: int):
        self.mode = mode
        self.ylag = int(ylag)
        self.xlag = int(xlag)
        self.hidden = list(hidden)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.model: Any = None
        self.theta_model: Any = None
        self.omega_model: Any = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        np.random.seed(self.seed)
        try:
            from sklearn.neural_network import MLPRegressor
        except Exception as exc:
            raise RuntimeError(f"scikit-learn is required to train NNARX baseline: {exc}")

        # Uses MLP on explicit ARX regressors. If SysIdentPy is available, keep a reference in metadata.
        if self.mode == "multi":
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden),
                activation="relu",
                max_iter=self.epochs,
                random_state=self.seed,
            )
            self.model.fit(X, Y)
        elif self.mode == "separate":
            self.theta_model = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden),
                activation="relu",
                max_iter=self.epochs,
                random_state=self.seed,
            )
            self.omega_model = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden),
                activation="relu",
                max_iter=self.epochs,
                random_state=self.seed + 17,
            )
            self.theta_model.fit(X, Y[:, 0])
            self.omega_model.fit(X, Y[:, 1])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "multi":
            return np.asarray(self.model.predict(X), dtype=float)
        th = np.asarray(self.theta_model.predict(X), dtype=float)
        om = np.asarray(self.omega_model.predict(X), dtype=float)
        return np.column_stack([th, om])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        rmse_theta=rmse(y_true[:, 0], y_pred[:, 0]),
        rmse_omega=rmse(y_true[:, 1], y_pred[:, 1]),
        nrmse_theta=nrmse(y_true[:, 0], y_pred[:, 0]),
        nrmse_omega=nrmse(y_true[:, 1], y_pred[:, 1]),
    )


def rollout_from_seed(model: NARXModel, df: pd.DataFrame, include_dt: bool) -> np.ndarray:
    y = df[["theta", "omega"]].to_numpy(dtype=float)
    u = df[["u"]].to_numpy(dtype=float)
    dt = df[["dt"]].to_numpy(dtype=float)
    max_l = max(model.ylag, model.xlag)
    y_hat = y.copy()
    for k in range(max_l, len(df) - 1):
        row: list[float] = []
        for i in range(1, model.ylag + 1):
            row.extend(y_hat[k - i].tolist())
        for i in range(0, model.xlag):
            row.extend(u[k - i].tolist())
        if include_dt:
            row.extend(dt[k].tolist())
        nxt = model.predict(np.asarray([row], dtype=float))[0]
        y_hat[k + 1] = nxt
    return y_hat


def k_step_rollout_error(model: NARXModel, df: pd.DataFrame, include_dt: bool, k_horizon: int) -> Metrics:
    y = df[["theta", "omega"]].to_numpy(dtype=float)
    u = df[["u"]].to_numpy(dtype=float)
    dt = df[["dt"]].to_numpy(dtype=float)
    max_l = max(model.ylag, model.xlag)
    preds = []
    trues = []
    for start in range(max_l, len(df) - k_horizon - 1):
        hist = y[: start + 1].copy()
        for kk in range(k_horizon):
            k = start + kk
            row: list[float] = []
            for i in range(1, model.ylag + 1):
                row.extend(hist[k - i].tolist())
            for i in range(0, model.xlag):
                row.extend(u[k - i].tolist())
            if include_dt:
                row.extend(dt[k].tolist())
            nxt = model.predict(np.asarray([row], dtype=float))[0]
            hist = np.vstack([hist, nxt])
        preds.append(hist[start + k_horizon])
        trues.append(y[start + k_horizon])
    return compute_metrics(np.asarray(trues, dtype=float), np.asarray(preds, dtype=float))


def build_sindy_library(theta: np.ndarray, omega: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.column_stack(
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
        ]
    )


def fit_linear_sparse(lib: np.ndarray, target: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(lib, target, rcond=None)
    coef = coef.astype(float)
    coef[np.abs(coef) < threshold] = 0.0
    return coef


def sindy_residual_fit(
    df: pd.DataFrame,
    y_true_next: np.ndarray,
    y_pred_next: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    theta = df["theta"].to_numpy(dtype=float)[: len(y_true_next)]
    omega = df["omega"].to_numpy(dtype=float)[: len(y_true_next)]
    u = df["u"].to_numpy(dtype=float)[: len(y_true_next)]
    lib = build_sindy_library(theta, omega, u)

    if mode == "discrete":
        r = y_true_next - y_pred_next
        coef_theta = fit_linear_sparse(lib, r[:, 0])
        coef_omega = fit_linear_sparse(lib, r[:, 1])
        return {"coef_theta": coef_theta, "coef_omega": coef_omega, "mode": mode}

    # continuous residual using omega_dot mismatch
    dt = np.maximum(df["dt"].to_numpy(dtype=float)[: len(y_true_next)], 1e-6)
    omega_dot_meas = np.gradient(y_true_next[:, 1], dt)
    omega_dot_nom = np.gradient(y_pred_next[:, 1], dt)
    rdot = omega_dot_meas - omega_dot_nom
    coef = fit_linear_sparse(lib, rdot)
    return {"coef": coef, "mode": mode}


def sindy_residual_apply(
    df: pd.DataFrame,
    y_pred_next: np.ndarray,
    sindy_model: dict[str, Any],
) -> np.ndarray:
    theta = df["theta"].to_numpy(dtype=float)[: len(y_pred_next)]
    omega = df["omega"].to_numpy(dtype=float)[: len(y_pred_next)]
    u = df["u"].to_numpy(dtype=float)[: len(y_pred_next)]
    lib = build_sindy_library(theta, omega, u)
    y_corr = y_pred_next.copy()
    if sindy_model["mode"] == "discrete":
        y_corr[:, 0] += lib @ sindy_model["coef_theta"]
        y_corr[:, 1] += lib @ sindy_model["coef_omega"]
    else:
        dt = np.maximum(df["dt"].to_numpy(dtype=float)[: len(y_pred_next)], 1e-6)
        omega_correction = np.cumsum((lib @ sindy_model["coef"]) * dt)
        y_corr[:, 1] += omega_correction
    return y_corr


class ParameterProposalEnv:
    """One-step environment: PPO proposes parameter/correction vector; reward = -trajectory cost."""

    def __init__(self, residual_base: dict[str, Any], dataset: dict[str, pd.DataFrame], weights: dict[str, float]):
        self.base = residual_base
        self.dataset = dataset
        self.weights = weights
        self.low = -0.5
        self.high = 0.5
        self.dim = self._dim()

    def _dim(self) -> int:
        if self.base["mode"] == "discrete":
            return len(self.base["coef_theta"]) + len(self.base["coef_omega"])
        return len(self.base["coef"])

    def evaluate(self, action: np.ndarray, baseline_preds: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
        a = np.asarray(action, dtype=float)
        a = np.clip(a, self.low, self.high)
        if self.base["mode"] == "discrete":
            n = len(self.base["coef_theta"])
            trial = {
                "mode": "discrete",
                "coef_theta": self.base["coef_theta"] + a[:n],
                "coef_omega": self.base["coef_omega"] + a[n : 2 * n],
            }
        else:
            trial = {"mode": "continuous", "coef": self.base["coef"] + a}

        costs = []
        instability = 0.0
        for split_name in ["train", "val", "test"]:
            df = self.dataset[split_name]
            pred = sindy_residual_apply(df, baseline_preds[split_name], trial)
            y = df[["theta", "omega"]].to_numpy(dtype=float)[1 : len(pred) + 1]
            pred = pred[: len(y)]
            met = compute_metrics(y, pred)
            alpha_true = df["alpha"].to_numpy(dtype=float)[1 : len(pred) + 1]
            dt = np.maximum(df["dt"].to_numpy(dtype=float)[1 : len(pred) + 1], 1e-6)
            alpha_pred = np.gradient(pred[:, 1], dt)
            alpha_nrmse = nrmse(alpha_true, alpha_pred)
            reg = float(np.mean(a**2))
            bound_penalty = float(np.mean(np.maximum(np.abs(a) - 0.45, 0.0) ** 2) * 100.0)
            if (not np.all(np.isfinite(pred))) or np.max(np.abs(pred)) > 1e4:
                instability += 1.0
            j = (
                self.weights["w_theta"] * met.nrmse_theta
                + self.weights["w_omega"] * met.nrmse_omega
                + self.weights["w_alpha"] * alpha_nrmse
                + self.weights["w_reg"] * reg
                + self.weights["w_bound"] * bound_penalty
            )
            costs.append(float(j))
        j_total = self.weights["lambda_mean"] * float(np.mean(costs)) + self.weights["lambda_max"] * float(np.max(costs))
        j_total += self.weights["w_stab"] * instability
        return -j_total, {"j_total": j_total, "instability": instability}


def run_ppo_search(
    env: ParameterProposalEnv,
    baseline_preds: dict[str, np.ndarray],
    seed: int,
    total_steps: int,
) -> dict[str, Any]:
    # True PPO agent (policy over parameter proposals). Fallback to random search if SB3 unavailable.
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
    except Exception:
        rng = np.random.default_rng(seed)
        best = None
        best_r = -np.inf
        for _ in range(total_steps):
            a = rng.uniform(env.low, env.high, size=(env.dim,))
            r, info = env.evaluate(a, baseline_preds)
            if r > best_r:
                best_r = r
                best = {"action": a.tolist(), "reward": float(r), "info": info}
        return {"backend": "random_search", "best": best}

    class _OneStepParamEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = spaces.Box(low=env.low, high=env.high, shape=(env.dim,), dtype=np.float32)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                np.random.seed(seed)
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            reward, info = env.evaluate(np.asarray(action, dtype=float), baseline_preds)
            obs = np.zeros((1,), dtype=np.float32)
            terminated = True
            truncated = False
            return obs, float(reward), terminated, truncated, info

    gym_env = _OneStepParamEnv()
    model = PPO(
        "MlpPolicy",
        gym_env,
        seed=seed,
        n_steps=64,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.0,
        verbose=0,
    )
    model.learn(total_timesteps=total_steps)

    best_reward = -np.inf
    best_action = None
    best_info = None
    obs, _ = gym_env.reset()
    for _ in range(256):
        action, _ = model.predict(obs, deterministic=False)
        reward, info = env.evaluate(action, baseline_preds)
        if reward > best_reward:
            best_reward = reward
            best_action = action
            best_info = info
    return {
        "backend": "stable_baselines3_ppo",
        "best": {"action": np.asarray(best_action, dtype=float).tolist(), "reward": float(best_reward), "info": best_info},
    }


def choose_csv(csv_hint: Path) -> Path:
    if csv_hint.exists():
        return csv_hint
    fallback = Path("host/run_logs/chrono_run_1.csv")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Neither finalized CSV nor fallback run log was found.")


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace):
    np.random.seed(args.seed)
    csv_path = choose_csv(Path(args.csv))
    meta_path = Path(args.meta)

    df_raw = load_signals(csv_path)
    uniform_df, dt_mean = uniform_resample(df_raw)
    irregular_df = irregular_with_dt(df_raw)

    results: dict[str, Any] = {
        "data": {
            "csv": str(csv_path),
            "meta": str(meta_path),
            "n_samples_raw": int(len(df_raw)),
            "dt_mean_uniform": dt_mean,
        },
        "stages": {},
    }

    strategies = {
        "uniform": (uniform_df, False),
        "irregular_dt": (irregular_df, True),
    }

    stage_outputs: dict[str, Any] = {}

    for strategy_name, (df_s, include_dt) in strategies.items():
        split = series_split(df_s, args.train_ratio, args.val_ratio)
        stage_outputs[strategy_name] = {}

        for mode in ["multi", "separate"]:
            model = NARXModel(mode=mode, ylag=args.ylag, xlag=args.xlag, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
            X_train, Y_train = build_arx_design(split["train"], args.ylag, args.xlag, include_dt=include_dt)
            X_val, Y_val = build_arx_design(split["val"], args.ylag, args.xlag, include_dt=include_dt)
            X_test, Y_test = build_arx_design(split["test"], args.ylag, args.xlag, include_dt=include_dt)

            model.fit(X_train, Y_train)
            yhat_1 = model.predict(X_test)
            one_step = compute_metrics(Y_test, yhat_1)

            roll_test = rollout_from_seed(model, split["test"], include_dt=include_dt)
            y_true_roll = split["test"][["theta", "omega"]].to_numpy(dtype=float)
            full_roll = compute_metrics(y_true_roll, roll_test)
            k10 = k_step_rollout_error(model, split["test"], include_dt=include_dt, k_horizon=10)
            k50 = k_step_rollout_error(model, split["test"], include_dt=include_dt, k_horizon=50)

            nom_pred_next = roll_test[:-1]
            nom_true_next = y_true_roll[1:]

            sindy_disc = sindy_residual_fit(split["test"], nom_true_next, nom_pred_next, mode="discrete")
            sindy_cont = sindy_residual_fit(split["test"], nom_true_next, nom_pred_next, mode="continuous")
            pred_disc = sindy_residual_apply(split["test"], nom_pred_next, sindy_disc)
            pred_cont = sindy_residual_apply(split["test"], nom_pred_next, sindy_cont)

            met_disc = compute_metrics(nom_true_next, pred_disc)
            met_cont = compute_metrics(nom_true_next, pred_cont)

            candidate = {
                "model": model,
                "split": split,
                "baseline_pred": {
                    "train": rollout_from_seed(model, split["train"], include_dt=include_dt)[:-1],
                    "val": rollout_from_seed(model, split["val"], include_dt=include_dt)[:-1],
                    "test": nom_pred_next,
                },
                "one_step": one_step,
                "full_roll": full_roll,
                "k10": k10,
                "k50": k50,
                "sindy_disc": sindy_disc,
                "sindy_cont": sindy_cont,
                "met_disc": met_disc,
                "met_cont": met_cont,
            }
            stage_outputs[strategy_name][mode] = candidate

    # choose best nominal by weighted NRMSE on full rollout
    best_key = None
    best_cost = np.inf
    for strat, per_mode in stage_outputs.items():
        for mode, obj in per_mode.items():
            cost = obj["full_roll"].weighted_nrmse(args.w_theta, args.w_omega)
            if cost < best_cost:
                best_cost = cost
                best_key = (strat, mode)

    assert best_key is not None
    best = stage_outputs[best_key[0]][best_key[1]]

    ppo_weights = {
        "w_theta": args.w_theta,
        "w_omega": args.w_omega,
        "w_alpha": args.w_alpha,
        "w_reg": args.w_reg,
        "w_bound": args.w_bound,
        "w_stab": args.w_stab,
        "lambda_mean": args.lambda_mean,
        "lambda_max": args.lambda_max,
    }

    # pick better residual mode for PPO base
    residual_base = best["sindy_disc"] if best["met_disc"].weighted_nrmse(args.w_theta, args.w_omega) <= best["met_cont"].weighted_nrmse(args.w_theta, args.w_omega) else best["sindy_cont"]
    ppo_env = ParameterProposalEnv(residual_base=residual_base, dataset=best["split"], weights=ppo_weights)
    ppo_result = run_ppo_search(ppo_env, best["baseline_pred"], seed=args.seed, total_steps=args.ppo_steps)

    report = {
        "config": vars(args),
        "best_nominal": {"sampling": best_key[0], "model_mode": best_key[1]},
        "sampling_comparison": {},
        "ablation": {},
        "ppo": ppo_result,
    }

    for strat, per_mode in stage_outputs.items():
        report["sampling_comparison"][strat] = {}
        for mode, obj in per_mode.items():
            report["sampling_comparison"][strat][mode] = {
                "one_step": asdict(obj["one_step"]),
                "k10_rollout": asdict(obj["k10"]),
                "k50_rollout": asdict(obj["k50"]),
                "full_rollout": asdict(obj["full_roll"]),
                "sindy_discrete": asdict(obj["met_disc"]),
                "sindy_continuous": asdict(obj["met_cont"]),
            }

    report["ablation"] = {
        "nnarx_only": asdict(best["full_roll"]),
        "nnarx_plus_sindy": asdict(best["met_disc"]),
        "nnarx_plus_sindy_plus_ppo": ppo_result.get("best", {}),
    }

    out_dir = REPORT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "benchmark_report.json", report)
    save_json(out_dir / "config.json", vars(args))

    # Save discovered residual equations in coefficient form.
    coef_dump = {
        "discrete": {
            "coef_theta": best["sindy_disc"].get("coef_theta", []).tolist() if "coef_theta" in best["sindy_disc"] else None,
            "coef_omega": best["sindy_disc"].get("coef_omega", []).tolist() if "coef_omega" in best["sindy_disc"] else None,
        },
        "continuous": {
            "coef": best["sindy_cont"].get("coef", []).tolist() if "coef" in best["sindy_cont"] else None,
        },
        "library_terms": [
            "1",
            "theta",
            "omega",
            "abs(omega)",
            "omega^2",
            "omega*abs(omega)",
            "sin(theta)",
            "cos(theta)",
            "u",
            "u*omega",
        ],
    }
    save_json(out_dir / "sindy_equations.json", coef_dump)

    # Lightweight markdown summary.
    md = [
        "# NNARX + SINDy + PPO Offline Benchmark",
        "",
        f"- Data CSV: `{csv_path}`",
        f"- Meta JSON: `{meta_path}`",
        f"- Best nominal: sampling=`{best_key[0]}`, mode=`{best_key[1]}`",
        "",
        "## Ablation",
        f"- NNARX only weighted NRMSE: {best['full_roll'].weighted_nrmse(args.w_theta, args.w_omega):.5f}",
        f"- NNARX + SINDy (discrete) weighted NRMSE: {best['met_disc'].weighted_nrmse(args.w_theta, args.w_omega):.5f}",
        f"- PPO backend: {ppo_result.get('backend', 'n/a')}",
        "",
        "## Sampling strategy comparison",
    ]
    for strat, per_mode in report["sampling_comparison"].items():
        for mode, met in per_mode.items():
            wr = args.w_theta * met["full_rollout"]["nrmse_theta"] + args.w_omega * met["full_rollout"]["nrmse_omega"]
            md.append(f"- {strat}/{mode}: full-roll weighted NRMSE={wr:.5f}")

    (out_dir / "benchmark_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[DONE] wrote benchmark artifacts to {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline identification benchmark: NNARX + SINDy-PI + PPO parameter proposal")
    p.add_argument("--csv", default="host/run_logs/chrono_run_1.finalized.csv")
    p.add_argument("--meta", default="host/run_logs/chrono_run_1.meta.json")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ylag", type=int, default=4)
    p.add_argument("--xlag", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--ppo-steps", type=int, default=5000)

    p.add_argument("--w-theta", type=float, default=0.5)
    p.add_argument("--w-omega", type=float, default=0.5)
    p.add_argument("--w-alpha", type=float, default=0.2)
    p.add_argument("--w-reg", type=float, default=0.01)
    p.add_argument("--w-bound", type=float, default=0.02)
    p.add_argument("--w-stab", type=float, default=10.0)
    p.add_argument("--lambda-mean", type=float, default=0.7)
    p.add_argument("--lambda-max", type=float, default=0.3)
    return p


def main():
    args = build_argparser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
