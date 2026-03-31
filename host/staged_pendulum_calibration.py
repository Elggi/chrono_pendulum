#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chrono_core.calibration_io import apply_calibration_json
from chrono_core.config import BridgeConfig
from chrono_core.pendulum_rl_env import (
    PendulumRLEnv,
    build_init_params,
    compute_error_features,
    load_replay_csv,
    simplified_loss,
    simulate_trajectory,
    weighted_loss,
)
from chrono_core.pendulum_rl_plots import (
    plot_param_convergence,
    plot_stage1_regression_summary,
    plot_stage123_regression_summary,
    plot_training_curves,
)


K_U_INTERPRETATION = "effective input-to-torque gain"


@dataclass
class StageContext:
    stage1: dict[str, float] | None = None
    stage2: dict[str, float] | None = None
    stage3: dict[str, float] | None = None


def _input(prompt: str, default: str | None = None) -> str:
    raw = input(prompt).strip()
    if raw == "" and default is not None:
        return default
    return raw


def list_csv_logs(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("*.csv"))


def list_json_logs(base_dir: Path) -> list[Path]:
    return sorted([p for p in base_dir.glob("*.json") if p.is_file()])


def choose_one_csv(items: list[Path], title: str) -> Path:
    if not items:
        raise ValueError("No CSV files found in run_logs.")
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


def choose_many_csv(items: list[Path], title: str) -> list[Path]:
    if not items:
        raise ValueError("No CSV files found in run_logs.")
    print(title)
    for i, item in enumerate(items, start=1):
        print(f"[{i}] {item.name}")
    raw = _input("Select indices (comma separated, Enter to skip): ", "")
    if raw.strip() == "":
        return []
    while True:
        try:
            idx = []
            for tok in raw.split(","):
                tok = tok.strip()
                if tok:
                    idx.append(int(tok))
            chosen = [items[i - 1] for i in idx]
            print(f"[INFO] selected_stage4_csv: {[str(p) for p in chosen]}")
            return chosen
        except (ValueError, IndexError):
            raw = _input("Invalid selection. Re-enter comma separated indices (or Enter to skip): ", "")
            if raw.strip() == "":
                return []


def choose_optional_json(items: list[Path], title: str) -> Path | None:
    print(title)
    options = [None] + items
    for i, item in enumerate(options, start=1):
        if item is None:
            print(f"[{i}] 없음 (JSON 미적용)")
        else:
            print(f"[{i}] {item.name}")
    while True:
        raw = _input("#? ", "1")
        try:
            idx = int(raw)
            picked = options[idx - 1]
            print(f"[INFO] Selected JSON: {picked if picked is not None else '없음'}")
            return picked
        except (ValueError, IndexError):
            print("Invalid selection. Please choose a valid index.")


def prompt_stage4_hyperparams(default_episodes: int) -> dict:
    print("Interactive mode (press Enter to keep default):")
    episodes = int(_input(f"num_episodes [{default_episodes}]: ", str(default_episodes)))
    gamma = float(_input("gamma [0.995]: ", "0.995"))
    lam = float(_input("lam [0.98]: ", "0.98"))
    kl_targ = float(_input("kl_targ [0.003]: ", "0.003"))
    batch_size = int(_input("batch_size [20]: ", "20"))
    device = _input("device [cpu/cuda] [cpu]: ", "cpu")
    dr = _input("domain_randomization (ON/OFF) [OFF]: ", "OFF").strip().upper()
    domain_randomization = dr in ("ON", "Y", "YES", "TRUE", "1")
    return {
        "episodes": episodes,
        "gamma": gamma,
        "gae_lambda": lam,
        "target_kl": kl_targ,
        "batch_size": batch_size,
        "device": device,
        "domain_randomization": domain_randomization,
    }


def load_stage_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_stage_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_geometry(cfg: BridgeConfig):
    m_total = cfg.rod_mass + cfg.imu_mass
    j_rod = (1.0 / 3.0) * cfg.rod_mass * (cfg.rod_length ** 2)
    j_imu = cfg.imu_mass * (cfg.r_imu ** 2)
    return float(m_total), float(j_rod + j_imu)


def _stage_metadata_base(stage: int, csv_path: Path, excitation_type: str, active_params: list[str], fixed_params: list[str],
                         theta_source: str, omega_source: str, alpha_source: str, input_source: str):
    return {
        "stage": stage,
        "csv_path": str(csv_path),
        "excitation_type": excitation_type,
        "active_params": active_params,
        "fixed_params": fixed_params,
        "theta_source": theta_source,
        "omega_source": omega_source,
        "alpha_source": alpha_source,
        "input_source": input_source,
        "target_source": f"J*{alpha_source}",
        "K_u_interpretation": K_U_INTERPRETATION,
    }


def _load_regression_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = ["theta_real", "omega_real", "hw_pwm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required measured columns in {csv_path}: {missing}")

    theta = pd.to_numeric(df["theta_real"], errors="coerce").to_numpy(dtype=float)
    omega = pd.to_numeric(df["omega_real"], errors="coerce").to_numpy(dtype=float)

    if "alpha_real" not in df.columns:
        raise ValueError(f"Missing required measured column 'alpha_real' in {csv_path}")
    alpha = pd.to_numeric(df["alpha_real"], errors="coerce").to_numpy(dtype=float)
    alpha_source = "alpha_real"

    input_source = "hw_pwm"
    u = pd.to_numeric(df["hw_pwm"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(u).any() or np.nanmax(np.abs(u)) < 1e-9:
        if "cmd_u_raw" in df.columns:
            u = pd.to_numeric(df["cmd_u_raw"], errors="coerce").to_numpy(dtype=float)
            input_source = "cmd_u_raw(fallback_from_hw_pwm)"
            print("[WARN] hw_pwm unavailable/unusable. Falling back to cmd_u_raw.")
        else:
            raise ValueError("Neither usable hw_pwm nor cmd_u_raw found.")

    mask = np.isfinite(theta) & np.isfinite(omega) & np.isfinite(alpha) & np.isfinite(u)
    if int(mask.sum()) < 16:
        raise ValueError(f"Not enough valid samples in {csv_path}. valid={int(mask.sum())}")

    return {
        "theta": np.unwrap(theta[mask]),
        "omega": omega[mask],
        "alpha": alpha[mask],
        "u": u[mask],
        "alpha_source": alpha_source,
        "input_source": input_source,
        "sample_count": int(mask.sum()),
    }


def _fit_stage1(data: dict, cfg: BridgeConfig, k_bounds: tuple[float, float]):
    m_total, j_pivot = _compute_geometry(cfg)
    l_bounds = (0.01, float(cfg.link_L))
    y = j_pivot * data["alpha"]
    X = np.column_stack([data["u"], -(m_total * cfg.gravity) * np.sin(data["theta"])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    k_u = float(np.clip(beta[0], k_bounds[0], k_bounds[1]))
    l_com = float(np.clip(beta[1], l_bounds[0], l_bounds[1]))
    y_hat = X @ np.array([k_u, l_com], dtype=float)
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    return {
        "K_u": k_u,
        "l_com": l_com,
        "rmse": rmse,
        "J_pivot": j_pivot,
        "m_total": m_total,
        "y_true": y,
        "y_pred": y_hat,
        "bounds": {"K_u": [k_bounds[0], k_bounds[1]], "l_com": [l_bounds[0], l_bounds[1]]},
    }


def _fit_stage2(data: dict, cfg: BridgeConfig, prev: dict[str, float], omega_deadband: float, b_bounds: tuple[float, float]):
    m_total, j_pivot = _compute_geometry(cfg)
    residual_b = j_pivot * data["alpha"] - prev["K_u"] * data["u"] + m_total * cfg.gravity * prev["l_com"] * np.sin(data["theta"])
    omega = data["omega"]
    if omega_deadband > 0.0:
        mask = np.abs(omega) >= omega_deadband
    else:
        mask = np.ones_like(omega, dtype=bool)
    om = omega[mask]
    rb = residual_b[mask]
    if len(om) < 8:
        raise ValueError("Too few samples after omega deadband filtering for Stage 2")
    b_eq = float(np.clip(-(om @ rb) / max(om @ om, 1e-12), b_bounds[0], b_bounds[1]))
    rb_hat = -b_eq * om
    return {
        "b_eq": b_eq,
        "rmse": float(np.sqrt(np.mean((rb_hat - rb) ** 2))),
        "used_ratio": float(len(om) / len(omega)),
        "omega_deadband": float(omega_deadband),
        "bounds": {"b_eq": [float(b_bounds[0]), float(b_bounds[1])]},
    }


def _fit_stage3(data: dict, cfg: BridgeConfig, prev: dict[str, float], tanh_eps: float, high_speed_ref: float, tau_bounds: tuple[float, float]):
    m_total, j_pivot = _compute_geometry(cfg)
    omega = data["omega"]
    residual_tau = (
        j_pivot * data["alpha"]
        - prev["K_u"] * data["u"]
        + prev["b_eq"] * omega
        + m_total * cfg.gravity * prev["l_com"] * np.sin(data["theta"])
    )
    reg = np.tanh(omega / max(tanh_eps, 1e-6))
    weights = 1.0 / (1.0 + (np.abs(omega) / max(high_speed_ref, 1e-6)) ** 2)
    num = -np.sum(weights * reg * residual_tau)
    den = max(np.sum(weights * reg * reg), 1e-12)
    tau_eq = float(np.clip(num / den, tau_bounds[0], tau_bounds[1]))
    pred = -tau_eq * reg
    low_speed_ratio = float(np.mean(np.abs(omega) <= high_speed_ref))
    return {
        "tau_eq": tau_eq,
        "rmse": float(np.sqrt(np.mean((pred - residual_tau) ** 2))),
        "low_speed_ratio": low_speed_ratio,
        "high_speed_ref": float(high_speed_ref),
        "tanh_eps": float(tanh_eps),
        "bounds": {"tau_eq": [float(tau_bounds[0]), float(tau_bounds[1])]},
    }


def _print_init_state(init_params: dict[str, float], loaded: bool):
    if loaded:
        print("[INFO] init_policy: loaded_from_json (fresh defaults are not overriding loaded values)")
    else:
        print("[INFO] init_policy: fresh_untrained_defaults")
    print(f"  - K_u_init: {init_params['K_u']}")
    print(f"  - l_com_init: {init_params['l_com']}")
    print(f"  - b_eq_init: {init_params['b_eq']}")
    print(f"  - tau_eq_init: {init_params['tau_eq']}")


def _prompt_bounds(name: str, lo: float, hi: float) -> tuple[float, float]:
    print(f"[INPUT] {name} bounds (current: [{lo}, {hi}])")
    new_lo = float(_input(f"  - {name}_lower [{lo}]: ", str(lo)))
    new_hi = float(_input(f"  - {name}_upper [{hi}]: ", str(hi)))
    if new_lo > new_hi:
        print(f"[WARN] lower > upper for {name}. Swapping values.")
        new_lo, new_hi = new_hi, new_lo
    return float(new_lo), float(new_hi)


def _prompt_stage_decision() -> str:
    print("A) Accept and move on to next stage")
    print("B) Rerun with different bounds")
    print("C) quit")
    while True:
        choice = _input("Select action [A/B/C]: ", "A").strip().upper()
        if choice in ("A", "B", "C"):
            return choice
        print("Invalid choice. Please select A, B, or C.")


def run_stage1(cfg: BridgeConfig, csv_path: Path, outdir: Path, k_bounds: tuple[float, float]):
    print("[INFO] stage1_dataset_policy:")
    print(f"  - csv_path: {csv_path}")
    print("  - excitation_type: sin")
    print("  - active_params: ['K_u', 'l_com']")

    data = _load_regression_data(csv_path)
    print("[INFO] stage1_regression_sources:")
    print("  - theta_source: theta_real")
    print("  - omega_source: omega_real")
    print(f"  - alpha_source: {data['alpha_source']}")
    print(f"  - input_source: {data['input_source']}")
    print(f"  - target_source: J*{data['alpha_source']}")
    print(f"  - K_u_interpretation: {K_U_INTERPRETATION}")

    res = _fit_stage1(data, cfg, k_bounds)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "J*alpha = K_u*u - m*g*l_com*sin(theta)",
        "identified_params": {"K_u": res["K_u"], "l_com": res["l_com"]},
        "fixed_params": {"b_eq": cfg.b_eq_init, "tau_eq": cfg.tau_eq_init},
        "metrics": {"rmse": res["rmse"], "sample_count": data["sample_count"]},
        "bounds": res["bounds"],
        "metadata": _stage_metadata_base(1, csv_path, "sin", ["K_u", "l_com"], [], "theta_real", "omega_real", data["alpha_source"], data["input_source"]),
    }
    save_stage_json(outdir / "stage1_params.json", payload)
    plot_stage1_regression_summary(res["y_true"], res["y_pred"], outdir / "stage1_regression_fit_summary.png")
    return payload


def run_stage2(cfg: BridgeConfig, csv_path: Path, outdir: Path, stage1_payload: dict, omega_deadband: float, b_bounds: tuple[float, float]):
    print("[INFO] stage2_dataset_policy:")
    print(f"  - csv_path: {csv_path}")
    print("  - excitation_type: square")
    print("  - active_params: ['b_eq']")
    print("  - fixed_params: ['K_u', 'l_com']")

    prev = stage1_payload["identified_params"]
    data = _load_regression_data(csv_path)
    res = _fit_stage2(data, cfg, prev=prev, omega_deadband=omega_deadband, b_bounds=b_bounds)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "residual_b = J*alpha - K_u*u + m*g*l_com*sin(theta) ~= -b_eq*omega",
        "identified_params": {"b_eq": res["b_eq"]},
        "fixed_params": {"K_u": prev["K_u"], "l_com": prev["l_com"]},
        "metrics": {"rmse": res["rmse"], "omega_deadband": res["omega_deadband"], "used_ratio": res["used_ratio"]},
        "bounds": res["bounds"],
        "metadata": {
            **_stage_metadata_base(2, csv_path, "square", ["b_eq"], ["K_u", "l_com"], "theta_real", "omega_real", data["alpha_source"], data["input_source"]),
            "filtering_policy": {
                "omega_deadband_applied": bool(omega_deadband > 0.0),
                "omega_deadband": float(omega_deadband),
            },
        },
    }
    save_stage_json(outdir / "stage2_params.json", payload)
    return payload


def run_stage3(cfg: BridgeConfig, csv_path: Path, outdir: Path, stage1_payload: dict, stage2_payload: dict, tanh_eps: float, high_speed_ref: float, tau_bounds: tuple[float, float]):
    print("[INFO] stage3_dataset_policy:")
    print(f"  - csv_path: {csv_path}")
    print("  - excitation_type: burst")
    print("  - active_params: ['tau_eq']")
    print("  - fixed_params: ['K_u', 'l_com', 'b_eq']")

    data = _load_regression_data(csv_path)
    prev = {
        "K_u": stage1_payload["identified_params"]["K_u"],
        "l_com": stage1_payload["identified_params"]["l_com"],
        "b_eq": stage2_payload["identified_params"]["b_eq"],
    }
    res = _fit_stage3(data, cfg, prev=prev, tanh_eps=tanh_eps, high_speed_ref=high_speed_ref, tau_bounds=tau_bounds)
    print(f"[INFO] stage3_low_speed_usage: ratio={res['low_speed_ratio']:.4f} (|omega| <= {high_speed_ref})")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "residual_tau = J*alpha - K_u*u + b_eq*omega + m*g*l_com*sin(theta) ~= -tau_eq*tanh(omega/eps)",
        "identified_params": {"tau_eq": res["tau_eq"]},
        "fixed_params": prev,
        "metrics": {
            "rmse": res["rmse"],
            "low_speed_ratio": res["low_speed_ratio"],
            "high_speed_ref": res["high_speed_ref"],
            "tanh_eps": res["tanh_eps"],
        },
        "bounds": res["bounds"],
        "metadata": {
            **_stage_metadata_base(3, csv_path, "burst", ["tau_eq"], ["K_u", "l_com", "b_eq"], "theta_real", "omega_real", data["alpha_source"], data["input_source"]),
            "filtering_policy": {
                "high_speed_down_weighting": True,
                "high_speed_ref": float(high_speed_ref),
            },
        },
    }
    save_stage_json(outdir / "stage3_params.json", payload)
    return payload


def evaluate_dataset_mode(env: PendulumRLEnv, params: dict[str, float], loss_mode: str):
    losses = []
    for traj in env.trajectories:
        sim = simulate_trajectory(traj, params, env.cfg, delay_sec=params.get("delay_sec", traj.delay_sec_est))
        feat = compute_error_features(traj, sim, align_shift_sec=0.0)
        losses.append(simplified_loss(feat, {"theta": 1.0, "omega": 1.0, "alpha": 1.0}) if loss_mode == "simplified" else weighted_loss(feat, env.reward_weights))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_rmse_metrics(env: PendulumRLEnv, params: dict[str, float]) -> dict[str, float]:
    rmse_theta = []
    rmse_omega = []
    rmse_alpha = []
    for traj in env.trajectories:
        sim = simulate_trajectory(traj, params, env.cfg, delay_sec=params.get("delay_sec", traj.delay_sec_est))
        feat = compute_error_features(traj, sim, align_shift_sec=0.0)
        rmse_theta.append(float(feat["rmse_theta"]))
        rmse_omega.append(float(feat["rmse_omega"]))
        rmse_alpha.append(float(feat["rmse_alpha"]))
    if len(rmse_theta) == 0:
        return {"rmse_theta": np.nan, "rmse_omega": np.nan, "rmse_alpha": np.nan}
    return {
        "rmse_theta": float(np.mean(rmse_theta)),
        "rmse_omega": float(np.mean(rmse_omega)),
        "rmse_alpha": float(np.mean(rmse_alpha)),
    }


def _train_rl(env, val_env, episodes: int, seed: int = 7, ppo_cfg: dict | None = None):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class _SB3Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(wrapped.state_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(wrapped.action_dim,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            return np.asarray(self.wrapped.reset(), dtype=np.float32), {}

        def step(self, action):
            obs, rew, done, info = self.wrapped.step(np.asarray(action, dtype=float))
            return np.asarray(obs, dtype=np.float32), float(rew), bool(done), False, info

    ppo_cfg = ppo_cfg or {}
    model = PPO(
        "MlpPolicy",
        _SB3Env(env),
        seed=seed,
        n_steps=32,
        batch_size=int(ppo_cfg.get("batch_size", 20)),
        gamma=float(ppo_cfg.get("gamma", 0.995)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.98)),
        target_kl=float(ppo_cfg.get("target_kl", 0.003)),
        device=str(ppo_cfg.get("device", "cpu")),
        verbose=0,
    )
    history = {
        "episode_reward": [],
        "train_loss": [],
        "val_loss": [],
        "rmse_theta": [],
        "rmse_omega": [],
        "rmse_alpha": [],
        "val_rmse_theta": [],
        "val_rmse_omega": [],
        "val_rmse_alpha": [],
    }
    param_hist = {
        "current_eval_params_per_episode": {k: [] for k in env.param_keys},
        "global_best_train_params_so_far": {k: [] for k in env.param_keys},
        "global_best_val_params_so_far": {k: [] for k in env.param_keys},
    }
    best_params = env.center.copy()
    best_val = float("inf")
    t0 = time.time()
    for ep in range(1, episodes + 1):
        model.learn(total_timesteps=max(1, env.max_refine_steps), progress_bar=False, reset_num_timesteps=False)
        obs, _ = _SB3Env(env).reset()
        done = False
        total_reward = 0.0
        info = {"params": env.center}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = _SB3Env(env).step(action)
            total_reward += rew
            done = bool(terminated or truncated)
        cur = dict(info.get("params", {}))
        train_loss = evaluate_dataset_mode(env, cur, env.loss_mode)
        val_loss = evaluate_dataset_mode(val_env, cur, val_env.loss_mode)
        train_rmse = evaluate_rmse_metrics(env, cur)
        val_rmse = evaluate_rmse_metrics(val_env, cur)
        history["episode_reward"].append(float(total_reward))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["rmse_theta"].append(float(train_rmse["rmse_theta"]))
        history["rmse_omega"].append(float(train_rmse["rmse_omega"]))
        history["rmse_alpha"].append(float(train_rmse["rmse_alpha"]))
        history["val_rmse_theta"].append(float(val_rmse["rmse_theta"]))
        history["val_rmse_omega"].append(float(val_rmse["rmse_omega"]))
        history["val_rmse_alpha"].append(float(val_rmse["rmse_alpha"]))
        if val_loss < best_val:
            best_val = float(val_loss)
            best_params = cur.copy()
        for k in env.param_keys:
            param_hist["current_eval_params_per_episode"][k].append(float(cur[k]))
            param_hist["global_best_train_params_so_far"][k].append(float(cur[k]))
            param_hist["global_best_val_params_so_far"][k].append(float(best_params.get(k, cur[k])))
        if ep % 10 == 0 or ep == 1 or ep == episodes:
            elapsed = time.time() - t0
            print(
                f"[RL] ep {ep}/{episodes} | train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} | best_val={best_val:.5f} | elapsed={elapsed:.1f}s"
            )
    return best_params, best_val, history, param_hist


def run_stage4(
    cfg: BridgeConfig,
    outdir: Path,
    stage1_payload: dict,
    stage2_payload: dict,
    stage3_payload: dict,
    stage4_csv: list[Path],
    episodes: int,
    domain_randomization: bool = False,
    ppo_cfg: dict | None = None,
):
    print("[INFO] stage4_dataset_policy:")
    print("  - mode: PPO fine-tuning")
    print("  - init_from_regression: True")

    init_params = {
        "K_u": float(stage1_payload["identified_params"]["K_u"]),
        "l_com": float(stage1_payload["identified_params"]["l_com"]),
        "b_eq": float(stage2_payload["identified_params"]["b_eq"]),
        "tau_eq": float(stage3_payload["identified_params"]["tau_eq"]),
    }
    print("[INFO] stage4_ppo_init:")
    print(f"  - K_u_init_from_stage1: {init_params['K_u']}")
    print(f"  - l_com_init_from_stage1: {init_params['l_com']}")
    print(f"  - b_eq_init_from_stage2: {init_params['b_eq']}")
    print(f"  - tau_eq_init_from_stage3: {init_params['tau_eq']}")
    print("  - fine_tuning_mode: bounded_refinement")

    trajectories = [load_replay_csv(p, cfg) for p in stage4_csv]
    spans = {"K_u": 0.25, "l_com": 0.20, "b_eq": 0.30, "tau_eq": 0.30}
    bounds = {}
    hard = {"K_u": (1e-6, 1.0), "l_com": (0.01, float(cfg.link_L)), "b_eq": (0.0, float(cfg.b_eq_max)), "tau_eq": (0.0, float(cfg.tau_eq_max))}
    for k, c in init_params.items():
        r = max(abs(c) * spans[k], 1e-6)
        lo, hi = hard[k]
        bounds[k] = (max(lo, c - r), min(hi, c + r))

    env = PendulumRLEnv(trajectories=trajectories, cfg=cfg, init_params=init_params, learn_delay=False,
                        domain_randomization=domain_randomization, max_refine_steps=12, action_step_frac=0.05, init_noise_frac=0.0,
                        param_keys_override=["K_u", "l_com", "b_eq", "tau_eq"], bounds_override=bounds, loss_mode="full")
    val_env = PendulumRLEnv(trajectories=trajectories, cfg=cfg, init_params=init_params, learn_delay=False,
                            domain_randomization=False, max_refine_steps=12, action_step_frac=0.05, init_noise_frac=0.0,
                            param_keys_override=["K_u", "l_com", "b_eq", "tau_eq"], bounds_override=bounds, loss_mode="full")
    best_params, best_val, history, param_hist = _train_rl(env, val_env, episodes=episodes, ppo_cfg=ppo_cfg)

    payload = {
        "stage": 4,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "PPO fine-tuning",
        "role": "final_refiner_not_initial_identifier",
        "identified_params": {k: float(best_params[k]) for k in ["K_u", "l_com", "b_eq", "tau_eq"]},
        "metrics": {"best_val_loss": float(best_val)},
        "metadata": {
            "stage": 4,
            "csv_path": [str(p) for p in stage4_csv],
            "excitation_type": "ppo_refinement",
            "active_params": ["K_u", "l_com", "b_eq", "tau_eq"],
            "fixed_params": [],
            "fine_tuning_mode": "bounded_refinement",
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
            "K_u_interpretation": K_U_INTERPRETATION,
        },
    }
    save_stage_json(outdir / "stage4_params.json", payload)

    stage_dir = outdir / "stage4_plots"
    stage_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, stage_dir)
    plot_param_convergence(param_hist, stage_dir)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="Sequential staged identification + PPO fine-tuning")
    p.add_argument("--run_logs", type=Path, default=Path(__file__).resolve().parent / "run_logs")
    p.add_argument("--mode", choices=["regression", "rl", "all"], default="all")
    p.add_argument("--calibration_json", type=Path, default=None)
    p.add_argument("--k_u_min", type=float, default=1e-6)
    p.add_argument("--k_u_max", type=float, default=1.0)
    p.add_argument("--omega_deadband", type=float, default=0.05)
    p.add_argument("--tanh_eps", type=float, default=0.2)
    p.add_argument("--high_speed_ref", type=float, default=3.0)
    p.add_argument("--ppo_episodes", type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = BridgeConfig()

    cfg.K_u_init = 1.0e-5
    cfg.b_eq_init = 0.0
    cfg.tau_eq_init = 0.0
    cfg.l_com_init = 0.5 * float(cfg.link_L)

    outdir = args.run_logs
    outdir.mkdir(parents=True, exist_ok=True)

    csv_files = list_csv_logs(outdir)
    json_files = list_json_logs(outdir)

    loaded = False
    calib_json = args.calibration_json
    if calib_json is None and args.mode == "rl":
        print("--------------------------------")
        calib_json = choose_optional_json(json_files, "[INFO] Calibration JSON (optional) 선택")
    if calib_json is not None and calib_json.exists():
        apply_calibration_json(cfg, str(calib_json))
        loaded = True

    init_params = build_init_params(cfg, calibration=None, parameter_json=None)
    _print_init_state(init_params, loaded=loaded)

    stage1 = None
    stage2 = None
    stage3 = None

    if args.mode in ("regression", "all"):
        print("[INFO] Running regression stages 1->3 with per-stage iterative confirmation.")

        k_bounds = (args.k_u_min, args.k_u_max)
        while True:
            print("--------------------------------")
            print("[INFO] Stage 1 (sin) setup")
            stage1_csv = choose_one_csv(csv_files, "Select Stage 1 CSV (sin excitation):")
            k_bounds = _prompt_bounds("K_u", k_bounds[0], k_bounds[1])
            stage1 = run_stage1(cfg, stage1_csv, outdir, k_bounds=k_bounds)
            print(f"[INFO] Stage 1 result: K_u={stage1['identified_params']['K_u']}, l_com={stage1['identified_params']['l_com']}")
            action = _prompt_stage_decision()
            if action == "A":
                break
            if action == "C":
                print("[INFO] Quit requested during Stage 1.")
                return

        b_bounds = (0.0, float(cfg.b_eq_max))
        while True:
            print("--------------------------------")
            print("[INFO] Stage 2 (square) setup")
            stage2_csv = choose_one_csv(csv_files, "Select Stage 2 CSV (square excitation):")
            b_bounds = _prompt_bounds("b_eq", b_bounds[0], b_bounds[1])
            stage2 = run_stage2(cfg, stage2_csv, outdir, stage1_payload=stage1, omega_deadband=args.omega_deadband, b_bounds=b_bounds)
            print(f"[INFO] Stage 2 result: b_eq={stage2['identified_params']['b_eq']}")
            action = _prompt_stage_decision()
            if action == "A":
                break
            if action == "C":
                print("[INFO] Quit requested during Stage 2.")
                return

        tau_bounds = (0.0, float(cfg.tau_eq_max))
        while True:
            print("--------------------------------")
            print("[INFO] Stage 3 (burst) setup")
            stage3_csv = choose_one_csv(csv_files, "Select Stage 3 CSV (burst excitation):")
            tau_bounds = _prompt_bounds("tau_eq", tau_bounds[0], tau_bounds[1])
            stage3 = run_stage3(cfg, stage3_csv, outdir, stage1_payload=stage1, stage2_payload=stage2,
                                tanh_eps=args.tanh_eps, high_speed_ref=args.high_speed_ref, tau_bounds=tau_bounds)
            print(f"[INFO] Stage 3 result: tau_eq={stage3['identified_params']['tau_eq']}")
            action = _prompt_stage_decision()
            if action == "A":
                break
            if action == "C":
                print("[INFO] Quit requested during Stage 3.")
                return

        plot_stage123_regression_summary(
            {"stage1": stage1, "stage2": stage2, "stage3": stage3},
            outdir / "stage123_regression_summary.png",
        )
        print(f"[INFO] Saved: {outdir / 'stage123_regression_summary.png'}")

    if args.mode in ("rl", "all"):
        print("[INFO] Running Stage 4 PPO fine-tuning only.")
        print("--------------------------------")
        param_json = choose_optional_json(json_files, "[INFO] Parameter JSON (optional) 선택")
        if param_json is not None:
            print("[INFO] Stage4 uses stage1/2/3 regression JSON as initialization source; selected parameter JSON is kept for operator traceability only.")
        ppo_cfg = prompt_stage4_hyperparams(args.ppo_episodes)
        if stage1 is None:
            stage1_path = outdir / "stage1_params.json"
            stage2_path = outdir / "stage2_params.json"
            stage3_path = outdir / "stage3_params.json"
            if not (stage1_path.exists() and stage2_path.exists() and stage3_path.exists()):
                raise FileNotFoundError("Stage4 requires stage1_params.json, stage2_params.json, stage3_params.json in run_logs.")
            stage1 = load_stage_json(stage1_path)
            stage2 = load_stage_json(stage2_path)
            stage3 = load_stage_json(stage3_path)
        stage4_csv = choose_many_csv(csv_files, "Select Stage 4 PPO CSV list (Enter => use Stage3 csv policy).")
        if not stage4_csv:
            stage4_csv = [choose_one_csv(csv_files, "Select fallback CSV for Stage 4 PPO:")]
        run_stage4(cfg, outdir, stage1_payload=stage1, stage2_payload=stage2, stage3_payload=stage3,
                   stage4_csv=stage4_csv, episodes=ppo_cfg["episodes"],
                   domain_randomization=ppo_cfg["domain_randomization"], ppo_cfg=ppo_cfg)


if __name__ == "__main__":
    main()
