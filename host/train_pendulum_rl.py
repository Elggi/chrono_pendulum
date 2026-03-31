#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chrono_core.calibration_io import apply_calibration_json
from chrono_core.config import BridgeConfig
from chrono_core.log_schema import PENDULUM_LOG_COLUMNS
from chrono_core.pendulum_rl_env import (
    PARAM_KEYS,
    PendulumRLEnv,
    build_init_params,
    load_replay_csv,
    simulate_trajectory,
    split_trajectories,
    weighted_loss,
    compute_error_features,
)
from chrono_core.pendulum_rl_plots import (
    plot_delay_diagnostics,
    plot_overlay,
    plot_param_convergence,
    plot_rl_dashboard,
    plot_training_curves,
)


@dataclass
class TrainConfig:
    num_episodes: int = 1000
    gamma: float = 0.995
    lam: float = 0.98
    kl_targ: float = 0.003
    batch_size: int = 20


def train_with_sb3(env, val_env, args, history, param_hist):
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        raise SystemExit(f"SB3 backend requested but dependencies are missing: {exc}")

    class _SB3ReplayEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, wrapped_env):
            super().__init__()
            self.wrapped = wrapped_env
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.wrapped.state_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.wrapped.action_dim,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            obs = self.wrapped.reset()
            return np.asarray(obs, dtype=np.float32), {}

        def step(self, action):
            obs, rew, done, info = self.wrapped.step(np.asarray(action, dtype=float))
            return np.asarray(obs, dtype=np.float32), float(rew), bool(done), False, info

    sb3_env = _SB3ReplayEnv(env)
    monitor_path = str(Path(args.outdir) / "sb3_monitor.csv")
    sb3_env = Monitor(sb3_env, filename=monitor_path)
    steps_per_episode = max(1, int(env.max_refine_steps))
    tb_dir = args.tensorboard_log if args.tensorboard_log else str(Path(args.outdir) / "tensorboard")
    model = PPO(
        "MlpPolicy",
        sb3_env,
        device=args.device,
        seed=args.seed,
        n_steps=max(env.max_refine_steps * 4, 32),
        batch_size=min(64, max(16, env.max_refine_steps * 2)),
        learning_rate=3e-4,
        gamma=args.gamma,
        tensorboard_log=tb_dir,
        verbose=0,
    )
    best_val = float("inf")
    best_params = env.best_params.copy()
    t_start = time.time()
    for ep in range(1, int(args.num_episodes) + 1):
        model.learn(
            total_timesteps=steps_per_episode,
            progress_bar=False,
            reset_num_timesteps=False,
            tb_log_name="ppo_pendulum",
        )
        train_loss, train_rmse = evaluate_dataset(env, env.best_params)
        val_loss, val_rmse = evaluate_dataset(val_env, env.best_params)
        history["reward"].append(float(-train_loss))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["rmse_theta"].append(float(train_rmse["theta"]))
        history["rmse_omega"].append(float(train_rmse["omega"]))
        history["rmse_alpha"].append(float(train_rmse["alpha"]))
        history["val_rmse_theta"].append(float(val_rmse["theta"]))
        history["val_rmse_omega"].append(float(val_rmse["omega"]))
        history["val_rmse_alpha"].append(float(val_rmse["alpha"]))
        for k in env.param_keys:
            param_hist[k].append(float(env.best_params[k]))
        if val_loss < best_val:
            best_val = float(val_loss)
            best_params = env.best_params.copy()
        if ep == 1 or ep % max(1, int(args.log_every_episodes)) == 0 or ep == int(args.num_episodes):
            elapsed = time.time() - t_start
            print(
                f"[RL] ep {ep}/{args.num_episodes} | "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} | "
                f"best_val={best_val:.5f} | elapsed={elapsed:.1f}s"
            )
    return best_val, best_params


def gather_csv_paths(csv: str | None, csv_dir: str | None):
    out = []
    if csv:
        out.append(Path(csv))
    if csv_dir:
        out.extend(sorted(Path(csv_dir).glob("*.csv")))
    uniq = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def maybe_prompt(args, parser):
    if not sys.stdin.isatty():
        return args

    def ask_int(name, cur):
        v = input(f"{name} [{cur}]: ").strip()
        return cur if v == "" else int(v)

    def ask_float(name, cur):
        v = input(f"{name} [{cur}]: ").strip()
        return cur if v == "" else float(v)

    def ask_bool(name, cur):
        d = "ON" if cur else "OFF"
        v = input(f"{name} (ON/OFF) [{d}]: ").strip().lower()
        if v == "":
            return cur
        return v in ("on", "1", "y", "yes", "true")

    print("Interactive mode (press Enter to keep default):")
    args.num_episodes = ask_int("num_episodes", args.num_episodes)
    args.gamma = ask_float("gamma", args.gamma)
    args.lam = ask_float("lam", args.lam)
    args.kl_targ = ask_float("kl_targ", args.kl_targ)
    args.batch_size = ask_int("batch_size", args.batch_size)
    args.device = input(f"device [cpu/cuda] [{args.device}]: ").strip() or args.device
    args.learn_delay = ask_bool("learn_delay", args.learn_delay)
    args.domain_randomization = ask_bool("domain_randomization", args.domain_randomization)
    return args


def evaluate_dataset(env: PendulumRLEnv, params: dict[str, float]):
    losses = []
    rmses = {"theta": [], "omega": [], "alpha": []}
    for traj in env.trajectories:
        d = params.get("delay_sec", traj.delay_sec_est)
        sim = simulate_trajectory(traj, params, env.cfg, delay_sec=d)
        feat = compute_error_features(traj, sim, align_shift_sec=0.0)
        losses.append(weighted_loss(feat, env.reward_weights))
        rmses["theta"].append(feat["rmse_theta"])
        rmses["omega"].append(feat["rmse_omega"])
        rmses["alpha"].append(feat["rmse_alpha"])
    return float(np.mean(losses)), {k: float(np.mean(v)) for k, v in rmses.items()}


def sanitize_metric(x: float):
    return float(x) if np.isfinite(x) else 0.0


def sanitize_dict(metrics: dict[str, float]):
    return {k: sanitize_metric(v) for k, v in metrics.items()}


def save_history_csv(history: dict, outpath: Path):
    keys = ["reward", "train_loss", "val_loss", "rmse_theta", "rmse_omega", "rmse_alpha",
            "val_rmse_theta", "val_rmse_omega", "val_rmse_alpha"]
    n = len(history.get("reward", []))
    with outpath.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["episode", *keys])
        for i in range(n):
            wr.writerow([i + 1, *[sanitize_metric(history.get(k, [0.0] * n)[i]) for k in keys]])


def save_best_replay_csv(outpath: Path, traj, sim: dict[str, np.ndarray], params: dict[str, float], loss: float, delay_sec: float, cfg: BridgeConfig):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(PENDULUM_LOG_COLUMNS)
        best_cost = float(loss)
        j_rod = (1.0 / 3.0) * cfg.rod_mass * (cfg.rod_length ** 2)
        j_imu = cfg.imu_mass * (cfg.r_imu ** 2)
        j_total = j_rod + j_imu
        for i in range(len(traj.t)):
            wr.writerow([
                0.0, traj.t[i], "replay",
                traj.cmd_u[i], sim["cmd_delayed"][i], traj.hw_pwm[i], traj.delay_sec_est, sim["tau_motor"][i] - sim["tau_res"][i],
                sim["theta"][i], sim["omega"][i], sim["alpha"][i],
                "", "",
                traj.theta_real[i], traj.omega_real[i], traj.alpha_real[i],
                delay_sec * 1000.0,
                params["l_com"], params["b_eq"], params["tau_eq"], params["K_u"],
                j_rod, j_imu, j_total,
                sim["tau_motor"][i], sim["tau_res"][i], sim["tau_visc"][i], sim["tau_coul"][i],
                loss, best_cost,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, sim["omega"][i],
                0.0, 0.0, 0.0,
                loss, 1, 1, json.dumps(params),
            ])


def main():
    ap = argparse.ArgumentParser(description="Offline PPO-style replay calibration for pendulum digital twin")
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--csv_dir", default="")
    ap.add_argument("--outdir", default="rl_calibration_out")

    ap.add_argument("-n", "--num_episodes", type=int, default=1000)
    ap.add_argument("-g", "--gamma", type=float, default=0.995)
    ap.add_argument("-l", "--lam", type=float, default=0.98)
    ap.add_argument("-k", "--kl_targ", type=float, default=0.003)
    ap.add_argument("-b", "--batch_size", type=int, default=20)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--learn_delay", action="store_true", default=False)
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--delay_jitter_ms", type=float, default=3.0)
    ap.add_argument("--domain_randomizationON", dest="domain_randomization", action="store_true")
    ap.add_argument("--domain_randomizationOFF", dest="domain_randomization", action="store_false")
    ap.set_defaults(domain_randomization=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_refine_steps", type=int, default=12)
    ap.add_argument("--log_every_episodes", type=int, default=10)
    ap.add_argument("--tensorboard_log", type=str, default="")

    args = maybe_prompt(ap.parse_args(), ap)

    csv_paths = gather_csv_paths(args.csv, args.csv_dir)
    if not csv_paths:
        raise SystemExit("No CSV files provided. Use --csv or --csv_dir.")

    cfg = BridgeConfig()
    calib = apply_calibration_json(cfg, args.calibration_json)
    param_data = None
    if args.parameter_json:
        with open(args.parameter_json, "r", encoding="utf-8") as f:
            param_data = json.load(f)

    init_params = build_init_params(cfg, calibration=calib, parameter_json=param_data)
    tr_paths, va_paths, te_paths = split_trajectories(csv_paths, seed=args.seed)

    train_traj = [load_replay_csv(p, cfg, delay_override=args.delay_override) for p in tr_paths]
    val_traj = [load_replay_csv(p, cfg, delay_override=args.delay_override) for p in va_paths]
    test_traj = [load_replay_csv(p, cfg, delay_override=args.delay_override) for p in te_paths]

    env = PendulumRLEnv(
        trajectories=train_traj,
        cfg=cfg,
        init_params=init_params,
        learn_delay=args.learn_delay,
        delay_jitter_ms=args.delay_jitter_ms,
        domain_randomization=args.domain_randomization,
        seed=args.seed,
        max_refine_steps=args.max_refine_steps,
    )
    val_env = PendulumRLEnv(
        trajectories=val_traj,
        cfg=cfg,
        init_params=init_params,
        learn_delay=args.learn_delay,
        delay_jitter_ms=0.0,
        domain_randomization=False,
        seed=args.seed + 1,
        max_refine_steps=args.max_refine_steps,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "initial_params.json", "w", encoding="utf-8") as f:
        json.dump(init_params, f, indent=2)

    history = {"reward": [], "train_loss": [], "val_loss": [], "rmse_theta": [], "rmse_omega": [], "rmse_alpha": [],
               "val_rmse_theta": [], "val_rmse_omega": [], "val_rmse_alpha": []}
    param_hist = {k: [] for k in env.param_keys}
    best_val, best_params = train_with_sb3(env, val_env, args, history, param_hist)
    print(f"[SB3] train_loss={history['train_loss'][-1]:.5f} val_loss={history['val_loss'][-1]:.5f}")

    if not args.learn_delay:
        best_params.pop("delay_sec", None)

    with open(outdir / "final_params_rl.json", "w", encoding="utf-8") as f:
        json.dump({"model_init": best_params, "best_validation_loss": best_val}, f, indent=2)

    delay_map = {t.name: float(t.delay_sec_est) for t in train_traj + val_traj + test_traj}
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "calibration_json": args.calibration_json,
        "cpr": None if not np.isfinite(cfg.cpr) else float(cfg.cpr),
        "dataset_files": [str(p) for p in csv_paths],
        "reward_weights": env.reward_weights,
        "randomization": {
            "domain_randomization": args.domain_randomization,
            "delay_jitter_ms": args.delay_jitter_ms,
        },
        "best_validation_score": best_val,
        "delay_estimates_sec": delay_map,
        "tensorboard_log": args.tensorboard_log if args.tensorboard_log else str(outdir / "tensorboard"),
        "sb3_monitor_csv": str(outdir / "sb3_monitor.csv"),
    }
    with open(outdir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(outdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    save_history_csv(history, outdir / "history.csv")

    plot_training_curves(history, outdir)
    plot_param_convergence(param_hist, outdir)
    plot_delay_diagnostics(delay_map, outdir)
    plot_rl_dashboard(history, param_hist, outdir)

    # Representative overlay from first test trajectory
    rep = test_traj[0]
    rep_delay = best_params.get("delay_sec", rep.delay_sec_est)
    sim = simulate_trajectory(rep, best_params, cfg, delay_sec=rep_delay)
    rep_feat = compute_error_features(rep, sim, align_shift_sec=0.0)
    rep_loss = weighted_loss(rep_feat, env.reward_weights)
    save_best_replay_csv(outdir / "replay_best.csv", rep, sim, best_params, rep_loss, rep_delay, cfg)
    plot_overlay(rep.t, rep.theta_real, sim["theta"], "theta [rad]", outdir / "overlay_theta.png")
    plot_overlay(rep.t, rep.omega_real, sim["omega"], "omega [rad/s]", outdir / "overlay_omega.png")
    plot_overlay(rep.t, rep.alpha_real, sim["alpha"], "alpha [rad/s^2]", outdir / "overlay_alpha.png")
    plot_overlay(rep.t, rep.hw_pwm, sim["cmd_delayed"], "PWM", outdir / "overlay_pwm_aligned.png")

    print("Saved outputs in", outdir)
    print(f"[INFO] SB3 Monitor CSV: {outdir / 'sb3_monitor.csv'}")
    print(f"[INFO] TensorBoard logdir: {args.tensorboard_log if args.tensorboard_log else str(outdir / 'tensorboard')}")
    print(f"[INFO] TensorBoard run: tensorboard --logdir \"{args.tensorboard_log if args.tensorboard_log else str(outdir / 'tensorboard')}\"")


if __name__ == "__main__":
    main()
