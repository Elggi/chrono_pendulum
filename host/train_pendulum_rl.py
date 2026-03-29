#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from chrono_core.calibration_io import apply_calibration_json
from chrono_core.config import BridgeConfig
from chrono_core.pendulum_rl_env import (
    RL_PARAM_KEYS,
    PendulumReplayCalibrationEnv,
    deterministic_prefit,
    load_replay_trajectories,
    params_from_cfg_and_json,
    replay_loss,
    simulate_replay,
    split_trajectories,
)
from chrono_core.pendulum_rl_plots import (
    plot_delay_diagnostics,
    plot_param_convergence,
    plot_rmse_curves,
    plot_training_curves,
)


def prompt_or_default(name: str, current, cast):
    s = input(f"{name} [{current}]: ").strip()
    return current if s == "" else cast(s)


def parse_args():
    ap = argparse.ArgumentParser(description="Offline RL replay calibration for 1-DOF pendulum digital twin")
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default=None)
    ap.add_argument("--csv", action="append", default=[])
    ap.add_argument("--csv_dir", default=None)
    ap.add_argument("--outdir", default="host/rl_outputs")

    ap.add_argument("-n", "--num_episodes", type=int, default=None)
    ap.add_argument("--renderON", dest="render", action="store_true")
    ap.add_argument("--renderOFF", dest="render", action="store_false")
    ap.set_defaults(render=False)
    ap.add_argument("-g", "--gamma", type=float, default=None)
    ap.add_argument("-l", "--lam", type=float, default=None)
    ap.add_argument("-k", "--kl_targ", type=float, default=None)
    ap.add_argument("-b", "--batch_size", type=int, default=None)

    ap.add_argument("--prefitON", dest="prefit", action="store_true")
    ap.add_argument("--prefitOFF", dest="prefit", action="store_false")
    ap.set_defaults(prefit=True)
    ap.add_argument("--learn_delay", action="store_true")
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--delay_jitter_ms", type=float, default=3.0)
    ap.add_argument("--domain_randomizationON", dest="dr", action="store_true")
    ap.add_argument("--domain_randomizationOFF", dest="dr", action="store_false")
    ap.set_defaults(dr=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interactive", action="store_true")
    return ap.parse_args()


def collect_csv_paths(args) -> list[Path]:
    paths = [Path(p) for p in args.csv]
    if args.csv_dir:
        paths.extend(sorted(Path(args.csv_dir).glob("*.csv")))
    if not paths:
        raise ValueError("No CSV files provided. Use --csv or --csv_dir")
    return paths


def dataset_metrics(trajs, params):
    losses = []
    agg = {"rmse_theta": [], "rmse_omega": [], "rmse_alpha": []}
    for tr in trajs:
        sim = simulate_replay(tr.df, params, tr.delay_sec_est)
        r = replay_loss(tr.df, sim)
        losses.append(r.loss)
        for k in agg:
            agg[k].append(r.metrics[k])
    return float(np.mean(losses)), {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    args = parse_args()

    defaults = {"num_episodes": 1000, "gamma": 0.995, "lam": 0.98, "kl_targ": 0.003, "batch_size": 20}
    if args.interactive:
        print("[interactive mode] press Enter to keep default")
        for k, v in defaults.items():
            if getattr(args, k) is None:
                setattr(args, k, prompt_or_default(k, v, type(v)))
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = BridgeConfig()
    apply_calibration_json(cfg, args.calibration_json)
    params = params_from_cfg_and_json(cfg, args.parameter_json)
    csv_paths = collect_csv_paths(args)
    train_f, val_f, test_f = split_trajectories(csv_paths, seed=args.seed)

    train_traj, delay_map = load_replay_trajectories(train_f, delay_override=args.delay_override)
    val_traj, _ = load_replay_trajectories(val_f, delay_override=args.delay_override)
    test_traj, _ = load_replay_trajectories(test_f, delay_override=args.delay_override)

    (outdir / "params_initial.json").write_text(json.dumps(params, indent=2))

    if args.prefit:
        params = deterministic_prefit(train_traj, params, iters=100, seed=args.seed)
        (outdir / "params_prefit.json").write_text(json.dumps(params, indent=2))

    env = PendulumReplayCalibrationEnv(
        trajectories=train_traj,
        init_params=params,
        learn_delay=args.learn_delay,
        delay_jitter_ms=args.delay_jitter_ms,
        domain_randomization=args.dr,
        max_refine_steps=max(5, args.batch_size),
        seed=args.seed,
    )

    model = PPO("MlpPolicy", env, verbose=0, gamma=args.gamma, batch_size=args.batch_size,
                n_steps=max(64, args.batch_size * 4), seed=args.seed)

    history = []
    best_val = float("inf")
    best_params = dict(params)

    for ep in range(1, args.num_episodes + 1):
        model.learn(total_timesteps=env.max_refine_steps, reset_num_timesteps=False)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(act)
            ep_reward += float(r)

        candidate = {k: float(v) for k, v in zip(env.param_keys, env.params)}
        train_loss, trm = dataset_metrics(train_traj, candidate)
        val_loss, vm = dataset_metrics(val_traj, candidate)
        if val_loss < best_val:
            best_val = val_loss
            best_params = dict(candidate)
        row = {
            "episode": ep,
            "reward": ep_reward,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "rmse_theta": trm["rmse_theta"],
            "rmse_omega": trm["rmse_omega"],
            "rmse_alpha": trm["rmse_alpha"],
            "val_rmse_theta": vm["rmse_theta"],
            "val_rmse_omega": vm["rmse_omega"],
            "val_rmse_alpha": vm["rmse_alpha"],
        }
        for k in RL_PARAM_KEYS:
            row[f"param_{k}"] = candidate.get(k, np.nan)
        if args.learn_delay:
            row["param_delay_sec"] = candidate.get("delay_sec", np.nan)
        history.append(row)

    test_loss, test_metrics = dataset_metrics(test_traj, best_params)

    (outdir / "params_final_rl.json").write_text(json.dumps(best_params, indent=2))
    (outdir / "training_history.json").write_text(json.dumps(history, indent=2))
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "dataset": {"train": [str(p) for p in train_f], "val": [str(p) for p in val_f], "test": [str(p) for p in test_f]},
        "reward_weights": {"theta": 4.0, "omega": 2.0, "alpha": 1.0, "pwm": 0.05},
        "randomization": {"enabled": args.dr, "delay_jitter_ms": args.delay_jitter_ms},
        "best_validation_score": best_val,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "delay_estimates_sec": delay_map,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    plot_training_curves(history, outdir)
    plot_rmse_curves(history, outdir)
    plot_param_convergence(history, outdir)
    plot_delay_diagnostics(delay_map, outdir)

    print(json.dumps({"best_val": best_val, "test_loss": test_loss, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
