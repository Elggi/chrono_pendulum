#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_env.chrono_param_env import ChronoParamEnv
from rl_utils.data_loader import load_from_csvs, load_from_npz
from rl_utils.reward import RewardWeights


class ProgressCallback(BaseCallback):
    def __init__(self, log_csv: Path):
        super().__init__()
        self.log_csv = log_csv
        self.rows: list[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue
            row = {
                "timesteps": int(self.num_timesteps),
                "reward": float(info.get("reward", 0.0)),
                "rmse_theta": float(info.get("rmse_theta", 0.0)),
                "rmse_omega": float(info.get("rmse_omega", 0.0)),
                "rmse_alpha": float(info.get("rmse_alpha", 0.0)),
                "J": float(info.get("params", {}).get("J", 0.0)) if isinstance(info.get("params"), dict) else 0.0,
                "b_eq": float(info.get("params", {}).get("b_eq", 0.0)) if isinstance(info.get("params"), dict) else 0.0,
                "tau_eq": float(info.get("params", {}).get("tau_eq", 0.0)) if isinstance(info.get("params"), dict) else 0.0,
                "K_I": float(info.get("params", {}).get("K_I", 0.0)) if isinstance(info.get("params"), dict) else 0.0,
            }
            self.rows.append(row)
            if len(self.rows) % 10 == 0:
                print(
                    f"[Step {self.num_timesteps}] Reward={row['reward']:.4f} "
                    f"RMSE(theta/omega/alpha)=({row['rmse_theta']:.4f}, {row['rmse_omega']:.4f}, {row['rmse_alpha']:.4f}) "
                    f"Params J={row['J']:.4f} b={row['b_eq']:.4f} tau0={row['tau_eq']:.4f} KI={row['K_I']:.4f}"
                )
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.log_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def make_env_fn(trajs, motor_json: str, calib_json: str, weights: RewardWeights):
    def _f():
        return ChronoParamEnv(trajs, motor_json, calib_json, reward_weights=weights)

    return _f


def plot_logs(log_csv: Path, out_dir: Path) -> None:
    import pandas as pd

    if not log_csv.exists():
        return
    df = pd.read_csv(log_csv)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["timesteps"], df["reward"])
    ax.set_title("Reward curve")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("reward")
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png")
    plt.close(fig)

    param_cols = ["J", "b_eq", "tau_eq", "K_I"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in param_cols:
        ax.plot(df["timesteps"], df[c], label=c)
    ax.legend(); ax.set_title("Parameter evolution")
    ax.set_xlabel("timesteps")
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_evolution.png")
    plt.close(fig)

    df[["timesteps", *param_cols]].to_csv(out_dir / "parameter_evolution.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PPO parameter finetuning with Chrono solver")
    ap.add_argument("--csvs", nargs="*", default=[])
    ap.add_argument("--npz", default="")
    ap.add_argument("--motor_torque_json", required=True)
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--total_timesteps", type=int, default=2000)
    ap.add_argument("--outdir", default="host/rl_results/ppo")
    ap.add_argument("--w_theta", type=float, default=1.0)
    ap.add_argument("--w_omega", type=float, default=1.0)
    ap.add_argument("--w_alpha", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.npz:
        trajs = load_from_npz(args.npz)
        sources = [args.npz]
    else:
        if not args.csvs:
            raw = input("CSV paths for PPO training (comma-separated): ").strip()
            args.csvs = [s.strip() for s in raw.split(",") if s.strip()]
        trajs = load_from_csvs(args.csvs)
        sources = [str(Path(c).resolve()) for c in args.csvs]

    weights = RewardWeights(w_theta=args.w_theta, w_omega=args.w_omega, w_alpha=args.w_alpha)

    env_fns = [make_env_fn(trajs, args.motor_torque_json, args.calibration_json, weights) for _ in range(max(1, args.num_envs))]
    vec_env = DummyVecEnv(env_fns) if args.num_envs == 1 else SubprocVecEnv(env_fns)

    log_csv = out_dir / "ppo_training_log.csv"
    cb = ProgressCallback(log_csv)
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=64, batch_size=64)
    model.learn(total_timesteps=args.total_timesteps, callback=cb)

    eval_env = ChronoParamEnv(trajs, args.motor_torque_json, args.calibration_json, reward_weights=weights)
    obs, _ = eval_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, info = eval_env.step(action)

    # update motor_torque canonical output
    motor_path = Path(args.motor_torque_json)
    motor = json.loads(motor_path.read_text(encoding="utf-8"))
    motor.setdefault("dynamic_parameters", {}).update(info["params"])
    motor.setdefault("identified_models", {})
    motor["identified_models"]["ppo_finetuning"] = {
        "final_parameters": info["params"],
        "best_reward": float(info.get("best_reward", reward)),
        "training_steps": int(args.total_timesteps),
        "source_data": sources,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fit_metrics": {
            "rmse_theta": float(info.get("rmse_theta", 0.0)),
            "rmse_omega": float(info.get("rmse_omega", 0.0)),
            "rmse_alpha": float(info.get("rmse_alpha", 0.0)),
        },
        "equation": "J*alpha = K_I*I - b_eq*omega - tau_eq*tanh(omega/eps) - m*g*l_com*sin(theta)",
    }
    motor_path.write_text(json.dumps(motor, indent=2), encoding="utf-8")

    # simple best sim-vs-real plot from first trajectory
    tr = trajs[0]
    th, om, al = eval_env._simulate_with_params(tr, np.array([info["params"]["J"], info["params"]["b_eq"], info["params"]["tau_eq"], info["params"]["K_I"]]))
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(tr.t, tr.theta, label="theta_real"); ax[0].plot(tr.t, th, "--", label="theta_sim"); ax[0].legend()
    ax[1].plot(tr.t, tr.omega, label="omega_real"); ax[1].plot(tr.t, om, "--", label="omega_sim"); ax[1].legend()
    ax[2].plot(tr.t, tr.alpha, label="alpha_real"); ax[2].plot(tr.t, al, "--", label="alpha_sim"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out_dir / "best_sim_vs_real_plot.png"); plt.close(fig)

    plot_logs(log_csv, out_dir)

    model.save(str(out_dir / "ppo_param_finetuner"))
    print(json.dumps({"best_reward": reward, "params": info["params"], "outdir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
