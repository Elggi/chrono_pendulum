#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
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


class GaussianPolicy:
    """Small numpy policy used for PPO-style episodic updates."""

    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0.0, 0.05, size=(act_dim, obs_dim))
        self.b = np.zeros(act_dim, dtype=float)
        self.log_std = np.full(act_dim, -0.5, dtype=float)

    def mean(self, obs: np.ndarray):
        return np.tanh(self.W @ obs + self.b)

    def sample(self, obs: np.ndarray):
        mu = self.mean(obs)
        std = np.exp(self.log_std)
        act = mu + std * self.rng.normal(size=mu.shape)
        return np.clip(act, -1.0, 1.0), mu, std

    def log_prob(self, a: np.ndarray, mu: np.ndarray, std: np.ndarray):
        z = (a - mu) / (std + 1e-9)
        return -0.5 * np.sum(z * z + 2.0 * np.log(std + 1e-9) + np.log(2 * np.pi))

    def ppo_update(self, batch, lr: float = 1e-2, clip_eps: float = 0.2):
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dls = np.zeros_like(self.log_std)
        kl_terms = []
        for item in batch:
            obs, act, adv, old_mu, old_std, old_lp = item
            mu = self.mean(obs)
            std = np.exp(self.log_std)
            lp = self.log_prob(act, mu, std)
            ratio = np.exp(lp - old_lp)
            ratio_c = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            w = ratio if abs(ratio - 1.0) < abs(ratio_c - 1.0) else ratio_c

            grad_lp_mu = (act - mu) / (std ** 2 + 1e-9)
            grad_mu_raw = (1.0 - mu ** 2)
            dW += np.outer(grad_lp_mu * grad_mu_raw * adv * w, obs)
            db += grad_lp_mu * grad_mu_raw * adv * w
            dls += ((act - mu) ** 2 / (std ** 2 + 1e-9) - 1.0) * adv * w
            kl_terms.append(np.mean(np.log(std / (old_std + 1e-9)) + (old_std ** 2 + (old_mu - mu) ** 2) / (2 * std ** 2 + 1e-9) - 0.5))

        n = max(len(batch), 1)
        self.W += lr * dW / n
        self.b += lr * db / n
        self.log_std += lr * 0.1 * dls / n
        self.log_std = np.clip(self.log_std, -3.0, 0.7)
        return float(np.mean(kl_terms)) if kl_terms else 0.0


def deterministic_prefit(env: PendulumRLEnv, iters: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = env.best_params.copy()
    best_loss = env.best_loss
    keys = list(env.param_keys)

    for _ in range(iters):
        cand = best.copy()
        k = keys[int(rng.integers(0, len(keys)))]
        lo, hi = env.bounds[k]
        cand[k] = float(np.clip(cand[k] + rng.normal(0.0, 0.08 * (hi - lo)), lo, hi))
        loss, _ = env._rollout_loss(cand)
        if loss < best_loss:
            best, best_loss = cand, loss
    return best, best_loss


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
    args.prefit = ask_bool("prefit", args.prefit)
    args.learn_delay = ask_bool("learn_delay", args.learn_delay)
    args.domain_randomization = ask_bool("domain_randomization", args.domain_randomization)
    return args


def evaluate_dataset(env: PendulumRLEnv, params: dict[str, float]):
    losses = []
    rmses = {"theta": [], "omega": [], "alpha": []}
    for traj in env.trajectories:
        d = params.get("delay_sec", traj.delay_sec_est)
        sim = simulate_trajectory(traj, params, env.cfg, delay_sec=d)
        feat = compute_error_features(traj, sim)
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


def save_best_replay_csv(outpath: Path, traj, sim: dict[str, np.ndarray], params: dict[str, float], loss: float, delay_sec: float):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(PENDULUM_LOG_COLUMNS)
        best_cost = float(loss)
        for i in range(len(traj.t)):
            wr.writerow([
                0.0, traj.t[i], "replay",
                traj.cmd_u[i], sim["cmd_delayed"][i], traj.hw_pwm[i], traj.delay_sec_est, sim["tau_motor"][i] - sim["tau_res"][i],
                sim["theta"][i], sim["omega"][i], sim["alpha"][i],
                "", "",
                traj.theta_real[i], traj.omega_real[i], traj.alpha_real[i],
                delay_sec * 1000.0,
                params["l_com"], params["b_eq"], params["tau_eq"], params["k_t"], params["i0"], params["R"], params["k_e"],
                traj.bus_v[i], traj.bus_v[i], traj.current_a[i], traj.current_a[i], traj.power_w[i],
                sim["tau_motor"][i], sim["tau_res"][i], sim["tau_visc"][i], sim["tau_coul"][i], sim["i_pred"][i], sim["v_applied"][i],
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

    ap.add_argument("--prefitON", dest="prefit", action="store_true")
    ap.add_argument("--prefitOFF", dest="prefit", action="store_false")
    ap.set_defaults(prefit=True)
    ap.add_argument("--learn_delay", action="store_true", default=False)
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--delay_jitter_ms", type=float, default=3.0)
    ap.add_argument("--domain_randomizationON", dest="domain_randomization", action="store_true")
    ap.add_argument("--domain_randomizationOFF", dest="domain_randomization", action="store_false")
    ap.set_defaults(domain_randomization=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_refine_steps", type=int, default=12)

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

    if args.prefit:
        env.reset()
        pref_params, pref_loss = deterministic_prefit(env, iters=120, seed=args.seed)
        with open(outdir / "prefit_params.json", "w", encoding="utf-8") as f:
            json.dump({"loss": pref_loss, "model_init": pref_params}, f, indent=2)
        init_params.update(pref_params)
        env.center.update(pref_params)
        val_env.center.update(pref_params)

    policy = GaussianPolicy(obs_dim=env.state_dim, act_dim=env.action_dim, seed=args.seed)

    history = {"reward": [], "train_loss": [], "val_loss": [], "rmse_theta": [], "rmse_omega": [], "rmse_alpha": [],
               "val_rmse_theta": [], "val_rmse_omega": [], "val_rmse_alpha": []}
    param_hist = {k: [] for k in env.param_keys}
    best_val = float("inf")
    best_params = dict(init_params)

    for ep in range(1, args.num_episodes + 1):
        obs = env.reset()
        traj_batch = []
        ep_reward = 0.0
        done = False
        while not done:
            act, mu, std = policy.sample(obs)
            old_lp = policy.log_prob(act, mu, std)
            nobs, rew, done, info = env.step(act)
            traj_batch.append((obs, act, rew, mu, std, old_lp))
            ep_reward += rew
            obs = nobs

        rews = np.array([x[2] for x in traj_batch], dtype=float)
        adv = (rews - np.mean(rews)) / (np.std(rews) + 1e-9)
        batch = []
        for i, item in enumerate(traj_batch):
            batch.append((item[0], item[1], adv[i], item[3], item[4], item[5]))
        kl = policy.ppo_update(batch, lr=8e-3)
        if kl > args.kl_targ * 2.0:
            policy.ppo_update(batch, lr=3e-3)

        train_loss, train_rmse = evaluate_dataset(env, env.best_params)
        val_loss, val_rmse = evaluate_dataset(val_env, env.best_params)
        ep_reward = sanitize_metric(ep_reward)
        train_loss = sanitize_metric(train_loss)
        val_loss = sanitize_metric(val_loss)
        train_rmse = sanitize_dict(train_rmse)
        val_rmse = sanitize_dict(val_rmse)

        history["reward"].append(float(ep_reward))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["rmse_theta"].append(train_rmse["theta"])
        history["rmse_omega"].append(train_rmse["omega"])
        history["rmse_alpha"].append(train_rmse["alpha"])
        history["val_rmse_theta"].append(val_rmse["theta"])
        history["val_rmse_omega"].append(val_rmse["omega"])
        history["val_rmse_alpha"].append(val_rmse["alpha"])

        for k in env.param_keys:
            param_hist[k].append(float(env.best_params[k]))

        if val_loss < best_val:
            best_val = val_loss
            best_params = env.best_params.copy()

        if ep % max(1, args.batch_size) == 0:
            print(f"ep={ep:4d} reward={ep_reward: .4f} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

    if not args.learn_delay:
        best_params.pop("delay_sec", None)

    with open(outdir / "final_params_rl.json", "w", encoding="utf-8") as f:
        json.dump({"model_init": best_params, "best_validation_loss": best_val}, f, indent=2)

    delay_map = {t.name: float(t.delay_sec_est) for t in train_traj + val_traj + test_traj}
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "dataset_files": [str(p) for p in csv_paths],
        "reward_weights": env.reward_weights,
        "randomization": {
            "domain_randomization": args.domain_randomization,
            "delay_jitter_ms": args.delay_jitter_ms,
        },
        "best_validation_score": best_val,
        "delay_estimates_sec": delay_map,
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
    rep_feat = compute_error_features(rep, sim)
    rep_loss = weighted_loss(rep_feat, env.reward_weights)
    save_best_replay_csv(outdir / "replay_best.csv", rep, sim, best_params, rep_loss, rep_delay)
    plot_overlay(rep.t, rep.theta_real, sim["theta"], "theta [rad]", outdir / "overlay_theta.png")
    plot_overlay(rep.t, rep.omega_real, sim["omega"], "omega [rad/s]", outdir / "overlay_omega.png")
    plot_overlay(rep.t, rep.alpha_real, sim["alpha"], "alpha [rad/s^2]", outdir / "overlay_alpha.png")
    plot_overlay(rep.t, rep.hw_pwm, sim["cmd_delayed"], "PWM", outdir / "overlay_pwm_aligned.png")

    print("Saved outputs in", outdir)


if __name__ == "__main__":
    main()
