#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from chrono_core.pendulum_rl_env import (
    PARAM_NAMES_BASE,
    PendulumRLEnv,
    chunk_single_csv,
    initial_params_from_files,
    load_trajectories,
    simulate_trajectory,
    split_trajectories,
)
from chrono_core.pendulum_rl_plots import (
    plot_delay_diagnostics,
    plot_overlay,
    plot_parameter_convergence,
    plot_training_curves,
)


def maybe_prompt(value, text, cast):
    if value is not None:
        return value
    raw = input(f"{text} [default={cast.__name__ if hasattr(cast,'__name__') else ''}]: ").strip()
    if raw == "":
        return None
    return cast(raw)


def bool_prompt_if_none(v, text, default):
    if v is not None:
        return v
    raw = input(f"{text} (y/n, default={'y' if default else 'n'}): ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "1", "on", "true")


def discover_csvs(csv, csv_dir):
    paths = []
    if csv:
        paths.append(csv)
    if csv_dir:
        paths.extend(str(p) for p in sorted(Path(csv_dir).glob("*.csv")))
    uniq = []
    for p in paths:
        if p not in uniq:
            uniq.append(p)
    return uniq


def prefit_random_search(env: PendulumRLEnv, iters: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = dict(env.base_params)
    best["delay_offset_sec"] = 0.0
    env.params = dict(best)
    best_metrics = env._eval_set(env.train_trajectories, progress=0.0)

    for _ in range(iters):
        cand = dict(best)
        for k in PARAM_NAMES_BASE:
            cand[k] = cand[k] * (1.0 + rng.uniform(-0.12, 0.12))
        if env.learn_delay:
            cand["delay_offset_sec"] = float(rng.uniform(-0.015, 0.015))
        env.params = cand
        env._clip_params()
        m = env._eval_set(env.train_trajectories, progress=0.0)
        if m["weighted_loss"] < best_metrics["weighted_loss"]:
            best = dict(env.params)
            best_metrics = m

    env.base_params.update({k: best[k] for k in PARAM_NAMES_BASE})
    if env.learn_delay:
        env.base_params["delay_offset_sec"] = best.get("delay_offset_sec", 0.0)
    return best, best_metrics


def build_torch_agent(state_dim, action_dim):
    import torch
    import torch.nn as nn

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, action_dim))
            self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.7)
            self.v = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

        def dist(self, s):
            mu = self.pi(s)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mu, std)

    return Agent()


def train_ppo(env: PendulumRLEnv, num_episodes: int, gamma: float, lam: float, kl_targ: float, batch_size: int, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = build_torch_agent(env.state_dim, env.action_dim)
    optim = torch.optim.Adam(agent.parameters(), lr=3e-4)

    history = {
        "episode_reward": [], "train_loss": [], "val_loss": [],
        "rmse_theta": [], "rmse_omega": [], "rmse_alpha": [],
        "val_rmse_theta": [], "val_rmse_omega": [], "val_rmse_alpha": [],
    }
    param_history = []

    for ep in range(num_episodes):
        batch = []
        ep_rewards = []
        for _ in range(batch_size):
            s = env.reset()
            done = False
            traj = []
            while not done:
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    dist = agent.dist(st)
                    a = dist.sample().squeeze(0).cpu().numpy()
                    logp = dist.log_prob(torch.tensor(a, dtype=torch.float32)).sum().item()
                    v = agent.v(st).item()
                a = np.clip(a, -1.0, 1.0)
                s2, r, done, info = env.step(a)
                traj.append((s, a, r, logp, v, done))
                s = s2
                ep_rewards.append(r)

            # GAE
            adv, ret = [], []
            gae = 0.0
            next_v = 0.0
            for t in reversed(range(len(traj))):
                _, _, r, _, v, d = traj[t]
                delta = r + gamma * next_v * (1.0 - float(d)) - v
                gae = delta + gamma * lam * (1.0 - float(d)) * gae
                adv.append(gae)
                ret.append(gae + v)
                next_v = v
            adv.reverse(); ret.reverse()
            for i, row in enumerate(traj):
                s0, a0, _, lp0, _, _ = row
                batch.append((s0, a0, lp0, ret[i], adv[i]))

        s_b = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
        a_b = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32)
        lp_old = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32)
        ret_b = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32)
        adv_b = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        for _ in range(8):
            dist = agent.dist(s_b)
            lp = dist.log_prob(a_b).sum(dim=1)
            ratio = torch.exp(lp - lp_old)
            clip = 0.2
            obj1 = ratio * adv_b
            obj2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_b
            pi_loss = -torch.min(obj1, obj2).mean()

            v_pred = agent.v(s_b).squeeze(1)
            v_loss = ((v_pred - ret_b) ** 2).mean()
            entropy = dist.entropy().sum(dim=1).mean()

            loss = pi_loss + 0.5 * v_loss - 0.005 * entropy
            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                kl = (lp_old - lp).mean().abs().item()
            if kl > 1.5 * kl_targ:
                break

        trm = env._eval_set(env.train_trajectories, progress=1.0)
        vm = env._eval_set(env.val_trajectories, progress=1.0) if env.val_trajectories else trm

        history["episode_reward"].append(float(np.mean(ep_rewards) if ep_rewards else 0.0))
        history["train_loss"].append(trm["weighted_loss"])
        history["val_loss"].append(vm["weighted_loss"])
        history["rmse_theta"].append(trm["rmse_theta"])
        history["rmse_omega"].append(trm["rmse_omega"])
        history["rmse_alpha"].append(trm["rmse_alpha"])
        history["val_rmse_theta"].append(vm["rmse_theta"])
        history["val_rmse_omega"].append(vm["rmse_omega"])
        history["val_rmse_alpha"].append(vm["rmse_alpha"])

        param_history.append(dict(env.params))
        if (ep + 1) % max(1, num_episodes // 10) == 0:
            print(f"[EP {ep+1}/{num_episodes}] reward={history['episode_reward'][-1]:.4f} train_loss={trm['weighted_loss']:.4f} val={vm['weighted_loss']:.4f}")

    return env.params, history, param_history


def to_param_json(params: dict):
    return {
        "model_init": {
            "l_com": float(params["l_com"]),
            "J_cm_base": float(params["J_cm_base"]),
            "b_eq": float(params["b_eq"]),
            "tau_eq": float(params["tau_eq"]),
            "k_t": float(params["k_t"]),
            "i0": float(params["i0"]),
            "R": float(params["R"]),
            "k_e": float(params["k_e"]),
        }
    }


def main():
    ap = argparse.ArgumentParser(description="Offline RL replay calibration for pendulum digital twin")
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csv_dir", default=None)
    ap.add_argument("--out_dir", default="./run_logs/rl_calibration")
    ap.add_argument("--chunk_single_csv_sec", type=float, default=8.0)

    ap.add_argument("-n", "--num_episodes", type=int, default=1000)
    ap.add_argument("--renderON", action="store_true")
    ap.add_argument("--renderOFF", action="store_true")
    ap.add_argument("-g", "--gamma", type=float, default=0.995)
    ap.add_argument("-l", "--lam", type=float, default=0.98)
    ap.add_argument("-k", "--kl_targ", type=float, default=0.003)
    ap.add_argument("-b", "--batch_size", type=int, default=20)

    ap.add_argument("--prefitON", action="store_true")
    ap.add_argument("--prefitOFF", action="store_true")
    ap.add_argument("--learn_delay", action="store_true")
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--delay_jitter_ms", type=float, default=2.0)
    ap.add_argument("--domain_randomizationON", action="store_true")
    ap.add_argument("--domain_randomizationOFF", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        args.num_episodes = int(input(f"num_episodes [{args.num_episodes}]: ") or args.num_episodes)
        args.gamma = float(input(f"gamma [{args.gamma}]: ") or args.gamma)
        args.lam = float(input(f"lam [{args.lam}]: ") or args.lam)
        args.kl_targ = float(input(f"kl_targ [{args.kl_targ}]: ") or args.kl_targ)
        args.batch_size = int(input(f"batch_size [{args.batch_size}]: ") or args.batch_size)

    prefit = (not args.prefitOFF) if not args.prefitON else True
    dom_rand = (not args.domain_randomizationOFF) if not args.domain_randomizationON else True

    csv_paths = discover_csvs(args.csv, args.csv_dir)
    if not csv_paths:
        raise ValueError("At least one --csv or --csv_dir/*.csv is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(csv_paths) == 1:
        chunk_dir = out_dir / "chunks"
        csv_paths = chunk_single_csv(csv_paths[0], str(chunk_dir), chunk_sec=args.chunk_single_csv_sec)

    train_paths, val_paths, test_paths = split_trajectories(csv_paths, seed=args.seed)
    train_tr = load_trajectories(train_paths, delay_override=args.delay_override)
    val_tr = load_trajectories(val_paths, delay_override=args.delay_override) if val_paths else []
    test_tr = load_trajectories(test_paths, delay_override=args.delay_override) if test_paths else []

    base_params, cfg = initial_params_from_files(args.calibration_json, args.parameter_json)

    env = PendulumRLEnv(
        train_trajectories=train_tr,
        val_trajectories=val_tr,
        base_params=base_params,
        cfg=cfg,
        learn_delay=args.learn_delay,
        delay_jitter_ms=args.delay_jitter_ms,
        domain_randomization=dom_rand,
        seed=args.seed,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "initial_params.json", "w", encoding="utf-8") as f:
        json.dump(to_param_json(base_params), f, indent=2)

    if prefit:
        pbest, pmetrics = prefit_random_search(env, iters=100, seed=args.seed)
        with open(run_dir / "prefit_params.json", "w", encoding="utf-8") as f:
            json.dump(to_param_json(pbest), f, indent=2)
        print(f"[prefit] weighted_loss={pmetrics['weighted_loss']:.6f}")

    final_params, history, param_history = train_ppo(
        env,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        lam=args.lam,
        kl_targ=args.kl_targ,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    with open(run_dir / "final_params.json", "w", encoding="utf-8") as f:
        json.dump(to_param_json(final_params), f, indent=2)

    meta = {
        "timestamp": ts,
        "training_settings": {
            "num_episodes": args.num_episodes,
            "gamma": args.gamma,
            "lam": args.lam,
            "kl_targ": args.kl_targ,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "prefit": prefit,
            "learn_delay": bool(args.learn_delay),
            "delay_jitter_ms": float(args.delay_jitter_ms),
            "domain_randomization": dom_rand,
        },
        "dataset_files": {"train": train_paths, "val": val_paths, "test": test_paths},
        "reward_weights": env.weights.__dict__,
        "randomization_ranges": {
            "theta0_offset": "±0.08 rad (curriculum)",
            "omega0_offset": "±0.4 rad/s (curriculum)",
            "delay_jitter_ms": f"±{args.delay_jitter_ms}",
            "pwm_scale": "±6% (curriculum)",
            "bus_scale": "±3% (curriculum)",
            "friction_scale": "±8% (curriculum)",
        },
        "best_validation_score": float(min(history["val_loss"]) if history["val_loss"] else min(history["train_loss"])),
        "per_trajectory_delay_estimates": [{"csv": tr.path, "delay_sec": tr.delay_est, "quality_corr": tr.delay_quality_corr} for tr in (train_tr + val_tr + test_tr)],
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, str(run_dir))
    plot_parameter_convergence(param_history, str(run_dir))
    plot_delay_diagnostics(train_tr + val_tr + test_tr, str(run_dir))

    representative = (val_tr[0] if val_tr else train_tr[0]) if train_tr else None
    if representative is not None:
        delay = representative.delay_est + (final_params.get("delay_offset_sec", 0.0) if args.learn_delay else 0.0)
        sim = simulate_trajectory(representative, final_params, cfg, max(delay, 0.0), randomization=None)
        plot_overlay(representative, sim, str(run_dir / "overlay_representative.png"))

    print(f"Done. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
