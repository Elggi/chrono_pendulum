#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from chrono_core.config import BridgeConfig
from chrono_core.pendulum_rl_env import PendulumRLEnv, shift_with_delay
from chrono_core.pendulum_rl_plots import save_delay_plots, save_overlay_plot, save_training_plots
from chrono_core.replay_io import (
    PARAM_NAMES_DEFAULT,
    build_params_from_calibration,
    gather_csv_paths,
    load_replay_trajectory,
)


def prompt_if_none(v, text, caster, default):
    if v is not None:
        return v
    raw = input(f"{text} [{default}]: ").strip()
    if raw == "":
        return default
    return caster(raw)


def parse_args():
    ap = argparse.ArgumentParser(description="Offline RL replay calibration for 1-DOF pendulum")
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--csv_dir", default="")
    ap.add_argument("--outdir", default="host/rl_runs")

    ap.add_argument("-n", "--num_episodes", type=int)
    ap.add_argument("--renderON", action="store_true")
    ap.add_argument("--renderOFF", action="store_true")
    ap.add_argument("-g", "--gamma", type=float)
    ap.add_argument("-l", "--lam", type=float)
    ap.add_argument("-k", "--kl_targ", type=float)
    ap.add_argument("-b", "--batch_size", type=int)

    ap.add_argument("--prefitON", action="store_true")
    ap.add_argument("--prefitOFF", action="store_true")
    ap.add_argument("--learn_delay", action="store_true")
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--delay_jitter_ms", type=float, default=0.0)
    ap.add_argument("--domain_randomizationON", action="store_true")
    ap.add_argument("--domain_randomizationOFF", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--non_interactive", action="store_true")
    return ap.parse_args()




def chunk_single_trajectory(traj, chunks=6):
    n = len(traj.t)
    if n < chunks * 20:
        return [traj]
    out = []
    edges = np.linspace(0, n, chunks + 1, dtype=int)
    for i in range(chunks):
        a, b = edges[i], edges[i + 1]
        if b - a < 10:
            continue
        out.append(type(traj)(
            path=f"{traj.path}#chunk{i}", t=traj.t[a:b], dt=traj.dt[a:b],
            cmd_u=traj.cmd_u[a:b], hw_pwm=traj.hw_pwm[a:b],
            theta=traj.theta[a:b], omega=traj.omega[a:b], alpha=traj.alpha[a:b],
            bus_v=traj.bus_v[a:b], current=traj.current[a:b], power=traj.power[a:b],
            delay_sec_est=traj.delay_sec_est,
        ))
    return out or [traj]
def prefit_coordinate_search(env: PendulumRLEnv, iterations: int = 60):
    best = env.center.copy()
    base_state = env.reset()
    best_loss = env.prev_loss
    for _ in range(iterations):
        for i in range(len(best)):
            for sgn in (-1.0, 1.0):
                cand = best.copy()
                cand[i] += sgn * 0.8 * env.scales[i]
                cand = np.clip(cand, env.lb, env.ub)
                m = env._compute_metrics(cand)
                loss = env._weighted_loss(m, cand)
                if loss < best_loss:
                    best_loss = loss
                    best = cand
    env.center = best.copy()
    env.reset()
    return best, best_loss


class SimpleGaussianPolicy:
    def __init__(self, obs_dim, act_dim, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0.0, 0.05, size=(act_dim, obs_dim))
        self.b = np.zeros(act_dim, dtype=float)
        self.log_std = np.full(act_dim, -1.0, dtype=float)

    def sample(self, obs):
        mu = np.tanh(self.W @ obs + self.b)
        std = np.exp(self.log_std)
        act = np.clip(mu + self.rng.normal(0.0, std), -1.0, 1.0)
        return act, mu, std

    def update(self, batch, lr=2e-3):
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        gls = np.zeros_like(self.log_std)
        for item in batch:
            obs, act, mu, std, adv = item
            var = np.maximum(std ** 2, 1e-6)
            dmu = (act - mu) / var * adv
            dz = dmu * (1.0 - mu * mu)
            gW += np.outer(dz, obs)
            gb += dz
            gls += (((act - mu) ** 2 / var) - 1.0) * adv
        n = max(1, len(batch))
        self.W += lr * gW / n
        self.b += lr * gb / n
        self.log_std += 0.5 * lr * gls / n
        self.log_std = np.clip(self.log_std, -3.0, 0.5)


def run_training(args):
    np.random.seed(args.seed)
    cfg = BridgeConfig()
    init_params = build_params_from_calibration(cfg, args.calibration_json, args.parameter_json or None)
    csv_paths = gather_csv_paths(args.csv or None, args.csv_dir or None)
    trajs = [load_replay_trajectory(p, args.delay_override) for p in csv_paths]
    if len(trajs) == 1:
        trajs = chunk_single_trajectory(trajs[0], chunks=8)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(trajs))
    rng.shuffle(idx)
    n_train = max(1, int(0.7 * len(trajs)))
    n_val = max(1, int(0.15 * len(trajs))) if len(trajs) > 2 else 0
    train = [trajs[i] for i in idx[:n_train]]
    val = [trajs[i] for i in idx[n_train:n_train + n_val]] if n_val > 0 else []
    test = [trajs[i] for i in idx[n_train + n_val:]]
    if not test:
        test = train[-1:]

    env = PendulumRLEnv(cfg, train, init_params, learn_delay=args.learn_delay,
                        domain_randomization=args.domain_randomizationON and not args.domain_randomizationOFF,
                        delay_jitter_ms=args.delay_jitter_ms, seed=args.seed)

    outdir = Path(args.outdir) / time.strftime("run_%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    initial_params = {k: init_params[k] for k in PARAM_NAMES_DEFAULT}
    if args.learn_delay:
        initial_params["delay_sec"] = float(np.mean([t.delay_sec_est for t in trajs]))

    prefit_params = None
    if args.prefitON and not args.prefitOFF:
        p, loss = prefit_coordinate_search(env)
        prefit_params = {k: float(v) for k, v in zip(env.param_names, p)}
        with open(outdir / "prefit_params.json", "w", encoding="utf-8") as f:
            json.dump({"params": prefit_params, "loss": loss}, f, indent=2)

    obs = env.reset()
    policy = SimpleGaussianPolicy(len(obs), len(env.param_names), seed=args.seed)
    hist = {"episode": [], "reward": [], "loss": [], "rmse_theta": [], "rmse_omega": [], "rmse_alpha": [], "params": []}
    val_hist = {"loss": [], "rmse_theta": [], "rmse_omega": [], "rmse_alpha": []}

    for ep in range(args.num_episodes):
        obs = env.reset()
        transitions = []
        total_r = 0.0
        done = False
        while not done:
            act, mu, std = policy.sample(obs)
            nxt, reward, done, info = env.step(act)
            transitions.append((obs, act, mu, std, reward))
            obs = nxt
            total_r += reward
        returns = []
        g = 0.0
        for _, _, _, _, r in reversed(transitions):
            g = r + args.gamma * g
            returns.append(g)
        returns = list(reversed(returns))
        adv = np.array(returns, dtype=float)
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        batch = [(o, a, m, s, ad) for (o, a, m, s, _), ad in zip(transitions, adv)]
        policy.update(batch)

        hist["episode"].append(ep + 1)
        hist["reward"].append(float(total_r))
        hist["loss"].append(float(info["loss"]))
        hist["rmse_theta"].append(float(info["rmse_theta"]))
        hist["rmse_omega"].append(float(info["rmse_omega"]))
        hist["rmse_alpha"].append(float(info["rmse_alpha"]))
        hist["params"].append(env.params.copy().tolist())

        if val:
            venv = PendulumRLEnv(cfg, val, {k: float(v) for k, v in zip(env.param_names, env.params)},
                                 learn_delay=args.learn_delay, domain_randomization=False, seed=args.seed)
            venv.reset()
            vinfo = venv.metrics
            val_hist["loss"].append(float(venv.prev_loss))
            val_hist["rmse_theta"].append(float(vinfo["rmse_theta"]))
            val_hist["rmse_omega"].append(float(vinfo["rmse_omega"]))
            val_hist["rmse_alpha"].append(float(vinfo["rmse_alpha"]))

    best_ep = int(np.argmin(hist["loss"]))
    best_params_vec = np.array(hist["params"][best_ep], dtype=float)
    best_params = {k: float(v) for k, v in zip(env.param_names, best_params_vec)}

    save_training_plots(str(outdir), hist, env.param_names, val_hist if val else None)
    delays = {t.path: t.delay_sec_est for t in trajs}
    rep = trajs[0]
    rep_cmd_del = shift_with_delay(rep.t, rep.cmd_u, rep.delay_sec_est)
    save_delay_plots(str(outdir), delays, (rep.t, rep.cmd_u, rep_cmd_del, rep.hw_pwm))

    # test overlay using best parameters
    tenv = PendulumRLEnv(cfg, test, best_params, learn_delay=args.learn_delay, domain_randomization=False, seed=args.seed)
    tenv.reset()
    tr, sim, _ = tenv.metrics["details"][0]
    save_overlay_plot(str(outdir), tr.t, sim, {"theta": tr.theta, "omega": tr.omega, "alpha": tr.alpha}, "Sim vs Real (test)", "sim_vs_real_overlay.png")

    with open(outdir / "initial_params.json", "w", encoding="utf-8") as f:
        json.dump(initial_params, f, indent=2)
    with open(outdir / "final_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "settings": {
                "num_episodes": args.num_episodes,
                "gamma": args.gamma,
                "lam": args.lam,
                "kl_targ": args.kl_targ,
                "batch_size": args.batch_size,
                "prefit": bool(args.prefitON and not args.prefitOFF),
                "learn_delay": args.learn_delay,
                "domain_randomization": bool(args.domain_randomizationON and not args.domain_randomizationOFF),
                "delay_jitter_ms": args.delay_jitter_ms,
                "seed": args.seed,
            },
            "dataset": {"train": [t.path for t in train], "val": [t.path for t in val], "test": [t.path for t in test]},
            "reward_weights": vars(env.reward_weights),
            "randomization_scales": env.scales.tolist(),
            "best_validation_score": min(val_hist["loss"]) if val and val_hist["loss"] else None,
            "delay_estimates_sec": delays,
        }, f, indent=2)

    print(f"[RL] Outputs saved to {outdir}")


def main():
    args = parse_args()
    if not args.non_interactive:
        args.num_episodes = prompt_if_none(args.num_episodes, "num_episodes", int, 1000)
        args.gamma = prompt_if_none(args.gamma, "gamma", float, 0.995)
        args.lam = prompt_if_none(args.lam, "lam", float, 0.98)
        args.kl_targ = prompt_if_none(args.kl_targ, "kl_targ", float, 0.003)
        args.batch_size = prompt_if_none(args.batch_size, "batch_size", int, 20)
    else:
        args.num_episodes = args.num_episodes or 1000
        args.gamma = args.gamma or 0.995
        args.lam = args.lam or 0.98
        args.kl_targ = args.kl_targ or 0.003
        args.batch_size = args.batch_size or 20
    run_training(args)


if __name__ == "__main__":
    main()
