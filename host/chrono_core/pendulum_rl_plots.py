#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training_curves(history: dict, outdir: Path):
    ep = np.arange(1, len(history.get("episode_reward", [])) + 1)
    if len(ep) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, history["episode_reward"], label="mean episode reward (SB3 Monitor)")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("True Episode Reward")
    ax.grid(True, alpha=0.3)
    _save(fig, outdir / "episode_reward.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, history["train_loss"], label="train weighted loss")
    if history.get("val_loss"):
        ax.plot(ep, history["val_loss"], label="val weighted loss")
    ax.set_xlabel("episode")
    ax.set_ylabel("weighted loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir / "loss_convergence.png")

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, key in enumerate(["rmse_theta", "rmse_omega", "rmse_alpha"]):
        axs[i].plot(ep, history.get(key, []), label=f"train {key}")
        vk = f"val_{key}"
        if history.get(vk):
            axs[i].plot(ep, history[vk], label=f"val {key}")
        axs[i].set_ylabel(key)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best")
    axs[-1].set_xlabel("episode")
    _save(fig, outdir / "rmse_convergence.png")


def plot_param_convergence(param_history: dict, outdir: Path):
    if not param_history:
        return
    current = param_history.get("current_eval_params_per_episode", {})
    best_train = param_history.get("global_best_train_params_so_far", {})
    best_val = param_history.get("global_best_val_params_so_far", {})
    keys = list(current.keys())
    if not keys:
        return
    fig, axs = plt.subplots(len(keys), 1, figsize=(8, 2.6 * len(keys)), sharex=True)
    if len(keys) == 1:
        axs = [axs]
    ep = np.arange(1, len(current[keys[0]]) + 1)
    for ax, key in zip(axs, keys):
        ax.plot(ep, current.get(key, []), label=f"{key} current-eval")
        ax.plot(ep, best_train.get(key, []), label=f"{key} best-train-so-far")
        ax.plot(ep, best_val.get(key, []), label=f"{key} best-val-so-far")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axs[-1].set_xlabel("episode")
    _save(fig, outdir / "parameter_convergence.png")


def plot_delay_diagnostics(delay_map: dict[str, float], outdir: Path):
    if not delay_map:
        return
    names = list(delay_map.keys())
    vals_ms = np.array([delay_map[n] * 1000.0 for n in names], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(names)), vals_ms, marker="o")
    ax.set_ylabel("estimated delay [ms]")
    ax.set_xlabel("trajectory index")
    ax.grid(True, alpha=0.3)
    _save(fig, outdir / "delay_per_trajectory.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals_ms, bins=min(10, max(3, len(vals_ms))))
    ax.set_xlabel("delay [ms]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    _save(fig, outdir / "delay_histogram.png")


def plot_overlay(t, real, sim, ylabel: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, real, label="real")
    ax.plot(t, sim, label="sim")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save(fig, outpath)


def plot_rl_dashboard(history: dict, param_history: dict, outdir: Path):
    ep = np.arange(1, len(history.get("episode_reward", [])) + 1)
    if len(ep) == 0:
        return
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    ax = axs[0, 0]
    ax.plot(ep, history.get("episode_reward", []), color="tab:blue")
    ax.set_title("Episode Reward (from SB3 Monitor)")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)

    ax = axs[0, 1]
    ax.plot(ep, history.get("train_loss", []), label="train")
    if history.get("val_loss"):
        ax.plot(ep, history.get("val_loss"), label="val")
    ax.set_title("Weighted Loss")
    ax.set_xlabel("episode")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axs[1, 0]
    for key in ("rmse_theta", "rmse_omega", "rmse_alpha"):
        vals = history.get(key, [])
        if vals:
            ax.plot(ep, vals, label=key.replace("rmse_", ""))
    ax.set_title("Train RMSE")
    ax.set_xlabel("episode")
    ax.set_ylabel("rmse")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axs[1, 1]
    current = param_history.get("current_eval_params_per_episode", {})
    best_val = param_history.get("global_best_val_params_so_far", {})
    keys = list(current.keys())[:4]
    for key in keys:
        vals_cur = current.get(key, [])
        vals_best = best_val.get(key, [])
        if vals_cur:
            ax.plot(ep, vals_cur, label=f"{key} current")
        if vals_best:
            ax.plot(ep, vals_best, linestyle="--", label=f"{key} best-val")
    ax.set_title("Parameter Trends (current vs best-val)")
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    _save(fig, outdir / "rl_dashboard.png")
