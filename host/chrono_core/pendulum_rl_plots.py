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
    ep = np.arange(1, len(history.get("reward", [])) + 1)
    if len(ep) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, history["reward"], label="episode reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
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


def plot_param_convergence(param_history: dict[str, list[float]], outdir: Path):
    if not param_history:
        return
    keys = list(param_history.keys())
    fig, axs = plt.subplots(len(keys), 1, figsize=(8, 2.4 * len(keys)), sharex=True)
    if len(keys) == 1:
        axs = [axs]
    ep = np.arange(1, len(param_history[keys[0]]) + 1)
    for ax, key in zip(axs, keys):
        ax.plot(ep, param_history[key], label=key)
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
