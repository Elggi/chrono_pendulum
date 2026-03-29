from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list[dict], outdir: Path):
    ep = np.array([h["episode"] for h in history])
    loss = np.array([h["train_loss"] for h in history])
    val = np.array([h["val_loss"] for h in history])
    r = np.array([h["reward"] for h in history])
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(ep, loss, label="train loss")
    ax[0].plot(ep, val, label="val loss")
    ax[0].set_ylabel("weighted loss")
    ax[0].legend()
    ax[1].plot(ep, r, label="episode reward")
    ax[1].set_ylabel("reward")
    ax[1].set_xlabel("episode")
    ax[1].legend()
    _save(fig, outdir / "training_curves.png")


def plot_rmse_curves(history: list[dict], outdir: Path):
    ep = np.array([h["episode"] for h in history])
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, key in enumerate(["rmse_theta", "rmse_omega", "rmse_alpha"]):
        ax[i].plot(ep, [h.get(key, np.nan) for h in history], label=f"train {key}")
        ax[i].plot(ep, [h.get(f"val_{key}", np.nan) for h in history], label=f"val {key}")
        ax[i].legend()
    ax[-1].set_xlabel("episode")
    _save(fig, outdir / "rmse_curves.png")


def plot_param_convergence(history: list[dict], outdir: Path):
    keys = [k for k in history[0].keys() if k.startswith("param_")]
    ep = np.array([h["episode"] for h in history])
    fig, ax = plt.subplots(len(keys), 1, figsize=(8, 2 * len(keys)), sharex=True)
    if len(keys) == 1:
        ax = [ax]
    for i, k in enumerate(keys):
        ax[i].plot(ep, [h[k] for h in history])
        ax[i].set_ylabel(k.replace("param_", ""))
    ax[-1].set_xlabel("episode")
    _save(fig, outdir / "param_convergence.png")


def plot_delay_diagnostics(delay_map: dict[str, float], outdir: Path):
    names = list(delay_map.keys())
    vals = np.array([delay_map[n] * 1000.0 for n in names])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(len(names)), vals, marker="o")
    ax[0].set_title("Estimated delay per trajectory (ms)")
    ax[0].set_xlabel("trajectory")
    ax[0].set_ylabel("delay [ms]")
    ax[1].hist(vals, bins=min(10, max(3, len(vals))))
    ax[1].set_title("Delay stability histogram")
    ax[1].set_xlabel("delay [ms]")
    _save(fig, outdir / "delay_diagnostics.png")
