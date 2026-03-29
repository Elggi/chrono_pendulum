#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ep = np.arange(1, len(history.get("episode_reward", [])) + 1)
    if len(ep) == 0:
        return

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()

    ax[0].plot(ep, history["episode_reward"], label="episode_reward")
    ax[0].set_title("Episode reward")
    ax[0].grid(True)

    ax[1].plot(ep, history["train_loss"], label="train weighted loss")
    if history.get("val_loss"):
        ax[1].plot(ep, history["val_loss"], label="val weighted loss")
    ax[1].set_title("Weighted total loss vs episode")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(ep, history["rmse_theta"], label="train")
    if history.get("val_rmse_theta"):
        ax[2].plot(ep, history["val_rmse_theta"], label="val")
    ax[2].set_title("RMSE theta")
    ax[2].legend()
    ax[2].grid(True)

    ax[3].plot(ep, history["rmse_omega"], label="train")
    if history.get("val_rmse_omega"):
        ax[3].plot(ep, history["val_rmse_omega"], label="val")
    ax[3].plot(ep, history["rmse_alpha"], label="alpha(train)")
    if history.get("val_rmse_alpha"):
        ax[3].plot(ep, history["val_rmse_alpha"], label="alpha(val)")
    ax[3].set_title("RMSE omega/alpha")
    ax[3].legend()
    ax[3].grid(True)

    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_parameter_convergence(param_history: list[dict], out_dir: str):
    if not param_history:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(param_history[0].keys())
    ep = np.arange(1, len(param_history) + 1)

    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for i, k in enumerate(keys):
        axes[i].plot(ep, [x[k] for x in param_history])
        axes[i].set_title(k)
        axes[i].grid(True)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out / "parameter_convergence.png", dpi=140)
    plt.close(fig)


def plot_delay_diagnostics(trajectories, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    delays_ms = [1000.0 * tr.delay_est for tr in trajectories]
    quality = [tr.delay_quality_corr for tr in trajectories]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(delays_ms, marker="o")
    ax[0].set_title("Estimated delay per trajectory [ms]")
    ax[0].grid(True)

    ax[1].hist(delays_ms, bins=min(20, max(4, len(delays_ms))))
    ax[1].set_title("Delay histogram [ms]")
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(out / "delay_estimates.png", dpi=140)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(quality, marker="x")
    ax2.set_title("Delay alignment quality (corr proxy)")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(out / "delay_quality.png", dpi=140)
    plt.close(fig2)


def plot_overlay(tr, sim, out_path: str):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax = ax.ravel()
    t = tr.t
    ax[0].plot(t, tr.cmd_u, label="cmd_u_raw")
    ax[0].plot(t, sim["u_aligned"], label="cmd_u_delayed")
    ax[0].plot(t, tr.hw_pwm, label="hw_pwm")
    ax[0].set_title("Input alignment")
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot(t, tr.theta_real, label="theta real")
    ax[1].plot(t, sim["theta"], label="theta sim")
    ax[1].set_title("Theta overlay")
    ax[1].legend(); ax[1].grid(True)

    ax[2].plot(t, tr.omega_real, label="omega real")
    ax[2].plot(t, sim["omega"], label="omega sim")
    ax[2].set_title("Omega overlay")
    ax[2].legend(); ax[2].grid(True)

    ax[3].plot(t, tr.alpha_real, label="alpha real")
    ax[3].plot(t, sim["alpha"], label="alpha sim")
    ax[3].set_title("Alpha overlay")
    ax[3].legend(); ax[3].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
