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
        y_train = history.get(key, [])
        if len(y_train) == len(ep):
            axs[i].plot(ep, y_train, label=f"train {key}")
        vk = f"val_{key}"
        y_val = history.get(vk, [])
        if len(y_val) == len(ep):
            axs[i].plot(ep, y_val, label=f"val {key}")
        axs[i].set_ylabel(key)
        axs[i].grid(True, alpha=0.3)
        if len(axs[i].lines) > 0:
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


def plot_stage1_regression_summary(y_true, y_pred, outpath: Path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(y_true, label="target J*alpha", alpha=0.8)
    axs[0].plot(y_pred, label="regression fit", alpha=0.8)
    axs[0].set_xlabel("sample")
    axs[0].set_ylabel("torque [N*m]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].scatter(y_true, y_pred, s=6, alpha=0.4)
    y_min = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
    y_max = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
    axs[1].plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1.0)
    axs[1].set_xlabel("target J*alpha")
    axs[1].set_ylabel("predicted")
    axs[1].grid(True, alpha=0.3)
    _save(fig, outpath)


def plot_stage123_regression_summary(stage_payloads: dict[str, dict], outpath: Path):
    """Render one consolidated figure for Stage 1~3 regression results."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    s1 = stage_payloads.get("stage1", {})
    s1_id = s1.get("identified_params", {})
    s1_m = s1.get("metrics", {})
    axs[0].axis("off")
    axs[0].set_title("Stage 1 (sin): fit [K_u, l_com]")
    txt1 = (
        f"K_u: {s1_id.get('K_u', np.nan):.8f}\n"
        f"l_com: {s1_id.get('l_com', np.nan):.8f}\n"
        f"rmse: {s1_m.get('rmse', np.nan):.6f}\n"
        f"samples: {s1_m.get('sample_count', np.nan)}"
    )
    axs[0].text(0.02, 0.55, txt1, fontsize=12, family="monospace")

    s2 = stage_payloads.get("stage2", {})
    s2_id = s2.get("identified_params", {})
    s2_m = s2.get("metrics", {})
    axs[1].axis("off")
    axs[1].set_title("Stage 2 (square): fit [b_eq]")
    txt2 = (
        f"b_eq: {s2_id.get('b_eq', np.nan):.8f}\n"
        f"rmse: {s2_m.get('rmse', np.nan):.6f}\n"
        f"omega_deadband: {s2_m.get('omega_deadband', np.nan)}\n"
        f"used_ratio: {s2_m.get('used_ratio', np.nan):.4f}"
    )
    axs[1].text(0.02, 0.55, txt2, fontsize=12, family="monospace")

    s3 = stage_payloads.get("stage3", {})
    s3_id = s3.get("identified_params", {})
    s3_m = s3.get("metrics", {})
    axs[2].axis("off")
    axs[2].set_title("Stage 3 (burst): fit [tau_eq]")
    txt3 = (
        f"tau_eq: {s3_id.get('tau_eq', np.nan):.8f}\n"
        f"rmse: {s3_m.get('rmse', np.nan):.6f}\n"
        f"low_speed_ratio: {s3_m.get('low_speed_ratio', np.nan):.4f}\n"
        f"high_speed_ref: {s3_m.get('high_speed_ref', np.nan)}"
    )
    axs[2].text(0.02, 0.55, txt3, fontsize=12, family="monospace")

    _save(fig, outpath)
