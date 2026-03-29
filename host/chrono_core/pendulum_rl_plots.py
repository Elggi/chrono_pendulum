from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_training_plots(outdir: str, history: dict, param_names: list[str], val_history: dict | None = None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(history.get("episode", [])) + 1)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()
    ax[0].plot(x, history.get("loss", []), label="train loss")
    if val_history and val_history.get("loss"):
        ax[0].plot(x[:len(val_history["loss"])], val_history["loss"], label="val loss")
    ax[0].set_title("Weighted total loss")
    ax[0].grid(True); ax[0].legend()

    for i, k in enumerate(["rmse_theta", "rmse_omega", "rmse_alpha"], start=1):
        ax[i].plot(x, history.get(k, []), label=f"train {k}")
        if val_history and val_history.get(k):
            ax[i].plot(x[:len(val_history[k])], val_history[k], label=f"val {k}")
        ax[i].grid(True); ax[i].legend(); ax[i].set_title(k)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=140)
    plt.close(fig)

    params = np.asarray(history.get("params", []), dtype=float)
    if params.size > 0:
        fig, ax = plt.subplots(len(param_names), 1, figsize=(10, 2.2 * len(param_names)), sharex=True)
        if len(param_names) == 1:
            ax = [ax]
        for i, name in enumerate(param_names):
            ax[i].plot(x, params[:, i])
            ax[i].set_ylabel(name)
            ax[i].grid(True)
        ax[-1].set_xlabel("episode")
        fig.tight_layout()
        fig.savefig(out / "parameter_convergence.png", dpi=140)
        plt.close(fig)


def save_delay_plots(outdir: str, delays: dict[str, float], representative):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(delays.keys())
    vals_ms = [1000.0 * delays[k] for k in names]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.arange(len(vals_ms)), vals_ms, marker="o")
    ax[0].set_title("Estimated delay per trajectory [ms]")
    ax[0].grid(True)
    ax[1].hist(vals_ms, bins=min(12, max(3, len(vals_ms))))
    ax[1].set_title("Delay stability histogram [ms]")
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(out / "delay_diagnostics.png", dpi=140)
    plt.close(fig)

    if representative is not None:
        t, cmd, cmd_del, hw = representative
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, cmd, label="cmd_raw")
        ax.plot(t, cmd_del, label="cmd_delayed")
        ax.plot(t, hw, label="hw_pwm")
        ax.grid(True); ax.legend(); ax.set_title("Command alignment diagnostic")
        fig.tight_layout()
        fig.savefig(out / "delay_alignment_example.png", dpi=140)
        plt.close(fig)


def save_overlay_plot(outdir: str, t, sim, real, title: str, filename: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, key in enumerate(["theta", "omega", "alpha"]):
        ax[i].plot(t, real[key], label=f"real {key}")
        ax[i].plot(t, sim[key], label=f"sim {key}")
        ax[i].grid(True); ax[i].legend()
    ax[0].set_title(title)
    ax[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=140)
    plt.close(fig)
