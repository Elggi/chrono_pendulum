"""Lightweight IMU viewer and sim-vs-real overlays (RViz replacement)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_overlay(real_csv: Path, sim_csv: Path, out_png: Path) -> Path:
    """Generate theta/omega overlay plot for validation reports."""
    real = pd.read_csv(real_csv)
    sim = pd.read_csv(sim_csv)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(real["t"], real["theta"], label="real")
    axes[0].plot(sim["t"], sim["theta"], label="sim", linestyle="--")
    axes[0].set_ylabel("theta [rad]")
    axes[0].legend()

    axes[1].plot(real["t"], real["omega"], label="real")
    axes[1].plot(sim["t"], sim["omega"], label="sim", linestyle="--")
    axes[1].set_ylabel("omega [rad/s]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
