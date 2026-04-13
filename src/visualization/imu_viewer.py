"""Lightweight IMU viewer and sim-vs-real overlays (RViz replacement)."""

from __future__ import annotations

import argparse
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


def main() -> None:
    """CLI entrypoint for quick real-vs-sim overlay generation."""
    parser = argparse.ArgumentParser(description="IMU overlay plotter")
    parser.add_argument("--real", type=Path, required=True, help="Real trajectory CSV (t/theta/omega)")
    parser.add_argument("--sim", type=Path, required=True, help="Sim trajectory CSV (t/theta/omega)")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    out = plot_overlay(args.real, args.sim, args.out)
    print(f"Saved overlay: {out}")


if __name__ == "__main__":
    main()
