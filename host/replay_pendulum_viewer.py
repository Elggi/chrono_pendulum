#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str, fallback: float = 0.0):
    if name not in df.columns:
        return np.full(len(df), float(fallback), dtype=float)
    v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    v[~np.isfinite(v)] = fallback
    return v


def load_csv(path: str):
    df = pd.read_csv(path)
    if "wall_elapsed" in df.columns:
        t = _col(df, "wall_elapsed")
    elif "wall_time" in df.columns:
        tw = _col(df, "wall_time")
        t = tw - tw[0] if len(tw) else tw
    else:
        t = np.arange(len(df), dtype=float) * 0.01

    pwm = _col(df, "cmd_u_raw")
    th_sim = _col(df, "theta")
    th_real = _col(df, "theta_real", np.nan)
    if not np.isfinite(th_real).any():
        th_real = _col(df, "theta")

    return t, pwm, th_sim, th_real


def tip(theta: float, length: float):
    return np.array([length * math.sin(theta), -length * math.cos(theta), 0.0], dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Replay CSV as 3D pendulum viewer")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--link_length", type=float, default=0.285)
    args = ap.parse_args()

    t, pwm, th_sim, th_real = load_csv(args.csv)
    if len(t) == 0:
        raise SystemExit("CSV is empty")

    fig = plt.figure(figsize=(14, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)

    t0 = t[0]
    t_rel = t - t0

    def update(i: int):
        i = min(max(i, 0), len(t_rel) - 1)
        ax3d.cla(); ax2.cla(); ax3.cla()

        ts = tip(th_sim[i], args.link_length)
        tr = tip(th_real[i], args.link_length)

        ax3d.set_title("Replay Orientation (3D)")
        ax3d.set_xlim(-args.link_length, args.link_length)
        ax3d.set_ylim(-args.link_length, args.link_length)
        ax3d.set_zlim(-0.2, 0.2)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.view_init(elev=20, azim=35)
        ax3d.plot([0, ts[0]], [0, ts[1]], [0, ts[2]], "-o", linewidth=4, color="tab:blue", label="sim")
        ax3d.plot([0, tr[0]], [0, tr[1]], [0, tr[2]], "-o", linewidth=3, color="tab:orange", label="real")
        ax3d.legend(loc="upper right")

        ax2.plot(t_rel, pwm, color="tab:green", label="PWM input")
        ax2.axvline(t_rel[i], color="k", linestyle="--", alpha=0.4)
        ax2.grid(True)
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("PWM")
        ax2.legend(loc="upper right")

        ax3.plot(t_rel, th_sim, color="tab:blue", label="sim theta")
        ax3.plot(t_rel, th_real, color="tab:orange", label="real theta")
        ax3.axvline(t_rel[i], color="k", linestyle="--", alpha=0.4)
        ax3.grid(True)
        ax3.set_xlabel("time [s]")
        ax3.set_ylabel("orientation [rad]")
        ax3.legend(loc="upper right")
        fig.suptitle(f"{args.csv} | t={t_rel[i]:.3f}s | pwm={pwm[i]:.1f}", fontsize=12)

    if len(t_rel) > 1:
        dt = float(np.median(np.diff(t_rel)))
        interval_ms = max(1, int(1000.0 * max(dt, 1e-3)))
    else:
        interval_ms = int(1000.0 / max(args.fps, 1))

    ani = FuncAnimation(fig, update, frames=len(t_rel), interval=interval_ms, repeat=False, cache_frame_data=False)
    fig._ani = ani
    plt.show()


if __name__ == "__main__":
    main()
