#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple replay terminal for chrono run logs")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = pd.to_numeric(df.get("time", df.get("wall_elapsed")), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    theta = pd.to_numeric(df.get("theta", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    omega = pd.to_numeric(df.get("omega", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    current = pd.to_numeric(df.get("input_current", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(t) < 2:
        raise SystemExit("Not enough samples")
    t0 = t[0]
    wall = time.time()
    i = 0
    while i < len(t):
        target = (time.time() - wall) * max(args.speed, 1e-6)
        while i + 1 < len(t) and (t[i + 1] - t0) <= target:
            i += 1
        print(f"t={t[i]-t0:7.3f}s | I={current[i]:7.4f}A | theta={theta[i]:7.4f}rad | omega={omega[i]:7.4f}rad/s", end="\r")
        if i >= len(t) - 1:
            break
        time.sleep(0.002)
    print("\n[OK] replay finished")


if __name__ == "__main__":
    main()
