#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Replay-only CLI: export replay csv and optionally plot")
    ap.add_argument("--csv", required=True, help="source run log csv")
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--parameter_json", default="")
    ap.add_argument("--out_csv", default="./rl_results/replay_best.csv")
    ap.add_argument("--delay_override", type=float, default=None)
    ap.add_argument("--plot", action="store_true", help="open plot_pendulum after export")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(base_dir, "replay_pendulum_export.py")
    cmd = [sys.executable, export_script,
           "--calibration_json", args.calibration_json,
           "--csv", args.csv,
           "--out_csv", args.out_csv]
    if args.parameter_json:
        cmd += ["--parameter_json", args.parameter_json]
    if args.delay_override is not None:
        cmd += ["--delay_override", str(args.delay_override)]

    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(f"Replay export failed with exit code {rc}")

    print(f"[INFO] replay export done: {args.out_csv}")
    if args.plot:
        plot_script = os.path.join(base_dir, "plot_pendulum.py")
        cmd2 = [sys.executable, plot_script, "--csv", args.out_csv]
        os.execv(sys.executable, cmd2)


if __name__ == "__main__":
    main()
