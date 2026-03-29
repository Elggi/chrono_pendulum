#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy compatibility wrapper.

`train_pendulum_rl.py` is the primary offline replay calibration entrypoint.
This wrapper keeps older menu/scripts working while forwarding options.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="[DEPRECATED] compatibility wrapper for train_pendulum_rl.py")
    ap.add_argument("--csv", required=True, help="single replay CSV")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo", help="kept for compatibility; train_pendulum_rl uses PPO-style flow")
    ap.add_argument("--outdir", default="rl_results")
    ap.add_argument("--timesteps", type=int, default=40000, help="maps to num_episodes approximately")
    ap.add_argument("--calibration-json", default="", help="if omitted, tries host/run_logs/calibration_latest.json")
    ap.add_argument("--parameter-json", default="")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    train_script = base_dir / "train_pendulum_rl.py"

    calib = args.calibration_json.strip()
    if not calib:
        fallback = base_dir / "run_logs" / "calibration_latest.json"
        if fallback.exists():
            calib = str(fallback)

    if not calib:
        raise SystemExit("[ERROR] calibration json is required. Provide --calibration-json or place run_logs/calibration_latest.json")

    num_episodes = max(50, int(args.timesteps // 40))
    cmd = [
        sys.executable,
        str(train_script),
        "--calibration_json", calib,
        "--csv", args.csv,
        "--outdir", args.outdir,
        "--num_episodes", str(num_episodes),
        "--seed", str(args.seed),
        "--renderOFF",
    ]
    if args.parameter_json.strip():
        cmd += ["--parameter_json", args.parameter_json.strip()]

    print("[WARN] RL_fitting.py is deprecated. Forwarding to train_pendulum_rl.py")
    print("[INFO] command:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
