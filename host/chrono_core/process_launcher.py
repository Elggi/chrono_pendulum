#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_chrono_pendulum_process(
    host_dir: Path,
    calibration_json: str = "",
    parameter_json: str = "",
    duration_sec: float = 8.0,
    headless: bool = True,
    mode: str = "host",
) -> int:
    script = host_dir / "chrono_pendulum.py"
    cmd = [sys.executable, str(script), "--mode", str(mode), "--duration", str(float(duration_sec))]
    if headless:
        cmd.append("--headless")
    if calibration_json:
        cmd.extend(["--calibration-json", str(calibration_json), "--radius-json", str(calibration_json)])
    if parameter_json:
        cmd.extend(["--parameter-json", str(parameter_json)])
    print(f"[INFO] launch_chrono_process: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(host_dir))
    return int(proc.returncode)
