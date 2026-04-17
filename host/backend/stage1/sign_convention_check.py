#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

HOST_DIR = Path(__file__).resolve().parents[2]
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from chrono_core.config import BridgeConfig
from chrono_core.dynamics import PendulumModel


def _simulate_early_omega(theta0: float, dt: float = 0.002, steps: int = 8) -> float:
    cfg = BridgeConfig(theta0_deg=0.0, omega0=0.0)
    model = PendulumModel(cfg)
    model.set_theta_kinematic(theta0, 0.0)
    for _ in range(int(steps)):
        model.apply_torque(0.0)
        model.step(float(dt))
    return float(model.get_omega())


def _assert_restore_direction(theta0: float) -> tuple[float, float]:
    omega_after = _simulate_early_omega(theta0)
    expected = -math.sin(float(theta0))
    if abs(expected) > 1e-9 and math.copysign(1.0, omega_after) != math.copysign(1.0, expected):
        raise AssertionError(
            f"restoring-direction mismatch: theta0={theta0:.6f}, omega_after={omega_after:.6f}, expected_sign={math.copysign(1.0, expected):+.0f}"
        )
    return omega_after, expected


def run_sign_convention_checks() -> None:
    cfg = BridgeConfig(theta0_deg=0.0, omega0=0.0)
    model = PendulumModel(cfg)
    diag_pos = model.sign_convention_diagnostic(theta_rad=+0.4)
    diag_neg = model.sign_convention_diagnostic(theta_rad=-0.4)

    # Geometry sanity: x-offset should track sin(theta).
    if diag_pos["com_rx"] <= 0.0:
        raise AssertionError(f"positive theta must place COM at +x. diag={diag_pos}")
    if diag_neg["com_rx"] >= 0.0:
        raise AssertionError(f"negative theta must place COM at -x. diag={diag_neg}")

    # Gravity restoring sign sanity.
    if diag_pos["gravity_torque_sign_proxy"] >= 0.0:
        raise AssertionError(f"positive theta should have negative restoring gravity torque. diag={diag_pos}")
    if diag_neg["gravity_torque_sign_proxy"] <= 0.0:
        raise AssertionError(f"negative theta should have positive restoring gravity torque. diag={diag_neg}")

    omega_pos, expected_pos = _assert_restore_direction(+0.35)
    omega_neg, expected_neg = _assert_restore_direction(-0.35)

    print("[sign-check] PASS")
    print("[sign-check] convention:")
    print("  theta=0 => rod along -Y, theta>0 => CCW (+Z), omega>0 => CCW")
    print("[sign-check] geometry/torque diagnostics:")
    print(f"  theta=+0.4 -> {asdict_like(diag_pos)}")
    print(f"  theta=-0.4 -> {asdict_like(diag_neg)}")
    print("[sign-check] restoring-direction:")
    print(f"  theta0=+0.35 rad: omega_after={omega_pos:+.6f}, expected_sign_from_-sin(theta)={math.copysign(1.0, expected_pos):+.0f}")
    print(f"  theta0=-0.35 rad: omega_after={omega_neg:+.6f}, expected_sign_from_-sin(theta)={math.copysign(1.0, expected_neg):+.0f}")


def asdict_like(d: dict[str, float]) -> str:
    keys = ["theta", "com_rx", "com_ry", "expected_rx_from_theta", "gravity_torque_sign_proxy"]
    return ", ".join(f"{k}={d[k]:+.6f}" for k in keys)


if __name__ == "__main__":
    run_sign_convention_checks()
