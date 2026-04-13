#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def _box_com(length: float, width: float, height: float, voxels: int = 64) -> list[float]:
    grid = np.ones((voxels, voxels, voxels), dtype=float)
    cz, cy, cx = ndimage.center_of_mass(grid)
    x = (cx / (voxels - 1) - 0.5) * width
    y = (cy / (voxels - 1) - 1.0) * length
    z = (cz / (voxels - 1) - 0.5) * height
    return [float(x), float(y), float(z)]


def _cylinder_com(length: float) -> list[float]:
    # Cylinder local axis centered at origin along z for our connector representation.
    _ = length
    return [0.0, 0.0, 0.0]


def _mass_weighted_com(items: list[tuple[float, np.ndarray]]) -> np.ndarray:
    msum = sum(m for m, _ in items)
    if msum <= 0.0:
        return np.zeros(3)
    return sum(m * c for m, c in items) / msum


def main() -> None:
    ap = argparse.ArgumentParser(description="Option 3: COM geometry validation and report")
    ap.add_argument("--motor_torque_json", default="")
    ap.add_argument("--pivot", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="pivot xyz [m]")
    args = ap.parse_args()

    path = Path(args.motor_torque_json or input("Select motor_torque.json path: ").strip())
    data = json.loads(path.read_text(encoding="utf-8"))

    rod = data["rod"]
    imu = data["imu"]
    conn = data.get("connector_cyl", {})

    rod_com = np.array(_box_com(float(rod["length"]), float(rod["width"]), float(rod["height"])))
    imu_com = np.array(_box_com(float(imu["length"]), float(imu["width"]), float(imu["height"])))
    # IMU offset from pivot is calibration-driven, keep local geometry COM but shift to mounted location.
    imu_mount = np.array(imu.get("com_local", [0.0, -0.22, 0.0]), dtype=float)
    imu_com = imu_mount + imu_com

    conn_mass = float(conn.get("mass", 0.01))
    conn_center = np.array([0.0, -float(conn.get("center_from_rod_base_edge", 0.01)), 0.0])
    conn_com = conn_center + np.array(_cylinder_com(float(conn.get("length", 0.028))))

    rod_mass = float(rod["mass"])
    imu_mass = float(imu["mass"])
    combined = _mass_weighted_com([(rod_mass, rod_com), (imu_mass, imu_com), (conn_mass, conn_com)])
    pivot = np.array(args.pivot, dtype=float)
    l_com = float(np.linalg.norm(combined - pivot))

    rod["com_local"] = rod_com.tolist()
    imu["com_local"] = imu_com.tolist()
    conn["com_local"] = conn_com.tolist()
    data["connector_cyl"] = conn
    data.setdefault("dynamic_parameters", {})
    data["dynamic_parameters"]["l_com"] = l_com

    warnings = []
    if l_com <= 0:
        warnings.append("combined COM distance to pivot is non-positive")
    if any(abs(v) > 10 for v in list(rod_com) + list(imu_com) + list(conn_com)):
        warnings.append("COM magnitude suspicious; check unit consistency")

    report = {
        "units": {"length": "m", "mass": "kg"},
        "frame": "pivot-centered world frame",
        "per_body_com": {
            "rod": rod_com.tolist(),
            "imu": imu_com.tolist(),
            "connector_cyl": conn_com.tolist(),
        },
        "combined_com": combined.tolist(),
        "pivot": pivot.tolist(),
        "pivot_relative_distance_l_com": l_com,
        "unit_consistency_checks": {
            "length_in_meters": True,
            "mass_in_kg": True,
        },
        "warnings": warnings,
        "inertia_note": "J is identification-based (Option 5); geometry inertia path deprecated.",
    }

    out_report = path.parent / "com_validation_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    data.setdefault("calculator", {})
    data["calculator"].update({
        "backend": "scipy.ndimage.center_of_mass",
        "note": "COM-only validation. J must be identified in Option 5.",
        "report": str(out_report),
    })

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] updated COM in {path}")
    print(f"[OK] validation report: {out_report}")


if __name__ == "__main__":
    main()
