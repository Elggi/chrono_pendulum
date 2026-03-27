#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""수동 회전 기반 캘리브레이션 도구.

- CPR: IMU/엔코더를 구독하며 `imu_viewer.py`의 SharedState 로직을 재사용해 자동 계산
- r  : orientation 기반 tip 좌표에서 실시간 반지름 추정
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from statistics import mean

import rclpy

from imu_viewer import SharedState, ViewerNode


class CprCollector:
    """imu_viewer.py 내부 상태 추적 로직(SharedState)을 그대로 활용한 CPR 수집기."""

    def __init__(self, imu_topic: str, enc_topic: str):
        self.state = SharedState()
        self._node = None
        self._thread = None
        self._running = False
        self.imu_topic = imu_topic
        self.enc_topic = enc_topic

    def start(self):
        if not rclpy.ok():
            rclpy.init()

        self._node = ViewerNode(self.state, self.imu_topic, self.enc_topic)
        self._running = True

        def _spin_loop():
            while self._running and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.1)

        self._thread = threading.Thread(target=_spin_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def wait_for_imu(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self.state.lock:
                if self.state.has_init and self.state.seq > 0:
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> dict:
        with self.state.lock:
            samples = list(self.state.cpr_samples)
            tip_hist = [tip.tolist() for tip in self.state.tip_hist]
            tip0 = self.state.tip0.tolist()
            return {
                "cpr_samples": samples,
                "last_cpr": self.state.last_cpr,
                "full_rotations": int(self.state.rev_index),
                "angle_unwrapped_rad": float(self.state.angle_unwrapped),
                "tip_hist": tip_hist,
                "tip0": tip0,
            }


def _collect_cpr_and_r_from_imu(args) -> tuple[list[dict], float, list[dict], float]:
    collector = CprCollector(imu_topic=args.imu_topic, enc_topic=args.hw_enc_topic)
    viewer_proc = maybe_launch_imu_viewer(args)

    try:
        collector.start()
        print("\n[CPR] IMU/엔코더 기반 자동 CPR 수집")
        if not collector.wait_for_imu(timeout_sec=args.imu_wait_sec):
            raise RuntimeError("IMU 데이터를 받지 못했습니다. 토픽 연결 상태를 확인하세요.")

        print("- 준비 완료. 수동으로 진자를 1회전 이상 회전시키세요.")
        input("- 회전을 시작하려면 Enter를 누르세요...")
        print("  [수집중] 회전 완료 후 Enter를 누르면 종료됩니다.")
        input()

        snap = collector.snapshot()
        cpr_samples = snap["cpr_samples"]
        if not cpr_samples:
            raise RuntimeError("full rotation이 감지되지 않아 CPR 샘플이 없습니다.")

        cpr_trials = [
            {
                "trial": idx,
                "cpr": float(cpr),
            }
            for idx, cpr in enumerate(cpr_samples, start=1)
        ]
        mean_cpr = float(mean(cpr_samples))

        r_trials, mean_r = _estimate_r_trials_from_snapshot(snap)
        if not r_trials:
            raise RuntimeError("orientation 데이터로 r 샘플을 만들지 못했습니다.")

        print(f"[INFO] 감지된 full rotation 수: {snap['full_rotations']}")
        print(f"[INFO] CPR 샘플 수: {len(cpr_samples)}")
        print(f"[INFO] r 샘플 수: {len(r_trials)}")
        return cpr_trials, mean_cpr, r_trials, mean_r
    finally:
        collector.stop()
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()


def _estimate_r_trials_from_snapshot(snapshot: dict) -> tuple[list[dict], float | None]:
    tip_hist = snapshot.get("tip_hist", [])
    tip0 = snapshot.get("tip0")
    if not tip_hist or tip0 is None:
        return [], None

    ref = tip0
    ref_norm = (ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2) ** 0.5
    if ref_norm < 1e-9:
        return [], None

    instant_samples = []
    parallel_samples = []

    for idx, tip in enumerate(tip_hist, start=1):
        tx, ty, tz = float(tip[0]), float(tip[1]), float(tip[2])
        tip_norm = (tx * tx + ty * ty + tz * tz) ** 0.5
        if tip_norm < 1e-9:
            continue

        r_instant = tip_norm
        instant_samples.append(
            {
                "trial": idx,
                "method": "tip_norm",
                "radius_m": r_instant,
            }
        )

        dot = tx * ref[0] + ty * ref[1] + tz * ref[2]
        cos_angle = dot / (tip_norm * ref_norm)
        if cos_angle < -0.99:
            dx = tx - ref[0]
            dy = ty - ref[1]
            dz = tz - ref[2]
            chord = (dx * dx + dy * dy + dz * dz) ** 0.5
            r_parallel = chord * 0.5
            parallel_samples.append(
                {
                    "trial": idx,
                    "method": "anti_parallel_chord",
                    "radius_m": r_parallel,
                }
            )

    trials = parallel_samples if parallel_samples else instant_samples
    if not trials:
        return [], None
    mean_r = float(mean(item["radius_m"] for item in trials))
    return trials, mean_r


def run_calibration(args) -> None:
    print("=== Manual Rotation Calibration (CPR/r from IMU+orientation) ===")

    cpr_trials, mean_cpr, r_trials, mean_r = _collect_cpr_and_r_from_imu(args)

    result = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "manual_rotation_with_orientation",
        "summary": {
            "mean_cpr": mean_cpr,
            "mean_radius_m": mean_r,
            "trial_count_cpr": len(cpr_trials),
            "trial_count_r": len(r_trials),
        },
        "cpr_trials": cpr_trials,
        "radius_trials": r_trials,
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== Calibration Result ===")
    print(f"mean CPR      : {mean_cpr:.6f}")
    print(f"mean radius r : {mean_r:.6f} m")
    print(f"JSON saved    : {args.output_json}")


def build_argparser():
    ap = argparse.ArgumentParser(description="CPR/r 캘리브레이션 (IMU CPR + orientation r)")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--hw-enc-topic", default="/hw/enc")
    ap.add_argument("--output-json", default="./run_logs/calibration_latest.json")
    ap.add_argument("--imu-wait-sec", type=float, default=5.0)
    ap.add_argument("--no-imu-viewer", action="store_true")
    return ap


def maybe_launch_imu_viewer(args):
    if args.no_imu_viewer:
        return None

    viewer_path = os.path.join(os.path.dirname(__file__), "imu_viewer.py")
    if not os.path.exists(viewer_path):
        return None

    return subprocess.Popen(
        [
            sys.executable,
            viewer_path,
            "--imu_topic",
            args.imu_topic,
            "--enc_topic",
            args.hw_enc_topic,
        ]
    )


def main():
    args = build_argparser().parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
