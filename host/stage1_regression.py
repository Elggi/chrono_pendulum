#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chrono_core.sindy_pipeline import (
    build_dataset_diagnostics,
    build_dataset_summary,
    build_residual_timeseries,
    export_llm_package,
    load_trajectories,
    preprocess_for_pysindy,
)


@dataclass
class RegressConfig:
    current_zero_thresh_a: float = 0.03
    tanh_eps: float = 0.05


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _estimate_free_decay(trajs, mgl: float, cfg: RegressConfig) -> dict:
    A_parts, y_parts, used = [], [], 0
    for tr in trajs:
        mask = np.abs(tr.current) <= cfg.current_zero_thresh_a
        if int(mask.sum()) < 16:
            continue
        used += int(mask.sum())
        A_parts.append(np.column_stack([tr.alpha[mask], tr.omega[mask], np.tanh(tr.omega[mask] / max(cfg.tanh_eps, 1e-6))]))
        y_parts.append(-mgl * np.sin(tr.theta[mask]))
    if not A_parts:
        raise RuntimeError("No free-decay samples found.")
    A = np.vstack(A_parts)
    y = np.concatenate(y_parts)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"J": float(x[0]), "b_eq": float(x[1]), "tau_eq": float(x[2]), "num_samples": used}


def _estimate_k_i(trajs, J: float, b_eq: float, tau_eq: float, mgl: float, cfg: RegressConfig) -> dict:
    A_parts, y_parts = [], []
    for tr in trajs:
        mask = np.abs(tr.current) > cfg.current_zero_thresh_a
        if int(mask.sum()) < 8:
            continue
        lhs = J * tr.alpha[mask] + b_eq * tr.omega[mask] + tau_eq * np.tanh(tr.omega[mask] / max(cfg.tanh_eps, 1e-6)) + mgl * np.sin(tr.theta[mask])
        A_parts.append(tr.current[mask][:, None])
        y_parts.append(lhs)
    if not A_parts:
        raise RuntimeError("No driven-current samples found for K_I estimation.")
    A = np.vstack(A_parts)
    y = np.concatenate(y_parts)
    return {"K_I": float(np.linalg.lstsq(A, y, rcond=None)[0][0]), "num_samples": int(len(y))}


def _joint_refine(trajs, mgl: float, cfg: RegressConfig) -> dict:
    A_parts, y_parts = [], []
    for tr in trajs:
        A_parts.append(np.column_stack([tr.current, -tr.omega, -np.tanh(tr.omega / max(cfg.tanh_eps, 1e-6)), -tr.alpha]))
        y_parts.append(mgl * np.sin(tr.theta))
    A = np.vstack(A_parts)
    y = np.concatenate(y_parts)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"K_I": float(x[0]), "b_eq": float(x[1]), "tau_eq": float(x[2]), "J": float(x[3])}


def _simulate(tr, params: dict, mgl: float, cfg: RegressConfig):
    n = len(tr.t)
    theta = np.zeros(n)
    omega = np.zeros(n)
    if n == 0:
        return theta, omega
    theta[0], omega[0] = tr.theta[0], tr.omega[0]
    for k in range(n - 1):
        dt = max(float(tr.t[k + 1] - tr.t[k]), 1e-4)
        domega = (
            params["K_I"] * tr.current[k]
            - params["b_eq"] * omega[k]
            - params["tau_eq"] * np.tanh(omega[k] / max(cfg.tanh_eps, 1e-6))
            - mgl * np.sin(theta[k])
        ) / max(params["J"], 1e-8)
        omega[k + 1] = omega[k] + dt * domega
        theta[k + 1] = theta[k] + dt * omega[k]
    return theta, omega


def _make_overlay_plots(tr, params: dict, mgl: float, cfg: RegressConfig, out_dir: Path) -> tuple[dict, dict]:
    theta_sim, omega_sim = _simulate(tr, params, mgl, cfg)
    alpha_sim = np.gradient(omega_sim, tr.t, edge_order=1) if len(tr.t) > 1 else np.zeros_like(omega_sim)

    overlays = {}
    for name, y_real, y_sim, ylabel in [
        ("theta", tr.theta, theta_sim, "theta [rad]"),
        ("omega", tr.omega, omega_sim, "omega [rad/s]"),
        ("alpha", tr.alpha, alpha_sim, "alpha [rad/s^2]"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(tr.t, y_real, label=f"{name}_real")
        ax.plot(tr.t, y_sim, "--", label=f"{name}_sim")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        p = out_dir / f"overlay_{name}.png"
        fig.savefig(p)
        plt.close(fig)
        overlays[name] = str(p)

    metrics = {
        "rmse_theta": _rmse(tr.theta, theta_sim),
        "rmse_omega": _rmse(tr.omega, omega_sim),
        "rmse_alpha": _rmse(tr.alpha, alpha_sim),
    }
    return metrics, overlays


def run_stage(args: argparse.Namespace) -> None:
    calib_path = Path(args.calibration_json or input("Select calibration_latest.json: ").strip())
    motor_path = Path(args.motor_torque_json or input("Select motor_torque.json: ").strip())
    csvs = args.csvs or [s.strip() for s in (input("CSV paths comma-separated: ") or "").split(",") if s.strip()]
    if not csvs:
        raise SystemExit("No CSV files provided")
    mode = args.mode or (input("Identification mode [staged_auto/free_decay/driven_current]: ").strip() or "staged_auto")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RegressConfig(current_zero_thresh_a=args.current_zero_thresh_a, tanh_eps=args.tanh_eps)
    motor = json.loads(motor_path.read_text(encoding="utf-8"))
    _ = json.loads(calib_path.read_text(encoding="utf-8"))

    trajs = load_trajectories(csvs)
    summary = build_dataset_summary(trajs, source_type=args.data_source)
    diagnostics = build_dataset_diagnostics(trajs)

    g = float(motor.get("gravity", 9.81))
    total_mass = float(motor["rod"]["mass"] + motor["imu"]["mass"] + motor.get("connector_cyl", {}).get("mass", 0.0))
    l_com = float(motor.get("dynamic_parameters", {}).get("l_com", abs(motor["rod"].get("com_local", [0, -0.14, 0])[1])))
    mgl = total_mass * g * l_com

    if mode == "free_decay":
        identified = _estimate_free_decay(trajs, mgl, cfg)
        identified["K_I"] = float(motor.get("dynamic_parameters", {}).get("K_I", 0.0))
    elif mode == "driven_current":
        dyn = motor.get("dynamic_parameters", {})
        identified = {"J": float(dyn.get("J", 0.02)), "b_eq": float(dyn.get("b_eq", 0.01)), "tau_eq": float(dyn.get("tau_eq", 0.0))}
        identified.update(_estimate_k_i(trajs, identified["J"], identified["b_eq"], identified["tau_eq"], mgl, cfg))
    else:
        s1 = _estimate_free_decay(trajs, mgl, cfg)
        s2 = _estimate_k_i(trajs, s1["J"], s1["b_eq"], s1["tau_eq"], mgl, cfg)
        identified = {**s1, **s2, **_joint_refine(trajs, mgl, cfg)}

    metrics, overlays = _make_overlay_plots(trajs[0], identified, mgl, cfg, out_dir)
    residual_df = build_residual_timeseries(trajs, identified, mgl, tanh_eps=cfg.tanh_eps)
    residual_csv = out_dir / "residual_timeseries.csv"
    residual_df.to_csv(residual_csv, index=False)

    residual_corr = {
        "corr_theta": float(np.corrcoef(residual_df["residual_alpha"], residual_df["theta"])[0, 1]) if len(residual_df) > 2 else 0.0,
        "corr_omega": float(np.corrcoef(residual_df["residual_alpha"], residual_df["omega"])[0, 1]) if len(residual_df) > 2 else 0.0,
        "corr_current": float(np.corrcoef(residual_df["residual_alpha"], residual_df["input_current"])[0, 1]) if len(residual_df) > 2 else 0.0,
        "near_zero_velocity_residual_std": float(residual_df.loc[np.abs(residual_df["omega"]) < 0.1, "residual_alpha"].std()),
        "high_speed_residual_std": float(residual_df.loc[np.abs(residual_df["omega"]) > 1.0, "residual_alpha"].std()),
    }

    regression_result = {
        "model_equation": "J*alpha = K_I*I - b_eq*omega - tau_eq*tanh(omega/tanh_eps) - m*g*l_com*sin(theta)",
        "known_constants": {"m_total": total_mass, "g": g, "l_com": l_com},
        "estimated_parameters": {
            "J_eff": float(identified["J"]),
            "b": float(identified["b_eq"]),
            "tau_0": float(identified["tau_eq"]),
            "K_I": float(identified["K_I"]),
        },
        "fit_metrics": metrics,
        "files_used": [str(Path(c).resolve()) for c in csvs],
        "residual_diagnostics": residual_corr,
        "dataset_key_diagnostics": {
            "free_decay_present": summary["experiment_regimes"]["free_decay_present"],
            "driven_data_present": summary["experiment_regimes"]["driven_data_present"],
            "theta_drift_suspicion": diagnostics["theta_drift_suspicion"],
        },
        "developer_hypothesis": "Residual structure correlated with omega/current suggests Option 6 greybox residual_alpha target.",
    }

    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "dataset_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (out_dir / "regression_result.json").write_text(json.dumps(regression_result, indent=2), encoding="utf-8")

    pysindy_npz = out_dir / "pysindy_input_arrays.npz"
    preprocess_summary = preprocess_for_pysindy(
        trajs,
        residual_df,
        out_npz=pysindy_npz,
        out_summary_json=out_dir / "pysindy_preprocess_summary.json",
        alpha_mode=args.alpha_mode,
    )

    export_llm_package(
        package_dir=out_dir / "llm_package",
        dataset_summary=summary,
        dataset_diagnostics=diagnostics,
        regression_result=regression_result,
        residual_df=residual_df,
        overlay_paths=overlays,
        pysindy_npz=pysindy_npz,
    )

    motor.setdefault("dynamic_parameters", {})
    motor["dynamic_parameters"].update({
        "J": float(identified["J"]),
        "K_I": float(identified["K_I"]),
        "b_eq": float(identified["b_eq"]),
        "tau_eq": float(identified["tau_eq"]),
        "l_com": float(l_com),
    })
    motor.setdefault("stage_metadata", {})
    motor["stage_metadata"].update({
        "active_equation": regression_result["model_equation"],
        "last_updated_stage": "option5_regression",
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "identification_mode": mode,
        "files_used": [str(Path(c).resolve()) for c in csvs],
    })
    motor_path.write_text(json.dumps(motor, indent=2), encoding="utf-8")

    print(json.dumps({"regression_result": regression_result, "preprocess_summary": preprocess_summary}, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Option 5 staged physics regression + LLM handoff package")
    ap.add_argument("--calibration_json", default="")
    ap.add_argument("--motor_torque_json", default="")
    ap.add_argument("--csvs", nargs="*", default=[])
    ap.add_argument("--mode", choices=["free_decay", "driven_current", "staged_auto"], default="")
    ap.add_argument("--data_source", choices=["real", "sim", "mixed"], default="mixed")
    ap.add_argument("--output_dir", default="host/run_logs")
    ap.add_argument("--current_zero_thresh_a", type=float, default=0.03)
    ap.add_argument("--tanh_eps", type=float, default=0.05)
    ap.add_argument("--alpha_mode", choices=["provided", "recompute_from_omega"], default="provided")
    return ap.parse_args()


if __name__ == "__main__":
    run_stage(parse_args())
