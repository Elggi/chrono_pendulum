from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TrajectoryData:
    trajectory_id: str
    source_file: str
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    current: np.ndarray


def _to_num(series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr


def _fill(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float).copy()
    if len(x) == 0:
        return x
    m = np.isfinite(x)
    if not m.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[~m] = np.interp(idx[~m], idx[m], x[m])
    return x


def load_trajectories(csv_paths: list[str]) -> list[TrajectoryData]:
    out: list[TrajectoryData] = []
    for i, p in enumerate(csv_paths):
        df = pd.read_csv(p)
        t = _to_num(df["time"] if "time" in df.columns else df["wall_elapsed"])
        theta = _to_num(df["theta"]) if "theta" in df.columns else np.zeros(len(df))
        omega = _to_num(df["omega"]) if "omega" in df.columns else np.zeros(len(df))
        if "alpha" in df.columns:
            alpha = _to_num(df["alpha"])
        else:
            alpha = np.gradient(_fill(omega), _fill(t), edge_order=1) if len(df) > 1 else np.zeros(len(df))
        if "input_current" in df.columns:
            current = _to_num(df["input_current"])
        elif "ina_current_signed_mA" in df.columns:
            current = _to_num(df["ina_current_signed_mA"]) * 0.001
        else:
            current = np.zeros(len(df))
        out.append(TrajectoryData(
            trajectory_id=f"traj_{i}",
            source_file=str(Path(p).resolve()),
            t=_fill(t),
            theta=_fill(theta),
            omega=_fill(omega),
            alpha=_fill(alpha),
            current=_fill(current),
        ))
    return out


def build_dataset_summary(trajs: list[TrajectoryData], source_type: str = "mixed") -> dict:
    dts = np.concatenate([np.diff(t.t) for t in trajs if len(t.t) > 1]) if trajs else np.array([])
    nominal_fs = float(1.0 / np.median(dts)) if len(dts) > 0 else 0.0
    actual_fs = float(1.0 / np.mean(dts)) if len(dts) > 0 else 0.0
    total_duration = float(np.sum([max(0.0, tr.t[-1] - tr.t[0]) if len(tr.t) > 1 else 0.0 for tr in trajs]))
    all_current = np.concatenate([tr.current for tr in trajs]) if trajs else np.array([])
    all_theta = np.concatenate([tr.theta for tr in trajs]) if trajs else np.array([])
    all_omega = np.concatenate([tr.omega for tr in trajs]) if trajs else np.array([])
    all_alpha = np.concatenate([tr.alpha for tr in trajs]) if trajs else np.array([])

    zero_crossings = int(np.sum(np.diff(np.signbit(all_theta)) != 0)) if len(all_theta) > 1 else 0
    drift = float(np.median(np.abs(np.diff(all_theta, prepend=all_theta[0])))) if len(all_theta) > 1 else 0.0

    return {
        "data_source": source_type,
        "num_files": len(trajs),
        "total_duration_sec": total_duration,
        "sampling_rate_hz_nominal": nominal_fs,
        "sampling_rate_hz_actual": actual_fs,
        "signals_available": {"theta": True, "omega": True, "alpha": True, "input_current": True},
        "preprocessing": {
            "filtering_method": "from input filtered csv",
            "alpha_derivation_method": "provided alpha preferred; gradient fallback",
            "unwrapping": "not modified in preprocess",
        },
        "experiment_regimes": {
            "free_decay_present": bool(np.any(np.abs(all_current) < 0.03)),
            "driven_data_present": bool(np.any(np.abs(all_current) >= 0.03)),
            "multiple_amplitudes_present": bool(len(np.unique(np.round(np.abs(all_current), 2))) > 4),
            "zero_crossings_present": bool(zero_crossings > 0),
        },
        "noise_characteristics": {
            "theta_drift": drift,
            "omega_noise_std": float(np.std(all_omega)) if len(all_omega) else 0.0,
            "alpha_noise_std": float(np.std(all_alpha)) if len(all_alpha) else 0.0,
        },
        "anomalies_detected": [],
    }


def build_dataset_diagnostics(trajs: list[TrajectoryData]) -> dict:
    diag = {
        "trajectory_count": len(trajs),
        "free_decay_like_segments": [],
        "driven_current_segments": [],
        "multiple_amplitude_detection": False,
        "zero_crossing_counts": {},
        "theta_drift_suspicion": False,
        "omega_noise_estimate": 0.0,
        "alpha_noise_estimate": 0.0,
        "missing_values": 0,
        "irregular_timestamps": False,
        "duplicate_timestamps": False,
        "clipping_suspected": False,
        "unwrap_failure_suspected": False,
        "spike_discontinuity_suspected": False,
        "warnings": [],
        "confidence": {
            "free_decay_detection": "medium",
            "driven_detection": "high",
            "dt_quality": "medium",
        },
    }

    all_cur = []
    all_om = []
    all_al = []
    for tr in trajs:
        dt = np.diff(tr.t) if len(tr.t) > 1 else np.array([])
        if len(dt) > 0:
            if np.any(dt <= 0):
                diag["duplicate_timestamps"] = True
            if np.std(dt) > 0.5 * max(np.mean(dt), 1e-9):
                diag["irregular_timestamps"] = True
        zc = int(np.sum(np.diff(np.signbit(tr.theta)) != 0)) if len(tr.theta) > 1 else 0
        diag["zero_crossing_counts"][tr.trajectory_id] = zc
        free_idx = np.where(np.abs(tr.current) < 0.03)[0]
        drv_idx = np.where(np.abs(tr.current) >= 0.03)[0]
        if len(free_idx) > 0:
            diag["free_decay_like_segments"].append({"trajectory_id": tr.trajectory_id, "samples": int(len(free_idx))})
        if len(drv_idx) > 0:
            diag["driven_current_segments"].append({"trajectory_id": tr.trajectory_id, "samples": int(len(drv_idx))})
        all_cur.append(tr.current)
        all_om.append(tr.omega)
        all_al.append(tr.alpha)

    cur = np.concatenate(all_cur) if all_cur else np.array([])
    om = np.concatenate(all_om) if all_om else np.array([])
    al = np.concatenate(all_al) if all_al else np.array([])
    diag["multiple_amplitude_detection"] = bool(len(np.unique(np.round(np.abs(cur), 2))) > 4) if len(cur) else False
    diag["theta_drift_suspicion"] = bool(any(abs(tr.theta[-1] - tr.theta[0]) > 0.5 for tr in trajs if len(tr.theta) > 2))
    diag["omega_noise_estimate"] = float(np.std(om)) if len(om) else 0.0
    diag["alpha_noise_estimate"] = float(np.std(al)) if len(al) else 0.0
    diag["clipping_suspected"] = bool(np.any(np.abs(cur) > 4.5)) if len(cur) else False
    diag["spike_discontinuity_suspected"] = bool(np.any(np.abs(np.diff(al)) > 50.0)) if len(al) > 1 else False
    return diag


def build_residual_timeseries(trajs: list[TrajectoryData], params: dict, mgl: float, tanh_eps: float = 0.05) -> pd.DataFrame:
    rows = []
    J = max(float(params["J"]), 1e-8)
    KI = float(params["K_I"])
    b = float(params["b_eq"])
    tau = float(params["tau_eq"])
    for tr in trajs:
        alpha_model = (KI * tr.current - b * tr.omega - tau * np.tanh(tr.omega / max(tanh_eps, 1e-6)) - mgl * np.sin(tr.theta)) / J
        residual = tr.alpha - alpha_model
        for k in range(len(tr.t)):
            rows.append({
                "trajectory_id": tr.trajectory_id,
                "time": float(tr.t[k]),
                "theta": float(tr.theta[k]),
                "omega": float(tr.omega[k]),
                "alpha_data": float(tr.alpha[k]),
                "alpha_model": float(alpha_model[k]),
                "residual_alpha": float(residual[k]),
                "input_current": float(tr.current[k]),
            })
    return pd.DataFrame(rows)


def export_llm_package(
    package_dir: Path,
    dataset_summary: dict,
    dataset_diagnostics: dict,
    regression_result: dict,
    residual_df: pd.DataFrame,
    overlay_paths: dict,
    pysindy_npz: Path | None = None,
) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    (package_dir / "dataset_diagnostics.json").write_text(json.dumps(dataset_diagnostics, indent=2), encoding="utf-8")
    (package_dir / "regression_result.json").write_text(json.dumps(regression_result, indent=2), encoding="utf-8")
    residual_df.to_csv(package_dir / "residual_timeseries.csv", index=False)

    for name, src in overlay_paths.items():
        if src and Path(src).exists():
            dst = package_dir / Path(src).name
            dst.write_bytes(Path(src).read_bytes())
    if pysindy_npz is not None and pysindy_npz.exists():
        (package_dir / pysindy_npz.name).write_bytes(pysindy_npz.read_bytes())

    readme = (
        "llm_package contents:\n"
        "- dataset_summary.json: high-level dataset characteristics\n"
        "- dataset_diagnostics.json: rule-based signal diagnostics/flags\n"
        "- regression_result.json: Option 5 parameter/result handoff\n"
        "- residual_timeseries.csv: residual alpha time-series for SINDy\n"
        "- overlay PNG files: sim-vs-real comparison\n\n"
        "Next step: feed this package to external LLM for PySINDy hyperparameter recommendation\n"
        "for Option 6 (greybox residual) or Option 8 (blackbox full).\n"
    )
    (package_dir / "README.txt").write_text(readme, encoding="utf-8")


def preprocess_for_pysindy(
    trajs: list[TrajectoryData],
    residual_df: pd.DataFrame | None,
    out_npz: Path,
    out_summary_json: Path,
    alpha_mode: str = "provided",
) -> dict:
    warnings = []
    time_uniform = True
    t_list, x_list, u_list, y_full_list, y_resid_list = [], [], [], [], []

    residual_map: dict[str, np.ndarray] = {}
    if residual_df is not None and not residual_df.empty:
        for tid, g in residual_df.groupby("trajectory_id"):
            residual_map[str(tid)] = pd.to_numeric(g["residual_alpha"], errors="coerce").to_numpy(dtype=float)

    for tr in trajs:
        dt = np.diff(tr.t) if len(tr.t) > 1 else np.array([])
        if len(dt) > 0 and np.std(dt) > 0.05 * max(np.mean(dt), 1e-9):
            time_uniform = False
        alpha = tr.alpha
        if alpha_mode == "recompute_from_omega":
            alpha = np.gradient(tr.omega, tr.t, edge_order=1) if len(tr.t) > 1 else np.zeros_like(tr.omega)
            warnings.append(f"{tr.trajectory_id}: alpha recomputed from omega")

        t_list.append(tr.t)
        x_list.append(np.column_stack([tr.theta, tr.omega]))
        u_list.append(tr.current.reshape(-1, 1))
        y_full_list.append(alpha)
        y_resid_list.append(residual_map.get(tr.trajectory_id, np.zeros_like(alpha)))

    np.savez(
        out_npz,
        t=np.array(t_list, dtype=object),
        X=np.array(x_list, dtype=object),
        U=np.array(u_list, dtype=object),
        X_dot=np.array(y_full_list, dtype=object),
        full_alpha=np.array(y_full_list, dtype=object),
        residual_alpha=np.array(y_resid_list, dtype=object),
        trajectory_id=np.array([tr.trajectory_id for tr in trajs], dtype=object),
        source_files=np.array([tr.source_file for tr in trajs], dtype=object),
    )

    summary = {
        "num_trajectories": len(trajs),
        "trajectory_lengths": [int(len(tr.t)) for tr in trajs],
        "time_uniform": time_uniform,
        "input_current_present": True,
        "alpha_mode": alpha_mode,
        "alpha_provided": alpha_mode == "provided",
        "warnings": warnings,
    }
    out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_llm_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def update_motor_torque_sindy_entry(
    motor_path: Path,
    stage: str,
    mode: str,
    target_type: str,
    config_used: dict,
    active_terms: list[str],
    coefficients: list[float],
    equation_human: str,
    fit_metrics: dict,
    source_files: list[str],
) -> None:
    motor = json.loads(motor_path.read_text(encoding="utf-8"))
    motor.setdefault("identified_models", {})
    entries = motor["identified_models"].setdefault("sindy_runs", [])
    entries.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "mode": mode,
            "sindy_mode": mode,
            "target_type": target_type,
            "feature_library": config_used.get("library"),
            "optimizer": config_used.get("optimizer"),
            "threshold": config_used.get("threshold"),
            "active_terms": active_terms,
            "coefficients": coefficients,
            "equation_human": equation_human,
            "fit_metrics": fit_metrics,
            "source_files": source_files,
            "config_used": config_used,
        }
    )
    motor.setdefault("stage_metadata", {})
    motor["stage_metadata"]["last_updated_stage"] = stage
    motor_path.write_text(json.dumps(motor, indent=2), encoding="utf-8")
