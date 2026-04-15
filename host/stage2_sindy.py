#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from chrono_core.config import BridgeConfig
from chrono_core.model_parameter_io import load_model_parameter_json
from stage2_dataset import Stage2Trajectory, load_trajectories
from stage2_feature_map import DEFAULT_FEATURES, build_feature_matrix
from stage2_residual_target import build_residual_target, known_params_from_model_json


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y - yhat) ** 2)))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    den = float(np.sum((y - np.mean(y)) ** 2))
    if den <= 1e-12:
        return 0.0
    num = float(np.sum((y - yhat) ** 2))
    return float(1.0 - (num / den))


def _stlsq(phi: np.ndarray, y: np.ndarray, threshold: float, max_iter: int = 12) -> np.ndarray:
    coef = np.linalg.lstsq(phi, y, rcond=None)[0]
    active = np.abs(coef) >= float(threshold)
    for _ in range(max_iter):
        if not np.any(active):
            return np.zeros_like(coef)
        coef_active = np.linalg.lstsq(phi[:, active], y, rcond=None)[0]
        coef_new = np.zeros_like(coef)
        coef_new[active] = coef_active
        active_new = np.abs(coef_new) >= float(threshold)
        coef = coef_new
        if np.array_equal(active, active_new):
            break
        active = active_new
    return coef


def _fit_sparse(phi: np.ndarray, y: np.ndarray, threshold: float) -> tuple[np.ndarray, str]:
    try:
        import pysindy as ps  # type: ignore

        opt = ps.STLSQ(threshold=float(threshold), alpha=0.0, fit_intercept=False)
        opt.fit(phi, y)
        coef = np.asarray(opt.coef_, dtype=float).reshape(-1)
        return coef, "pysindy.STLSQ"
    except Exception:
        coef = _stlsq(phi, y, threshold=float(threshold), max_iter=12)
        return coef, "fallback_stlsq"


def _equation_string(names: list[str], coefs: np.ndarray, precision: int = 6) -> str:
    terms = []
    for n, c in zip(names, coefs):
        if abs(float(c)) <= 0.0:
            continue
        terms.append(f"{float(c):.{precision}g}*{n}")
    return "tau_residual = " + (" + ".join(terms) if terms else "0")


@dataclass
class Stage2Metrics:
    rmse: float
    r2: float


@dataclass
class Stage2Result:
    method: str
    timestamp_utc: str
    source_csvs: list[str]
    feature_library: list[str]
    optimizer: str
    active_terms: list[dict[str, float]]
    equation: str
    metrics_overall: Stage2Metrics
    metrics_per_trajectory: dict[str, Stage2Metrics]
    model_parameter_json: str
    output_equation_txt: str
    output_coeff_csv: str
    output_overlay_csv: str


def run_stage2(
    *,
    csv_paths: list[Path],
    model_parameter_json: Path,
    outdir: Path,
    features: list[str],
    threshold: float,
) -> Stage2Result:
    outdir.mkdir(parents=True, exist_ok=True)
    trajs: list[Stage2Trajectory] = load_trajectories(csv_paths)
    model_data = load_model_parameter_json(str(model_parameter_json))
    if model_data is None:
        model_data = {}
    cfg = BridgeConfig()
    known = known_params_from_model_json(model_data, cfg)

    phis = []
    ys = []
    per_traj = {}
    overlay_rows: list[dict[str, float | str]] = []
    for tr in trajs:
        rt = build_residual_target(tr, known)
        fm = build_feature_matrix(tr.theta, tr.omega, tr.motor_input_a, features)
        phis.append(fm.phi)
        ys.append(rt.tau_residual_target)
        per_traj[tr.name] = {
            "traj": tr,
            "target": rt.tau_residual_target,
            "phi": fm.phi,
        }
    names = list(features)

    phi_all = np.vstack(phis)
    y_all = np.concatenate(ys)
    coefs, optimizer_name = _fit_sparse(phi_all, y_all, threshold=float(threshold))
    yhat_all = phi_all @ coefs

    metrics_per: dict[str, Stage2Metrics] = {}
    for name, rec in per_traj.items():
        y = rec["target"]
        yhat = rec["phi"] @ coefs
        metrics_per[name] = Stage2Metrics(rmse=_rmse(y, yhat), r2=_r2(y, yhat))
        tr = rec["traj"]
        for i in range(len(tr.t)):
            overlay_rows.append(
                {
                    "trajectory": name,
                    "t": float(tr.t[i]),
                    "tau_residual_target": float(y[i]),
                    "tau_residual_pred": float(yhat[i]),
                    "theta": float(tr.theta[i]),
                    "omega": float(tr.omega[i]),
                    "motor_input_a": float(tr.motor_input_a[i]),
                }
            )

    active_terms = []
    for n, c in zip(names, coefs):
        if abs(float(c)) >= 1e-12:
            active_terms.append({"feature": str(n), "coeff": float(c)})

    eq = _equation_string(names, coefs, precision=8)
    overall = Stage2Metrics(rmse=_rmse(y_all, yhat_all), r2=_r2(y_all, yhat_all))

    coeff_csv = outdir / "stage2_coefficients.csv"
    with coeff_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["feature", "coeff", "active"])
        for n, c in zip(names, coefs):
            wr.writerow([n, float(c), int(abs(float(c)) >= 1e-12)])

    overlay_csv = outdir / "stage2_overlay.csv"
    with overlay_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(overlay_rows[0].keys()))
        wr.writeheader()
        wr.writerows(overlay_rows)

    eq_txt = outdir / "stage2_equation.txt"
    eq_txt.write_text(eq + "\n", encoding="utf-8")

    # Update model_parameter.json as canonical registry
    model_data.setdefault("version", 1)
    model_data.setdefault("known", {})
    model_data.setdefault("torque_model", {})
    model_data["torque_model"].setdefault("motor", {"enabled": True, "equation": "tau_motor = K_i * I_filtered_A", "params": {}})
    model_data["torque_model"].setdefault(
        "resistance",
        {"enabled": True, "equation": "tau_res = b_eq*omega + tau_eq*tanh(omega/eps)", "params": {}},
    )
    model_data["torque_model"]["residual_terms"] = active_terms
    model_data.setdefault("stage_outputs", {})
    model_data["stage_outputs"]["stage2"] = {
        "method": "greybox_residual_torque_sindy",
        "source_csvs": [str(p) for p in csv_paths],
        "feature_library": list(features),
        "optimizer": optimizer_name,
        "equation": eq,
        "active_terms": active_terms,
        "metrics_overall": asdict(overall),
        "metrics_per_trajectory": {k: asdict(v) for k, v in metrics_per.items()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_data["stage_outputs"].setdefault("stage1", None)
    model_data["stage_outputs"].setdefault("stage3", None)
    model_parameter_json.write_text(json.dumps(model_data, indent=2, ensure_ascii=False), encoding="utf-8")

    result = Stage2Result(
        method="greybox_residual_torque_sindy",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_csvs=[str(p) for p in csv_paths],
        feature_library=list(features),
        optimizer=optimizer_name,
        active_terms=active_terms,
        equation=eq,
        metrics_overall=overall,
        metrics_per_trajectory=metrics_per,
        model_parameter_json=str(model_parameter_json),
        output_equation_txt=str(eq_txt),
        output_coeff_csv=str(coeff_csv),
        output_overlay_csv=str(overlay_csv),
    )
    return result


def save_stage2_summary(result: Stage2Result, outdir: Path) -> Path:
    path = outdir / "stage2_summary.json"
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return path

