#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import traceback
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


def _fit_sparse(
    phi: np.ndarray,
    y: np.ndarray,
    threshold: float,
    feature_names: list[str],
    phi_trajs: list[np.ndarray] | None = None,
    y_trajs: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    try:
        import pysindy as ps

        # Use full PySINDy model API with identity library on prebuilt greybox features.
        # We fit a multi-output system where only output-0 carries the residual target.
        # This allows direct sparse regression in the SINDy workflow while keeping
        # runtime-compatible feature names.
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=float(threshold), alpha=0.0, normalize_columns=True),
            feature_library=ps.IdentityLibrary(),
        )
        if phi_trajs and y_trajs and len(phi_trajs) == len(y_trajs):
            x_list = []
            xdot_list = []
            for p, yt in zip(phi_trajs, y_trajs):
                x_i = np.asarray(p, dtype=float)
                xd_i = np.zeros_like(x_i, dtype=float)
                xd_i[:, 0] = np.asarray(yt, dtype=float).reshape(-1)
                x_list.append(x_i)
                xdot_list.append(xd_i)
            try:
                model.fit(x=x_list, t=1.0, x_dot=xdot_list, multiple_trajectories=True)
            except TypeError:
                # Compatibility path for older PySINDy APIs.
                x_cat = np.vstack(x_list)
                xd_cat = np.vstack(xdot_list)
                model.fit(x=x_cat, t=1.0, x_dot=xd_cat)
        else:
            x = np.asarray(phi, dtype=float)
            x_dot = np.zeros_like(x, dtype=float)
            x_dot[:, 0] = np.asarray(y, dtype=float).reshape(-1)
            model.fit(x=x, t=1.0, x_dot=x_dot)
        coef_mat = np.asarray(model.coefficients(), dtype=float)
        coef = coef_mat[0, :]
        return coef, "pysindy.SINDy(STLSQ+IdentityLibrary)"
    except Exception as exc:
        print("[ERROR] PySINDy fitting failed.")
        print(f"[ERROR] exception: {exc}")
        print("[ERROR] traceback:")
        print(traceback.format_exc())
        raise RuntimeError("Stage2 requires a working PySINDy installation (expected version: 2.1.0).") from exc


def _equation_string(names: list[str], coefs: np.ndarray, precision: int = 6) -> str:
    terms = []
    for n, c in zip(names, coefs):
        if abs(float(c)) <= 0.0:
            continue
        terms.append(f"{float(c):.{precision}g}*{n}")
    return "tau_residual = " + (" + ".join(terms) if terms else "0")


def _equation_string_alpha(names: list[str], coefs: np.ndarray, precision: int = 6) -> str:
    terms = []
    for n, c in zip(names, coefs):
        if abs(float(c)) <= 0.0:
            continue
        terms.append(f"{float(c):.{precision}g}*{n}")
    return "alpha_residual = " + (" + ".join(terms) if terms else "0")


def _simulate_rollout_with_alpha_residual(
    traj: Stage2Trajectory,
    known: Any,
    coefs_alpha: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, RolloutMetrics]:
    theta = np.asarray(traj.theta, dtype=float)
    omega = np.asarray(traj.omega, dtype=float)
    t = np.asarray(traj.t, dtype=float)
    motor_input = np.asarray(traj.motor_input_a, dtype=float)
    n = len(t)
    th_sim = np.zeros(n, dtype=float)
    om_sim = np.zeros(n, dtype=float)
    th_sim[0] = float(theta[0])
    om_sim[0] = float(omega[0])

    stable = True
    diverged_at = -1
    for k in range(1, n):
        dt = float(max(t[k] - t[k - 1], 1e-5))
        th = float(th_sim[k - 1])
        om = float(om_sim[k - 1])
        mi = float(motor_input[k - 1])
        fm = build_feature_matrix(
            np.asarray([th], dtype=float),
            np.asarray([om], dtype=float),
            np.asarray([mi], dtype=float),
            feature_names,
        )
        alpha_res = float(fm.phi[0] @ coefs_alpha)
        tau_motor = float(known.K_i) * mi
        tau_gravity = float(known.m_total * known.g * known.l_com) * math.sin(th)
        tau_visc = float(known.b_eq) * om
        tau_coul = float(known.tau_eq) * math.tanh(om / max(float(known.eps), 1e-9))
        alpha_known = (tau_motor - tau_gravity - tau_visc - tau_coul) / max(float(known.j_total), 1e-12)
        alpha = float(alpha_known + alpha_res)
        om_next = om + alpha * dt
        th_next = th + om_next * dt
        om_sim[k] = om_next
        th_sim[k] = th_next
        if not np.isfinite(om_next) or not np.isfinite(th_next) or abs(om_next) > 120.0 or abs(th_next) > 40.0:
            stable = False
            diverged_at = k
            om_sim[k:] = np.nan
            th_sim[k:] = np.nan
            break

    valid = np.isfinite(th_sim) & np.isfinite(om_sim)
    if int(np.sum(valid)) < 2:
        m = RolloutMetrics(theta_rmse=float("inf"), omega_rmse=float("inf"), stable=False, diverged_at_index=max(diverged_at, 0))
        return th_sim, om_sim, m
    m = RolloutMetrics(
        theta_rmse=_rmse(theta[valid], th_sim[valid]),
        omega_rmse=_rmse(omega[valid], om_sim[valid]),
        stable=bool(stable),
        diverged_at_index=int(diverged_at),
    )
    return th_sim, om_sim, m


@dataclass
class Stage2Metrics:
    rmse: float
    r2: float


@dataclass
class RolloutMetrics:
    theta_rmse: float
    omega_rmse: float
    stable: bool
    diverged_at_index: int


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
    rollout_metrics_per_trajectory: dict[str, RolloutMetrics]
    rollout_stable_all: bool
    deployment_readiness: str
    model_parameter_json: str
    output_equation_txt: str
    output_coeff_csv: str
    output_overlay_csv: str
    output_overlay_per_trajectory: dict[str, dict[str, str]]


def run_stage2(
    *,
    csv_paths: list[Path],
    model_parameter_json: Path,
    outdir: Path,
    features: list[str],
    threshold: float,
) -> Stage2Result:
    outdir.mkdir(parents=True, exist_ok=True)
    print("[Stage2] ===============================================")
    print("[Stage2] Greybox Residual-Torque SINDy (PySINDy 2.1.0-compatible path)")
    print(f"[Stage2] outdir: {outdir}")
    print(f"[Stage2] model_parameter_json: {model_parameter_json}")
    print(f"[Stage2] num_input_csv: {len(csv_paths)}")
    for i, p in enumerate(csv_paths, start=1):
        print(f"[Stage2]   csv[{i}]: {p}")
    print(f"[Stage2] feature_library: {features}")
    print(f"[Stage2] threshold: {threshold}")
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
    overlay_artifacts: dict[str, dict[str, str]] = {}
    for tr in trajs:
        rt = build_residual_target(tr, known)
        fm = build_feature_matrix(tr.theta, tr.omega, tr.motor_input_a, features)
        phis.append(fm.phi)
        ys.append(rt.alpha_residual_target)
        per_traj[tr.name] = {
            "traj": tr,
            "target": rt.alpha_residual_target,
            "phi": fm.phi,
            "target_tau_equiv": rt.tau_residual_target_equiv,
        }
    names = list(features)

    phi_all = np.vstack(phis)
    y_all = np.concatenate(ys)
    coefs, optimizer_name = _fit_sparse(
        phi_all,
        y_all,
        threshold=float(threshold),
        feature_names=names,
        phi_trajs=phis,
        y_trajs=ys,
    )
    yhat_all = phi_all @ coefs
    print(f"[Stage2] optimizer_used: {optimizer_name}")

    metrics_per: dict[str, Stage2Metrics] = {}
    rollout_metrics_per: dict[str, RolloutMetrics] = {}
    for name, rec in per_traj.items():
        y = rec["target"]
        yhat = rec["phi"] @ coefs
        metrics_per[name] = Stage2Metrics(rmse=_rmse(y, yhat), r2=_r2(y, yhat))
        tr = rec["traj"]
        traj_rows: list[dict[str, float | str]] = []
        for i in range(len(tr.t)):
            row = {
                "trajectory": name,
                "t": float(tr.t[i]),
                "alpha_residual_target": float(y[i]),
                "alpha_residual_pred": float(yhat[i]),
                "theta": float(tr.theta[i]),
                "omega": float(tr.omega[i]),
                "motor_input_a": float(tr.motor_input_a[i]),
            }
            overlay_rows.append(row)
            traj_rows.append(row)

        # Rollout-aware validation (deployment-style dynamics integration)
        th_sim, om_sim, rm = _simulate_rollout_with_alpha_residual(tr, known, coefs, names)
        rollout_metrics_per[name] = rm
        for i in range(len(traj_rows)):
            traj_rows[i]["theta_rollout"] = float(th_sim[i]) if np.isfinite(th_sim[i]) else float("nan")
            traj_rows[i]["omega_rollout"] = float(om_sim[i]) if np.isfinite(om_sim[i]) else float("nan")

        per_csv = outdir / f"stage2_overlay_{name}.csv"
        with per_csv.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
            wr.writeheader()
            wr.writerows(traj_rows)

        per_png = outdir / f"stage2_overlay_{name}.png"
        try:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            axs[0].plot(tr.t, y, label="alpha_residual_target")
            axs[0].plot(tr.t, yhat, label="alpha_residual_pred")
            axs[0].set_ylabel("alpha_residual [rad/s^2]")
            axs[0].grid(alpha=0.25)
            axs[0].legend()
            axs[1].plot(tr.t, tr.theta, label="theta_meas")
            axs[1].plot(tr.t, tr.omega, label="omega_meas")
            axs[1].plot(tr.t, th_sim, "--", label="theta_rollout")
            axs[1].plot(tr.t, om_sim, "--", label="omega_rollout")
            axs[1].set_xlabel("time [s]")
            axs[1].set_ylabel("state")
            axs[1].grid(alpha=0.25)
            axs[1].legend()
            fig.tight_layout()
            fig.savefig(per_png, dpi=160)
            plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Failed to save overlay plot for trajectory '{name}': {exc}")
            print(traceback.format_exc())

        overlay_artifacts[name] = {
            "overlay_csv": str(per_csv),
            "overlay_png": str(per_png),
            "rollout_stable": str(rm.stable),
            "rollout_theta_rmse": f"{rm.theta_rmse:.6e}",
            "rollout_omega_rmse": f"{rm.omega_rmse:.6e}",
        }

    active_terms = []
    for n, c in zip(names, coefs):
        if abs(float(c)) >= 1e-12:
            active_terms.append({"feature": str(n), "coeff": float(c)})

    eq = _equation_string_alpha(names, coefs, precision=8)
    coefs_tau = -float(known.j_total) * np.asarray(coefs, dtype=float)
    eq_tau = _equation_string(names, coefs_tau, precision=8)
    overall = Stage2Metrics(rmse=_rmse(y_all, yhat_all), r2=_r2(y_all, yhat_all))
    print(f"[Stage2] discovered_equation: {eq}")
    print(f"[Stage2] mapped_runtime_equation: {eq_tau}")
    print(f"[Stage2] overall_metrics: rmse={overall.rmse:.6e}, r2={overall.r2:.6f}")
    for k, v in metrics_per.items():
        print(f"[Stage2] per_trajectory[{k}]: rmse={v.rmse:.6e}, r2={v.r2:.6f}")
    for k, v in rollout_metrics_per.items():
        print(
            f"[Stage2] rollout[{k}]: stable={v.stable}, "
            f"theta_rmse={v.theta_rmse:.6e}, omega_rmse={v.omega_rmse:.6e}, diverged_at={v.diverged_at_index}"
        )

    rollout_stable_all = all(m.stable for m in rollout_metrics_per.values()) if rollout_metrics_per else False
    if rollout_stable_all and overall.r2 >= 0.5:
        deployment_readiness = "safe_for_runtime_deployment"
    elif rollout_stable_all:
        deployment_readiness = "rollout_stable_but_regression_weak"
    else:
        deployment_readiness = "regression_good_only_or_unstable_rollout"

    coeff_csv = outdir / "stage2_coefficients.csv"
    with coeff_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["feature", "coeff", "active"])
        for n, c in zip(names, coefs_tau):
            wr.writerow([n, float(c), int(abs(float(c)) >= 1e-12)])

    overlay_csv = outdir / "stage2_overlay.csv"
    with overlay_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(overlay_rows[0].keys()))
        wr.writeheader()
        wr.writerows(overlay_rows)

    eq_txt = outdir / "stage2_equation.txt"
    eq_txt.write_text(
        eq + "\n\n# mapped torque residual for runtime\n" + eq_tau + "\n",
        encoding="utf-8",
    )

    # Update model_parameter.json as canonical registry
    model_data.setdefault("version", 1)
    model_data.setdefault("known", {})
    model_data.setdefault("torque_model", {})
    model_data["torque_model"].setdefault("motor", {"enabled": True, "equation": "tau_motor = K_i * I_filtered_A", "params": {}})
    model_data["torque_model"].setdefault(
        "resistance",
        {"enabled": True, "equation": "tau_res = b_eq*omega + tau_eq*tanh(omega/eps)", "params": {}},
    )
    active_terms_tau = []
    for n, c in zip(names, coefs_tau):
        if abs(float(c)) >= 1e-12:
            active_terms_tau.append({"feature": str(n), "coeff": float(c)})
    model_data["torque_model"]["residual_terms"] = active_terms_tau
    model_data.setdefault("stage_outputs", {})
    model_data["stage_outputs"]["stage2"] = {
        "method": "greybox_residual_acceleration_sindy",
        "source_csvs": [str(p) for p in csv_paths],
        "feature_library": list(features),
        "optimizer": optimizer_name,
        "formulation": "residual_acceleration_sindy",
        "equation_alpha_residual": eq,
        "equation_tau_residual_runtime": eq_tau,
        "active_terms_alpha": active_terms,
        "active_terms_tau_runtime": active_terms_tau,
        "metrics_overall": asdict(overall),
        "metrics_per_trajectory": {k: asdict(v) for k, v in metrics_per.items()},
        "rollout_metrics_per_trajectory": {k: asdict(v) for k, v in rollout_metrics_per.items()},
        "rollout_stable_all": bool(rollout_stable_all),
        "deployment_readiness": deployment_readiness,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_data["stage_outputs"].setdefault("stage1", None)
    model_data["stage_outputs"].setdefault("stage3", None)
    model_parameter_json.write_text(json.dumps(model_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[Stage2] updated model_parameter.json:")
    print(f"[Stage2]   torque_model.residual_terms <- {active_terms_tau}")
    print(f"[Stage2]   stage_outputs.stage2.method <- greybox_residual_acceleration_sindy")
    print(f"[Stage2] deployment_readiness: {deployment_readiness}")

    result = Stage2Result(
        method="greybox_residual_acceleration_sindy",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_csvs=[str(p) for p in csv_paths],
        feature_library=list(features),
        optimizer=optimizer_name,
        active_terms=active_terms_tau,
        equation=eq_tau,
        metrics_overall=overall,
        metrics_per_trajectory=metrics_per,
        rollout_metrics_per_trajectory=rollout_metrics_per,
        rollout_stable_all=bool(rollout_stable_all),
        deployment_readiness=deployment_readiness,
        model_parameter_json=str(model_parameter_json),
        output_equation_txt=str(eq_txt),
        output_coeff_csv=str(coeff_csv),
        output_overlay_csv=str(overlay_csv),
        output_overlay_per_trajectory=overlay_artifacts,
    )
    print("[Stage2] output artifacts:")
    print(f"[Stage2]   equation_txt: {eq_txt}")
    print(f"[Stage2]   coefficients_csv: {coeff_csv}")
    print(f"[Stage2]   overlay_csv_combined: {overlay_csv}")
    for name, paths in overlay_artifacts.items():
        print(f"[Stage2]   overlay[{name}].csv: {paths.get('overlay_csv', '')}")
        print(f"[Stage2]   overlay[{name}].png: {paths.get('overlay_png', '')}")
    print("[Stage2] ===============================================")
    return result


def save_stage2_summary(result: Stage2Result, outdir: Path) -> Path:
    path = outdir / "stage2_summary.json"
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
