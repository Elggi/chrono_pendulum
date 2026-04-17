#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import sys

HOST_DIR = Path(__file__).resolve().parents[2]
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))
from typing import Any

import numpy as np

from chrono_core.config import BridgeConfig
from chrono_core.dynamics import PendulumModel, compute_model_torque_and_electrics
from chrono_core.model_parameter_io import load_model_parameter_json
from backend.stage2.dataset import Stage2Trajectory, load_trajectories
from stage2_settings import (
    DEFAULT_FEATURES,
    parse_feature_list,
    build_feature_matrix,
    known_params_from_model_json_with_trace,
    compute_residual_target,
    evaluate_residual_from_terms,
)


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
        print("[WARN] PySINDy path failed; falling back to deterministic STLSQ.")
        print(f"[WARN] exception: {exc}")
        print("[WARN] traceback:")
        print(traceback.format_exc())
        coef = _stlsq(phi, y, threshold=float(threshold), max_iter=12)
        return coef, "fallback_stlsq"


def _equation_string(names: list[str], coefs: np.ndarray, precision: int = 6) -> str:
    terms = []
    for n, c in zip(names, coefs):
        if abs(float(c)) <= 0.0:
            continue
        terms.append(f"{float(c):.{precision}g}*{n}")
    return "tau_residual = " + (" + ".join(terms) if terms else "0")


def evaluate_residual_torque(theta: float, omega: float, motor_input: float, active_terms: list[dict[str, float]], eps: float) -> float:
    return evaluate_residual_from_terms(theta, omega, motor_input, active_terms, eps)


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
    output_overlay_per_trajectory: dict[str, dict[str, str]]


def _rollout_identified_trajectory(
    tr: Stage2Trajectory,
    *,
    cfg: BridgeConfig,
    known: Any,
    residual_terms: list[dict[str, float]],
) -> dict[str, np.ndarray]:
    n = len(tr.t)
    model = PendulumModel(cfg)
    model.set_theta_kinematic(float(tr.theta[0]), float(tr.omega[0]))
    p = {
        "K_i": float(known.K_i),
        "b_eq": float(known.b_eq),
        "tau_eq": float(known.tau_eq),
        "residual_terms": list(residual_terms),
    }
    theta_sim = np.zeros(n, dtype=float)
    omega_sim = np.zeros(n, dtype=float)
    tau_motor_sim = np.zeros(n, dtype=float)
    tau_visc_sim = np.zeros(n, dtype=float)
    tau_coul_sim = np.zeros(n, dtype=float)
    tau_residual_sim = np.zeros(n, dtype=float)
    tau_res_sim = np.zeros(n, dtype=float)
    tau_net_sim = np.zeros(n, dtype=float)

    for k in range(n):
        theta_model = float(model.get_theta())
        out = compute_model_torque_and_electrics(
            motor_input=float(tr.motor_input_a[k]),
            theta=theta_model,
            omega=model.get_omega(),
            bus_v=float("nan"),
            p=p,
            cfg=cfg,
            cmd_u_for_duty=0.0,
        )
        theta_sim[k] = theta_model
        omega_sim[k] = float(model.get_omega())
        tau_motor_sim[k] = float(out["tau_motor"])
        tau_visc_sim[k] = float(out["tau_visc"])
        tau_coul_sim[k] = float(out["tau_coul"])
        tau_residual_sim[k] = float(out["tau_residual"])
        tau_res_sim[k] = float(out["tau_res"])
        tau_net_sim[k] = float(out["tau_net"])
        if k < n - 1:
            dt = float(max(tr.t[k + 1] - tr.t[k], cfg.step))
            model.apply_torque(out["tau_net"])
            model.step(dt)

    dt = np.diff(tr.t, prepend=tr.t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.maximum(dt, 1e-6)
    alpha_sim = np.gradient(omega_sim, dt)
    return {
        "theta_sim": theta_sim,
        "omega_sim": omega_sim,
        "alpha_sim": alpha_sim,
        "tau_motor_sim": tau_motor_sim,
        "tau_visc_sim": tau_visc_sim,
        "tau_coul_sim": tau_coul_sim,
        "tau_residual_sim": tau_residual_sim,
        "tau_res_sim": tau_res_sim,
        "tau_net_sim": tau_net_sim,
    }


def run_stage2(
    *,
    csv_paths: list[Path],
    model_parameter_json: Path,
    latest_model_parameter_json: Path,
    outdir: Path,
    features: list[str],
    threshold: float,
    target_mode: str = "greybox",
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
    known, known_trace = known_params_from_model_json_with_trace(model_data)
    print("[Stage2] residual target torque model:")
    if str(target_mode).strip().lower() == "blackbox":
        print("[Stage2]   tau_target = tau_total + tau_gravity")
    else:
        print("[Stage2]   tau_target = tau_total + tau_visc + tau_coul + tau_gravity - tau_motor")
    print("[Stage2]   gravity term is explicitly added to cancel measured gravity effect for Chrono injection.")
    print(f"[Stage2]   tau_motor = K_i * I_filtered_A, K_i={known.K_i:.9g}")
    print(f"[Stage2]   tau_gravity = m_total * g * l_com * sin(theta), m_total={known.m_total:.9g}, g={known.g:.9g}, l_com={known.l_com:.9g}")
    print(f"[Stage2]   tau_visc = b_eq * omega, b_eq={known.b_eq:.9g}")
    print(f"[Stage2]   tau_coul = tau_eq * tanh(omega/eps), tau_eq={known.tau_eq:.9g}, eps={known.eps:.9g}")
    print(f"[Stage2]   tau_total = J_total * alpha, J_total={known.j_total:.9g}")
    print("[Stage2]   alpha source = SmoothedFiniteDifference(d/dt of filtered omega), CSV alpha is ignored.")
    print("[Stage2]   known-parameter initialization (value | source):")
    print(f"[Stage2]     m_total={known.m_total:.9g} | {known_trace.get('m_total','unknown')}")
    print(f"[Stage2]     j_total={known.j_total:.9g} | {known_trace.get('j_total','unknown')}")
    print(f"[Stage2]     l_com={known.l_com:.9g} | {known_trace.get('l_com','unknown')}")
    print(f"[Stage2]     g={known.g:.9g} | {known_trace.get('g','unknown')}")
    print(f"[Stage2]     K_i={known.K_i:.9g} | {known_trace.get('K_i','unknown')}")
    print(f"[Stage2]     b_eq={known.b_eq:.9g} | {known_trace.get('b_eq','unknown')}")
    print(f"[Stage2]     tau_eq={known.tau_eq:.9g} | {known_trace.get('tau_eq','unknown')}")
    print(f"[Stage2]     eps={known.eps:.9g} | {known_trace.get('eps','unknown')}")
    print("[Stage2]   NOTE: feature 'motor_input' means measured input current [A].")

    features, parse_warnings = parse_feature_list(features)
    for w in parse_warnings:
        print(f"[Stage2][WARN] {w}")
    print(f"[Stage2] feature_library(sanitized): {features}")
    phis = []
    ys = []
    per_traj = {}
    overlay_rows: list[dict[str, float | str]] = []
    overlay_artifacts: dict[str, dict[str, str]] = {}
    for tr in trajs:
        dt_med = float(np.median(np.diff(tr.t))) if len(tr.t) >= 2 else float("nan")
        print(f"[Stage2] trajectory[{tr.name}]: samples={len(tr.t)}, dt_median={dt_med:.6g}s, alpha=SmoothedFiniteDifference(omega)")
        rt = compute_residual_target(tr.theta, tr.omega, tr.alpha, tr.motor_input_a, known, target_mode=target_mode)
        _, phi = build_feature_matrix(tr.theta, tr.omega, tr.motor_input_a, features, eps=known.eps)
        phis.append(phi)
        ys.append(rt.tau_residual_target)
        per_traj[tr.name] = {
            "traj": tr,
            "target": rt.tau_residual_target,
            "phi": phi,
        }
    names = list(features)

    phi_all = np.vstack(phis)
    y_all = np.concatenate(ys)
    if any(str(f).strip() == "1" for f in names):
        raise ValueError("Stage2 disallows constant feature '1'. Remove it from --features.")
    n_all = len(y_all)
    idx = np.arange(n_all, dtype=int)
    tr_idx = idx[idx % 2 == 0]
    va_idx = idx[idx % 2 == 1]
    if len(va_idx) < 4:
        tr_idx = idx
        va_idx = idx
    threshold_grid = sorted(set([max(float(threshold) / 10.0, 1e-8), float(threshold), float(threshold) * 10.0]))
    candidate_rows: list[dict[str, float | int | str]] = []
    best = None
    for th in threshold_grid:
        coef_i, opt_i = _fit_sparse(phi_all[tr_idx], y_all[tr_idx], threshold=th, feature_names=names)
        yhat_va = phi_all[va_idx] @ coef_i
        rmse_va = _rmse(y_all[va_idx], yhat_va)
        active_n = int(np.sum(np.abs(coef_i) >= 1e-12))
        rank_key = (rmse_va, active_n)
        candidate_rows.append(
            {
                "threshold": float(th),
                "optimizer": str(opt_i),
                "rmse_validation": float(rmse_va),
                "active_terms": int(active_n),
            }
        )
        if best is None or rank_key < best["rank_key"]:
            best = {"rank_key": rank_key, "coef": coef_i, "optimizer": opt_i, "threshold": float(th)}
    assert best is not None
    coefs = np.asarray(best["coef"], dtype=float)
    optimizer_name = str(best["optimizer"])
    selected_threshold = float(best["threshold"])
    yhat_all = phi_all @ coefs
    print(f"[Stage2] optimizer_used: {optimizer_name}")
    print(f"[Stage2] selected_threshold: {selected_threshold}")
    identified_terms = [{"feature": str(n), "coeff": float(c)} for n, c in zip(names, coefs) if abs(float(c)) >= 1e-12]

    metrics_per: dict[str, Stage2Metrics] = {}
    for name, rec in per_traj.items():
        y = rec["target"]
        yhat = rec["phi"] @ coefs
        metrics_per[name] = Stage2Metrics(rmse=_rmse(y, yhat), r2=_r2(y, yhat))
        tr = rec["traj"]
        sim_rollout = _rollout_identified_trajectory(
            tr,
            cfg=cfg,
            known=known,
            residual_terms=identified_terms,
        )
        traj_rows: list[dict[str, float | str]] = []
        for i in range(len(tr.t)):
            row = {
                "trajectory": name,
                "t": float(tr.t[i]),
                "tau_residual_target": float(y[i]),
                "tau_residual_pred": float(yhat[i]),
                "theta_real": float(tr.theta[i]),
                "theta_sim": float(sim_rollout["theta_sim"][i]),
                "omega_real": float(tr.omega[i]),
                "omega_sim": float(sim_rollout["omega_sim"][i]),
                "alpha_real": float(tr.alpha[i]),
                "alpha_sim": float(sim_rollout["alpha_sim"][i]),
                "motor_input_a": float(tr.motor_input_a[i]),
                "tau_motor_sim": float(sim_rollout["tau_motor_sim"][i]),
                "tau_visc_sim": float(sim_rollout["tau_visc_sim"][i]),
                "tau_coul_sim": float(sim_rollout["tau_coul_sim"][i]),
                "tau_residual_sim": float(sim_rollout["tau_residual_sim"][i]),
                "tau_res_sim": float(sim_rollout["tau_res_sim"][i]),
                "tau_net_sim": float(sim_rollout["tau_net_sim"][i]),
            }
            overlay_rows.append(row)
            traj_rows.append(row)

        per_csv = outdir / f"stage2_overlay_{name}.csv"
        with per_csv.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
            wr.writeheader()
            wr.writerows(traj_rows)

        per_png = outdir / f"stage2_overlay_{name}.png"
        try:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(tr.t, y, label="tau_residual_target")
            axs[0].plot(tr.t, yhat, label="tau_residual_pred")
            axs[0].set_ylabel("tau_residual [Nm]")
            axs[0].grid(alpha=0.25)
            axs[0].legend()
            axs[1].plot(tr.t, tr.theta, label="theta_real")
            axs[1].plot(tr.t, sim_rollout["theta_sim"], label="theta_sim")
            axs[1].set_ylabel("theta [rad]")
            axs[1].grid(alpha=0.25)
            axs[1].legend()
            axs[2].plot(tr.t, tr.omega, label="omega_real")
            axs[2].plot(tr.t, sim_rollout["omega_sim"], label="omega_sim")
            axs[2].set_xlabel("time [s]")
            axs[2].set_ylabel("omega [rad/s]")
            axs[2].grid(alpha=0.25)
            axs[2].legend()
            fig.tight_layout()
            fig.savefig(per_png, dpi=160)
            plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Failed to save overlay plot for trajectory '{name}': {exc}")
            print(traceback.format_exc())

        overlay_artifacts[name] = {
            "overlay_csv": str(per_csv),
            "overlay_png": str(per_png),
        }

    active_terms = list(identified_terms)

    eq = _equation_string(names, coefs, precision=8)
    overall = Stage2Metrics(rmse=_rmse(y_all, yhat_all), r2=_r2(y_all, yhat_all))
    print(f"[Stage2] discovered_equation: {eq}")
    print(f"[Stage2] overall_metrics: rmse={overall.rmse:.6e}, r2={overall.r2:.6f}")
    for k, v in metrics_per.items():
        print(f"[Stage2] per_trajectory[{k}]: rmse={v.rmse:.6e}, r2={v.r2:.6f}")

    coeff_csv = outdir / "stage2_coefficients.csv"
    with coeff_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["feature", "coeff", "active"])
        for n, c in zip(names, coefs):
            wr.writerow([n, float(c), int(abs(float(c)) >= 1e-12)])

    candidate_csv = outdir / "stage2_candidate_models.csv"
    with candidate_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["threshold", "optimizer", "rmse_validation", "active_terms"])
        wr.writeheader()
        wr.writerows(candidate_rows)

    overlay_csv = outdir / "stage2_overlay.csv"
    with overlay_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(overlay_rows[0].keys()))
        wr.writeheader()
        wr.writerows(overlay_rows)

    eq_txt = outdir / "stage2_equation.txt"
    eq_txt.write_text(eq + "\n", encoding="utf-8")
    residual_eq_txt = outdir / "residual_torque_equation.txt"
    residual_eq_txt.write_text(eq + "\n", encoding="utf-8")

    feature_ranges: dict[str, dict[str, float]] = {}
    torque_contribution_ranges: dict[str, dict[str, float]] = {}
    for j, name in enumerate(names):
        col = phi_all[:, j]
        feature_ranges[name] = {"min": float(np.nanmin(col)), "max": float(np.nanmax(col))}
        contrib = float(coefs[j]) * col
        torque_contribution_ranges[name] = {
            "min": float(np.nanmin(contrib)),
            "max": float(np.nanmax(contrib)),
            "max_abs": float(np.nanmax(np.abs(contrib))),
        }
    active_abs = [v["max_abs"] for k, v in torque_contribution_ranges.items() if abs(float(dict(zip(names, coefs))[k])) >= 1e-12]
    med_abs = float(np.median(active_abs)) if active_abs else 0.0
    dominance_warnings = []
    if med_abs > 0.0:
        for n, v in torque_contribution_ranges.items():
            if v["max_abs"] > 5.0 * med_abs:
                msg = f"term '{n}' dominates: max_abs={v['max_abs']:.6g} (>5x median {med_abs:.6g})"
                dominance_warnings.append(msg)
                print(f"[Stage2][WARN] {msg}")

    residual_fit_csv = outdir / "residual_torque_fit.csv"
    with residual_fit_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(overlay_rows[0].keys()))
        wr.writeheader()
        wr.writerows(overlay_rows)

    residual_model_json = outdir / "residual_torque_model.json"
    residual_diag_json = outdir / "residual_torque_diagnostics.json"

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
        "target_mode": str(target_mode),
        "source_csvs": [str(p) for p in csv_paths],
        "feature_library": list(features),
        "optimizer": optimizer_name,
        "threshold": float(selected_threshold),
        "equation": eq,
        "active_terms": active_terms,
        "metrics_overall": asdict(overall),
        "metrics_per_trajectory": {k: asdict(v) for k, v in metrics_per.items()},
        "feature_ranges": feature_ranges,
        "torque_contribution_ranges": torque_contribution_ranges,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_data["stage_outputs"].setdefault("stage1", None)
    model_data["stage_outputs"].setdefault("stage3", None)
    payload = json.dumps(model_data, indent=2, ensure_ascii=False)
    model_parameter_json.write_text(payload, encoding="utf-8")
    latest_model_parameter_json.write_text(payload, encoding="utf-8")
    print("[Stage2] updated model_parameter.json:")
    print(f"[Stage2]   torque_model.residual_terms <- {active_terms}")
    print(f"[Stage2]   stage_outputs.stage2.method <- greybox_residual_torque_sindy")
    print(f"[Stage2]   latest synchronized <- {latest_model_parameter_json}")

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
        output_overlay_per_trajectory=overlay_artifacts,
    )
    print("[Stage2] output artifacts:")
    print(f"[Stage2]   equation_txt: {eq_txt}")
    print(f"[Stage2]   coefficients_csv: {coeff_csv}")
    print(f"[Stage2]   overlay_csv_combined: {overlay_csv}")
    print(f"[Stage2]   candidate_models_csv: {candidate_csv}")
    print(f"[Stage2]   residual_model_json: {residual_model_json}")
    print(f"[Stage2]   residual_diagnostics_json: {residual_diag_json}")
    for name, paths in overlay_artifacts.items():
        print(f"[Stage2]   overlay[{name}].csv: {paths.get('overlay_csv', '')}")
        print(f"[Stage2]   overlay[{name}].png: {paths.get('overlay_png', '')}")
    print("[Stage2] ===============================================")

    residual_payload = {
        "target_mode": str(target_mode),
        "feature_library": list(features),
        "optimizer": optimizer_name,
        "threshold": float(selected_threshold),
        "active_terms": active_terms,
        "equation": eq,
        "metrics_overall": asdict(overall),
        "metrics_per_trajectory": {k: asdict(v) for k, v in metrics_per.items()},
        "feature_ranges": feature_ranges,
        "torque_contribution_ranges": torque_contribution_ranges,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    residual_model_json.write_text(json.dumps(residual_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    residual_diag_json.write_text(
        json.dumps(
            {
                "candidate_models": candidate_rows,
                "dominance_warnings": dominance_warnings,
                "sanity_report": {
                    "median_active_term_max_abs_torque": med_abs,
                    "num_warnings": len(dominance_warnings),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


def save_stage2_summary(result: Stage2Result, outdir: Path) -> Path:
    path = outdir / "stage2_summary.json"
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
