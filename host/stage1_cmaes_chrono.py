#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from cmaes import CMA

from chrono_core.calibration_io import apply_calibration_json
from chrono_core.config import BridgeConfig
from chrono_core.dynamics import PendulumModel, compute_model_torque_and_electrics
from chrono_core.model_parameter_io import load_model_parameter_json, extract_runtime_overrides


def _pick_col(cols: dict[str, np.ndarray], candidates: list[str]) -> np.ndarray:
    for c in candidates:
        if c in cols:
            return cols[c]
    raise KeyError(f"Missing required columns. tried={candidates}, have={list(cols.keys())}")


def load_free_decay_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows:
        raise ValueError(f"No rows in csv: {path}")
    cols: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k in cols.keys():
            try:
                cols[k].append(float(r[k]))
            except Exception:
                cols[k].append(float("nan"))
    arr = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    t = _pick_col(arr, ["wall_elapsed", "t", "time", "time_sec"])
    theta = _pick_col(arr, ["theta_imu_filtered_unwrapped", "theta", "theta_imu"])
    omega = _pick_col(arr, ["omega_imu_filtered", "omega", "omega_imu"])
    n = len(t)
    i_ma = None
    for c in ["I_filtered_mA", "ina_current_corr_mA", "ina_current_raw_mA", "current_mA", "ina_current_signed_online_mA"]:
        if c in arr:
            i_ma = np.asarray(arr[c], dtype=float)
            break
    if i_ma is None:
        i_ma = np.zeros(n, dtype=float)
    motor_input = i_ma / 1000.0  # A
    return {"t": t, "theta": theta, "omega": omega, "motor_input": motor_input}


def _first_hold_start(mask: np.ndarray, min_len: int) -> int | None:
    if min_len <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if idx.size > 0 else None
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if bool(ok) else 0
        if run >= min_len:
            return int(i - min_len + 1)
    return None


def trim_long_tail(
    dataset: dict[str, np.ndarray],
    *,
    tail_theta_abs_deg: float,
    tail_omega_abs: float,
    tail_hold_sec: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    t = np.asarray(dataset["t"], dtype=float)
    theta = np.asarray(dataset["theta"], dtype=float)
    omega = np.asarray(dataset["omega"], dtype=float)
    n = len(t)
    if n < 4:
        return {"t": t, "theta": theta, "omega": omega}, {"trimmed": 0.0, "n_before": float(n), "n_after": float(n)}

    theta_eps = math.radians(max(float(tail_theta_abs_deg), 0.0))
    omega_eps = max(float(tail_omega_abs), 0.0)
    dt_med = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt_med) or dt_med <= 0.0:
        dt_med = 1e-2
    min_len = max(2, int(round(max(float(tail_hold_sec), 0.0) / dt_med)))

    quiet = np.isfinite(theta) & np.isfinite(omega) & (np.abs(theta) <= theta_eps) & (np.abs(omega) <= omega_eps)
    cut_start = _first_hold_start(quiet, min_len=min_len)
    if cut_start is None:
        return {"t": t, "theta": theta, "omega": omega}, {"trimmed": 0.0, "n_before": float(n), "n_after": float(n)}
    if cut_start <= 2:
        return {"t": t, "theta": theta, "omega": omega}, {"trimmed": 0.0, "n_before": float(n), "n_after": float(n)}

    trimmed = {"t": t[:cut_start], "theta": theta[:cut_start], "omega": omega[:cut_start]}
    return trimmed, {"trimmed": 1.0, "n_before": float(n), "n_after": float(len(trimmed["t"]))}


def build_cfg(calibration_json: str, parameter_json: str) -> tuple[BridgeConfig, dict[str, Any], dict[str, Any]]:
    cfg = BridgeConfig()
    cfg.enable_render = False
    cfg.enable_imu_viewer = False
    calib = apply_calibration_json(cfg, calibration_json)
    param_data = load_model_parameter_json(parameter_json)
    runtime = extract_runtime_overrides(param_data, cfg) if param_data is not None else {}
    if "r_imu" in runtime:
        cfg.r_imu = float(runtime["r_imu"])
    if "gravity" in runtime:
        cfg.gravity = float(runtime["gravity"])
    return cfg, ({} if calib is None else calib), runtime


def rollout_loss_for_candidate(
    candidate: np.ndarray,
    dataset: dict[str, np.ndarray],
    cfg_dict: dict[str, Any],
    base_params: dict[str, Any],
    w_theta: float,
    w_omega: float,
    fit_ki_with_motor_input: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    cfg = BridgeConfig(**cfg_dict)
    model = PendulumModel(cfg)

    t = dataset["t"]
    theta_real = dataset["theta"]
    omega_real = dataset["omega"]
    theta0 = float(theta_real[0])
    omega0 = 0.0
    model.set_theta_kinematic(theta0, omega0)

    p = dict(base_params)
    if fit_ki_with_motor_input:
        p["K_i"] = float(candidate[0])
        p["b_eq"] = float(candidate[1])
        p["tau_eq"] = float(candidate[2])
    else:
        p["b_eq"] = float(candidate[0])
        p["tau_eq"] = float(candidate[1])
    motor_input_arr = np.asarray(dataset.get("motor_input", np.zeros_like(theta_real)), dtype=float)
    theta_sim = np.zeros_like(theta_real)
    omega_sim = np.zeros_like(omega_real)
    theta_sim[0] = model.get_theta()
    omega_sim[0] = model.get_omega()

    for k in range(1, len(t)):
        dt = float(max(t[k] - t[k - 1], cfg.step))
        out = compute_model_torque_and_electrics(
            motor_input=float(motor_input_arr[k - 1]),
            theta=model.get_theta(),
            omega=model.get_omega(),
            bus_v=float("nan"),
            p=p,
            cfg=cfg,
            cmd_u_for_duty=0.0,
        )
        model.apply_torque(out["tau_net"])
        model.step(dt)
        theta_sim[k] = model.get_theta()
        omega_sim[k] = model.get_omega()

    rmse_theta = float(np.sqrt(np.nanmean((theta_sim - theta_real) ** 2)))
    rmse_omega = float(np.sqrt(np.nanmean((omega_sim - omega_real) ** 2)))
    loss = float(w_theta * rmse_theta + w_omega * rmse_omega)
    return loss, theta_sim, omega_sim


def _worker(args):
    cand, datasets, cfg_dict, base_params, w_theta, w_omega, fit_ki_with_motor_input = args
    losses = []
    for ds in datasets:
        loss, _, _ = rollout_loss_for_candidate(
            candidate=np.asarray(cand, dtype=float),
            dataset=ds,
            cfg_dict=cfg_dict,
            base_params=base_params,
            w_theta=w_theta,
            w_omega=w_omega,
            fit_ki_with_motor_input=fit_ki_with_motor_input,
        )
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("inf")


def update_model_parameter_json(
    path: Path,
    cfg: BridgeConfig,
    best_b: float,
    best_tau: float,
    best_ki: float | None,
    best_loss: float,
    generations: int,
    popsize: int,
    input_csvs: list[str],
    fit_ki_with_motor_input: bool,
):
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.setdefault("version", 1)
    if not isinstance(data.get("known"), dict):
        data["known"] = {}
    known = data["known"]
    m_total = float(cfg.rod_mass + cfg.imu_mass)
    l_com_total = float(cfg.l_com_init)
    # Simple effective inertia around pivot for prior-known dynamics in Stage2.
    j_eff = float((cfg.rod_mass * (cfg.link_L ** 2) / 3.0) + (cfg.imu_mass * (cfg.r_imu ** 2)))
    known["mass_total_kg"] = m_total
    known["l_com_total_m"] = l_com_total
    known["inertia_total_kgm2"] = j_eff
    known["gravity_mps2"] = float(cfg.gravity)
    known["r_imu_m"] = float(cfg.r_imu)

    if not isinstance(data.get("torque_model"), dict):
        data["torque_model"] = {}
    tm = data["torque_model"]
    if not isinstance(tm.get("motor"), dict):
        tm["motor"] = {"enabled": True, "equation": "tau_motor = K_i * I_filtered_A", "params": {}}
    if not isinstance(tm["motor"].get("params"), dict):
        tm["motor"]["params"] = {}
    if fit_ki_with_motor_input and best_ki is not None:
        tm["motor"]["params"]["K_i"] = float(best_ki)
    else:
        tm["motor"]["params"]["K_i"] = float(tm["motor"]["params"].get("K_i", cfg.K_i_init))
    if not isinstance(tm.get("resistance"), dict):
        tm["resistance"] = {"enabled": True, "equation": "tau_res = b_eq*omega + tau_eq*tanh(omega/eps)", "params": {}}
    if not isinstance(tm["resistance"].get("params"), dict):
        tm["resistance"]["params"] = {}
    tm["resistance"]["params"]["b_eq"] = float(best_b)
    tm["resistance"]["params"]["tau_eq"] = float(best_tau)
    tm["resistance"]["params"].setdefault("eps", float(cfg.tanh_eps))
    if not isinstance(tm.get("residual_terms"), list):
        tm["residual_terms"] = []

    if not isinstance(data.get("stage_outputs"), dict):
        data["stage_outputs"] = {}
    data["stage_outputs"]["stage1"] = {
        "method": "cmaes_chrono_headless",
        "mode": "actuation_with_K_i_fit" if fit_ki_with_motor_input else "free_decay_passive_fit",
        "best_loss": float(best_loss),
        "best_params": (
            {"K_i": float(best_ki), "b_eq": float(best_b), "tau_eq": float(best_tau)}
            if (fit_ki_with_motor_input and best_ki is not None)
            else {"b_eq": float(best_b), "tau_eq": float(best_tau)}
        ),
        "generations": int(generations),
        "population_size": int(popsize),
        "input_csvs": list(input_csvs),
    }
    data["stage_outputs"].setdefault("stage2", None)
    data["stage_outputs"].setdefault("stage3", None)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Stage1 CMA-ES on headless Chrono free-decay replay (fit b_eq, tau_eq)")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="free-decay training csv (multi trajectory allowed)")
    ap.add_argument("--calibration-json", type=str, default="host/run_logs/calibration_latest.json")
    ap.add_argument("--model-parameter-json", type=Path, default=Path("host/model_parameter.latest.json"))
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--max-generations", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--w-theta", type=float, default=1.0)
    ap.add_argument("--w-omega", type=float, default=0.1)
    ap.add_argument("--b-min", type=float, default=0.0)
    ap.add_argument("--b-max", type=float, default=5.0)
    ap.add_argument("--tau-min", type=float, default=0.0)
    ap.add_argument("--tau-max", type=float, default=2.0)
    ap.add_argument("--tail-theta-abs-deg", type=float, default=0.8, help="tail trim threshold for |theta| [deg]")
    ap.add_argument("--tail-omega-abs", type=float, default=0.25, help="tail trim threshold for |omega| [rad/s]")
    ap.add_argument("--tail-hold-sec", type=float, default=0.8, help="required hold duration for tail trimming [s]")
    ap.add_argument("--fit-ki-with-motor-input", action="store_true", help="fit K_i together with b_eq/tau_eq using current input from CSV")
    ap.add_argument("--ki-min", type=float, default=1e-7)
    ap.add_argument("--ki-max", type=float, default=5e-3)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_datasets = [load_free_decay_csv(p) for p in args.csv]
    datasets = []
    for i, ds in enumerate(raw_datasets):
        trimmed, info = trim_long_tail(
            ds,
            tail_theta_abs_deg=float(args.tail_theta_abs_deg),
            tail_omega_abs=float(args.tail_omega_abs),
            tail_hold_sec=float(args.tail_hold_sec),
        )
        datasets.append(trimmed)
        print(
            f"[dataset {i}] tail_trim={bool(int(info['trimmed']))} "
            f"n_before={int(info['n_before'])} n_after={int(info['n_after'])}"
        )
        if len(trimmed["t"]) < 4:
            raise ValueError(f"Dataset too short after preprocessing: idx={i}, samples={len(trimmed['t'])}")
    cfg, _, runtime = build_cfg(args.calibration_json, str(args.model_parameter_json))
    cfg_dict = asdict(cfg)

    ki0 = float(runtime.get("K_i", cfg.K_i_init))
    b0 = float(runtime.get("b_eq", cfg.b_eq_init))
    tau0 = float(runtime.get("tau_eq", cfg.tau_eq_init))
    base_params = {
        "K_i": float(runtime.get("K_i", cfg.K_i_init)),
        "residual_terms": list(runtime.get("residual_terms", [])),
    }

    if args.fit_ki_with_motor_input:
        optimizer = CMA(
            mean=np.array([ki0, b0, tau0], dtype=float),
            sigma=float(args.sigma),
            bounds=np.array(
                [[args.ki_min, args.ki_max], [args.b_min, args.b_max], [args.tau_min, args.tau_max]],
                dtype=float,
            ),
            population_size=int(args.popsize),
        )
        best_x = np.array([ki0, b0, tau0], dtype=float)
    else:
        optimizer = CMA(
            mean=np.array([b0, tau0], dtype=float),
            sigma=float(args.sigma),
            bounds=np.array([[args.b_min, args.b_max], [args.tau_min, args.tau_max]], dtype=float),
            population_size=int(args.popsize),
        )
        best_x = np.array([b0, tau0], dtype=float)

    best_loss = float("inf")
    progress_rows: list[dict[str, float]] = []
    for gen in range(int(args.max_generations)):
        candidates = [optimizer.ask() for _ in range(optimizer.population_size)]
        payloads = [
            (
                np.asarray(x, dtype=float),
                datasets,
                cfg_dict,
                base_params,
                float(args.w_theta),
                float(args.w_omega),
                bool(args.fit_ki_with_motor_input),
            )
            for x in candidates
        ]
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            losses = list(ex.map(_worker, payloads))
        optimizer.tell([(x, float(l)) for x, l in zip(candidates, losses)])

        gen_best_i = int(np.argmin(losses))
        if losses[gen_best_i] < best_loss:
            best_loss = float(losses[gen_best_i])
            best_x = np.asarray(candidates[gen_best_i], dtype=float)
        row = {"gen": float(gen), "best_loss": float(np.min(losses)), "mean_loss": float(np.mean(losses))}
        if args.fit_ki_with_motor_input:
            row["best_K_i_so_far"] = float(best_x[0])
            row["best_b_eq_so_far"] = float(best_x[1])
            row["best_tau_eq_so_far"] = float(best_x[2])
            print(
                f"[gen {gen:03d}] best={min(losses):.6f} mean={float(np.mean(losses)):.6f} "
                f"| best_params(K_i={best_x[0]:.8f}, b_eq={best_x[1]:.6f}, tau_eq={best_x[2]:.6f})"
            )
        else:
            row["best_b_eq_so_far"] = float(best_x[0])
            row["best_tau_eq_so_far"] = float(best_x[1])
            print(
                f"[gen {gen:03d}] best={min(losses):.6f} mean={float(np.mean(losses)):.6f} "
                f"| best_params(b_eq={best_x[0]:.6f}, tau_eq={best_x[1]:.6f})"
            )
        progress_rows.append(row)

    progress_csv = args.outdir / "stage1_cmaes_progress.csv"
    with progress_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["gen", "best_loss", "mean_loss"]
        if args.fit_ki_with_motor_input:
            fieldnames += ["best_K_i_so_far", "best_b_eq_so_far", "best_tau_eq_so_far"]
        else:
            fieldnames += ["best_b_eq_so_far", "best_tau_eq_so_far"]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(progress_rows)

    # Evaluate best candidate on first trajectory for replay/visual artifact output.
    primary_ds = datasets[0]
    _, theta_sim, omega_sim = rollout_loss_for_candidate(
        candidate=best_x,
        dataset=primary_ds,
        cfg_dict=cfg_dict,
        base_params=base_params,
        w_theta=float(args.w_theta),
        w_omega=float(args.w_omega),
        fit_ki_with_motor_input=bool(args.fit_ki_with_motor_input),
    )
    if len(theta_sim) >= 2 and len(primary_ds["t"]) >= 2:
        theta0 = float(theta_sim[0])
        omega1 = float(omega_sim[1])
        expected = -math.sin(theta0)
        if abs(expected) > 1e-6 and np.sign(omega1) != np.sign(expected):
            print(
                "[audit] initial omega sign mismatch "
                f"(theta0={theta0:.6f}, omega_sim[1]={omega1:.6f}, expected_sign={np.sign(expected):+.0f})"
            )

    out_csv = args.outdir / "stage1_cmaes_rollout.csv"
    dt = np.diff(primary_ds["t"], prepend=primary_ds["t"][0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.maximum(dt, 1e-6)
    alpha_sim = np.gradient(omega_sim, dt)
    alpha_real = np.gradient(primary_ds["omega"], dt)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "wall_elapsed",
                "theta_real",
                "theta",
                "theta_imu_filtered_unwrapped",
                "omega_real",
                "omega",
                "omega_imu_filtered",
                "alpha_real",
                "alpha_from_linear_accel_filtered",
                "cmd_u_raw",
                "hw_pwm",
                "ina_current_raw_mA",
                "current_mA",
                "ina_current_corr_mA",
                "I_filtered_mA",
            ]
        )
        for i in range(len(primary_ds["t"])):
            mi_a = float(np.asarray(primary_ds.get("motor_input", np.zeros_like(primary_ds["t"])), dtype=float)[i])
            wr.writerow(
                [
                    primary_ds["t"][i],
                    primary_ds["theta"][i],
                    theta_sim[i],
                    primary_ds["theta"][i],
                    primary_ds["omega"][i],
                    omega_sim[i],
                    primary_ds["omega"][i],
                    alpha_real[i],
                    alpha_sim[i],
                    0.0,
                    0.0,
                    mi_a * 1000.0,
                    mi_a * 1000.0,
                    0.0,
                    0.0,
                ]
            )

    # Optional overlay plot artifact.
    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axs[0].plot(primary_ds["t"], primary_ds["theta"], label="theta_real")
        axs[0].plot(primary_ds["t"], theta_sim, label="theta_sim")
        axs[0].set_ylabel("theta [rad]")
        axs[0].legend()
        axs[0].grid(alpha=0.25)
        axs[1].plot(primary_ds["t"], primary_ds["omega"], label="omega_real")
        axs[1].plot(primary_ds["t"], omega_sim, label="omega_sim")
        axs[1].set_ylabel("omega [rad/s]")
        axs[1].set_xlabel("time [s]")
        axs[1].legend()
        axs[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.outdir / "stage1_cmaes_overlay.png", dpi=160)
        plt.close(fig)

        if progress_rows:
            pg = np.asarray([r["gen"] for r in progress_rows], dtype=float)
            pb = np.asarray([r["best_loss"] for r in progress_rows], dtype=float)
            pm = np.asarray([r["mean_loss"] for r in progress_rows], dtype=float)
            pbeq = np.asarray([r["best_b_eq_so_far"] for r in progress_rows], dtype=float)
            ptau = np.asarray([r["best_tau_eq_so_far"] for r in progress_rows], dtype=float)

            fig2, axs2 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            axs2[0].plot(pg, pb, label="best_loss")
            axs2[0].plot(pg, pm, label="mean_loss")
            axs2[0].set_ylabel("loss (weighted RMSE)")
            axs2[0].grid(alpha=0.25)
            axs2[0].legend()
            axs2[1].plot(pg, pbeq, label="best_b_eq_so_far")
            axs2[1].plot(pg, ptau, label="best_tau_eq_so_far")
            axs2[1].set_ylabel("parameter value")
            axs2[1].set_xlabel("generation")
            axs2[1].grid(alpha=0.25)
            axs2[1].legend()
            fig2.tight_layout()
            fig2.savefig(args.outdir / "stage1_cmaes_progress.png", dpi=160)
            plt.close(fig2)
    except Exception:
        pass

    out_result = args.outdir / "stage1_cmaes_result.json"
    out_result.write_text(
        json.dumps(
            {
                "method": "cmaes_chrono_headless",
                "best_loss": float(best_loss),
                "mode": "actuation_with_K_i_fit" if args.fit_ki_with_motor_input else "free_decay_passive_fit",
                "best_params": (
                    {"K_i": float(best_x[0]), "b_eq": float(best_x[1]), "tau_eq": float(best_x[2])}
                    if args.fit_ki_with_motor_input
                    else {"b_eq": float(best_x[0]), "tau_eq": float(best_x[1])}
                ),
                "population_size": int(args.popsize),
                "max_generations": int(args.max_generations),
                "input_csvs": [str(p) for p in args.csv],
                "num_trajectories": int(len(datasets)),
                "parallel_workers": int(args.workers),
                "headless_chrono": True,
                "loss_type": "weighted_rmse",
                "loss_weights": {"w_theta": float(args.w_theta), "w_omega": float(args.w_omega)},
                "preprocess": {
                    "tail_theta_abs_deg": float(args.tail_theta_abs_deg),
                    "tail_omega_abs": float(args.tail_omega_abs),
                    "tail_hold_sec": float(args.tail_hold_sec),
                },
                "output_rollout_csv": str(out_csv),
                "output_progress_csv": str(progress_csv),
                "output_progress_png": str(args.outdir / "stage1_cmaes_progress.png"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    update_model_parameter_json(
        path=args.model_parameter_json,
        cfg=cfg,
        best_b=float(best_x[1]) if args.fit_ki_with_motor_input else float(best_x[0]),
        best_tau=float(best_x[2]) if args.fit_ki_with_motor_input else float(best_x[1]),
        best_ki=float(best_x[0]) if args.fit_ki_with_motor_input else None,
        best_loss=float(best_loss),
        generations=int(args.max_generations),
        popsize=int(args.popsize),
        input_csvs=[str(p) for p in args.csv],
        fit_ki_with_motor_input=bool(args.fit_ki_with_motor_input),
    )
    if args.fit_ki_with_motor_input:
        print(f"[DONE] best K_i={best_x[0]:.8f}, b_eq={best_x[1]:.6f}, tau_eq={best_x[2]:.6f}, loss={best_loss:.6f}")
    else:
        print(f"[DONE] best b_eq={best_x[0]:.6f}, tau_eq={best_x[1]:.6f}, loss={best_loss:.6f}")
    print(f"[DONE] saved rollout: {out_csv}")
    print(f"[DONE] updated model-parameter json: {args.model_parameter_json}")


if __name__ == "__main__":
    main()
