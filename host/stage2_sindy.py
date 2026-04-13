#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

from chrono_core.sindy_pipeline import (
    load_llm_config,
    load_trajectories,
    preprocess_for_pysindy,
    update_motor_torque_sindy_entry,
)


def _build_library(cfg: dict):
    lib_cfg = cfg.get("library", {"type": "polynomial", "degree": 3})
    if lib_cfg.get("type") == "fourier":
        return ps.FourierLibrary(n_frequencies=int(lib_cfg.get("n_frequencies", 3)))
    return ps.PolynomialLibrary(degree=int(lib_cfg.get("degree", 3)), include_bias=True)


def _build_optimizer(cfg: dict):
    opt = cfg.get("optimizer", {"type": "stlsq", "alpha": 0.001})
    threshold = float(cfg.get("threshold", opt.get("threshold", 0.05)))
    if opt.get("type", "stlsq") == "sr3":
        return ps.SR3(threshold=threshold, nu=float(opt.get("nu", 1.0)))
    return ps.STLSQ(threshold=threshold, alpha=float(opt.get("alpha", 0.001)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Option 6 Greybox SINDy")
    ap.add_argument("--csvs", nargs="*", default=[])
    ap.add_argument("--motor_torque_json", required=True)
    ap.add_argument("--llm_config", default="llm_config.json")
    ap.add_argument("--target_type", choices=["residual_alpha", "full_alpha"], default="residual_alpha")
    ap.add_argument("--mode", choices=["standard_sindy", "sindy_pi"], default="standard_sindy")
    ap.add_argument("--output_dir", default="host/run_logs")
    args = ap.parse_args()

    cfg = load_llm_config(args.llm_config)
    target_type = cfg.get("target_type", args.target_type)
    mode = cfg.get("mode", args.mode)

    csvs = args.csvs
    if not csvs:
        raw = input("CSV paths for Option 6 (comma-separated): ").strip()
        csvs = [s.strip() for s in raw.split(",") if s.strip()]
    trajs = load_trajectories(csvs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arrays; residual may come from Option5 output if present.
    residual_csv = out_dir / "residual_timeseries.csv"
    residual_df = pd.read_csv(residual_csv) if residual_csv.exists() else pd.DataFrame()
    preprocess_for_pysindy(
        trajs,
        residual_df,
        out_npz=out_dir / "pysindy_input_arrays.npz",
        out_summary_json=out_dir / "pysindy_preprocess_summary.json",
        alpha_mode=cfg.get("differentiation_method", "provided"),
    )

    npz = np.load(out_dir / "pysindy_input_arrays.npz", allow_pickle=True)
    x_list = list(npz["X"])
    u_list = list(npz["U"])
    t_list = list(npz["t"])
    y_key = "residual_alpha" if target_type == "residual_alpha" else "full_alpha"
    y_list = list(npz[y_key])

    library = _build_library(cfg)
    optimizer = _build_optimizer(cfg)

    # model state derivative as alpha target for omega equation
    X = [np.column_stack([x[:, 0], x[:, 1]]) for x in x_list]
    U = [u for u in u_list]
    T = [t for t in t_list]

    model = ps.SINDy(feature_library=library, optimizer=optimizer, feature_names=["theta", "omega", "I"])
    if mode == "sindy_pi" and hasattr(ps, "SINDyPI"):
        model = ps.SINDyPI(feature_library=library)

    # Fit on concatenated for practical robustness.
    Xc = np.vstack(X)
    Uc = np.vstack(U)
    dt = float(np.median(np.diff(np.concatenate(T)))) if len(np.concatenate(T)) > 2 else 0.001
    model.fit(Xc, u=Uc, t=dt)

    pred = model.predict(Xc, u=Uc)
    rmse = float(np.sqrt(np.mean((pred - Xc) ** 2)))

    terms = model.get_feature_names()
    coeff = model.coefficients()
    active_terms = [terms[i] for i in range(len(terms)) if np.any(np.abs(coeff[:, i]) > 1e-12)]
    coefs = [float(np.max(np.abs(coeff[:, i]))) for i in range(len(terms)) if np.any(np.abs(coeff[:, i]) > 1e-12)]

    eq_lines: list[str] = []
    model.print(lambda s: eq_lines.append(s))
    equation_human = "\n".join(eq_lines)

    # prediction overlay (omega channel)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(Xc[:, 1], label="omega_true")
    ax.plot(pred[:, 1], label="omega_pred", alpha=0.8)
    ax.set_title("Option6 prediction overlay (omega)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "prediction_overlay_option6.png")
    plt.close(fig)

    report = {
        "stage": "option6_greybox",
        "mode": mode,
        "target_type": target_type,
        "fit_metrics": {"rmse_state": rmse},
        "active_terms": active_terms,
        "coefficients": coefs,
        "equation": equation_human,
        "source_files": [str(Path(c).resolve()) for c in csvs],
    }
    (out_dir / "sindy_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "discovered_equation.txt").write_text(equation_human, encoding="utf-8")
    (out_dir / "active_terms.json").write_text(json.dumps({"active_terms": active_terms, "coefficients": coefs}, indent=2), encoding="utf-8")
    pd.DataFrame(coeff).to_csv(out_dir / "coefficient_matrix.csv", index=False)

    update_motor_torque_sindy_entry(
        motor_path=Path(args.motor_torque_json),
        stage="option6_greybox",
        mode=mode,
        target_type=target_type,
        config_used=cfg,
        active_terms=active_terms,
        coefficients=coefs,
        equation_human=equation_human,
        fit_metrics={"rmse_state": rmse},
        source_files=[str(Path(c).resolve()) for c in csvs],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
