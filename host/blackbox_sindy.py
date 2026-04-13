#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

from chrono_core.sindy_pipeline import load_llm_config, load_trajectories, preprocess_for_pysindy, update_motor_torque_sindy_entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Option 8 Blackbox SINDy")
    ap.add_argument("--csvs", nargs="*", default=[])
    ap.add_argument("--motor_torque_json", required=True)
    ap.add_argument("--llm_config", default="llm_config.json")
    ap.add_argument("--mode", choices=["standard_sindy", "sindy_pi"], default="standard_sindy")
    ap.add_argument("--output_dir", default="host/run_logs")
    args = ap.parse_args()

    cfg = load_llm_config(args.llm_config)
    mode = cfg.get("mode", args.mode)

    csvs = args.csvs
    if not csvs:
        raw = input("CSV paths for Option 8 (comma-separated): ").strip()
        csvs = [s.strip() for s in raw.split(",") if s.strip()]

    trajs = load_trajectories(csvs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocess_for_pysindy(
        trajs,
        residual_df=None,
        out_npz=out_dir / "pysindy_input_arrays.npz",
        out_summary_json=out_dir / "pysindy_preprocess_summary.json",
        alpha_mode=cfg.get("differentiation_method", "provided"),
    )

    npz = np.load(out_dir / "pysindy_input_arrays.npz", allow_pickle=True)
    X = np.vstack(list(npz["X"]))
    U = np.vstack(list(npz["U"]))
    T = np.concatenate(list(npz["t"]))
    dt = float(np.median(np.diff(T))) if len(T) > 2 else 0.001

    lib_type = cfg.get("library", {}).get("type", "polynomial")
    if lib_type == "fourier":
        library = ps.FourierLibrary(n_frequencies=int(cfg.get("library", {}).get("n_frequencies", 3)))
    else:
        library = ps.PolynomialLibrary(degree=int(cfg.get("library", {}).get("degree", 4)), include_bias=True)

    threshold = float(cfg.get("threshold", 0.02))
    if mode == "sindy_pi" and hasattr(ps, "SINDyPI"):
        model = ps.SINDyPI(feature_library=library)
    else:
        optimizer = ps.STLSQ(threshold=threshold, alpha=float(cfg.get("optimizer", {}).get("alpha", 0.001)))
        model = ps.SINDy(feature_library=library, optimizer=optimizer, feature_names=["theta", "omega", "I"])

    model.fit(X, u=U, t=dt)
    pred = model.predict(X, u=U)
    rmse = float(np.sqrt(np.mean((pred - X) ** 2)))

    terms = model.get_feature_names()
    coeff = model.coefficients()
    active_terms = [terms[i] for i in range(len(terms)) if np.any(np.abs(coeff[:, i]) > 1e-12)]
    coefs = [float(np.max(np.abs(coeff[:, i]))) for i in range(len(terms)) if np.any(np.abs(coeff[:, i]) > 1e-12)]

    eq_lines: list[str] = []
    model.print(lambda s: eq_lines.append(s))
    equation_human = "\n".join(eq_lines)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(X[:, 1], label="omega_true")
    ax.plot(pred[:, 1], label="omega_pred", alpha=0.8)
    ax.set_title("Option8 prediction overlay (omega)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "prediction_overlay_option8.png")
    plt.close(fig)

    report = {
        "stage": "option8_blackbox",
        "mode": mode,
        "target_type": "full_alpha",
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
        stage="option8_blackbox",
        mode=mode,
        target_type="full_alpha",
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
