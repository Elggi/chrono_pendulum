#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mnode.data import build_training_tensors, load_dataset
from mnode.models import ResidualMLP


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate experimental MBD-NODE residual model")
    ap.add_argument("--csvs", nargs="*", default=[])
    ap.add_argument("--npz", default="")
    ap.add_argument("--motor_torque_json", required=True)
    ap.add_argument("--model_dir", default="host/models/nn_residual")
    args = ap.parse_args()

    dataset = load_dataset(args.csvs, npz=args.npz or None)
    X, y, scaler = build_training_tensors(dataset, args.motor_torque_json)

    model_dir = Path(args.model_dir)
    schema = json.loads((model_dir / "model_schema.json").read_text(encoding="utf-8"))
    model = ResidualMLP(in_dim=len(schema["input_features"]), hidden_dim=int(schema.get("hidden_dim", 64)), out_dim=1)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy().reshape(-1)
    y_true = y.reshape(-1)
    rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y_true[:2000], label="tau_residual_true")
    ax.plot(pred[:2000], label="tau_residual_pred", alpha=0.8)
    ax.legend(); ax.set_title(f"Residual torque prediction RMSE={rmse:.5f}")
    fig.tight_layout(); fig.savefig(model_dir / "eval_overlay.png")
    plt.close(fig)

    (model_dir / "human_readable_report.txt").write_text(
        f"Experimental MBD-NODE residual model\nRMSE: {rmse:.6f}\nSamples: {len(y_true)}\n",
        encoding="utf-8",
    )
    print(json.dumps({"rmse_tau_residual": rmse, "samples": int(len(y_true))}, indent=2))


if __name__ == "__main__":
    main()
