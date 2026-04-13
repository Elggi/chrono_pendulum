#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from mnode.config import MNodeConfig
from mnode.data import build_training_tensors, load_dataset
from mnode.losses import mse_loss
from mnode.models import ResidualMLP
from mnode.reports import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimental MBD-NODE style residual torque training")
    ap.add_argument("--csvs", nargs="*", default=[])
    ap.add_argument("--npz", default="")
    ap.add_argument("--motor_torque_json", required=True)
    ap.add_argument("--outdir", default="host/models/nn_residual")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    if not args.csvs and not args.npz:
        raw = input("CSV paths for mnode training (comma-separated): ").strip()
        args.csvs = [s.strip() for s in raw.split(",") if s.strip()]

    cfg = MNodeConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_dim, val_split=args.val_split)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.csvs, npz=args.npz or None)
    X, y, scaler = build_training_tensors(dataset, args.motor_torque_json)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)

    n_val = max(1, int(len(ds) * cfg.val_split))
    n_train = max(1, len(ds) - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=min(cfg.batch_size, n_train), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, n_val), shuffle=False)

    model = ResidualMLP(in_dim=3, hidden_dim=cfg.hidden_dim, out_dim=1)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    log_rows = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optim.zero_grad()
            pred = model(xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            optim.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(float(mse_loss(model(xb), yb).item()))
        tr = float(np.mean(train_losses)) if train_losses else 0.0
        va = float(np.mean(val_losses)) if val_losses else 0.0
        log_rows.append({"epoch": epoch, "train_loss": tr, "val_loss": va})

        improved = va < best_val
        if improved:
            best_val = va
            torch.save(model.state_dict(), outdir / "best_model.pt")
        print(f"[Epoch {epoch:03d}] train_loss={tr:.6f} val_loss={va:.6f} {'*best*' if improved else ''}")

    with (outdir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader(); w.writerows(log_rows)

    write_json(outdir / "scaler.json", scaler)
    schema = {
        "type": "nn_torque_residual",
        "backend": "pytorch",
        "input_features": scaler["features"],
        "output": "tau_learned",
        "hidden_dim": cfg.hidden_dim,
        "mode": cfg.mode,
    }
    write_json(outdir / "model_schema.json", schema)
    summary = {
        "best_val_loss": best_val,
        "epochs": cfg.epochs,
        "train_samples": n_train,
        "val_samples": n_val,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": [d["source"] for d in dataset],
    }
    write_json(outdir / "training_summary.json", summary)
    write_json(outdir / "validation_metrics.json", {"best_val_loss": best_val})

    # update motor_torque.json references
    motor_path = Path(args.motor_torque_json)
    motor = json.loads(motor_path.read_text(encoding="utf-8"))
    motor["learned_residual"] = {
        "enabled": True,
        "type": "nn_torque_residual",
        "backend": "pytorch",
        "checkpoint_path": str((outdir / "best_model.pt").resolve()),
        "normalization_path": str((outdir / "scaler.json").resolve()),
        "schema_path": str((outdir / "model_schema.json").resolve()),
        "input_features": scaler["features"],
        "output": "tau_learned",
        "trained_from": [d["source"] for d in dataset],
        "notes": "experimental MBD-NODE-style residual torque model",
    }
    motor_path.write_text(json.dumps(motor, indent=2), encoding="utf-8")
    print(f"[OK] saved artifacts to {outdir}")


if __name__ == "__main__":
    main()
