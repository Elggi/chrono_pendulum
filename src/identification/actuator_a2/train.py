"""Pipeline A-2: neural actuator/residual identification."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import save_json


@dataclass
class ActuatorTrainConfig:
    data_csv: Path
    out_dir: Path = Path("models/actuator")
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    features: tuple[str, ...] = ("u", "current", "theta", "omega")
    target: str = "tau_eff"


class ActuatorResidualNet(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_actuator_a2(cfg: ActuatorTrainConfig) -> Path:
    """Train actuator mapping on excitation-only data and save actuator_a2.pt."""
    df = pd.read_csv(cfg.data_csv)
    if "segment_type" in df.columns and not (df["segment_type"] == "excitation").all():
        raise ValueError("Actuator model requires excitation-only data")

    missing = set(cfg.features + (cfg.target,)) - set(df.columns)
    if missing:
        raise ValueError(f"Actuator dataset missing columns: {sorted(missing)}")

    x = torch.tensor(df[list(cfg.features)].to_numpy(dtype="float32"))
    y = torch.tensor(df[[cfg.target]].to_numpy(dtype="float32"))

    model = ActuatorResidualNet(x.shape[1])
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            opt.step()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.out_dir / "actuator_a2.pt"
    torch.save(model.state_dict(), model_path)
    save_json(cfg.out_dir / "config_snapshot.json", asdict(cfg) | {"data_csv": str(cfg.data_csv), "out_dir": str(cfg.out_dir)})
    save_json(cfg.out_dir / "feature_schema.json", {"inputs": list(cfg.features), "target": cfg.target})
    save_json(cfg.out_dir / "split_metadata.json", {"strategy": "trajectory_based", "source": str(cfg.data_csv)})
    save_json(cfg.out_dir / "normalization.json", {"note": "identity scaling"})
    return model_path
