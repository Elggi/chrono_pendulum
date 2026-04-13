"""Pipeline N: nominal passive model training from free-decay data only."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import save_json


@dataclass
class NominalTrainConfig:
    data_csv: Path
    out_dir: Path = Path("models/nominal")
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    state_columns: tuple[str, ...] = ("theta", "omega")
    dt_column: str = "dt"


class NominalMLP(nn.Module):
    """Small passive dynamics predictor for one-step mapping."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_supervised(df: pd.DataFrame, cfg: NominalTrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    f = list(cfg.state_columns) + [cfg.dt_column]
    if not set(f).issubset(df.columns):
        raise ValueError(f"Nominal data must include {f}")
    x = df[f].iloc[:-1].to_numpy(dtype="float32")
    y = df[list(cfg.state_columns)].iloc[1:].to_numpy(dtype="float32")
    return torch.tensor(x), torch.tensor(y)


def train_nominal_model(cfg: NominalTrainConfig) -> Path:
    """Train passive dynamics model from free-decay samples and save model.pt."""
    df = pd.read_csv(cfg.data_csv)
    if "segment_type" in df.columns and not (df["segment_type"] == "free_decay").all():
        raise ValueError("Nominal model must use free-decay data only")

    x, y = _build_supervised(df, cfg)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model = NominalMLP(in_dim=x.shape[1], out_dim=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    save_json(cfg.out_dir / "config_snapshot.json", asdict(cfg) | {"data_csv": str(cfg.data_csv), "out_dir": str(cfg.out_dir)})
    save_json(cfg.out_dir / "feature_schema.json", {"inputs": list(cfg.state_columns) + [cfg.dt_column], "targets": list(cfg.state_columns)})
    save_json(cfg.out_dir / "split_metadata.json", {"strategy": "trajectory_based", "source": str(cfg.data_csv)})
    save_json(cfg.out_dir / "normalization.json", {"note": "identity scaling"})

    return model_path


if __name__ == "__main__":
    train_nominal_model(NominalTrainConfig(data_csv=Path("data/processed/nominal_train.csv")))
