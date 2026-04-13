from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MNodeConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 64
    val_split: float = 0.2
    seed: int = 42
    mode: str = "residual_torque"  # residual_torque | full_dynamics
