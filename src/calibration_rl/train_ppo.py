"""Train SB3 PPO policy for calibration tuning (Pipeline C-1)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .env import CalibrationEnv, CalibrationTargets
from ..identification.common import save_json


@dataclass
class RLTrainConfig:
    data_csv: Path
    out_dir: Path = Path("models/rl")
    timesteps: int = 20_000


def train_rl_calibrator(cfg: RLTrainConfig) -> Path:
    """Train PPO policy that tunes calibration parameters for trajectory agreement."""
    df = pd.read_csv(cfg.data_csv)
    req = {"theta", "omega", "u"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(req - set(df.columns))}")

    targets = CalibrationTargets(
        theta=df["theta"].to_numpy(dtype=np.float32),
        omega=df["omega"].to_numpy(dtype=np.float32),
        u=df["u"].to_numpy(dtype=np.float32),
    )
    env = CalibrationEnv(targets)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=cfg.timesteps)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / "ppo_calibrator.zip"
    model.save(str(out_path))
    save_json(cfg.out_dir / "config_snapshot.json", asdict(cfg) | {"data_csv": str(cfg.data_csv), "out_dir": str(cfg.out_dir)})
    return out_path
