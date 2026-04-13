"""Shared helpers for training artifact persistence."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def normalize_config(config: object) -> dict:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    raise TypeError("Config must be a dataclass or dict")
