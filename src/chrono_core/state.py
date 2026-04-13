"""Shared state definitions for simulator and identification pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PendulumState:
    """Canonical pendulum state."""

    theta: float
    omega: float
    dt: float
