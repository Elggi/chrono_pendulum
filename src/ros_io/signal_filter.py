"""Lightweight causal IIR filters for online ROS signal conditioning."""

from __future__ import annotations


class CausalIIRFilter:
    """First-order causal low-pass filter (EMA style)."""

    def __init__(self, alpha: float = 0.18):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = float(alpha)
        self.state: float | None = None

    def reset(self, value: float | None = None) -> None:
        self.state = value

    def update(self, value: float) -> float:
        if self.state is None:
            self.state = float(value)
            return self.state
        self.state = self.alpha * float(value) + (1.0 - self.alpha) * self.state
        return self.state
