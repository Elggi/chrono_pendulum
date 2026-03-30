from collections import deque

import numpy as np


class RobustSignalFilter:
    def __init__(self, median_window: int, hampel_k: int, hampel_sigma: float, lpf_tau_sec: float):
        self.median_window = max(int(median_window), 1)
        self.hampel_k = max(int(hampel_k), 1)
        self.hampel_sigma = max(float(hampel_sigma), 0.1)
        self.lpf_tau_sec = max(float(lpf_tau_sec), 1e-4)
        self.raw = deque(maxlen=max(self.median_window, self.hampel_k))
        self.lpf_state = None

    def update(self, value: float, dt: float) -> float:
        val = float(value)
        self.raw.append(val)
        med_src = list(self.raw)[-self.median_window:]
        med = float(np.median(np.array(med_src, dtype=float)))

        hampel_src = np.array(list(self.raw)[-self.hampel_k:], dtype=float)
        center = float(np.median(hampel_src))
        mad = float(np.median(np.abs(hampel_src - center)))
        scale = 1.4826 * mad
        cleaned = med
        if scale > 1e-9 and abs(val - center) <= self.hampel_sigma * scale:
            cleaned = val

        if self.lpf_state is None:
            self.lpf_state = cleaned
        else:
            dt = max(float(dt), 1e-6)
            alpha = dt / (self.lpf_tau_sec + dt)
            self.lpf_state = (1.0 - alpha) * self.lpf_state + alpha * cleaned
        return float(self.lpf_state)
