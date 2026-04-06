from collections import deque

import numpy as np


def _sanitize_series(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if out.size == 0:
        return out
    finite = np.isfinite(out)
    if np.all(finite):
        return out
    if not np.any(finite):
        out[:] = 0.0
        return out
    idx = np.arange(out.size, dtype=float)
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def _safe_timebase(n: int, t: np.ndarray | None = None, dt: np.ndarray | float | None = None) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    if t is not None:
        ts = _sanitize_series(np.asarray(t, dtype=float).reshape(-1))
        if ts.size != n:
            ts = np.linspace(0.0, float(max(n - 1, 0)), n, dtype=float)
    elif dt is not None:
        if np.isscalar(dt):
            h = np.full(n, max(float(dt), 1e-6), dtype=float)
        else:
            h = _sanitize_series(np.asarray(dt, dtype=float).reshape(-1))
            if h.size != n:
                h = np.full(n, float(np.nanmedian(h)) if h.size > 0 else 1e-3, dtype=float)
            h = np.maximum(h, 1e-6)
        ts = np.cumsum(h)
        ts -= ts[0]
    else:
        ts = np.arange(n, dtype=float)

    # enforce strictly increasing timestamps
    for i in range(1, n):
        if (not np.isfinite(ts[i])) or ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1e-6
    return ts


def estimate_filtered_alpha_from_omega(
    omega: np.ndarray,
    t: np.ndarray | None = None,
    dt: np.ndarray | float | None = None,
) -> np.ndarray:
    """Estimate alpha from omega using smoothing + differentiation.

    Default method:
    1) Build a safe monotonic time base.
    2) Resample omega onto a uniform grid.
    3) If SciPy is available, use Savitzky-Golay derivative on uniform omega.
    4) Fallback to moving-average smoothing + numerical gradient.
    5) Interpolate the derivative back to the original time base.
    """
    om = _sanitize_series(np.asarray(omega, dtype=float).reshape(-1))
    n = om.size
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)

    ts = _safe_timebase(n, t=t, dt=dt)
    dt_med = max(float(np.nanmedian(np.diff(ts))), 1e-6)
    t_uniform = np.arange(0.0, dt_med * n, dt_med, dtype=float)[:n]
    om_uniform = np.interp(t_uniform, ts - ts[0], om)

    alpha_uniform = None
    try:
        from scipy.signal import savgol_filter  # type: ignore

        # ~110 ms window at 100 Hz, odd length and sufficiently large for polyorder 3.
        win = int(round(0.11 / dt_med))
        win = max(5, win)
        if (win % 2) == 0:
            win += 1
        win = min(win, n if (n % 2 == 1) else (n - 1))
        if win >= 5:
            alpha_uniform = savgol_filter(
                om_uniform,
                window_length=win,
                polyorder=3 if win >= 7 else 2,
                deriv=1,
                delta=dt_med,
                mode="interp",
            )
    except Exception:
        alpha_uniform = None

    if alpha_uniform is None:
        # SciPy-free fallback: smooth omega then differentiate.
        win = int(round(0.11 / dt_med))
        win = max(3, win)
        if (win % 2) == 0:
            win += 1
        pad = win // 2
        kernel = np.ones(win, dtype=float) / float(win)
        om_smooth = np.convolve(np.pad(om_uniform, (pad, pad), mode="edge"), kernel, mode="valid")[:n]
        alpha_uniform = np.gradient(om_smooth, t_uniform, edge_order=1)

    alpha = np.interp(ts - ts[0], t_uniform, alpha_uniform)
    alpha[~np.isfinite(alpha)] = 0.0
    return alpha


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


class CausalIIRFilter:
    """Simple real-time 1st-order causal IIR filter."""

    def __init__(self, alpha: float = 0.18):
        self.alpha = float(min(max(alpha, 1e-4), 1.0))
        self.state = None

    def reset(self, value: float = 0.0):
        self.state = float(value)

    def update(self, value: float) -> float:
        v = float(value)
        if self.state is None or not np.isfinite(self.state):
            self.state = v
        else:
            self.state = (1.0 - self.alpha) * self.state + self.alpha * v
        return float(self.state)


def causal_iir_filter_series(values: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    x = _sanitize_series(np.asarray(values, dtype=float))
    y = np.zeros_like(x, dtype=float)
    if x.size == 0:
        return y
    flt = CausalIIRFilter(alpha=alpha)
    flt.reset(float(x[0]))
    for i, v in enumerate(x):
        y[i] = flt.update(float(v))
    return y
