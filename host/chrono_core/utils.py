import math
import os
import re
import shutil
import sys
import time

import numpy as np
import pychrono as ch


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def now_wall():
    return time.time()


def terminal_status_line(msg: str, width: int | None = None):
    term_width = shutil.get_terminal_size((max(width or 120, 40), 24)).columns
    usable_width = max(20, min(width or term_width, term_width) - 1)
    sys.stdout.write("\r\033[2K" + msg[:usable_width].ljust(usable_width))
    sys.stdout.flush()


def sanitize_float(value, default=0.0, limit=np.finfo(np.float32).max * 0.99):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(clamp(value, -limit, limit))


def make_numbered_path(folder: str, prefix: str, ext: str = ".csv") -> str:
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    max_n = 0
    for name in os.listdir(folder):
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return os.path.join(folder, f"{prefix}{max_n + 1}{ext}")


def moving_average(x: np.ndarray, win: int):
    if win <= 1 or len(x) == 0:
        return x.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, kernel, mode="valid")
    return y[:len(x)]


def prbs_value(t: float, dt: float = 0.25, seed: int = 12345) -> float:
    if dt <= 1e-12:
        return 1.0
    k = int(t / dt)
    x = (1103515245 * (k + seed) + 12345) & 0x7FFFFFFF
    return 1.0 if (x & 1) else -1.0


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_to_np(q: ch.ChQuaterniond):
    return np.array([q.e0, q.e1, q.e2, q.e3], dtype=float)
