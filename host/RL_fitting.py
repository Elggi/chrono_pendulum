#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env


def safe_savgol(x, window=21, poly=3):
    n = len(x)
    if n < 5:
        return x.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        w = poly + 2
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 5:
        return x.copy()
    return savgol_filter(x, window_length=w, polyorder=min(poly, w - 2), mode="interp")


def compute_dt(df, time_col="sim_time", dt_col=""):
    if dt_col and dt_col in df.columns:
        dt = df[dt_col].to_numpy(dtype=float)
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
        return dt
    t = df[time_col].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
    return dt


def robust_std(x, floor=1e-6):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(1.4826 * mad, floor)


def deadzone(x, x0):
    return np.sign(x) * np.maximum(np.abs(x) - x0, 0.0)


def shift_signal_by_delay(u, t, delay_sec):
    y = np.zeros_like(u)
    for k in range(len(t)):
        target = t[k] - delay_sec
        if target <= t[0]:
            y[k] = u[0]
        elif target >= t[-1]:
            y[k] = u[-1]
        else:
            y[k] = np.interp(target, t, u)
    return y


def simulate_with_electromech(t, dt, pwm, bus_v, theta0, omega0, params, tanh_eps=0.05, pwm_limit=255.0):
    # params = [J, b, tau_c, mgl, k_t, i0, Rm, k_e, delay_sec]
    J, b, tau_c, mgl, k_t, i0, Rm, k_e, delay_sec = params
    J = max(J, 1e-6)
    Rm = max(Rm, 1e-6)
    delay_sec = max(delay_sec, 0.0)
    pwm_d = shift_signal_by_delay(pwm, t, delay_sec)
    n = len(t)
    theta = np.zeros(n, dtype=float)
    omega = np.zeros(n, dtype=float)
    alpha = np.zeros(n, dtype=float)
    current = np.zeros(n, dtype=float)
    voltage_pred = np.zeros(n, dtype=float)
    power_pred = np.zeros(n, dtype=float)
    theta[0] = theta0
    omega[0] = omega0

    for k in range(n - 1):
        h = float(dt[k])
        u = pwm_d[k]
        vbus = bus_v[k]

        def dyn(th, om):
            duty = np.clip(u / max(pwm_limit, 1e-9), -1.0, 1.0)
            vapplied = duty * vbus
            i = (vapplied - k_e * om) / Rm
            i = deadzone(np.array([i]), i0)[0]
            tau = k_t * i - b * om - tau_c * np.tanh(om / tanh_eps) - mgl * np.sin(th)
            a = tau / J
            return om, a, i, vapplied

        th, om = theta[k], omega[k]
        k1_th, k1_om, i1, v1 = dyn(th, om)
        k2_th, k2_om, _, _ = dyn(th + 0.5 * h * k1_th, om + 0.5 * h * k1_om)
        k3_th, k3_om, _, _ = dyn(th + 0.5 * h * k2_th, om + 0.5 * h * k2_om)
        k4_th, k4_om, _, _ = dyn(th + h * k3_th, om + h * k3_om)
        theta[k + 1] = th + (h / 6.0) * (k1_th + 2*k2_th + 2*k3_th + k4_th)
        omega[k + 1] = om + (h / 6.0) * (k1_om + 2*k2_om + 2*k3_om + k4_om)
        current[k] = i1
        voltage_pred[k] = v1
        power_pred[k] = v1 * i1

    current[-1] = current[-2] if n > 1 else 0.0
    voltage_pred[-1] = voltage_pred[-2] if n > 1 else 0.0
    power_pred[-1] = power_pred[-2] if n > 1 else 0.0
    alpha[:] = np.gradient(omega, dt)
    return {
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "current": current,
        "voltage": voltage_pred,
        "power": power_pred,
    }


class PendulumFitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, horizon=500, max_steps=25, tanh_eps=0.05, pwm_limit=255.0, seed=0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.horizon = min(horizon, len(df))
        self.max_steps = max_steps
        self.tanh_eps = tanh_eps
        self.pwm_limit = pwm_limit
        self.rng = np.random.default_rng(seed)
        self.param_lb = np.array([1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.05, 0.0, 0.0], dtype=float)
        self.param_ub = np.array([0.50, 5.0, 10.0, 20.0, 10.0, 2.0, 50.0, 5.0, 0.20], dtype=float)
        self.param_nom = np.array([0.010, 0.03, 0.08, 0.55, 0.25, 0.05, 2.0, 0.02, 0.03], dtype=float)
        self.delta_scale = np.array([0.01, 0.10, 0.10, 0.10, 0.05, 0.02, 0.20, 0.02, 0.01], dtype=float)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(15,), dtype=np.float32)
        self.w_theta = 5.0
        self.w_omega = 3.0
        self.w_alpha = 0.7
        self.w_v = 0.5
        self.w_i = 1.0
        self.w_p = 0.3
        self.params = self.param_nom.copy()
        self.step_idx = 0
        self.last_cost = None
        self.best_info = None

    def _sample_window(self):
        if len(self.df) <= self.horizon:
            i0 = 0
        else:
            i0 = self.rng.integers(0, len(self.df) - self.horizon)
        self.win = self.df.iloc[i0:i0+self.horizon].reset_index(drop=True)
        self.t = self.win["time"].to_numpy(dtype=float)
        self.dt = self.win["dt"].to_numpy(dtype=float)
        self.theta_meas = self.win["theta"].to_numpy(dtype=float)
        self.omega_meas = self.win["omega"].to_numpy(dtype=float)
        self.alpha_meas = self.win["alpha"].to_numpy(dtype=float)
        self.pwm_meas = self.win["pwm"].to_numpy(dtype=float)
        self.bus_v_meas = self.win["voltage"].to_numpy(dtype=float)
        self.current_meas = self.win["current"].to_numpy(dtype=float)
        self.power_meas = self.win["power"].to_numpy(dtype=float)

    def _randomize_params(self):
        rnd = self.param_nom.copy()
        rnd *= self.rng.uniform(0.7, 1.3, size=len(rnd))
        rnd[-1] = self.rng.uniform(0.0, 0.08)
        self.params = np.clip(rnd, self.param_lb, self.param_ub)

    def _cost(self, p):
        sim = simulate_with_electromech(self.t, self.dt, self.pwm_meas, self.bus_v_meas, self.theta_meas[0], self.omega_meas[0], p, tanh_eps=self.tanh_eps, pwm_limit=self.pwm_limit)
        e_theta = (sim["theta"] - self.theta_meas) / robust_std(self.theta_meas)
        e_omega = (sim["omega"] - self.omega_meas) / robust_std(self.omega_meas)
        e_alpha = (sim["alpha"] - self.alpha_meas) / robust_std(self.alpha_meas)
        e_v = (sim["voltage"] - self.bus_v_meas) / robust_std(self.bus_v_meas)
        e_i = (sim["current"] - self.current_meas) / robust_std(self.current_meas)
        e_p = (sim["power"] - self.power_meas) / robust_std(self.power_meas)
        cost = (
            self.w_theta * np.mean(e_theta**2) +
            self.w_omega * np.mean(e_omega**2) +
            self.w_alpha * np.mean(e_alpha**2) +
            self.w_v * np.mean(e_v**2) +
            self.w_i * np.mean(e_i**2) +
            self.w_p * np.mean(e_p**2)
        )
        return float(cost), {"cost": float(cost), "params": p.copy(), "sim": sim}

    def _obs(self):
        return np.array([
            self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
            self.params[5], self.params[6], self.params[7], self.params[8],
            self.last_cost if self.last_cost is not None else 0.0,
            self.theta_meas[0], self.omega_meas[0],
            np.mean(np.abs(self.pwm_meas)), np.mean(self.bus_v_meas), np.mean(np.abs(self.current_meas)),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_window()
        self._randomize_params()
        self.step_idx = 0
        self.last_cost, self.best_info = self._cost(self.params)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        self.params = np.clip(self.params + self.delta_scale * action, self.param_lb, self.param_ub)
        new_cost, info = self._cost(self.params)
        reward = float(self.last_cost - new_cost)
        if self.best_info is None or info["cost"] < self.best_info["cost"]:
            self.best_info = info
        self.last_cost = new_cost
        self.step_idx += 1
        terminated = self.step_idx >= self.max_steps
        truncated = False
        return self._obs(), reward, terminated, truncated, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="rl_results")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--episode-steps", type=int, default=25)
    ap.add_argument("--time-col", default="sim_time")
    ap.add_argument("--dt-col", default="")
    ap.add_argument("--theta-col", default="theta")
    ap.add_argument("--omega-col", default="omega")
    ap.add_argument("--alpha-col", default="alpha")
    ap.add_argument("--pwm-col", default="hw_pwm")
    ap.add_argument("--voltage-col", default="bus_v")
    ap.add_argument("--current-col", default="current_A")
    ap.add_argument("--power-col", default="power_W")
    ap.add_argument("--pwm-limit", type=float, default=255.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(args.csv)
    dt = compute_dt(df_raw, args.time_col, args.dt_col)
    theta = safe_savgol(df_raw[args.theta_col].to_numpy(dtype=float))
    omega = safe_savgol(df_raw[args.omega-col].to_numpy(dtype=float)) if args.omega_col in df_raw.columns else safe_savgol(np.gradient(theta, dt))
    alpha = safe_savgol(df_raw[args.alpha_col].to_numpy(dtype=float)) if args.alpha_col in df_raw.columns else safe_savgol(np.gradient(omega, dt))
    df = pd.DataFrame({
        "time": df_raw[args.time_col].to_numpy(dtype=float),
        "dt": dt,
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "pwm": df_raw[args.pwm_col].to_numpy(dtype=float),
        "voltage": df_raw[args.voltage_col].to_numpy(dtype=float),
        "current": df_raw[args.current_col].to_numpy(dtype=float),
        "power": df_raw[args.power_col].to_numpy(dtype=float),
    })

    env = PendulumFitEnv(df=df, horizon=args.horizon, max_steps=args.episode_steps, pwm_limit=args.pwm_limit)
    check_env(env, warn=True)

    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=256, batch_size=64, gamma=0.98, gae_lambda=0.95, clip_range=0.2)
    else:
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=128, gamma=0.98, train_freq=1, gradient_steps=1)

    model.learn(total_timesteps=args.timesteps)
    model_path = outdir / f"{args.algo}_pendulum_fit.zip"
    model.save(str(model_path))

    obs, _ = env.reset()
    best_cost = np.inf
    best_info = None
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if info["cost"] < best_cost:
            best_cost = info["cost"]
            best_info = info
        if done:
            break

    p = best_info["params"]
    result = {
        "best_cost": float(best_cost),
        "best_params": {
            "J": float(p[0]), "b": float(p[1]), "tau_c": float(p[2]), "mgl": float(p[3]),
            "k_t": float(p[4]), "i0": float(p[5]), "Rm": float(p[6]), "k_e": float(p[7]), "delay_sec": float(p[8]),
        },
        "recommended_cli": (
            f"python3 chrono_pendulum.py --J {p[0]:.6f} --b {p[1]:.6f} --tau-c {p[2]:.6f} --mgl {p[3]:.6f} "
            f"--k-t {p[4]:.6f} --i0 {p[5]:.6f} --R {p[6]:.6f} --k-e {p[7]:.6f} --delay-ms {1000.0*p[8]:.3f}"
        )
    }

    with open(outdir / "rl_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    sim = best_info["sim"]
    pred_df = pd.DataFrame({
        "time": env.t,
        "theta_meas": env.theta_meas,
        "omega_meas": env.omega_meas,
        "alpha_meas": env.alpha_meas,
        "voltage_meas": env.bus_v_meas,
        "current_meas": env.current_meas,
        "power_meas": env.power_meas,
        "pwm_meas": env.pwm_meas,
        "theta_sim": sim["theta"],
        "omega_sim": sim["omega"],
        "alpha_sim": sim["alpha"],
        "voltage_sim": sim["voltage"],
        "current_sim": sim["current"],
        "power_sim": sim["power"],
    })
    pred_df.to_csv(outdir / "rl_best_prediction.csv", index=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {model_path}")
    print(f"saved: {outdir / 'rl_result.json'}")
    print(f"saved: {outdir / 'rl_best_prediction.csv'}")


if __name__ == "__main__":
    main()
