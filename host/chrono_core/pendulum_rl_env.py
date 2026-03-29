import math
from dataclasses import dataclass

import numpy as np

from .config import BridgeConfig
from .replay_io import PARAM_NAMES_DEFAULT, ReplayTrajectory


@dataclass
class RewardWeights:
    theta: float = 5.0
    omega: float = 2.5
    alpha: float = 1.0
    pwm: float = 0.2
    current: float = 0.1
    power: float = 0.05
    param_reg: float = 0.01
    action_penalty: float = 0.03


def shift_with_delay(t: np.ndarray, x: np.ndarray, delay_sec: float):
    y = np.zeros_like(x)
    for i in range(len(t)):
        ts = t[i] - delay_sec
        if ts <= t[0]:
            y[i] = x[0]
        elif ts >= t[-1]:
            y[i] = x[-1]
        else:
            y[i] = np.interp(ts, t, x)
    return y


def simulate_trajectory(cfg: BridgeConfig, traj: ReplayTrajectory, params: dict, delay_sec: float):
    m_total = cfg.link_mass + cfg.imu_mass
    J_pivot = max(params["J_cm_base"] + m_total * (params["l_com"] ** 2), 1e-6)
    theta = np.zeros_like(traj.theta)
    omega = np.zeros_like(traj.omega)
    alpha = np.zeros_like(traj.alpha)
    current = np.zeros_like(traj.current)
    power = np.zeros_like(traj.power)
    tau_motor_hist = np.zeros_like(traj.theta)
    tau_res_hist = np.zeros_like(traj.theta)

    theta[0] = float(traj.theta[0])
    omega[0] = float(traj.omega[0])
    u_del = shift_with_delay(traj.t, traj.cmd_u, delay_sec)

    for k in range(len(theta) - 1):
        dt = float(max(traj.dt[k], 1e-4))
        u = float(u_del[k])
        vbus = float(traj.bus_v[k])

        duty = float(np.clip(u / max(cfg.pwm_limit, 1e-9), -1.0, 1.0))
        vapplied = duty * vbus

        def dyn(th, om):
            i_raw = (vapplied - params["k_e"] * om) / max(params["R"], 1e-6)
            i_eff = math.copysign(max(abs(i_raw) - max(params["i0"], 0.0), 0.0), i_raw)
            tau_motor = params["k_t"] * i_eff
            tau_res = params["b_eq"] * om + params["tau_eq"] * math.tanh(om / max(cfg.tanh_eps, 1e-9))
            tau_g = m_total * cfg.gravity * params["l_com"] * math.sin(th)
            a = (tau_motor - tau_res - tau_g) / J_pivot
            return om, a, i_eff, tau_motor, tau_res

        th = theta[k]
        om = omega[k]
        k1_t, k1_o, i1, tm1, tr1 = dyn(th, om)
        k2_t, k2_o, _, _, _ = dyn(th + 0.5 * dt * k1_t, om + 0.5 * dt * k1_o)
        k3_t, k3_o, _, _, _ = dyn(th + 0.5 * dt * k2_t, om + 0.5 * dt * k2_o)
        k4_t, k4_o, _, _, _ = dyn(th + dt * k3_t, om + dt * k3_o)
        theta[k + 1] = th + (dt / 6.0) * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)
        omega[k + 1] = om + (dt / 6.0) * (k1_o + 2 * k2_o + 2 * k3_o + k4_o)
        alpha[k] = k1_o
        current[k] = i1
        power[k] = vapplied * i1
        tau_motor_hist[k] = tm1
        tau_res_hist[k] = tr1

    alpha[-1] = alpha[-2] if len(alpha) > 1 else 0.0
    current[-1] = current[-2] if len(current) > 1 else 0.0
    power[-1] = power[-2] if len(power) > 1 else 0.0
    return {
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "current": current,
        "power": power,
        "cmd_delayed": u_del,
        "tau_motor": tau_motor_hist,
        "tau_res": tau_res_hist,
    }


class PendulumRLEnv:
    def __init__(self, cfg: BridgeConfig, trajectories: list[ReplayTrajectory], init_params: dict,
                 learn_delay: bool = False, reward_weights: RewardWeights | None = None,
                 domain_randomization: bool = True, delay_jitter_ms: float = 0.0, seed: int = 0,
                 max_refine_steps: int = 12):
        self.cfg = cfg
        self.trajs = trajectories
        self.learn_delay = learn_delay
        self.domain_randomization = domain_randomization
        self.delay_jitter_ms = delay_jitter_ms
        self.rng = np.random.default_rng(seed)
        self.max_refine_steps = max_refine_steps
        self.reward_weights = reward_weights or RewardWeights()
        self.param_names = list(PARAM_NAMES_DEFAULT) + (["delay_sec"] if learn_delay else [])
        self.center = np.array([init_params[k] for k in self.param_names], dtype=float)
        self.scales = np.array([0.03, 0.0007, 0.02, 0.02, 0.03, 0.02, 0.2, 0.01] + ([0.004] if learn_delay else []), dtype=float)
        self.lb = np.array([0.03, 0.0002, 0.0001, 0.0001, 0.01, 0.0, 0.2, 0.001] + ([0.0] if learn_delay else []), dtype=float)
        self.ub = np.array([0.35, 0.02, 1.0, 0.8, 1.0, 1.5, 20.0, 0.3] + ([0.25] if learn_delay else []), dtype=float)
        self.reset()

    def _vec_to_params(self, vec):
        p = {k: float(v) for k, v in zip(self.param_names, vec)}
        if "delay_sec" not in p:
            p["delay_sec"] = 0.0
        return p

    def _compute_metrics(self, vec):
        p = self._vec_to_params(vec)
        rmse_t = []; rmse_o = []; rmse_a = []; bias_t = []; bias_o = []; pwm_m = []; curr_m=[]; power_m=[]
        details = []
        for tr in self.active_trajs:
            d = tr.delay_sec_est
            if self.learn_delay:
                d = p["delay_sec"]
            elif self.delay_jitter_ms > 0.0:
                d += self.rng.uniform(-self.delay_jitter_ms, self.delay_jitter_ms) * 1e-3
            sim = simulate_trajectory(self.cfg, tr, p, d)
            et = sim["theta"] - tr.theta
            eo = sim["omega"] - tr.omega
            ea = sim["alpha"] - tr.alpha
            rmse_t.append(float(np.sqrt(np.mean(et**2))))
            rmse_o.append(float(np.sqrt(np.mean(eo**2))))
            rmse_a.append(float(np.sqrt(np.mean(ea**2))))
            bias_t.append(float(np.mean(et)))
            bias_o.append(float(np.mean(eo)))
            pwm_m.append(float(np.sqrt(np.mean((sim["cmd_delayed"] - tr.hw_pwm) ** 2))))
            if np.isfinite(tr.current).any():
                curr_m.append(float(np.sqrt(np.nanmean((sim["current"] - tr.current) ** 2))))
            if np.isfinite(tr.power).any():
                power_m.append(float(np.sqrt(np.nanmean((sim["power"] - tr.power) ** 2))))
            details.append((tr, sim, d))

        metrics = {
            "rmse_theta": float(np.mean(rmse_t)),
            "rmse_omega": float(np.mean(rmse_o)),
            "rmse_alpha": float(np.mean(rmse_a)),
            "bias_theta": float(np.mean(bias_t)),
            "bias_omega": float(np.mean(bias_o)),
            "pwm_mismatch": float(np.mean(pwm_m)) if pwm_m else 0.0,
            "current_mismatch": float(np.mean(curr_m)) if curr_m else 0.0,
            "power_mismatch": float(np.mean(power_m)) if power_m else 0.0,
            "details": details,
        }
        return metrics

    def _weighted_loss(self, metrics, vec, action=None):
        w = self.reward_weights
        loss = (
            w.theta * metrics["rmse_theta"] +
            w.omega * metrics["rmse_omega"] +
            w.alpha * metrics["rmse_alpha"] +
            w.pwm * metrics["pwm_mismatch"] +
            w.current * metrics["current_mismatch"] +
            w.power * metrics["power_mismatch"]
        )
        loss += w.param_reg * float(np.mean(((vec - self.center) / np.maximum(self.scales, 1e-6)) ** 2))
        if action is not None:
            loss += w.action_penalty * float(np.mean(action ** 2))
        return float(loss)

    def _state_vec(self, metrics):
        norm_params = (self.params - self.center) / np.maximum(self.scales, 1e-6)
        feats = np.array([
            metrics["rmse_theta"], metrics["rmse_omega"], metrics["rmse_alpha"],
            metrics["bias_theta"], metrics["bias_omega"], metrics["pwm_mismatch"],
            metrics["current_mismatch"], metrics["power_mismatch"],
        ], dtype=float)
        return np.concatenate([norm_params, feats]).astype(np.float32)

    def reset(self):
        self.steps = 0
        self.active_trajs = list(self.trajs)
        self.params = self.center.copy()
        if self.domain_randomization:
            self.params += self.rng.normal(0.0, self.scales * 0.5)
            self.params = np.clip(self.params, self.lb, self.ub)
        self.metrics = self._compute_metrics(self.params)
        self.prev_loss = self._weighted_loss(self.metrics, self.params)
        return self._state_vec(self.metrics)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        self.params = np.clip(self.params + action * self.scales, self.lb, self.ub)
        metrics = self._compute_metrics(self.params)
        new_loss = self._weighted_loss(metrics, self.params, action=action)
        improvement = self.prev_loss - new_loss
        reward = float(improvement - new_loss)
        if self.steps + 1 >= self.max_refine_steps and new_loss < 0.7 * self.prev_loss:
            reward += 0.2
        self.prev_loss = new_loss
        self.metrics = metrics
        self.steps += 1
        done = self.steps >= self.max_refine_steps
        info = {"loss": new_loss, **{k: v for k, v in metrics.items() if k != "details"}}
        return self._state_vec(metrics), reward, done, info
