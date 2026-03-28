#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Chrono pendulum Gymnasium wrapper + PPO trainer.

This script provides a PPO training entrypoint with CLI options similar to the
older TensorFlow demo (`num_episodes`, `gamma`, `lam`, `kl_targ`, `batch_size`,
`renderON/OFF`) while keeping Stable-Baselines3 PPO backend.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PendulumParams:
    J: float = 0.010
    b: float = 0.030
    tau_c: float = 0.080
    mgl: float = 0.550
    k_t: float = 0.250
    i0: float = 0.050
    R: float = 2.0
    k_e: float = 0.020
    tanh_eps: float = 0.05
    pwm_limit: float = 255.0
    bus_v: float = 12.0


class IrrlichtPendulumRenderer:
    """Optional full renderer using PyChrono + Irrlicht."""

    def __init__(self, params: PendulumParams, theta0: float, dt: float):
        import pychrono as ch
        import pychrono.irrlicht as irr

        self._ch = ch
        self._irr = irr
        self._dt = dt

        self.sys = ch.ChSystemNSC()
        self.sys.SetGravitationalAcceleration(ch.ChVector3d(0.0, -9.81, 0.0))

        self.base = ch.ChBody()
        self.base.SetFixed(True)
        self.base.SetPos(ch.ChVector3d(0.0, 0.0, 0.0))
        self.sys.Add(self.base)

        motor_radius = 0.020
        motor_length = 0.050
        link_L = 0.285
        link_W = 0.020
        link_T = 0.006
        link_mass = 0.200

        q_cyl_to_x = ch.QuatFromAngleZ(-math.pi / 2.0)
        motor_cyl = ch.ChVisualShapeCylinder(motor_radius, motor_length)
        self.base.AddVisualShape(motor_cyl, ch.ChFramed(ch.ChVector3d(0.0, 0.0, 0.0), q_cyl_to_x))

        self.link = ch.ChBody()
        self.link.SetMass(link_mass)
        izz_com = (1.0 / 12.0) * link_mass * (link_L ** 2 + link_W ** 2)
        self.link.SetInertiaXX(ch.ChVector3d(1e-5, 1e-5, izz_com))
        self.link.SetFrameCOMToRef(ch.ChFramed(ch.ChVector3d(0.0, -link_L / 2.0, 0.0), ch.QUNIT))
        self.link.SetPos(ch.ChVector3d(0.0, 0.0, motor_length / 2.0))
        self.link.SetRot(ch.QuatFromAngleZ(theta0))
        self.sys.Add(self.link)

        vis_link = ch.ChVisualShapeBox(link_W, link_L, link_T)
        self.link.AddVisualShape(vis_link, ch.ChFramed(ch.ChVector3d(0.0, -link_L / 2.0, 0.0), ch.QUNIT))

        self.vis = irr.ChVisualSystemIrrlicht()
        self.vis.AttachSystem(self.sys)
        self.vis.SetWindowSize(1280, 900)
        self.vis.SetWindowTitle("Chrono Pendulum RL (Irrlicht)")
        self.vis.Initialize()
        self.vis.AddCamera(ch.ChVector3d(0.4, 0.2, 0.4))
        self.vis.AddTypicalLights()

    def draw(self, theta: float, omega: float):
        self.link.SetRot(self._ch.QuatFromAngleZ(theta))
        self.link.SetAngVelLocal(self._ch.ChVector3d(0.0, 0.0, omega))

        if self.vis.Run():
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        time.sleep(self._dt)


class ChronoPendulumEnv(gym.Env):
    """1-DOF rotary pendulum env compatible with Gymnasium."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        params: PendulumParams,
        dt: float = 0.01,
        episode_seconds: float = 10.0,
        theta0_deg: float = -10.0,
        omega0: float = 0.0,
        action_repeat: int = 2,
        render_mode: str | None = None,
        wrong_dir_window_sec: float = 0.5,
        dir_check_cmd_thresh: float = 20.0,
        real_clockwise: bool = True,
        diverge_omega_thresh: float = 30.0,
        early_term_penalty: float = 50.0,
    ):
        super().__init__()
        self.params = params
        self.dt = float(dt)
        self.max_steps = int(max(1, episode_seconds / dt))
        self.theta0 = math.radians(theta0_deg)
        self.omega0 = float(omega0)
        self.action_repeat = int(max(1, action_repeat))
        self.render_mode = render_mode
        self.step_dt = self.dt * self.action_repeat
        self.wrong_dir_window_sec = float(max(0.05, wrong_dir_window_sec))
        self.dir_check_cmd_thresh = float(max(0.0, dir_check_cmd_thresh))
        self.real_clockwise = bool(real_clockwise)
        self.diverge_omega_thresh = float(max(1.0, diverge_omega_thresh))
        self.early_term_penalty = float(max(0.0, early_term_penalty))

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -40.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.theta = self.theta0
        self.omega = self.omega0
        self.step_count = 0
        self.last_cmd_u = 0.0
        self._renderer = None
        self._wrong_dir_time = 0.0
        self._diverge_time = 0.0

    def _get_obs(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta), self.omega], dtype=np.float32)

    def _motor_torque(self, cmd_u: float, omega: float) -> float:
        p = self.params
        u_eff = math.copysign(max(abs(cmd_u) - p.i0, 0.0), cmd_u)
        duty = np.clip(u_eff / max(p.pwm_limit, 1e-9), -1.0, 1.0)
        v_applied = duty * p.bus_v
        i_pred = (v_applied - p.k_e * omega) / max(p.R, 1e-6)
        tau_motor = p.k_t * i_pred
        tau_visc = p.b * omega
        tau_coul = p.tau_c * math.tanh(omega / max(p.tanh_eps, 1e-9))
        return tau_motor - tau_visc - tau_coul

    def _dynamics(self, theta: float, omega: float, cmd_u: float) -> tuple[float, float]:
        p = self.params
        tau_net = self._motor_torque(cmd_u, omega)
        alpha = (tau_net - p.mgl * math.sin(theta)) / max(p.J, 1e-6)
        return omega, alpha

    def _rk4_step(self, cmd_u: float):
        h = self.dt
        th0, om0 = self.theta, self.omega

        k1_th, k1_om = self._dynamics(th0, om0, cmd_u)
        k2_th, k2_om = self._dynamics(th0 + 0.5 * h * k1_th, om0 + 0.5 * h * k1_om, cmd_u)
        k3_th, k3_om = self._dynamics(th0 + 0.5 * h * k2_th, om0 + 0.5 * h * k2_om, cmd_u)
        k4_th, k4_om = self._dynamics(th0 + h * k3_th, om0 + h * k3_om, cmd_u)

        self.theta = wrap_to_pi(th0 + (h / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th))
        self.omega = om0 + (h / 6.0) * (k1_om + 2.0 * k2_om + 2.0 * k3_om + k4_om)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.theta = self.theta0 + self.np_random.uniform(-0.02, 0.02)
        self.omega = self.omega0 + self.np_random.uniform(-0.2, 0.2)
        self.step_count = 0
        self.last_cmd_u = 0.0
        self._wrong_dir_time = 0.0
        self._diverge_time = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        u_norm = float(np.clip(action[0], -1.0, 1.0))
        cmd_u = u_norm * self.params.pwm_limit
        self.last_cmd_u = cmd_u

        for _ in range(self.action_repeat):
            self._rk4_step(cmd_u)

        theta_err = wrap_to_pi(self.theta - math.pi)
        reward = 1.0 - (theta_err ** 2) - 0.02 * (self.omega ** 2) - 0.0008 * (cmd_u ** 2)

        self.step_count += 1
        terminated = False
        term_reason = ""

        # 1) Wrong rotation direction persistence check
        if abs(cmd_u) >= self.dir_check_cmd_thresh:
            expected_omega_sign = -np.sign(cmd_u) if self.real_clockwise else np.sign(cmd_u)
            if abs(expected_omega_sign) > 0 and np.sign(self.omega) != 0 and np.sign(self.omega) != expected_omega_sign:
                self._wrong_dir_time += self.step_dt
            else:
                self._wrong_dir_time = 0.0
        else:
            self._wrong_dir_time = 0.0

        if self._wrong_dir_time >= self.wrong_dir_window_sec:
            terminated = True
            term_reason = "wrong_rotation_direction"

        # 2) Divergence persistence check (too high angular velocity / non-finite)
        if not np.isfinite(self.theta) or not np.isfinite(self.omega):
            terminated = True
            term_reason = "non_finite_state"
        else:
            if abs(self.omega) > self.diverge_omega_thresh:
                self._diverge_time += self.step_dt
            else:
                self._diverge_time = 0.0
            if self._diverge_time >= self.wrong_dir_window_sec:
                terminated = True
                term_reason = "diverged_omega"

        if terminated:
            reward -= self.early_term_penalty

        truncated = self.step_count >= self.max_steps
        info = {
            "theta": float(self.theta),
            "omega": float(self.omega),
            "cmd_u": float(cmd_u),
            "theta_error": float(theta_err),
            "terminated_early": bool(terminated),
            "termination_reason": term_reason,
            "wrong_dir_time": float(self._wrong_dir_time),
            "diverge_time": float(self._diverge_time),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None

        if self._renderer is None:
            try:
                self._renderer = IrrlichtPendulumRenderer(self.params, self.theta, self.dt * self.action_repeat)
            except Exception:
                self._renderer = False

        if self._renderer:
            self._renderer.draw(self.theta, self.omega)
        else:
            msg = (
                f"step={self.step_count:04d} "
                f"theta={self.theta:+.3f} rad "
                f"omega={self.omega:+.3f} rad/s "
                f"cmd={self.last_cmd_u:+.1f} "
                f"[irrlicht unavailable -> text render]"
            )
            print(msg)
            time.sleep(self.dt * self.action_repeat)
        return None


def evaluate(model: PPO, env: ChronoPendulumEnv, episodes: int = 5, render: bool = False) -> dict:
    rewards = []
    final_theta_error = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_reward = 0.0
        info = {}
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        rewards.append(ep_reward)
        final_theta_error.append(abs(info.get("theta_error", math.pi)))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_final_theta_error_rad": float(np.mean(final_theta_error)),
    }


def main():
    ap = argparse.ArgumentParser(description="Chrono pendulum Gym wrapper + PPO trainer")

    # TensorFlow-demo style knobs (aliases included)
    ap.add_argument("-n", "--num_episodes", type=int, default=1000)
    ap.add_argument("-g", "--gamma", type=float, default=0.995)
    ap.add_argument("-l", "--lam", type=float, default=0.98)
    ap.add_argument("-k", "--kl_targ", type=float, default=0.003)
    ap.add_argument("-b", "--batch_size", type=int, default=20)
    ap.add_argument("--renderON", dest="render_on", action="store_true")
    ap.add_argument("--renderOFF", dest="render_on", action="store_false")
    ap.set_defaults(render_on=False)

    # Optional explicit timesteps (if omitted, computed from episodes)
    ap.add_argument("--timesteps", type=int, default=None)

    # Env/general options
    ap.add_argument("--outdir", default="rl_results")
    ap.add_argument("--episode-seconds", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--theta0-deg", type=float, default=-10.0)
    ap.add_argument("--omega0", type=float, default=0.0)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--wrong-dir-window-sec", type=float, default=0.5)
    ap.add_argument("--dir-check-cmd-thresh", type=float, default=20.0)
    ap.add_argument("--real-clockwise", dest="real_clockwise", action="store_true")
    ap.add_argument("--real-counterclockwise", dest="real_clockwise", action="store_false")
    ap.set_defaults(real_clockwise=True)
    ap.add_argument("--diverge-omega-thresh", type=float, default=30.0)
    ap.add_argument("--early-term-penalty", type=float, default=50.0)
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto")
    ap.add_argument("--seed", type=int, default=0)

    # Physical parameters (compatible with chrono_pendulum defaults)
    ap.add_argument("--J", type=float, default=0.010)
    ap.add_argument("--b", type=float, default=0.030)
    ap.add_argument("--tau-c", dest="tau_c", type=float, default=0.080)
    ap.add_argument("--mgl", type=float, default=0.550)
    ap.add_argument("--k-t", dest="k_t", type=float, default=0.250)
    ap.add_argument("--i0", type=float, default=0.050)
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--k-e", dest="k_e", type=float, default=0.020)
    ap.add_argument("--tanh-eps", type=float, default=0.05)
    ap.add_argument("--pwm-limit", type=float, default=255.0)
    ap.add_argument("--bus-v", type=float, default=12.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = PendulumParams(
        J=args.J,
        b=args.b,
        tau_c=args.tau_c,
        mgl=args.mgl,
        k_t=args.k_t,
        i0=args.i0,
        R=args.R,
        k_e=args.k_e,
        tanh_eps=args.tanh_eps,
        pwm_limit=args.pwm_limit,
        bus_v=args.bus_v,
    )

    env = ChronoPendulumEnv(
        params=params,
        dt=args.dt,
        episode_seconds=args.episode_seconds,
        theta0_deg=args.theta0_deg,
        omega0=args.omega0,
        action_repeat=args.action_repeat,
        render_mode="human" if args.render_on else None,
        wrong_dir_window_sec=args.wrong_dir_window_sec,
        dir_check_cmd_thresh=args.dir_check_cmd_thresh,
        real_clockwise=args.real_clockwise,
        diverge_omega_thresh=args.diverge_omega_thresh,
        early_term_penalty=args.early_term_penalty,
    )
    check_env(env, warn=True)

    total_timesteps = args.timesteps
    if total_timesteps is None:
        total_timesteps = int(max(1, args.num_episodes) * env.max_steps)

    n_steps = max(env.max_steps, args.batch_size)
    monitor_path = outdir / "ppo_monitor.csv"
    train_env = Monitor(env, filename=str(monitor_path))

    model = PPO(
        "MlpPolicy",
        train_env,
        device=args.device,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.lam,
        target_kl=args.kl_targ,
        clip_range=0.2,
        ent_coef=0.0,
    )
    model.learn(total_timesteps=total_timesteps)

    model_path = outdir / "ppo_chrono_pendulum.zip"
    model.save(str(model_path))

    eval_stats = evaluate(model, env, episodes=5, render=args.render_on)
    result = {
        "timesteps": int(total_timesteps),
        "num_episodes": int(args.num_episodes),
        "device": args.device,
        "ppo": {
            "gamma": float(args.gamma),
            "lam": float(args.lam),
            "kl_targ": float(args.kl_targ),
            "batch_size": int(args.batch_size),
            "n_steps": int(n_steps),
        },
        "params": asdict(params),
        "env": {
            "dt": float(args.dt),
            "episode_seconds": float(args.episode_seconds),
            "theta0_deg": float(args.theta0_deg),
            "omega0": float(args.omega0),
            "action_repeat": int(args.action_repeat),
            "max_steps": int(env.max_steps),
            "render_on": bool(args.render_on),
            "wrong_dir_window_sec": float(args.wrong_dir_window_sec),
            "dir_check_cmd_thresh": float(args.dir_check_cmd_thresh),
            "real_clockwise": bool(args.real_clockwise),
            "diverge_omega_thresh": float(args.diverge_omega_thresh),
            "early_term_penalty": float(args.early_term_penalty),
        },
        "eval": eval_stats,
        "artifacts": {
            "model": str(model_path),
            "monitor": str(monitor_path),
        },
    }

    result_path = outdir / "ppo_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {model_path}")
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
