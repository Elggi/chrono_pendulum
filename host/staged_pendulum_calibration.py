#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chrono_core.calibration_io import apply_calibration_json
from chrono_core.config import BridgeConfig
from chrono_core.pendulum_rl_env import (
    PendulumRLEnv,
    build_init_params,
    compute_error_features,
    load_replay_csv,
    simplified_loss,
    simulate_trajectory,
    split_trajectories,
    weighted_loss,
)
from chrono_core.pendulum_rl_plots import (
    plot_param_convergence,
    plot_stage1_regression_summary,
    plot_training_curves,
)


def _input(prompt: str, default: str | None = None) -> str:
    raw = input(prompt).strip()
    if raw == "" and default is not None:
        return default
    return raw


def _yn(prompt: str, default_yes: bool = True) -> bool:
    default_hint = "Y/n" if default_yes else "y/N"
    val = _input(f"{prompt} [{default_hint}] ", "y" if default_yes else "n").lower()
    return val in ("y", "yes", "1", "true")


def list_csv_logs(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("*.csv"))


def list_param_json(base_dir: Path) -> list[Path]:
    return sorted([p for p in base_dir.glob("*.json") if p.is_file()])


def choose_from_list(items: list[Path], title: str, min_count: int = 1, allow_multi: bool = True) -> list[Path]:
    if not items:
        print(f"No entries found for: {title}")
        return []
    print(title)
    for i, item in enumerate(items, start=1):
        print(f"[{i}] {item.name}")
    while True:
        raw = _input("Select CSV indices (comma separated): " if allow_multi else "Select index: ")
        try:
            if allow_multi:
                idx = [int(x.strip()) for x in raw.split(",") if x.strip()]
            else:
                idx = [int(raw)]
        except ValueError:
            print("Invalid input; please enter numeric indices.")
            continue
        if len(idx) < min_count:
            print(f"Please select at least {min_count} file(s).")
            continue
        try:
            chosen = [items[i - 1] for i in idx]
        except IndexError:
            print("Index out of range.")
            continue
        return chosen


def check_bound_hit(param: str, value: float, bounds: tuple[float, float], tol_frac: float = 0.02) -> dict[str, bool]:
    lo, hi = float(bounds[0]), float(bounds[1])
    span = max(hi - lo, 1e-12)
    near = tol_frac * span
    return {
        "lower_hit": abs(value - lo) <= 1e-12,
        "upper_hit": abs(value - hi) <= 1e-12,
        "lower_near": (value - lo) <= near,
        "upper_near": (hi - value) <= near,
        "warning": ((value - lo) <= near) or ((hi - value) <= near),
        "param": param,
    }


def _compute_geometry(cfg: BridgeConfig):
    m_total = cfg.rod_mass + cfg.imu_mass
    j_rod = (1.0 / 3.0) * cfg.rod_mass * (cfg.rod_length ** 2)
    j_imu = cfg.imu_mass * (cfg.r_imu ** 2)
    return float(m_total), float(j_rod + j_imu)


def _fit_stage1_least_squares(trajs, cfg: BridgeConfig, gravity_used: float, k_bounds: tuple[float, float]):
    m_total, j_pivot = _compute_geometry(cfg)
    l_bounds = (0.01, float(cfg.link_L))
    x_rows = []
    y_vals = []
    for traj in trajs:
        y = j_pivot * traj.alpha_real
        x1 = traj.cmd_u
        x2 = -(m_total * gravity_used) * np.sin(traj.theta_real)
        x_rows.append(np.column_stack([x1, x2]))
        y_vals.append(y)
    X = np.vstack(x_rows)
    y = np.concatenate(y_vals)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    k_u = float(np.clip(beta[0], k_bounds[0], k_bounds[1]))
    l_com = float(np.clip(beta[1], l_bounds[0], l_bounds[1]))
    y_hat = X @ np.array([k_u, l_com], dtype=float)
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    return {
        "K_u": k_u,
        "l_com": l_com,
        "rmse": rmse,
        "J_pivot": j_pivot,
        "m_total": m_total,
        "bounds": {"K_u": [k_bounds[0], k_bounds[1]], "l_com": [l_bounds[0], l_bounds[1]]},
        "y_true": y,
        "y_pred": y_hat,
    }


def save_stage_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_stage1(cfg: BridgeConfig, run_logs: Path):
    csv_files = list_csv_logs(run_logs)
    print("=== Stage 1 Regression ===")
    print("Available CSV logs:")
    selected_csv = choose_from_list(csv_files, "", min_count=1, allow_multi=True)
    if not selected_csv:
        return

    print("Use calibration JSON?\n[1] yes\n[2] no")
    use_calib = _input("> ", "1") == "1"
    calib_data = None
    model_init_mode = "fresh"
    gravity_used = float(cfg.gravity)
    if use_calib:
        json_files = list_param_json(run_logs)
        if json_files:
            picked = choose_from_list(json_files, "Select calibration JSON", min_count=1, allow_multi=False)
            if picked:
                calib_data = apply_calibration_json(cfg, str(picked[0]))
                model_init_mode = "calibration_geometry_only"
        else:
            print("No JSON files found; continuing without calibration JSON.")
        summary = calib_data.get("summary", {}) if isinstance(calib_data, dict) else {}
        g_eff = summary.get("g_eff_mps2") if isinstance(summary, dict) else None
        if g_eff is not None and _yn("Use calibrated gravity?", True):
            gravity_used = float(g_eff)

    lo = float(_input("Enter K_u minimum [default: 1e-6]: ", "1e-6"))
    hi = float(_input("Enter K_u maximum [default: 1.0]: ", "1.0"))
    while hi <= lo:
        print("K_u maximum must be greater than minimum.")
        lo = float(_input("Enter K_u minimum [default: 1e-6]: ", "1e-6"))
        hi = float(_input("Enter K_u maximum [default: 1.0]: ", "1.0"))

    trajs = [load_replay_csv(p, cfg) for p in selected_csv]
    if trajs:
        src = trajs[0]
        print("[INFO] stage1_regression_sources:")
        print(f"  - theta_source: {src.theta_source}")
        print(f"  - omega_source: {src.omega_source}")
        print(f"  - alpha_source: {src.alpha_source}")
        print(f"  - input_source: {src.input_source}")
        print(f"  - target_source: {src.target_source}")
        print(
            f"[INFO] model_init_mode: {model_init_mode} | "
            f"K_u_init={cfg.K_u_init:.6g}, b_eq_init={cfg.b_eq_init:.6g}, "
            f"tau_eq_init={cfg.tau_eq_init:.6g}, l_com_init={cfg.l_com_init:.6g}"
        )
        if "sim_fallback" in src.theta_source:
            print("[WARN] theta_real missing in some logs; using sim theta fallback for Stage1.")
    while True:
        res = _fit_stage1_least_squares(trajs, cfg, gravity_used, (lo, hi))
        bku = check_bound_hit("K_u", res["K_u"], (lo, hi))
        blc = check_bound_hit("l_com", res["l_com"], (0.01, cfg.link_L))
        print("\n=== Stage 1 Result ===")
        print(f"K_u      = {res['K_u']:.8f}")
        print(f"l_com    = {res['l_com']:.8f} m")
        print(f"l_com/L  = {res['l_com'] / max(cfg.link_L, 1e-9):.6f}")
        print(f"RMSE     = {res['rmse']:.8f}")
        print("\nBound hit check:")
        print(f"- K_u lower bound hit: {'YES' if bku['lower_hit'] else 'NO'}")
        print(f"- K_u upper bound hit: {'YES' if bku['upper_hit'] else 'NO'}")
        print(f"- l_com lower bound hit: {'YES' if blc['lower_hit'] else 'NO'}")
        print(f"- l_com upper bound hit: {'YES' if blc['upper_hit'] else 'NO'}")
        if bku["warning"] or blc["warning"]:
            print("[WARN] One or more parameters are at/near bounds.")

        print("\n[A] Accept and save as stage1_params.json\n[R] Rerun with new K_u bounds\n[Q] Quit without saving")
        action = _input("> ", "A").lower()
        if action == "a":
            payload = {
                "stage": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": "J*alpha = K_u*u - m*g*l_com*sin(theta)",
                "active_params": ["K_u", "l_com"],
                "fixed_params": {
                    "b_eq": float(cfg.b_eq_init),
                    "tau_eq": float(cfg.tau_eq_init),
                    "J_pivot": float(res["J_pivot"]),
                },
                "identified_params": {"K_u": float(res["K_u"]), "l_com": float(res["l_com"])},
                "bounds": res["bounds"],
                "metrics": {"rmse": float(res["rmse"])},
                "csv_files": [str(p) for p in selected_csv],
                "gravity_used": float(gravity_used),
                "model_init_mode": model_init_mode,
                "theta_source": src.theta_source if trajs else "theta_real",
                "omega_source": src.omega_source if trajs else "omega_real",
                "alpha_source": src.alpha_source if trajs else "real_alpha_filtered",
                "input_source": src.input_source if trajs else "cmd_u_raw",
                "target_source": src.target_source if trajs else "J*real_alpha_filtered",
                "b_eq_init": float(cfg.b_eq_init),
                "tau_eq_init": float(cfg.tau_eq_init),
                "l_com_init": float(cfg.l_com_init),
                "b_eq_trainable_in_stage2": True,
                "tau_eq_trainable_in_stage2": True,
            }
            save_stage_json(run_logs / "stage1_params.json", payload)
            plot_stage1_regression_summary(
                res["y_true"],
                res["y_pred"],
                run_logs / "stage1_regression_fit_summary.png",
            )
            print(f"Saved: {run_logs / 'stage1_params.json'}")
            return
        if action == "r":
            lo = float(_input("Enter K_u minimum [default: 1e-6]: ", str(lo)))
            hi = float(_input("Enter K_u maximum [default: 1.0]: ", str(hi)))
            continue
        if action == "q":
            return


@dataclass
class StageRLConfig:
    stage: int
    active_params: list[str]
    fixed_params: list[str]
    loss_mode: str
    out_json: str


def evaluate_dataset_mode(env: PendulumRLEnv, params: dict[str, float], loss_mode: str):
    losses = []
    for traj in env.trajectories:
        sim = simulate_trajectory(traj, params, env.cfg, delay_sec=params.get("delay_sec", traj.delay_sec_est))
        feat = compute_error_features(traj, sim, align_shift_sec=0.0)
        if loss_mode == "simplified":
            losses.append(simplified_loss(feat, {"theta": 1.0, "omega": 1.0, "alpha": 1.0}))
        else:
            losses.append(weighted_loss(feat, env.reward_weights))
    return float(np.mean(losses)) if losses else 0.0


def _train_rl(env, val_env, episodes: int, seed: int = 7):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class _SB3Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(wrapped.state_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(wrapped.action_dim,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            return np.asarray(self.wrapped.reset(), dtype=np.float32), {}

        def step(self, action):
            obs, rew, done, info = self.wrapped.step(np.asarray(action, dtype=float))
            return np.asarray(obs, dtype=np.float32), float(rew), bool(done), False, info

    model = PPO("MlpPolicy", _SB3Env(env), seed=seed, n_steps=32, batch_size=16, verbose=0)
    history = {"episode_reward": [], "train_loss": [], "val_loss": []}
    param_hist = {
        "current_eval_params_per_episode": {k: [] for k in env.param_keys},
        "global_best_train_params_so_far": {k: [] for k in env.param_keys},
        "global_best_val_params_so_far": {k: [] for k in env.param_keys},
    }
    best_params = env.center.copy()
    best_val = float("inf")
    best_train = float("inf")
    for ep in range(1, episodes + 1):
        model.learn(total_timesteps=max(1, env.max_refine_steps), progress_bar=False, reset_num_timesteps=False)
        obs, _ = _SB3Env(env).reset()
        done = False
        total_reward = 0.0
        info = {"params": env.center}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = _SB3Env(env).step(action)
            total_reward += rew
            done = bool(terminated or truncated)
        cur = dict(info.get("params", {}))
        train_loss = evaluate_dataset_mode(env, cur, env.loss_mode)
        val_loss = evaluate_dataset_mode(val_env, cur, val_env.loss_mode)
        history["episode_reward"].append(float(total_reward))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        if train_loss < best_train:
            best_train = float(train_loss)
        if val_loss < best_val:
            best_val = float(val_loss)
            best_params = cur.copy()
        for k in env.param_keys:
            param_hist["current_eval_params_per_episode"][k].append(float(cur[k]))
            param_hist["global_best_train_params_so_far"][k].append(float(cur[k]))
            param_hist["global_best_val_params_so_far"][k].append(float(best_params.get(k, cur[k])))
        if ep % 10 == 0 or ep == 1 or ep == episodes:
            print(f"[Stage RL] ep {ep}/{episodes} reward={total_reward:.5f} train={train_loss:.5f} val={val_loss:.5f}")
    return best_params, best_val, history, param_hist


def run_stage_rl(cfg: BridgeConfig, run_logs: Path, stage_cfg: StageRLConfig):
    print(f"=== Stage {stage_cfg.stage} RL Fine-tuning ===")
    json_files = list_param_json(run_logs)
    if stage_cfg.stage == 2:
        candidates = [p for p in json_files if p.name == "stage1_params.json"]
    else:
        candidates = [p for p in json_files if p.name == "stage2_params.json"]
    if not candidates:
        print("Required input parameter JSON not found.")
        return
    picked = choose_from_list(candidates, "Select input parameter JSON:", min_count=1, allow_multi=False)
    if not picked:
        return
    with picked[0].open("r", encoding="utf-8") as f:
        pjson = json.load(f)

    csv_files = list_csv_logs(run_logs)
    print("Available CSV logs:")
    selected_csv = choose_from_list(csv_files, "", min_count=3, allow_multi=True)
    if len(selected_csv) < 3:
        print("Stage 2/3 require at least 3 CSV files.")
        return

    print("Domain randomization:\n[1] OFF (debug)\n[2] ON  (research)")
    domain_randomization = _input("> ", "2") == "2"

    init_params = build_init_params(cfg, calibration=None, parameter_json=pjson)
    model_init_mode = "loaded_parameter_json"
    # Fill from stage-style schema.
    if isinstance(pjson.get("identified_params"), dict):
        for k, v in pjson["identified_params"].items():
            init_params[k] = float(v)
    if pjson.get("model_init_mode") == "fresh":
        model_init_mode = "fresh_from_stage_json"

    print("[INFO] stage2_rl_param_policy:")
    print(f"  - b_eq_initial_for_stage2: {init_params.get('b_eq', cfg.b_eq_init):.6g}")
    print(f"  - tau_eq_initial_for_stage2: {init_params.get('tau_eq', cfg.tau_eq_init):.6g}")
    print(f"  - b_eq_trainable_in_ppo: {('b_eq' in stage_cfg.active_params)}")
    print(f"  - tau_eq_trainable_in_ppo: {('tau_eq' in stage_cfg.active_params)}")

    hard_bounds = {
        "K_u": (1e-6, 1.0),
        "l_com": (0.01, float(cfg.link_L)),
        "b_eq": (0.0, float(cfg.b_eq_max)),
        "tau_eq": (0.0, float(cfg.tau_eq_max)),
    }

    tr_paths, va_paths, te_paths = split_trajectories(selected_csv, seed=7)
    train = [load_replay_csv(p, cfg) for p in tr_paths]
    val = [load_replay_csv(p, cfg) for p in va_paths]
    _ = [load_replay_csv(p, cfg) for p in te_paths]

    env = PendulumRLEnv(
        trajectories=train,
        cfg=cfg,
        init_params=init_params,
        learn_delay=False,
        delay_jitter_ms=3.0 if domain_randomization else 0.0,
        domain_randomization=domain_randomization,
        seed=7,
        max_refine_steps=12,
        action_step_frac=0.08,
        init_noise_frac=0.07,
        param_keys_override=stage_cfg.active_params,
        bounds_override={k: hard_bounds[k] for k in stage_cfg.active_params},
        loss_mode=stage_cfg.loss_mode,
    )
    val_env = PendulumRLEnv(
        trajectories=val,
        cfg=cfg,
        init_params=init_params,
        learn_delay=False,
        delay_jitter_ms=0.0,
        domain_randomization=False,
        seed=8,
        max_refine_steps=12,
        action_step_frac=0.08,
        init_noise_frac=0.0,
        param_keys_override=stage_cfg.active_params,
        bounds_override={k: hard_bounds[k] for k in stage_cfg.active_params},
        loss_mode=stage_cfg.loss_mode,
    )
    best_params, best_val, history, param_hist = _train_rl(env, val_env, episodes=100)

    fixed = {}
    for k in stage_cfg.fixed_params:
        fixed[k] = float(init_params[k])
        best_params[k] = float(init_params[k])

    for p in stage_cfg.active_params:
        hit = check_bound_hit(p, float(best_params[p]), hard_bounds[p])
        if hit["warning"]:
            print(f"[WARN] {p} is at/near bound {hard_bounds[p]} (value={best_params[p]:.6f})")

    payload = {
        "stage": stage_cfg.stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_params": stage_cfg.active_params,
        "fixed_params": fixed,
        "identified_params": {k: float(best_params[k]) for k in stage_cfg.active_params},
        "bounds": {k: list(hard_bounds[k]) for k in stage_cfg.active_params},
        "csv_files": [str(p) for p in selected_csv],
        "metrics": {
            "best_val_loss": float(best_val),
            "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
            "final_episode_reward": float(history["episode_reward"][-1]) if history["episode_reward"] else None,
        },
        "domain_randomization": bool(domain_randomization),
        "model_init_mode": model_init_mode,
        "theta_source": train[0].theta_source if train else "theta_real",
        "omega_source": train[0].omega_source if train else "omega_real",
        "alpha_source": train[0].alpha_source if train else "real_alpha_filtered",
        "input_source": train[0].input_source if train else "cmd_u_raw",
        "target_source": train[0].target_source if train else "J*real_alpha_filtered",
        "b_eq_init": float(init_params.get("b_eq", cfg.b_eq_init)),
        "tau_eq_init": float(init_params.get("tau_eq", cfg.tau_eq_init)),
        "l_com_init": float(init_params.get("l_com", cfg.l_com_init)),
        "b_eq_trainable_in_stage2": bool("b_eq" in stage_cfg.active_params),
        "tau_eq_trainable_in_stage2": bool("tau_eq" in stage_cfg.active_params),
    }
    save_stage_json(run_logs / stage_cfg.out_json, payload)

    stage_dir = run_logs / f"stage{stage_cfg.stage}_plots"
    stage_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, stage_dir)
    plot_param_convergence(param_hist, stage_dir)
    for src_name, dst_name in [
        ("episode_reward.png", f"stage{stage_cfg.stage}_reward_curve.png"),
        ("loss_convergence.png", f"stage{stage_cfg.stage}_loss_curve.png"),
        ("parameter_convergence.png", f"stage{stage_cfg.stage}_parameter_evolution.png"),
    ]:
        src = stage_dir / src_name
        if src.exists():
            src.replace(stage_dir / dst_name)
    print(f"Saved: {run_logs / stage_cfg.out_json}")


def show_csv(run_logs: Path):
    logs = list_csv_logs(run_logs)
    print("Available CSV logs:")
    for i, p in enumerate(logs, start=1):
        print(f"[{i}] {p.name}")


def show_json(run_logs: Path):
    files = list_param_json(run_logs)
    print("Available parameter JSON files:")
    for i, p in enumerate(files, start=1):
        print(f"[{i}] {p.name}")


def main():
    run_logs = Path(__file__).resolve().parent / "run_logs"
    cfg = BridgeConfig()

    while True:
        print("\n=== Pendulum Parameter Optimization ===\n")
        print("[1] Stage 1 Regression (K_u, l_com)")
        print("[2] Stage 2 RL Fine-tuning (K_u, l_com, b_eq, tau_eq)")
        print("[3] Stage 3 RL Fine-tuning (K_u, l_com, b_eq, tau_eq)")
        print("[4] Show available CSV logs")
        print("[5] Show available parameter JSON files")
        print("[Q] Quit")
        cmd = _input("> ", "Q").lower()
        if cmd == "1":
            run_stage1(cfg, run_logs)
        elif cmd == "2":
            run_stage_rl(
                cfg,
                run_logs,
                StageRLConfig(
                    stage=2,
                    active_params=["K_u", "l_com", "b_eq", "tau_eq"],
                    fixed_params=[],
                    loss_mode="simplified",
                    out_json="stage2_params.json",
                ),
            )
        elif cmd == "3":
            run_stage_rl(
                cfg,
                run_logs,
                StageRLConfig(
                    stage=3,
                    active_params=["K_u", "l_com", "b_eq", "tau_eq"],
                    fixed_params=[],
                    loss_mode="full",
                    out_json="stage3_params.json",
                ),
            )
        elif cmd == "4":
            show_csv(run_logs)
        elif cmd == "5":
            show_json(run_logs)
        elif cmd == "q":
            break
        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
