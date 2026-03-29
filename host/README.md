# Host Layer (`host/`)

This layer runs the host-side digital twin, online identification, logging, and visualization for the Chrono pendulum.

## Main runtime: `chrono_pendulum.py`

### What it does
- Builds the 1-DOF PyChrono pendulum model.
- Subscribes to hardware topics (`/cmd/u`, `/hw/*`, `/ina219/*`).
- Publishes simulated states (`/sim/*`).
- Runs delay estimation/compensation and online EKF-style identification.
- Saves CSV + meta JSON logs.

### Physical model (updated)

#### 1) Gravity representation
Gravity is **not** identified via surrogate `mgl` anymore.

Gravity is represented using COM physics with:
- total mass `m_total = link_mass + imu_mass = 0.220 kg`
- COM offset from pivot `l_com`
- system gravity `g`

#### 2) COM/inertia coupling
A single effective COM parameter is used:
- identified parameter: `l_com`
- configured base COM inertia: `J_cm_base`

Pivot inertia follows:

\[
J_{pivot} = J_{cm\_base} + m_{total} l_{com}^2
\]

Pivot-space dynamics:
- `J_pivot = J_cm_base + m_total*l_com^2` set directly on the rigid body inertia
- explicit gravity torque term `tau_gravity = m_total*g*l_com*sin(theta)` in motor-side torque composition

Visual geometry still uses `link_L`, while dynamic modeling uses `l_com` and fixed masses.

#### 3) Unified resistance block
A single resistance model is used consistently in simulation and identified model equations:

\[
\tau_{res}(\omega) = b_{eq}\,\omega + \tau_{eq}\,\tanh(\omega/\epsilon)
\]

With decomposition logged as:
- `tau_visc = b_eq * omega`
- `tau_coul = tau_eq * tanh(omega/eps)`
- `tau_res = tau_visc + tau_coul`

#### 4) Electrical model
Used model:
- `duty = clamp(u_eff / pwm_limit, -1, 1)`
- `v_applied = duty * v_bus_filtered`
- `i_raw = (v_applied - k_e * omega) / R`
- `i_eff = sign(i_raw) * max(|i_raw| - i0, 0)`
- `tau_motor = k_t * i_eff`

Net torque:

\[
\tau_{net} = \tau_{motor} - \tau_{res}
\]

Optional current clipping is configurable (`current_clip_enable`, `current_clip_A`).

#### 5) Delay compensation
Delay is estimated from `/cmd/u` and `/hw/pwm_applied`, then applied as:

\[
u_{sim}(t) = u_{cmd}(t-\Delta t_d)
\]

The implementation conceptually delays the faster command side to match hardware-applied PWM timing. The delayed command is used for:
- simulation input
- sim-real comparison basis
- EKF update input

## INA219 handling (robust/noise-aware)

INA219 is treated as an auxiliary/noisy source:
- bus voltage: robust filter (median/Hampel + LPF) then used by electrical model
- current/power: filtered/logged mainly for diagnostics, not required for estimator stability
- fallback nominal bus voltage is used when INA data is invalid or disabled

Config switches include:
- `electrical_use_ina_bus_voltage`
- `nominal_bus_voltage`
- `ina_*` robust filter window/threshold parameters

## Online identification (`chrono_core/estimation.py`)

Current EKF augmented state:
- `[theta, omega, l_com, b_eq, tau_eq, delay_sec]`

Removed from active identification flow:
- `mgl`

`blend_parameters_for_sim()` forwards the current identified parameters (`l_com`, `b_eq`, `tau_eq`, delay, electrical params) into simulation torque/electrical computation.

## Logging additions

CSV/meta logs now include (when available):
- `cmd_u_raw`, `cmd_u_delayed`, `hw_pwm`, `delay_sec_est`
- `bus_v_raw`, `bus_v_filtered`
- `current_raw_A`, `current_filtered_A`, `power_raw_W`
- `tau_motor`, `tau_res`, `tau_visc`, `tau_coul`
- `i_pred`, `v_applied`
- identified parameter snapshots (`l_com_est`, `b_eq_est`, `tau_eq_est`, ...)

## Modules
- `chrono_core/config.py`: runtime/config defaults
- `chrono_core/dynamics.py`: Chrono pendulum model + unified torque/electrical model
- `chrono_core/estimation.py`: delay estimator, robust filters, EKF, convergence monitor
- `chrono_core/calibration_io.py`: calibration/radius JSON loading
- `plot_pendulum.py`: log plotting
- `RL_fitting.py`: offline parameter optimization
- `imu_viewer.py`: IMU visualization

## Offline RL replay calibration workflow (new)

Use `train_pendulum_rl.py` for episodic/offline parameter calibration from recorded CSV logs.

### Inputs
- `--calibration_json <path>` (required)
- `--parameter_json <path>` (optional)
  - if omitted, initial parameters are built from `calibration.json` + `BridgeConfig` defaults
- `--csv <path>` (repeatable) or `--csv_dir <dir>`

### Core design
- This workflow **does not** control hardware in real-time.
- It replays logged trajectories and updates model parameters to reduce sim-vs-real mismatch.
- Default calibrated vector:
  - `[l_com, J_cm_base, b_eq, tau_eq, k_t, i0, R, k_e]`
- `delay_sec` is excluded by default and treated as replay-alignment metadata.

### Delay policy (default)
1. Estimate delay per trajectory from `cmd_u_raw` vs `hw_pwm`.
2. Replay with fixed aligned command `u_sim(t) = u_cmd(t - delay_est)`.
3. Compute loss/reward on aligned replay basis.

Advanced mode:
- `--learn_delay` enables joint learning of `delay_sec` as an extra parameter.

Robustness option:
- `--delay_jitter_ms` adds only small random delay jitter around pre-estimated delay during training domain randomization.

### Deterministic pre-fit + PPO fine-tune
- Optional pre-fit local random search (`--prefitON`/`--prefitOFF`) generates a strong initialization.
- PPO then performs episodic parameter-delta refinement in `PendulumReplayCalibrationEnv`.
- State includes normalized parameter vector + compact replay error features (`RMSE`, bias, loss progress).
- Reward follows improvement-based structure: improvement − weighted loss − action penalty.

### Domain randomization
Enable/disable with:
- `--domain_randomizationON` / `--domain_randomizationOFF`

Randomization includes bounded perturbations around calibrated centers and optional delay jitter.

### PPO-style interface
Supports Chrono-like arguments:
- `-n/--num_episodes`
- `-g/--gamma`
- `-l/--lam`
- `-k/--kl_targ`
- `-b/--batch_size`
- `--renderON/--renderOFF`

Additional calibration controls:
- `--prefitON/--prefitOFF`
- `--learn_delay`
- `--delay_override`
- `--delay_jitter_ms`
- `--seed`
- `--interactive` (prompt mode; pressing Enter keeps defaults)

### Train/validation/test split
- Multiple CSVs: split by file.
- Single CSV: current implementation reuses the single file for train/val/test.

### Outputs
Written under `--outdir`:
- `params_initial.json`
- `params_prefit.json` (if enabled)
- `params_final_rl.json`
- `metadata.json` (settings, files, reward weights, randomization ranges, best validation score, timestamp, per-trajectory delay estimates)
- `training_history.json`
- plots:
  - reward/loss curves
  - RMSE curves (`theta`, `omega`, `alpha` + validation)
  - parameter convergence
  - delay diagnostics

### Replay/export CSV compatibility
Use `replay_pendulum_export.py` to export replayed trajectories with the **same column names and order** used by `chrono_pendulum.py`, enabling direct use with existing `plot_pendulum.py` tooling.
