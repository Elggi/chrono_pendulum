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

## Online identification (`chrono_core/signal_filter.py`)

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
- `chrono_core/signal_filter.py`: robust bus/current signal filtering
- `chrono_core/calibration_io.py`: calibration/radius JSON loading
- `plot_pendulum.py`: log plotting
- `train_pendulum_rl.py`: offline replay RL calibration (primary)
- `imu_viewer.py`: IMU visualization

## Offline RL Replay Calibration (new)

Use `train_pendulum_rl.py` for episodic/offline calibration. This is separate from runtime online self-fitting.

### Inputs
- Required: `--calibration_json <path>`
- Optional: `--parameter_json <path>`
  - if omitted, initialization uses `calibration_json` + `BridgeConfig` defaults.
- Training logs: `--csv <path>` and/or `--csv_dir <dir>`

### Delay policy (default)
- Delay is pre-estimated per trajectory from `/cmd/u` vs `/hw/pwm_applied`.
- Replay always uses aligned input `u_sim(t) = u_cmd(t - delay_sec_est)`.
- RL default action space **does not include** delay.
- Optional advanced mode: `--learn_delay` to jointly optimize delay.
- Optional robustness jitter: `--delay_jitter_ms` (small randomization around pre-estimated delay).

### RL state/action/reward design
- Action: normalized parameter delta update for
  `[l_com, J_cm_base, b_eq, tau_eq, k_t, i0, R, k_e]` (+`delay_sec` only with `--learn_delay`).
- State: normalized parameter vector + compact replay-error features (`RMSE theta/omega/alpha`, bias, peak mismatch, delay-quality, current loss).
- Reward: improvement in weighted replay loss minus action regularization.

### Deterministic pre-fit
- Enabled by default (`--prefitON`).
- Performs local random/coordinate style search before PPO-style updates.
- Saves `prefit_params.json`.

### PPO-style CLI
Supports both argparse and interactive prompt mode:
- `-n/--num_episodes` (default `1000`)
- `--renderON/--renderOFF`
- `-g/--gamma` (default `0.995`)
- `-l/--lam` (default `0.98`)
- `-k/--kl_targ` (default `0.003`)
- `-b/--batch_size` (default `20`)
- plus calibration options (`--prefitON`, `--learn_delay`, `--delay_override`, `--domain_randomizationON/OFF`, `--seed`, ...)

### Domain randomization
Optional (`--domain_randomizationON` by default), with bounded perturbations around calibrated center values and delay jitter for robustness.

### Outputs
`train_pendulum_rl.py` writes:
- `initial_params.json`
- `prefit_params.json` (if enabled)
- `final_params_rl.json`
- `training_metadata.json` (settings, files, reward/randomization config, best score, per-trajectory delays)
- convergence/diagnostic plots:
  - episode reward
  - weighted loss (train/val)
  - RMSE theta/omega/alpha (train/val)
  - parameter convergence
  - delay per trajectory + histogram
  - representative overlays (sim vs real)

### Replay CSV export compatibility
Use `replay_pendulum_export.py` to export replay simulations with the **same CSV column names and order** as runtime `chrono_pendulum.py`, so outputs can be opened directly in `plot_pendulum.py`.
