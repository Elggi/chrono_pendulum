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

## Offline RL replay calibration pipeline (new)

This repository now supports a separate **offline/episodic replay-based calibration** workflow for model parameters.

### Runtime vs offline roles
- `chrono_pendulum.py`: online simulation + online self-fitting during live operation.
- `train_pendulum_rl.py`: offline replay calibration over logged CSV trajectories.
- `replay_pendulum_export.py`: offline replay/export using calibrated parameters with **runtime-compatible CSV schema**.

### Inputs
`train_pendulum_rl.py` supports:
- required: `--calibration_json <path>`
- optional: `--parameter_json <path>`
- training data: `--csv <path>` and/or `--csv_dir <dir>`

If `parameter.json` is absent, initial parameters are built from calibration JSON + config defaults.

### Calibrated parameter vector
Default calibrated vector:
- `[l_com, J_cm_base, b_eq, tau_eq, k_t, i0, R, k_e]`

By default, **delay is not in RL action space**.

Advanced mode:
- `--learn_delay` enables joint learning of `delay_sec`.

### Delay handling policy (default)
For each trajectory:
1. estimate delay from `/cmd/u` and `/hw/pwm_applied`
2. fix that delay during replay/alignment for training and evaluation
3. replay with aligned input: `u_sim(t) = u_cmd(t - delay_sec_est)`

Additional options:
- `--delay_override <sec>`
- `--delay_jitter_ms <float>` for small robustness randomization around pre-estimated delay

### RL environment design
- action: parameter delta (not motor command)
- state: normalized parameter vector + compact replay-error features
- reward: improvement minus weighted loss minus action penalty

Tracked mismatch terms include:
- RMSE theta / omega / alpha
- mean bias theta / omega
- aligned PWM mismatch
- optional soft current/power mismatches (robust to INA noise or missing data)

### Deterministic pre-fit
Optional `--prefitON` stage performs coordinate/local search before RL updates.

Outputs:
- `initial_params.json`
- `prefit_params.json` (if enabled)
- `final_params.json`
- `metadata.json` (settings, dataset split, reward weights, randomization, best validation score, delay estimates)

### PPO-style CLI and interactive prompts
The trainer supports PPO-style options:
- `-n/--num_episodes` (default 1000)
- `-g/--gamma` (default 0.995)
- `-l/--lam` (default 0.98)
- `-k/--kl_targ` (default 0.003)
- `-b/--batch_size` (default 20)
- `--renderON/--renderOFF`

If these are not supplied (and `--non_interactive` is not set), the script prompts in terminal and uses defaults on empty Enter.

### Domain randomization
Configurable randomization around calibration-centered values is supported (not arbitrary global ranges), including:
- initial state perturbations
- small delay jitter
- friction/parameter perturbation scale

Toggle using:
- `--domain_randomizationON`
- `--domain_randomizationOFF`

### Train/validation/test split
- Multiple CSV files: split by file.
- Single CSV file: split by reproducible time chunks/windows.

### Plot outputs
Training utilities generate:
- weighted total loss vs episode (train/validation)
- RMSE theta/omega/alpha vs episode (train/validation)
- parameter convergence
- delay diagnostics (per-trajectory and histogram)
- command alignment diagnostic plot
- sim-vs-real overlay

### Export CSV compatibility
`replay_pendulum_export.py` writes replay/evaluation CSVs in the **same column order/schema** as runtime logs from `chrono_pendulum.py`, so files can be loaded directly by `plot_pendulum.py`.
