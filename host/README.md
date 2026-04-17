# Host folder guide

This folder contains host-side runtime, calibration, replay, plotting, and RL optimization tools for the Chrono pendulum workflow.

## Top-level scripts

- `chrono_pendulum.py`  
  Main real-time Chrono runtime. Runs sim, reads ROS sensor streams, estimates runtime channels (`theta_imu/encoder`, `omega_imu/encoder`, `alpha_imu/linear/encoder` with `*_online` filters), exports offline-ID aliases (`theta_real`, `omega_real`, `alpha_real`), publishes sim topics, and writes synchronized logs.

- `pendulum_stack.sh`  
  Interactive launcher menu for common workflows (viewer, calibration, RL fitting, runtime, plotting, replay validation).

- `imu_viewer.py`  
  Live IMU orientation + tip trajectory viewer (with encoder counters/CPR helper display).

- `calibration.py`  
  Calibration utility for encoder CPR and geometry/radius measurements; writes calibration JSON artifacts used by runtime and training.

- `plot_pendulum.py`  
  Unified plotting dashboard for simulation/real trajectory comparison and error visualization.

- `backend/stage3/train_pendulum_rl.py`  
  Offline replay-based PPO optimization for surrogate model parameters.

- `backend/stage3/offline_id_pem_sindy_ppo.py`
  Three-stage offline identification benchmark:
  1) PEM nominal fit with irregular-sampling RK45 (plus uniform-resampled RK4 baseline),
  2) residual-only SINDy-PI discovery,
  3) PPO parameter proposal optimization (rollout-level reward, **not** control).
  Writes reproducible artifacts under `reports/PEM_SINDy_PPO/`.

- `replay_pendulum_cli.py`  
  CLI replay runner that re-simulates logged command streams with chosen parameter/calibration JSON.

- `replay_pendulum_viewer.py`  
  Visualization-oriented replay script for inspecting playback behavior.

- `replay_pendulum_export.py`  
  Export helper for replay outputs/artifacts used by downstream analysis.

## `chrono_core/` modules

- `config.py`  
  Central configuration dataclass (`BridgeConfig`) and defaults for runtime/training parameters.

- `dynamics.py`  
  Core pendulum dynamics and torque decomposition used by runtime/replay.

- `pendulum_rl_env.py`  
  RL environment logic (state/action/reward, replay integration, parameter bounds) for PPO training.

- `pendulum_rl_plots.py`  
  Plot/report utilities for RL results and training history visualization.

- `signal_filter.py`  
  Signal preprocessing/filter helpers used by estimation and analysis paths.

- `calibration_io.py`  
  Calibration JSON loading/application helpers (e.g., CPR/radius extraction).

- `log_schema.py`  
  CSV log column schema constants to keep runtime and analysis aligned.

- `utils.py`  
  Shared utility functions (time helpers, clamping, terminal status formatting, numeric sanitizers, path helpers).

- `__init__.py`  
  Package initializer for `chrono_core`.

## Data/output directories

- `run_logs/`  
  Runtime and calibration outputs (CSV + metadata JSON). Includes the latest calibration snapshots and run logs.


## Stage-wise GRU trajectory fitting

- `staged_pendulum_calibration.py`
  Stage-wise discrete-time trajectory fitting pipeline (Stage 1: sin, Stage 2: square, Stage 3: burst).
  Uses a **PyTorch GRU black-box dynamics learner** with real-data-only policy:
  - input source: `hw_pwm`
  - state source: `theta_real`, `omega_real`
  - target: `theta_next`, `omega_next`
  - optional one-step + rollout trajectory loss

  Example:

  ```bash
  python host/staged_pendulum_calibration.py --mode full     --stage1_csv host/run_logs/sin.csv     --stage2_csv host/run_logs/square.csv     --stage3_csv host/run_logs/burst.csv     --save_checkpoint
  ```

  Outputs include:
  - `trajectory_fit_summary.json` (pipeline summary)
  - `trajectory_model_params.json` (Chrono `--parameter-json` compatible wrapper + GRU checkpoint reference)
  - `stageN/stageN_loss_history.csv` and `stageN/stageN_loss_convergence.png` (epoch-wise loss logs/plot)
