# Chrono Pendulum Digital Twin (1-DOF Rotary)

This repository is a **physics-grounded sim-to-real framework** for a 1-DOF rotary pendulum.

## Core principles

- **Project Chrono** is the source of rigid-body truth: geometry, density-driven mass/inertia, gravity, contact, and joints.
- Actuation enters through **`ChLinkMotorRotationTorque`**.
- ML identifies what physics does not capture well: actuator mapping, delay/deadzone, friction mismatch, and residual dynamics.
- RL is used for **calibration parameter tuning**, not end-to-end controller replacement.

## Critical data interpretation rule

### Free-decay data
- Zero/near-zero input trajectories.
- Used for passive nominal state evolution.
- Trains `models/nominal/model.pt`.

### Excitation data
- Nonzero input trajectories (PWM/current/chirp/step).
- Used for actuator and residual identification.
- Trains `models/actuator/actuator_a2.pt` and sparse/regression artifacts.

If logs are mixed, use segmentation to extract input-off segments before nominal training.

## Pipelines

### Pipeline N (nominal passive)
- Module: `src/identification/nominal_free_decay/train.py`
- Learns `(theta, omega, dt) -> (theta_next, omega_next)` from free-decay only.
- Saves:
  - `models/nominal/model.pt`
  - `config_snapshot.json`
  - `normalization.json`
  - `feature_schema.json`
  - `split_metadata.json`

### Pipeline A-2 (neural actuator/residual)
- Module: `src/identification/actuator_a2/train.py`
- Learns effective torque from excitation data.
- Saves `models/actuator/actuator_a2.pt` plus reproducibility artifacts.

### Pipeline B (interpretable)
- Regression actuator fit: `src/identification/regression/fit_actuator_regression.py`
- Sparse residual discovery (SINDy-style STLSQ): `src/identification/sparse/fit_sindy.py`
- Sparse equations saved in JSON under `models/sparse/`.

### Pipeline C-1 (RL calibration)
- Environment: `src/calibration_rl/env.py`
- Trainer: `src/calibration_rl/train_ppo.py`
- PPO agent tunes calibration parameters for rollout agreement.
- Saves policy to `models/rl/ppo_calibrator.zip`.

## Directory structure

```text
configs/
data/raw data/processed data/splits
models/nominal models/actuator models/sparse models/rl
src/chrono_core/
src/ros_io/
src/preprocessing/
src/identification/nominal_free_decay/
src/identification/actuator_a2/
src/identification/regression/
src/identification/sparse/
src/calibration_rl/
src/visualization/
src/cli/
tests/
docs/
```

## Developer workflow

1. Add free-decay logs to `data/raw/free_decay/` and convert using `src/ros_io/ingest.py`.
2. Add excitation logs to `data/raw/excitation/` and convert similarly.
3. Segment mixed logs with `src/preprocessing/segmentation.py`.
4. Train nominal model:
   - `python -m src.cli.pipeline train-nominal --data data/processed/nominal_train.csv`
5. Train actuator A-2 model:
   - `python -m src.cli.pipeline train-actuator-a2 --data data/processed/excitation_train.csv`
6. Fit regression actuator law:
   - `python -m src.cli.pipeline fit-regression --data data/processed/excitation_train.csv`
7. Fit sparse residual equation:
   - `python -m src.cli.pipeline fit-sparse --data data/processed/residual_train.csv`
8. Run RL calibration:
   - `python -m src.cli.pipeline train-rl-calibrator --data data/processed/calibration_rollout.csv`
9. Validate via one-step and rollout metrics + overlays in `src/visualization/validation.py` and `src/visualization/imu_viewer.py`.

## Reproducibility

Every training pipeline must persist:
- model weights,
- config snapshot,
- normalization/scaler parameters,
- feature schema,
- trajectory-based split metadata.

## Dependencies

- Python 3.10+
- `pychrono`
- `numpy`, `pandas`, `torch`
- `gymnasium`, `stable-baselines3`
- `matplotlib`
- `pytest`
