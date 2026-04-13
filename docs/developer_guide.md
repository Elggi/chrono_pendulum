# Developer Guide

## Data onboarding

### Add free-decay datasets
1. Place source logs in `data/raw/free_decay/`.
2. Convert to canonical schema (`t, theta, omega, u, current`) using `src/ros_io/ingest.py`.
3. Label as `segment_type=free_decay` before training nominal model.

### Add excitation datasets
1. Place source logs in `data/raw/excitation/`.
2. Convert to canonical schema.
3. Label as `segment_type=excitation`.

### Segment mixed logs
Use `src/preprocessing/segmentation.py` to split near-zero-input sections for nominal training and forced sections for actuator/residual identification.

## Training

- Nominal free-decay: `python -m src.cli.pipeline train-nominal --data <csv>`
- Actuator A-2: `python -m src.cli.pipeline train-actuator-a2 --data <csv>`
- Regression actuator: `python -m src.cli.pipeline fit-regression --data <csv>`
- Sparse residual: `python -m src.cli.pipeline fit-sparse --data <csv>`
- RL calibration: `python -m src.cli.pipeline train-rl-calibrator --data <csv>`

## Validation

- One-step and rollout metrics: `src/visualization/validation.py`
- Free-decay/forced-response overlay plots: `src/visualization/imu_viewer.py`
- Torque comparison: compare measured `tau_eff` and model-generated torque residuals.
