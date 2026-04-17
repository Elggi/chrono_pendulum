# Host Architecture (Frontend / Backend Split)

## Frontend (user-editable control surface)
Located directly under `host/`:
- `chrono_pendulum.py`, `calibration.py`, `replay_pendulum_cli.py`, `plot_pendulum.py`, `pendulum_stack.sh`
- `model_parameter.latest.json`, `model_parameter.template.json`
- `stage2_settings.py` (**single editable place for Stage2 candidate library + target torque definition**)

## Backend (stage-specific implementation)
Located under `host/backend/`:
- `stage1/` : CMA-ES identification runtime
- `stage2/` : SINDy dataset + solver entrypoints
- `stage3/` : PPO/RL optimization and offline benchmark

This layout keeps user-facing configuration and operating scripts at top-level `host/`, while stage execution logic lives under `host/backend/stageN`.
