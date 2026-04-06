# Signal Pipeline Notes (Raw / Online / Offline)

## Raw signals (preserved in CSV)
- `theta_imu`, `omega_imu`, `alpha_imu`
- `theta_encoder`, `omega_encoder`, `alpha_encoder`
- `alpha_linear`
- `ina_current_raw_mA`, `ina_bus_voltage_v`, `ina_power_mw`
- `pwm_hw` (`/hw/pwm_applied`)

## Current processing
- `ina_current_corr_mA = ina_current_raw_mA - ina_current_offset_mA`
- `ina_current_signed_mA = sign(pwm_hw) * ina_current_corr_mA`

Sign source is **always** hardware PWM (`/hw/pwm_applied`), never `/cmd/u`.

## Online filter (runtime-safe)
Exponential smoothing is applied in `chrono_pendulum.py` and logged as:
- `*_online` columns (theta/omega/alpha/current)

## Offline filter (postprocessing)
`plot_pendulum.py` and replay tools compute offline smooth overlays
(Savitzky-Golay if available; moving-average fallback).

## Winner selection artifacts
- `plot_pendulum.py` exports:
  - `*.theta_candidates.png`
  - `*.omega_candidates.png`
  - `*.alpha_candidates.png`
  - `*.current_audit.png`
  - `*.signal_candidate_metrics.csv`
  - `*.signal_winner_summary.json`

## Sampling diagnostics
Computed from `wall_elapsed` timestamp differences:
- mean/std/min/max dt
- mean frequency

Saved in `chrono_run_*.meta.json` under `sampling_diagnostics` and printed in plotting/replay CLI summaries.

## Mandatory warmup initialization
- Runtime enforces `STATE_INIT -> STATE_WARMUP(1.0s) -> STATE_RUN`.
- During warmup, no training-valid rows are logged.
- Warmup computes:
  - `theta_offset_rad = median(unwrap(theta_warmup))`
  - `current_offset_mA = median(current | |pwm|<thr and |omega|<thr)` with fallback 26 mA
- Only after warmup does valid elapsed time start and corrected signals get logged.
