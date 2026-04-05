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

## Sampling diagnostics
Computed from `wall_elapsed` timestamp differences:
- mean/std/min/max dt
- mean frequency

Saved in `chrono_run_*.meta.json` under `sampling_diagnostics` and printed in plotting/replay CLI summaries.
