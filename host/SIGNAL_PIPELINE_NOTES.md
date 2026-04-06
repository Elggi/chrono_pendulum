# Signal Pipeline Notes (Raw / Online / Offline)

## Raw signals (preserved in CSV)
- `theta_imu`, `omega_imu`, `alpha_imu`
- `theta_encoder`, `omega_encoder`, `alpha_encoder`
- `alpha_linear`
- `ina_current_raw_mA`, `ina_bus_voltage_v`, `ina_power_mw`
- `pwm_hw` (`/hw/pwm_applied`)

## Runtime signal policy
- Canonical runtime channels are explicit provenance names:
  - `theta_imu`, `theta_imu_online`, `theta_encoder`, `theta_encoder_online`
  - `omega_imu`, `omega_imu_online`, `omega_encoder`, `omega_encoder_online`
  - `alpha_imu`, `alpha_imu_online`, `alpha_linear`, `alpha_linear_online`, `alpha_encoder`, `alpha_encoder_online`
- `theta_real`, `omega_real`, `alpha_real` are exported aliases for offline identification compatibility.

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

## Plotting
- `plot_pendulum.py` is visualization-only and consumes canonical runtime columns directly.
- Runtime now also exports `chrono_run_*.finalized.csv` containing offline-ID training signals only:
  - `I_filtered_mA`
  - `theta_imu_filtered_unwrapped`
  - `omega_imu_filtered`
  - `alpha_from_linear_accel_filtered`

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
  - `alpha_linear_offset = median(alpha_linear_warmup)` for pre-actuation baseline alignment
  - `current_offset_mA = median(current | |pwm|<thr and |omega|<thr)` with fallback 26 mA
- Only after warmup does valid elapsed time start and corrected signals get logged.
- At warmup→run transition, unwrap accumulators are seeded to the warmup offset so first run sample is zero-start aligned.
