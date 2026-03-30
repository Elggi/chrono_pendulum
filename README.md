# Chrono Pendulum (1-DOF Sim2Real Digital Twin)

This repository uses **Project Chrono 9.0.1** + ROS2 to run a 1-DOF pendulum digital twin.

## Introduction: Research Goal

The main goal is to build a **Sim2Real digital twin** that can:
- run a physically grounded pendulum model in Chrono,
- reconstruct real states from hardware IMU/encoder streams,
- identify/update compact surrogate dynamics parameters, and
- validate/optimize tracking quality with replay + PPO-based tuning.

Practically, this supports a repeatable workflow from data collection → calibration → policy/parameter optimization → real-time runtime comparison.

## Framework Flow

1. **Data acquisition / runtime**
   - `chrono_pendulum.py` runs simulation and logs synchronized sim/real channels.
   - ROS2 bridges commands and sensor streams.
2. **Calibration**
   - `calibration.py` provides geometric and sensor-derived constants (e.g., CPR/radius).
3. **Model parameter optimization**
   - `train_pendulum_rl.py` replays logs and tunes surrogate parameters.
4. **Validation and visualization**
   - `plot_pendulum.py` compares `theta/omega/alpha` between sim and real.
   - `replay_pendulum_cli.py` validates best parameters on replay data.
5. **Operational launcher**
   - `pendulum_stack.sh` provides menu-based orchestration of the whole pipeline.

## Recommended Versions

- Python: **3.10+** (ROS2 Humble-friendly)
- ROS2: **Humble**
- Project Chrono: **9.0.1**
- NumPy / Matplotlib / Pandas: recent stable releases compatible with Python 3.10+

> Note: Match Chrono and ROS2 environments first; most runtime issues come from version or ABI mismatches there.

## Adopted surrogate model

We use a geometry-fixed inertia surrogate:

<img width="463" height="48" alt="Screenshot from 2026-03-31 04-09-14" src="https://github.com/user-attachments/assets/63f41e68-6bc8-45f5-af6a-cda97c200834" />


- Control input: `u` (same command channel used in runtime)
- States: `theta`, `omega`, `alpha=theta_ddot`
- Identified parameters (only 4):
  1. `K_u` (motor gain)
  2. `b` (`b_eq`, viscous damping)
  3. `tau_eq` (Coulomb friction magnitude)
  4. `l_cg` (`l_com`, COM for gravity term)
- `eps` is configurable (`tanh_eps`) and **not identified by default**.

## Geometry-based fixed inertia

Inertia is computed once from geometry/calibration and kept fixed during identification:

<img width="149" height="33" alt="Screenshot from 2026-03-31 01-59-01" src="https://github.com/user-attachments/assets/65024e42-f0b8-40aa-8571-85f378dc06f5" />


<img width="159" height="62" alt="Screenshot from 2026-03-31 01-58-21" src="https://github.com/user-attachments/assets/27d31549-71d4-4d32-8eb8-2568cbca4f1d" />  <img width="156" height="34" alt="Screenshot from 2026-03-31 01-59-45" src="https://github.com/user-attachments/assets/a29fafeb-65f2-4da1-8081-e59978389ae3" />

Constants/config:
- `rod_mass = 0.2 kg`
- `rod_length = 0.285 m`
- `imu_mass = 0.02 kg`
- `r_imu` from calibration (`calibration.py` output) or config default
- `gravity`, `eps` in config

### Why fixed-geometry `J`?

- Avoids coupling between `J` and `l_cg` (identifiability issue when both vary together).
- Keeps `l_cg` physically interpretable in only the gravity torque term.
- Reduces optimizer/RL search dimension and improves parameter robustness.

## Real-state derivation

Runtime logs `theta_real`, `omega_real`, `alpha_real` from IMU-based reconstruction:

- `theta_real`: quaternion -> relative orientation -> pendulum tip direction -> planar angle
- `omega_real`: IMU `wz`
- `alpha_real` options (`--real-alpha-source`):
  - `omega_diff`: derivative of `omega_real`
  - `tangential_accel`: tangential acceleration / radius
  - `blend`: weighted mixture of both

## Config parameters required by this model

`BridgeConfig` includes:
- `rod_mass`
- `rod_length`
- `imu_mass`
- `r_imu`
- `tanh_eps` (eps)
- `gravity`

## Notes

- Electrical voltage/current/power motor modeling was removed from the surrogate dynamics path.
- `pendulum_stack.sh` monitoring can stay for Jetson runtime observability.

## PPO Algorithm: how it’s built

PPO training is implemented in the replay optimization path (`train_pendulum_rl.py`) with a custom pendulum replay environment (`chrono_core/pendulum_rl_env.py`).

High-level structure:
- **Environment objective**: minimize mismatch between simulated and measured trajectories (theta/omega/alpha, plus optional smoothness penalties).
<img width="987" height="141" alt="image" src="https://github.com/user-attachments/assets/eb1c06ff-0762-4e91-b76f-7dc785323409" />
- **Action meaning**: update candidate surrogate parameters (e.g., `l_com`, `b_eq`, `tau_eq`, `K_u`) within bounded ranges.
<img width="252" height="32" alt="image" src="https://github.com/user-attachments/assets/b21ad714-62b7-42b8-9e2d-7d33215b101d" />
<img width="330" height="91" alt="image" src="https://github.com/user-attachments/assets/5a43782a-8e95-4c62-93df-d3f76c61b8bc" />

- **Episode loop**:
  1. sample/init parameters,
  2. roll out replay dynamics over logged command inputs,
  3. compute reward from tracking error/cost,
  4. update policy/value networks with PPO clipped objective.
- **Outputs**:
  - best parameter JSON,
  - replay CSV for direct comparison plots,
  - optional dashboard/monitor artifacts.

This keeps online runtime lightweight while using offline RL to search robust parameter sets against real logs.
