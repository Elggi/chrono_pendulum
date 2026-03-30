# Chrono Pendulum (1-DOF Sim2Real Digital Twin)

This repository uses **Project Chrono 9.0.1** + ROS2 to run a 1-DOF pendulum digital twin.

## Adopted surrogate model

We use a geometry-fixed inertia surrogate:

\[
J\,\ddot{\theta} = K_u u - b\,\omega - \tau_{eq}\tanh\left(\frac{\omega}{\epsilon}\right) - m_{total} g l_{cg} \sin\theta
\]

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

\[
J = J_{rod} + J_{imu}
\]
\[
J_{rod} = \frac{1}{3} m_{rod}L^2, \quad J_{imu}=m_{imu}r_{imu}^2
\]

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
