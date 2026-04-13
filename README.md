# Chrono Pendulum Digital Twin Framework

## Vision

This repository is intended to evolve into a **physics-grounded sim-to-real digital twin framework** for a 1-DOF rotary pendulum built on **Project Chrono**. The goal is not to replace physics with a black-box model, but to combine:

1. **Chrono's inherent rigid-body solver** for geometry, mass, inertia, gravity, contact, and joint dynamics,
2. **`ChLinkMotorRotationTorque`** as the actuator interface,
3. **system identification pipelines** that estimate what Chrono does not already know precisely,
4. **ROS-based data collection** from the real hardware, and
5. **reinforcement learning fine-tuning** for final sim-to-real calibration.

The core design philosophy is:

- **Chrono should solve the rigid-body dynamics.**
- **Machine learning should identify unknown actuator behavior, friction mismatch, residual dynamics, and calibration parameters.**
- **The final simulator should remain interpretable, modular, and physically editable.**

---

## 1. System Overview

The physical system is a 1-DOF pendulum with:

- a rod,
- an IMU mass mounted on the rod,
- a connector or hub geometry,
- a motorized revolute actuation interface,
- real sensor logging through ROS,
- a Chrono simulation model with matching geometry.

The final framework should support the full loop:

```text
Real Hardware
  -> ROS logging
  -> curated datasets
  -> system identification / actuator learning / calibration
  -> Chrono digital twin update
  -> validation against real trajectories
  -> RL fine-tuning if needed
  -> improved simulator for replay, prediction, and control design
```

---

## 2. Physics and Modeling Philosophy

### 2.1 What Chrono should handle directly

The Chrono model should explicitly represent the pendulum using **EasyBody-style rigid bodies** whose:

- geometry,
- density,
- material,
- collision/contact settings,
- center of mass,
- mass,
- inertia

are derived from the body definition itself whenever possible.

This means the rod, IMU housing, and connector geometry should be modeled as actual bodies, not abstract point-mass placeholders unless simplification is explicitly required.

### 2.2 Gravity should not be reimplemented in actuator torque

If the pendulum body is built correctly in Chrono, then the gravity-induced restoring effect is already produced by the multibody solver. Therefore, the usual pendulum term

\[
\tau_g(\theta) = m g l_{\mathrm{com}} \sin\theta
\]

should **not** be manually added inside the motor torque model if the body is already represented as a rigid body under gravity.

Instead:

- gravity comes from Chrono,
- inertia comes from Chrono,
- geometric COM effects come from Chrono,
- contact effects come from Chrono,
- actuator torque enters through `ChLinkMotorRotationTorque`.

### 2.3 Why inertia may be unknown at the beginning

A key advantage of this framework is that Chrono does not require the user to begin with a manually derived scalar pendulum inertia model. If the geometry and density are defined consistently, the solver already has a physically meaningful inertia tensor.

This does **not** mean the model is perfect. It means the initial model is physically structured, and system identification can then calibrate:

- density corrections,
- actuator gain,
- friction mismatch,
- sensor alignment,
- residual dynamics.

---

## 3. Machine Learning Terminology for This Project

To avoid ambiguity, the following terms are used consistently.

### State

The simulator or learned model state describes the pendulum configuration and motion. A minimal state is

\[
x_t = [\theta_t, \omega_t]
\]

where:

- \(\theta_t\): pendulum angle,
- \(\omega_t\): angular velocity.

Optional state augmentation may include:

- filtered angular acceleration \(\alpha_t\),
- previous torques,
- sensor bias states,
- delay states,
- hidden recurrent memory.

### Input

An **input** is an externally measured or commanded quantity available from hardware logs, for example:

- PWM command,
- motor current,
- estimated current-derived torque request,
- explicit torque command.

We denote this as

\[
u_t
\]

### Action

In reinforcement learning, the **action** is the calibration adjustment chosen by the RL agent. This is **not** the same as motor input. Examples:

\[
a_t = [\Delta K_u, \Delta b, \Delta \tau_c, \Delta \rho_{\mathrm{rod}}, \ldots]
\]

So:

- **input** refers to real actuator excitation from hardware,
- **action** refers to RL-controlled simulator parameter updates.

### Output

The **output** is what the learned model predicts. Typical outputs are:

- next state \(x_{t+1}\),
- state increment \(\Delta x_t\),
- effective torque \(\tau_t\),
- residual torque \(\tau_{\mathrm{res},t}\).

### Features

**Features** are the variables actually fed into a regression or neural network. For example:

\[
\phi_t = [\theta_t, \omega_t, u_t, \Delta t, u_{t-1}, \omega_{t-1}]
\]

Features can be raw, filtered, delayed, windowed, or normalized.

---

## 4. Why Free-Decay and Excitation Data Must Be Distinguished

This distinction is central to the entire framework.

### 4.1 Free-decay data

**Free-decay data** means trajectories recorded with essentially no external actuation:

\[
u_t \approx 0
\]

The pendulum moves only due to:

- gravity,
- inertia,
- passive damping,
- friction,
- small unmodeled passive effects.

This data isolates the **passive plant dynamics**.

Typical information contained in free-decay trajectories:

- gravitational restoring behavior,
- passive damping,
- Coulomb-like friction,
- effective inertia seen in motion,
- natural trajectory evolution from initial conditions.

### 4.2 Excitation data

**Excitation data** means trajectories where external input is actively applied:

\[
u_t \neq 0
\]

Examples:

- PWM injection,
- current commands,
- chirps,
- steps,
- PRBS,
- user actuation sweeps.

This data contains both:

1. plant dynamics, and
2. actuator/input-path dynamics.

Typical information contained in excitation trajectories:

- effective torque generation,
- gain mismatch,
- deadzone,
- saturation,
- motor lag,
- backlash/compliance effects,
- transmission asymmetry,
- forced residual dynamics.

### 4.3 Critical project rule

If `model.pt` is intended to represent the **nominal plant dynamics model**, then it should be trained using **free-decay data only**, or at least **zero-input segments extracted from mixed logs**.

That is,

> **`model.pt` should denote the nominal zero-input dynamics model unless explicitly stated otherwise.**

This separation is important because if excitation data is used directly to train the nominal model, the learned model may entangle:

- rigid-body pendulum physics,
- actuator nonlinearities,
- friction mismatch,
- command-to-torque conversion,
- delay and deadzone.

That makes the digital twin harder to interpret and harder to couple cleanly to `ChLinkMotorRotationTorque`.

### 4.4 Recommended interpretation

- **Free-decay data** -> identify the passive nominal plant.
- **Excitation data** -> identify actuator mapping and forced residual terms.

In other words:

\[
\texttt{model.pt} = f_{\mathrm{nominal}}(x_t)
\]

should come from free-decay data, while actuator identification should use excitation data to build:

\[
\tau_{\mathrm{eff}} = g(u_t, x_t, \text{history})
\]

or residual terms such as

\[
\tau_{\mathrm{res}} = r(u_t, x_t, \text{history})
\]

---

## 5. Governing Conceptual Dynamics

The physical pendulum with actuation can be viewed conceptually as

\[
M(q)\ddot q + C(q, \dot q) + G(q) + F_{\mathrm{passive}}(q, \dot q) = \tau_{\mathrm{act}} + \tau_{\mathrm{res}}
\]

For this project, Chrono already handles the rigid-body side:

- \(M(q)\),
- geometric effects inside \(C(q, \dot q)\),
- gravity \(G(q)\),
- contact-related effects if enabled.

So the practical digital twin decomposition becomes:

\[
\tau_{\mathrm{cmd}} = \tau_{\mathrm{nominal\_act}} + \tau_{\mathrm{residual}}
\]

where:

- `Chrono solver` handles the mechanical evolution,
- `ChLinkMotorRotationTorque` injects \(\tau_{\mathrm{cmd}}\),
- identification methods estimate what that torque should be.

---

## 6. System Identification Pipelines

This framework supports several complementary pipelines.

### Pipeline N: Nominal Plant Model from Free-Decay

### Purpose
Learn the passive pendulum dynamics from zero-input motion.

### Data
- free-decay runs,
- or zero-input windows extracted from longer logs.

### Model target
Predict either:

\[
x_{t+1} = f_{\mathrm{nominal}}(x_t)
\]

or

\[
\Delta x_t = f_{\mathrm{nominal}}(x_t, \Delta t)
\]

### Recommended meaning of `model.pt`
This repository should use `model.pt` to mean:

> **the nominal passive dynamics model trained from free-decay data**

unless a different file naming convention is introduced.

### Candidate model classes
- LSTM,
- NNARX,
- TDNN,
- small MLP with time-window features,
- physics-informed residual network.

### Why this model matters
It gives a clean baseline for:

- passive replay,
- trajectory prior generation,
- residual decomposition,
- simulator validation.

---

## 7. Actuation Pipeline A-2: Neural Actuator / Residual Model

### Purpose
Learn the actuation-side mismatch using excitation data.

This is the preferred development interpretation for **Actuation Pipeline A = A-2**.

### Data
- current-injected runs,
- PWM excitation runs,
- chirp/step input runs,
- forced motion logs.

### Objective
Map command-side measurements to effective torque behavior:

\[
\tau_{\mathrm{eff},t} = g_\theta(u_t, x_t, h_t)
\]

where \(h_t\) may include history terms.

Possible forms:

1. **Direct effective torque model**

\[
\tau_{\mathrm{eff},t} = g_\theta(\theta_t, \omega_t, u_t, u_{t-1}, \omega_{t-1}, \ldots)
\]

2. **Nominal + neural residual**

\[
\tau_{\mathrm{eff},t} = \tau_{\mathrm{nominal},t} + \tau_{\mathrm{NN\_res},t}
\]

with

\[
\tau_{\mathrm{NN\_res},t} = r_\phi(\theta_t, \omega_t, u_t, \text{history})
\]

### Why A-2 is preferred
It preserves the Chrono structure while allowing actuator mismatch to be learned where physics is uncertain.

### Recommended deployment
At runtime, the torque passed to `ChLinkMotorRotationTorque` should be:

\[
\tau_{\mathrm{cmd},t} = \tau_{\mathrm{act\_nominal},t} + \tau_{\mathrm{act\_residual},t}
\]

---

## 8. Pipeline B: Sparse / Regression-Based Identification

Pipeline B is intended for interpretable equation discovery.

### B-1. Classical regression for nominal actuator law

Use excitation data to fit a low-order interpretable law such as

\[
\tau_{\mathrm{act}} \approx K_u u - b_\omega \omega - \tau_c \tanh(\omega / \varepsilon)
\]

or a richer physically motivated variant.

This is useful when you want:

- explicit coefficients,
- a small interpretable equation,
- a baseline before neural residuals.

### B-2. SINDy / SINDy-PI residual discovery

Use sparse regression on residuals to identify missing physics.

Given measured state evolution and a nominal simulator or nominal model, define residual torque or residual acceleration:

\[
r_t = y_t - \hat y_t^{\mathrm{nominal}}
\]

Then identify sparse governing terms:

\[
r_t \approx \Theta(z_t)\Xi
\]

where \(\Theta(z_t)\) is a candidate function library built from variables such as:

- \(\theta\),
- \(\omega\),
- \(u\),
- \(u\omega\),
- \(\sin\theta\),
- \(|\omega|\),
- delayed inputs,
- filtered current,
- hysteresis proxies.

### Why SINDy matters here
SINDy gives:

- sparse interpretable residual equations,
- better physical insight than a black-box network,
- a bridge between classical ID and ML.

### When to use SINDy-PI
Use SINDy-PI when the target structure is implicit or rational and ordinary sparse explicit SINDy is too restrictive.

---

## 9. Pipeline C-1: RL for Parameter / Calibration Fine-Tuning

Pipeline C should be implemented as **C-1: RL for parameter calibration**, not end-to-end policy learning for the physical pendulum task itself.

### Goal
The RL agent should tune simulator parameters so that simulated rollouts match real trajectories more closely.

### Action space
The action should modify simulator calibration parameters, for example:

\[
a_t = [K_u,\ b_{eq},\ \tau_{eq},\ \rho_{rod},\ \rho_{imu},\ \text{sensor bias},\ \text{delay scale},\ \ldots]
\]

or parameter deltas around a nominal configuration.

### Environment
The RL environment should be a Chrono-based calibration environment where:

- an episode replays a real trajectory or control log,
- the simulator runs with candidate parameters,
- the reward measures sim-to-real agreement.

### Reward example
A practical reward is the negative weighted rollout mismatch:

\[
R = -\sum_t \Big(
 w_\theta |\theta_t^{\mathrm{sim}} - \theta_t^{\mathrm{real}}|^2
+ w_\omega |\omega_t^{\mathrm{sim}} - \omega_t^{\mathrm{real}}|^2
+ w_\alpha |\alpha_t^{\mathrm{sim}} - \alpha_t^{\mathrm{real}}|^2
+ w_\tau |\tau_t^{\mathrm{sim}} - \tau_t^{\mathrm{est}}|^2
\Big)
\]

Additional penalties can be used for:

- unstable simulations,
- physically implausible densities,
- unrealistic gains,
- excessive parameter drift.

### Why PPO is appropriate
PPO is a reasonable first choice because it is stable, widely supported, and already compatible with Chrono/Gym-style environments.

---

## 10. Recommended Full Data-to-Simulator Workflow

```text
[Hardware pendulum]
    |
    v
[ROS logging]
    |
    +--> free-decay dataset
    |       -> Nominal plant learning
    |       -> model.pt
    |
    +--> excitation dataset
            -> actuator mapping / residual identification
            -> regression / SINDy / NN residual

[Chrono simulator]
    |
    +--> rigid bodies from geometry + density
    +--> gravity/contact/joint dynamics from Chrono
    +--> ChLinkMotorRotationTorque receives learned torque command
    |
    v
[Validation: sim vs real]
    |
    v
[RL fine-tuning: Pipeline C-1]
    |
    v
[Calibrated digital twin]
```

---

## 11. Proposed Repository Architecture (Future Target)

This is a **vision architecture**, not a statement about the current repository contents.

```text
chrono_pendulum/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── hardware/
│   │   ├── sensor_calibration.yaml
│   │   └── actuator_limits.yaml
│   ├── simulation/
│   │   ├── pendulum_geometry.yaml
│   │   ├── materials.yaml
│   │   ├── solver.yaml
│   │   └── runtime_modes.yaml
│   └── training/
│       ├── nominal_model.yaml
│       ├── actuator_a2.yaml
│       ├── sindy.yaml
│       └── rl_calibration.yaml
│
├── data/
│   ├── raw/
│   │   ├── rosbags/
│   │   └── exported_csv/
│   ├── processed/
│   │   ├── free_decay/
│   │   ├── excitation/
│   │   └── replay_ready/
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── nominal/
│   │   └── model.pt
│   ├── actuator/
│   │   ├── actuator_a2.pt
│   │   └── actuator_regression.json
│   ├── sparse/
│   │   └── sindy_equations.json
│   └── rl/
│       └── ppo_calibrator.zip
│
├── src/
│   ├── chrono_core/
│   │   ├── system_builder.py
│   │   ├── bodies.py
│   │   ├── joints.py
│   │   ├── materials.py
│   │   ├── actuator_interface.py
│   │   ├── runtime_modes.py
│   │   └── replay.py
│   │
│   ├── ros_io/
│   │   ├── log_reader.py
│   │   ├── bag_export.py
│   │   ├── topic_schema.py
│   │   └── sync.py
│   │
│   ├── preprocessing/
│   │   ├── segmentation.py
│   │   ├── feature_builder.py
│   │   ├── filters.py
│   │   ├── normalization.py
│   │   └── dataset_builder.py
│   │
│   ├── identification/
│   │   ├── nominal_free_decay/
│   │   │   ├── train_nominal.py
│   │   │   ├── infer_nominal.py
│   │   │   └── models.py
│   │   ├── actuator_a2/
│   │   │   ├── train_actuator_nn.py
│   │   │   ├── infer_torque.py
│   │   │   └── models.py
│   │   ├── regression/
│   │   │   └── fit_nominal_torque.py
│   │   ├── sparse/
│   │   │   ├── fit_sindy.py
│   │   │   ├── fit_sindy_pi.py
│   │   │   └── library.py
│   │   └── evaluation/
│   │       ├── rollout_metrics.py
│   │       └── compare_models.py
│   │
│   ├── calibration_rl/
│   │   ├── env.py
│   │   ├── reward.py
│   │   ├── train_ppo.py
│   │   └── evaluate_ppo.py
│   │
│   ├── visualization/
│   │   ├── imu_viewer.py
│   │   ├── plot_sim_vs_real.py
│   │   ├── replay_runs.py
│   │   └── diagnostics.py
│   │
│   └── cli/
│       ├── collect.py
│       ├── build_dataset.py
│       ├── train_nominal.py
│       ├── train_actuator.py
│       ├── fit_sparse.py
│       ├── train_rl.py
│       └── validate.py
│
├── scripts/
│   ├── run_free_decay_collection.sh
│   ├── run_excitation_collection.sh
│   ├── train_nominal.sh
│   ├── train_actuator.sh
│   ├── fit_sindy.sh
│   └── train_rl_calibrator.sh
│
├── tests/
│   ├── test_chrono_build.py
│   ├── test_dataset_splits.py
│   ├── test_nominal_model.py
│   ├── test_actuator_model.py
│   ├── test_sindy_pipeline.py
│   └── test_rl_env.py
│
└── docs/
    ├── architecture.md
    ├── dataset_protocol.md
    ├── sim2real_validation.md
    └── equations.md
```

---

## 12. Runtime Modes

The framework should support at least these simulation modes.

### Mode 1. Pure physics baseline
Chrono only, with a hand-written nominal actuator law.

### Mode 2. Physics + regression actuator
Chrono + interpretable fitted torque equation.

### Mode 3. Physics + neural actuator residual (A-2)
Chrono + neural residual torque model.

### Mode 4. Physics + sparse residual
Chrono + SINDy residual equation.

### Mode 5. RL-calibrated simulation
Chrono + selected actuator model + PPO-tuned parameters.

---

## 13. Theory Background for Each Identification Method

### 13.1 Classical regression
Classical regression assumes the unknown behavior can be represented by a chosen functional form with a small number of coefficients.

Example:

\[
\tau = K_u u - b\omega - \tau_c \tanh(\omega/\varepsilon)
\]

Advantages:
- interpretable,
- fast,
- low data requirement.

Weakness:
- misses complex hysteresis and unmodeled effects.

### 13.2 NNARX / TDNN / LSTM nominal or actuator models
These methods learn temporal input-output relations from sequences.

General form:

\[
\hat y_t = f_\theta(y_{t-1}, y_{t-2}, \ldots, u_t, u_{t-1}, \ldots)
\]

- **NNARX**: explicit lagged inputs to feedforward net.
- **TDNN**: temporal windows, useful for local history effects.
- **LSTM**: recurrent memory for longer-term dependencies.

Advantages:
- captures delays and history,
- flexible for actuator hysteresis.

Weakness:
- less interpretable than sparse equations.

### 13.3 SINDy
SINDy assumes nonlinear dynamics are sparse in a candidate library.

\[
\dot x = \Theta(x, u)\Xi
\]

Advantages:
- interpretable,
- compact,
- physically inspectable.

Weakness:
- sensitive to feature design and derivative quality.

### 13.4 SINDy-PI
SINDy-PI extends SINDy to implicit or rational dynamics.

Useful when the true residual structure is not easy to write as an explicit sparse RHS equation.

### 13.5 KBINN / Kalman-informed approaches
These approaches combine neural networks with filtering ideas to estimate states and parameters under noisy measurements.

Useful if:
- noise is strong,
- direct differentiation is unreliable,
- uncertainty-aware state reconstruction is needed.

### 13.6 RL parameter calibration
RL is used here **not** to learn the pendulum control policy from scratch, but to tune simulator parameters by optimizing long-horizon rollout agreement.

Advantages:
- directly optimizes sim-to-real fit,
- naturally handles multi-step mismatch.

Weakness:
- computationally expensive,
- requires careful reward design,
- should be the final stage, not the first stage.

---

## 14. ROS-Based Data Collection Requirements

The real hardware logging pipeline should export synchronized time-series including as many of the following as possible:

- timestamp,
- encoder angle,
- encoder velocity estimate,
- filtered IMU angle,
- filtered angular velocity,
- filtered angular acceleration,
- current,
- PWM,
- commanded input,
- calibration metadata,
- experiment tag (`free_decay`, `excitation`, `validation`, etc.).

Each run should also record metadata such as:

- hardware mode,
- calibration version,
- filtering version,
- pendulum geometry version,
- battery state if relevant,
- notes on whether external actuation was present.

---

## 15. Reproducibility and Required Dependencies

A reproducible environment should include at minimum:

### Core simulation
- Project Chrono / PyChrono built for the user's Python environment,
- NumPy,
- SciPy,
- pandas,
- matplotlib.

### ML and training
- PyTorch,
- scikit-learn,
- PySINDy,
- stable-baselines3,
- Gymnasium or compatible RL environment tooling.

### Data and filtering
- rosbag / ROS2 bag tools as needed,
- PyYAML,
- joblib,
- possibly filterpy if EKF/UKF utilities are desired.

### Suggested optional tools
- tensorboard,
- wandb or another experiment tracker,
- jupyter,
- pytest,
- black / ruff / mypy for code quality.

### Reproducibility practices
- fixed random seeds,
- versioned configs,
- train/val/test splits by trajectory,
- saved scaler parameters,
- saved feature schemas,
- full metadata with every exported model.

---

## 16. Validation Philosophy

The digital twin should not be judged by single-step prediction only.

Validation should include:

1. **one-step error**,
2. **multi-step rollout error**,
3. **free-decay replay quality**,
4. **forced-response replay quality**,
5. **parameter plausibility**,
6. **generalization to unseen trajectories**.

Key metrics may include:

- RMSE in \(\theta\),
- RMSE in \(\omega\),
- rollout drift,
- phase portrait agreement,
- torque consistency,
- energy decay consistency.

---

## 17. Design Intent Summary

This project should be implemented according to the following intent:

1. Build the pendulum in Chrono as a **real rigid-body system**, not as a purely abstract equation.
2. Use **geometry + density** so mass and inertia come from the modeled bodies whenever feasible.
3. Let **Chrono handle gravity, inertia, contact, and joint dynamics**.
4. Use **`ChLinkMotorRotationTorque`** as the only actuation insertion point.
5. Treat **free-decay data** as the source for the nominal passive model.
6. Treat **excitation data** as the source for actuator and residual identification.
7. Keep the architecture modular enough to swap:
   - regression actuator,
   - SINDy actuator,
   - neural residual actuator,
   - RL-calibrated parameter sets.
8. Prefer interpretability where possible, and use black-box learning only where necessary.

---

## 18. Canonical Naming Convention

To reduce ambiguity:

- `model.pt` -> nominal passive free-decay model,
- `actuator_a2.pt` -> neural actuator/residual model,
- `sindy_equations.json` -> sparse identified residual equations,
- `ppo_calibrator.zip` -> RL calibration policy,
- `chrono_params.json` -> simulator calibration parameter snapshot.

---

## 19. Final Practical Rule

When in doubt:

- if the question is **"what is the pendulum itself doing without being driven?"** -> use **free-decay data**;
- if the question is **"how does command/current actually become torque in the real system?"** -> use **excitation data**.

That rule should govern both repository architecture and model naming.
