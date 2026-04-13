# Chrono Pendulum Digital Twin: Multi-Method System Identification Framework

This project is a **Sim2Real digital-twin framework** for a 1-DOF actuated pendulum using **Project Chrono** as the authoritative physics engine and a **system-identification toolbox** (regression, SINDy, neural residual models, and RL calibration).

The core philosophy is:

> **Chrono is the plant** (rigid-body dynamics, contact, gravity, geometry, mass/inertia).  
> **System ID learns only what the physics model cannot know** (actuator losses, friction, small modeling errors, sensor/latency effects).

## Vision and architecture
### Key design decisions
1. Chrono plant uses torque actuation via ChLinkMotorRotationTorque. This motor applies torque without enforcing motion constraints (unlike angle/speed motors), which is the correct abstraction for real actuators and robust contact events.
2. Bodies are created using ChBodyEasy* (e.g., ChBodyEasyBox, ChBodyEasyCylinder) so that mass and inertia are computed automatically from geometry and density.
3. All identification methods share a single canonical dataset and unit convention (rad, rad/s, rad/s², A, N·m).
4. Any learned/identified model is packaged as an artifact:
   - nominal parameters (JSON)
   - discovered symbolic equation (text/JSON)
   - NN residual model (.pt + scaler/schema)
   - RL-calibrated parameter set (JSON)
### Core concepts (ML terminology)

We standardize terminology across pipelines:

State:
[ x(t) = [\theta(t), \ \omega(t)]^\top ] 

<img width="207" height="37" alt="image" src="https://github.com/user-attachments/assets/eca5ab74-b188-4fee-bd43-b15230204da0" />

Optionally include (\alpha(t)) if available, but treat it carefully (differentiation noise).

Input / action (actuation signal):
[ u(t) = I(t) ] 

<img width="118" height="32" alt="image" src="https://github.com/user-attachments/assets/0bc5588d-cde4-472b-ad92-c0b22d1551ef" />

where (I) is actuator current (amps). (PWM/voltage can be substituted if that is what hardware provides.)

Torque command applied to Chrono:
[ \tau(t) \in \mathbb{R} ] 

<img width="118" height="32" alt="image" src="https://github.com/user-attachments/assets/c061ad88-5e16-4a88-a0f5-142dd5ec589e" />

applied via a torque function attached to ChLinkMotorRotationTorque.

Feature vector (for regression/SINDy/NN):
[ \phi(t) = f([x(t),u(t)]) ] 

<img width="206" height="36" alt="image" src="https://github.com/user-attachments/assets/3fa5fc88-45ff-45d5-902f-688981e8edae" />

Example: ([\theta,\omega,I,\sin\theta,\tanh(\omega/\epsilon),\omega I,\ldots]).

<img width="286" height="40" alt="image" src="https://github.com/user-attachments/assets/ef4fb592-dd08-4ad5-b82d-03c1773f91af" />


### Chrono plant layer
Why ChBodyEasy* and geometry+density? 
The plant is built from rigid bodies whose mass and inertia should be derived from geometry. Chrono provides ChBodyEasyBox and ChBodyEasyCylinder where mass and inertia are set automatically depending on density and the shape is created at the center of mass (COM). The cylinder is created along the Y axis and centered at COM. (See Chrono API docs.)

This matters because it prevents “silent” mismatches between:

   - CAD dimensions vs configured mass values
   - manually typed inertia tensors that drift over time

### Why torque motors, not motion-enforcing motors
We use ChLinkMotorRotationTorque as the actuation port:

   - It applies torque between two frames/bodies.
   - Unlike ChLinkMotorRotationAngle and ChLinkMotorRotationSpeed, it does not enforce motion via constraints.
   - Torque is supplied via a ChFunction (nominally (\tau = f(t)), but Chrono docs explicitly allow custom functions that also depend on state information).

This makes the simulation stable when the pendulum collides or experiences hard constraints, and it matches the reality that motors produce torque, not perfect kinematic trajectories.

### ROS2 data collection and logging
Required topics (reference)
The framework assumes ROS2 topics from a hardware bridge node (or replay source), e.g.:

- IMU:
   - /imu/data (sensor_msgs/Imu)
- Encoder:
   - /hw/enc (counts or radians; project-defined)
- Current:
   - /ina219/current_ma (mA) or /input_current (A)
- Optional:
   - /hw/pwm_applied
   - /pendulum/state_estimate

### Logs (canonical CSV schema)
Each run produces a CSV log that includes at minimum:

   - time or wall_elapsed
   - theta (rad)
   - omega (rad/s)
   - alpha (rad/s²) — either sensor-derived or computed from (\omega)
   - input_current (A)
All identification pipelines consume these logs.

### System identification pipelines
We support multiple complementary ID methods. The point is not to pick one “best” method but to build a toolbox that can:

   - produce interpretable baseline models,
   - discover missing physics terms,
   - learn residuals when symbolic models are not enough,
   - and calibrate parameters for sim2real fidelity.


### Pipeline B: physics regression + SINDy
Pipeline B is the interpretable baseline.

### B1. Regression (nominal model)
A standard nominal 1-DOF equation is:

[ J\alpha = K_I I - b_{eq}\omega - \tau_{eq}\tanh\left(\frac{\omega}{\epsilon}\right)- m g l_{com}\sin\theta ]

<img width="439" height="42" alt="image" src="https://github.com/user-attachments/assets/b6882bcc-b2d9-48b1-80f2-537aa7bc1ebb" />


   - (J): effective inertia about the pivot axis
   - (K_I): current-to-torque gain
   - (b_{eq}): viscous friction coefficient
   - (\tau_{eq}): Coulomb friction magnitude (smoothed by (\tanh))
   - (m g l_{com}\sin\theta): gravity load torque term for a rigid rotating assembly
Theory: this is a linear regression problem in the parameters if you rearrange:

[ J\alpha + b_{eq}\omega + \tau_{eq}\tanh(\omega/\epsilon) + m g l_{com}\sin\theta = K_I I ]

<img width="439" height="42" alt="image" src="https://github.com/user-attachments/assets/711efaa4-2c63-488d-8ff2-af5b4c3b2b7c" />

or stack terms and solve with least squares. This yields a robust, interpretable baseline and a residual signal for further modeling.

### B2. SINDy (Sparse Identification of Nonlinear Dynamics)
SINDy aims to discover a parsimonious symbolic model by solving:

[ \dot{x} = \Theta(x,u),\xi ]

<img width="156" height="38" alt="image" src="https://github.com/user-attachments/assets/8e6d4cfb-ba50-4009-bbc5-168549118289" />

where:

   - (\Theta(\cdot)) is a library of candidate nonlinear features (polynomials, Fourier terms, (\sin\theta), (\tanh(\omega)), etc.)
   - (\xi) is a sparse coefficient vector found via sparse regression (e.g., STLSQ, SR3)

We recommend two variants:

1. Greybox SINDy (residual dynamics)
Fit SINDy on the residual: [ \alpha_{res}(t) = \alpha_{data}(t) - \alpha_{nom}(t) ]

<img width="262" height="35" alt="image" src="https://github.com/user-attachments/assets/361b4b08-3a53-4bf8-a41f-216926db2d44" />

This preserves interpretability while focusing discovery on “what’s missing.”

3. Blackbox SINDy (full dynamics)
Fit SINDy to the full dynamics if interpretability constraints are relaxed.

Theory: SINDy is sparse regression (compressed sensing intuition): you assume only a few terms in a large candidate set are active and use thresholding-based solvers to select them.

### Pipeline A-2: Actuator neural residual (current → extra torque)
Pipeline A-2 models unmodeled actuator effects as a learned residual torque:

[ \tau(t) = \tau_{nom}(x(t),I(t)) + \tau_{res}(x(t),I(t);\psi) ]

<img width="400" height="42" alt="image" src="https://github.com/user-attachments/assets/ce3af638-e8f8-402a-99b5-2daf75c92f20" />

where:

- (\tau_{nom}) is the nominal torque model (gain + friction)
- (\tau_{res}) is a neural network (small MLP) trained from data

A-2 residual training target
A common residual definition is:

[ \tau_{res,\ target}(t)
J\alpha(t)
\Big( K_I I(t)
b_{eq}\omega(t)
\tau_{eq}\tanh(\omega(t)/\epsilon)
m g l_{com}\sin\theta(t) \Big) ]

<img width="573" height="53" alt="image" src="https://github.com/user-attachments/assets/a0e4e920-b954-44bf-96a1-1e5576b190b2" />

This target has a practical interpretation: “the torque not explained by the nominal model.”

Features and scaling
A minimal feature vector is:

[ z(t) = [\theta(t),\ \omega(t),\ I(t)] ]

<img width="220" height="33" alt="image" src="https://github.com/user-attachments/assets/de3fc9ab-7c64-4f89-9f9e-13070d7cce51" />

In practice, scaling (standardization) is essential for stable training and cross-run generalization.

Theory: this is supervised regression with a universal function approximator. Residual learning is a standard grey-box technique: keep physics constraints explicit, learn only the mismatch.

### Pipeline C-1: RL calibration (PPO) for sim2real parameters
Pipeline C-1 treats calibration as a sequential decision process:

- Episode = simulate one or more logged trajectories using candidate parameters
- Action = propose parameter updates (e.g., (K_I, b_{eq}, \tau_{eq}, l_{com}))
- Reward = negative tracking error between simulation and real: [ r = -\mathrm{RMSE}(\theta) - \lambda_1\mathrm{RMSE}(\omega) - \lambda_2\mathrm{RMSE}(\alpha) ]

<img width="439" height="35" alt="image" src="https://github.com/user-attachments/assets/1e33a7dd-cf44-4196-a64d-fcc773747d31" />

Because this is fundamentally an optimization of parameters, RL is most valuable when:

- the objective is non-smooth (contacts, resets),
- the simulator is a black box w.r.t. those parameters,
- you want robust policies over a distribution of trajectories.

We use PPO (on-policy) primarily for stability; alternatives include CMA-ES or Bayesian optimization for purely static parameter tuning.

Theory: PPO is policy-gradient RL with a clipped surrogate objective to prevent destructive policy updates. Here, the “policy” outputs calibration parameters bounded in a physically plausible range.

Optional: RL residual tuning
You can also combine approaches:

[ \tau_{nom}(x,I;\theta) + \tau_{sindy}(x,I;\xi) + \tau_{nn}(x,I;\psi) ]

<img width="380" height="42" alt="image" src="https://github.com/user-attachments/assets/d1aab18f-90da-41e5-9011-741e33f3384c" />

and apply RL to tune selected scalars (e.g., gain scaling or residual weight) for robustness.

### Concise module map 
Package / module	|  Key classes / functions	  |  Responsibilities	|    Pipelines supported

chrono_pendulum/plant/chrono_plant.py  |	ChronoPendulumPlant	|  Build Chrono system with ChBodyEasy* bodies; joints; ChLinkMotorRotationTorque torque port; stepping; state extraction  |	A-2, B, C-1

chrono_pendulum/plant/geometry.py  | 	density_from_mass(), derived_inertia_report()  | 	Compute density so mass matches config; compute derived COM/inertia summaries for reporting  |	A-2 targets, B constants, C-1 optional

chrono_pendulum/plant/contact.py  |  make_contact_material()	|  NSC/SMC surface material creation and collision enable flags  |	Plant fidelity

chrono_pendulum/models/actuator_nominal.py	|   NominalTorqueModel	|   Implements (\tau_{nom}(x,u)) (gain + friction) used by A-2 and B	|   A-2, B, C-1

chrono_pendulum/models/actuator_nn_residual.py	|   ResidualTorqueNN	|   Loads .pt + scaler + schema; predicts (\tau_{res})	|   A-2

chrono_pendulum/models/actuator_compose.py	|   ComposedActuator	|   (\tau = \tau_{nom} + \tau_{res})	|   A-2

chrono_pendulum/data/csv_io.py	|   load_log_csvs()	|   One canonical CSV parser (columns, units, alpha derivation)	|   A-2, B, C-1

chrono_pendulum/pipelines/regression.py	|   fit_nominal_params()	|   Stage1 regression / least-squares identification	|   B

chrono_pendulum/pipelines/sindy.py   |	fit_sindy()	|   SINDy on residual or full dynamics; exports equation	|   B

chrono_pendulum/pipelines/a2_train.py	|   train_residual_nn()	|   Trains NN residual; writes artifacts; updates config	|   A-2

chrono_pendulum/pipelines/c1_env.py	|  ChronoCalibrationEnv	   |   Gym/Gymnasium env for parameter calibration	|   C-1

chrono_pendulum/pipelines/c1_train_ppo.py	|   train_ppo()	|   PPO trainer harness, logging and write-back	|   C-1

apps/ (or keep host/)	|  CLI scripts	|   Thin wrappers only (no core logic)	|   All


### Reproducibility (dependencies)
We recommend a pinned Python environment.

Core runtime
Python 3.10+
Project Chrono Python bindings (PyChrono)
ROS2 + rclpy + relevant message packages
Data + plotting
numpy, scipy, pandas, matplotlib
System identification
PySINDy
scikit-learn (scalers)
Neural residual
PyTorch (CPU is fine, GPU optional)
RL calibration
gymnasium
stable-baselines3
tensorboard (recommended)

### Target project layout (after refactor)
chrono_pendulum/
  plant/
  models/
  data/
  pipelines/
apps/
  chrono_pendulum_run.py
  imu_viewer.py
  replay.py
configs/
  motor_torque.json
  calibration_latest.json

### Workflow quickstart (conceptual)
1. Collect data (ROS2) → CSV logs
2. Run Regression (B1) → nominal params
3. Run SINDy (B2) → discovered terms/equation
4. Train NN residual (A-2) if needed → model.pt
5. Run RL calibration (C-1) for sim2real match → calibrated params
6. Use the assembled actuator model + Chrono plant for replay and prediction


<img width="998" height="626" alt="image" src="https://github.com/user-attachments/assets/febeb639-f8b5-4e7e-bb9b-77d95da3d4af" />

