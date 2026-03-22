## Files

- `chrono_pendulum.py`
  - PyChrono 기반 1-DOF Pendulum 시뮬레이션
  - ROS2 subscribe/publish
  - host keyboard control
  - INA219 전압/전류/전력 수신
  - `/cmd/u` 와 `/hw/pwm_applied` 기반 자동 delay compensation
  - online EKF-style parameter tuning
  - CPR 추정 및 meta json 저장
  - Chrono window + IMU viewer window 동시 실행

- `plot_pendulum.py`
  - 최신 또는 지정 CSV 로그 시각화
  - sim vs real 비교
  - electrical model vs INA219 비교
  - online parameter convergence plot
  - 가장 cost가 낮았던 시점의 parameter 출력
  - meta json에서 CPR 자동 사용 가능

- `RL_fitting.py`
  - PPO / SAC 기반 offline post fitting
  - domain randomization
  - J, b, tau_c, mgl, k_t, i0, Rm, k_e, delay 포함
  - 최적 파라미터 json 및 추천 CLI 출력

## Basic requirements

예상 환경:

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+
- `pychrono` 9.0.1
- `rclpy`
- `numpy`, `pandas`, `matplotlib`, `scipy`
- `gymnasium`, `stable_baselines3` (`RL_fitting.py`용)

## ROS topics expected

### Subscribed by `chrono_pendulum.py`

- `/cmd/u`
- `/hw/pwm_applied`
- `/hw/enc`
- `/hw/arduino_ms`
- `/ina219/bus_voltage_v`
- `/ina219/current_ma`
- `/ina219/power_mw`

### Published by `chrono_pendulum.py`

- `/sim/theta`
- `/sim/omega`
- `/sim/alpha`
- `/sim/tau`
- `/sim/cmd_used`
- `/sim/delay_ms`
- `/sim/status`
- `/imu/data`

## Example usage

### 1) External hardware-controlled mode

```bash
python3 chrono_pendulum.py
```

### 2) Host keyboard-controlled mode

```bash
python3 chrono_pendulum.py --host-control
```

### 3) Headless run

```bash
python3 chrono_pendulum.py --headless --no-imu-viewer --duration 15
```

### 4) Fixed initial parameters

```bash
python3 chrono_pendulum.py \
  --J 0.012 \
  --b 0.035 \
  --tau-c 0.09 \
  --mgl 0.60 \
  --k-t 0.23 \
  --i0 0.06 \
  --R 2.1 \
  --k-e 0.025
```

### 5) Plot latest log

```bash
python3 plot_pendulum.py --dir ./run_logs
```

### 6) Plot specific log with manual CPR

```bash
python3 plot_pendulum.py --csv ./run_logs/chrono_run_3.csv --counts-per-revolution 8192
```

### 7) RL fitting

```bash
python3 RL_fitting.py --csv ./run_logs/chrono_run_3.csv --algo ppo
```

또는

```bash
python3 RL_fitting.py --csv ./run_logs/chrono_run_3.csv --algo sac
```

## Notes

- `chrono_pendulum.py`는 Chrono의 실제 강체 동역학을 사용합니다.
- online fitting용 governing equation은 진단/추정 모델로 사용됩니다.
- `J`는 rise shape, alpha peak, omega slope에 큰 영향을 주므로 핵심 파라미터입니다.
- delay는 기본적으로 auto compensation이 켜져 있습니다. 고정 delay만 쓰고 싶으면 `--disable-auto-delay --delay-ms ...`를 사용하세요.
- CPR 자동 추정은 full rotation episode가 있을 때 가장 잘 작동합니다. 완전회전이 없는 실험은 `plot_pendulum.py --counts-per-revolution ...`로 수동 지정하세요.

## Output files

`chrono_pendulum.py` 실행 후:

- `./run_logs/chrono_run_N.csv`
- `./run_logs/chrono_run_N.meta.json`

`RL_fitting.py` 실행 후:

- `./rl_results/ppo_pendulum_fit.zip` 또는 `sac_pendulum_fit.zip`
- `./rl_results/rl_result.json`
- `./rl_results/rl_best_prediction.csv`
