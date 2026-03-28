# Chrono Pendulum: Sim2Real Digital Twin for a 1-DOF Rotary Pendulum

본 프로젝트는 **1-DOF 회전 진자(rotary pendulum)**를 대상으로, 실제 하드웨어(Arduino+IMU+INA219)와 물리 시뮬레이터(PyChrono)를 ROS2 기반으로 결합한 **Sim2Real 디지털 트윈 프레임워크**를 제안한다. 시스템은 (1) 실시간 제어 입력/센서 스트리밍, (2) 적응형 캘리브레이션(adaptive calibration), (3) 온라인 파라미터 피팅(EKF-like), (4) 오프라인 강화학습 기반 파인튜닝(PPO/SAC)으로 구성된다. 캘리브레이션 단계에서는 고정 PWM 스크립트 대신 IMU yaw/엔코더 피드백 기반 sweep를 수행해 안전 제약(회전수/입력 제한)을 유지하면서 CPR(counts per revolution)과 동역학 초기 파라미터를 추정한다. 결과적으로 본 스택은 단순 시뮬레이션 재현을 넘어, 하드웨어 변동·지연·마찰·센서 노이즈를 포함한 **현실 적합형(sim-to-real consistent) 모델 업데이트**를 가능하게 한다.

---

## Framework Flow

```text
[Jetson ROS sensor/actuation]
   ├─ hw_arduino_bridge (PWM/encoder/time)
   ├─ wheeltec IMU node (/imu/data)
   └─ optional keyboard/controller publisher (/cmd/u)
              │
              ▼
[ROS2 topics]
              │
              ├─ host/calibration.py
              │    (adaptive calibration, CPR estimate, safety return)
              │
              ├─ host/chrono_pendulum.py
              │    (PyChrono twin + online EKF-like fitting + delay compensation)
              │
              ├─ host/plot_pendulum.py
              │    (diagnostic visualization)
              │
              └─ host/RL_fitting.py
                   (offline PPO/SAC parameter refinement)
```

---

## Programs and Recommended Versions

> 아래는 프로젝트 운영 기준 권장 버전(실험 환경에 따라 조정 가능)

- OS: Ubuntu 22.04 LTS
- Python: 3.10+
- ROS2: Humble
- PyChrono: 9.0.1
- NumPy: 1.22.4 이상 (권장 1.26.x)
- Pandas: 1.5 이상 (2.0.x 호환)
- Matplotlib, SciPy
- RL(optional): gymnasium, stable-baselines3

---

## Applied Algorithms (수식 포함)

### 1) Pendulum dynamics surrogate (식별/피팅용)
시스템의 핵심 추정 모델은 아래 형태를 사용한다.

\[
J\dot{\omega} = \tau_m - b\omega - \tau_c\tanh\left(\frac{\omega}{\epsilon}\right) - mgl\sin\theta
\]

- \(J\): 등가 관성
- \(b\): 점성 마찰 계수
- \(\tau_c\): 쿨롱 마찰 항
- \(mgl\): 중력 토크 계수
- \(\tau_m\): 모터 토크 항 (PWM/전기 모델 기반)

### 2) Encoder-to-angle conversion
\[
\theta_{enc}(k) = s\cdot\frac{2\pi}{CPR}\left(enc(k)-enc(0)\right)+\theta_0
\]

### 3) Adaptive calibration sweep safety rule
- 입력 증가: \(u = \pm(u_0 + n\Delta u)\)
- 정지 조건(한 방향):
\[
N_{turn} = \max\left(\frac{|\Delta\psi_{imu}|}{2\pi},\;\frac{|\Delta enc|}{CPR_{guess}}\right) \ge N_{max}
\]

### 4) Delay compensation (online)
명령 적용 시점을 추정 지연 \(\hat d\) 만큼 보정하여 시뮬레이터 입력과 실기 응답을 정렬.

### 5) Online EKF-like fitting
상태/파라미터 벡터를 확장해 innovation 기반으로 파라미터를 점진 갱신.
(구현은 EKF 유사 구조이며 시스템 안정성 우선으로 공분산/게인을 제한)

### 6) Offline RL fine-tuning (PPO/SAC)
목표는 trajectory mismatch와 전기적 불일치를 최소화하는 파라미터 탐색:
\[
\min_{\phi}\; \mathcal{L}(\phi)=w_\theta\|\theta_{sim}-\theta_{real}\|^2 + w_\omega\|\omega_{sim}-\omega_{real}\|^2 + w_p\|P_{sim}-P_{real}\|^2 + \cdots
\]

---

## Sim2Real 연구로서의 의미

1. **현실 제약 내 식별**: 케이블 꼬임/과회전/과입력 위험을 안전 제약으로 내재화.
2. **모델-실기 동시 관측**: 동일 ROS 토픽 기반으로 sim/real을 정렬해 오차 원인을 분해.
3. **계층형 최적화**: 온라인(즉시 보정) + 오프라인(RL 전역 탐색) 결합.
4. **재현 가능 실험 체인**: 캘리브레이션→런→플롯→파인튜닝의 파이프라인 자동화.

---

## Repository Guide

- `host/README.md`: Host 파이프라인 상세 동작
- `jetson/README.md`: Jetson 제어/센서/노드 운영 상세
- `motor_controller.ino`, `ros2_ws/`는 본 상위 README에서 상세 생략 (요청사항 반영)
