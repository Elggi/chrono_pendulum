# Host Layer 상세 문서

이 폴더는 **디지털 트윈 시뮬레이션, 캘리브레이션, 로그 분석, RL 파인튜닝**의 중심 계층입니다.

## 1) `chrono_pendulum.py`

### 역할
- PyChrono 기반 1-DOF 물리 모델 생성
- ROS2로 하드웨어 신호(`/cmd/u`, `/hw/*`, `/ina219/*`) 수신
- 시뮬레이터 상태(`/sim/*`) 퍼블리시
- 온라인 EKF-like 파라미터 업데이트
- 지연 보상(auto delay compensation)

### Phase A 모듈화(동작 동일 리팩터)
- `chrono_pendulum.py`는 실행 orchestration(ROS spin/메인 루프/상태 출력/로그 기록) 중심으로 축소.
- 공통 유틸/설정/동역학/추정/캘리브레이션 JSON 로딩은 `host/chrono_core/` 하위 모듈로 분리:
  - `chrono_core/config.py`: `BridgeConfig`
  - `chrono_core/utils.py`: 유틸 함수(`clamp`, `sanitize_float`, `terminal_status_line` 등)
  - `chrono_core/dynamics.py`: `PendulumModel`, 토크/전기모델, `enc_to_theta`
  - `chrono_core/estimation.py`: `DelayCompensator`, `CPREstimator`, `OnlineParameterEKF`, `ObservationLPF`, `FitConvergenceMonitor`
  - `chrono_core/calibration_io.py`: calibration/radius JSON 로딩
- 목적: 기존 실행 결과를 유지하면서 파일 책임을 분리하고, 이후 RL 환경(`environment.py`)에서 재사용 가능한 단위 컴포넌트를 확보.

### 동작 요약
1. 초기 파라미터(`J,b,tau_c,mgl,k_t,i0,R,k_e`) 로딩
2. 입력 PWM 수신 후 지연 보상 큐를 거쳐 모델에 적용
3. 시뮬레이션 상태(각도/각속도/각가속도) 계산
4. 실측 신호(EST/ENC)를 LPF 후 EKF 입력으로 사용
5. LS 수렴 판정(window RMS + hold 시간) 만족 시 self-fitting LOCK
6. fitting OFF 모드에서는 pure simulation만 수행(파라미터 업데이트 없음)
5. CSV + meta JSON 기록

### Sim2Real 반영 포인트 (최신 코드 기준)

#### A) Low-pass observation filter (관측 LPF)
- EKF 갱신 입력은 raw 값이 아니라 필터된 \(\theta,\omega,\alpha\) 관측치 사용:
\[
\hat y_k = (1-\alpha) \hat y_{k-1} + \alpha y_k,\quad
\alpha = \frac{\Delta t}{\tau+\Delta t}
\]
- 센서 스파이크/미분 잡음을 줄여 EKF 파라미터 흔들림을 완화.

#### B) Smoothness regularization
- fitting cost에 입력 변화량 페널티 추가:
\[
\mathcal{L}=\cdots + w_{\Delta u}(\Delta u)^2 + w_{\Delta^2u}(\Delta^2u)^2
\]
- 급격한 제어 변화가 만든 과도응답에 과적합되는 현상을 억제.

#### C) Least-squares 수렴 판정 + self-fitting lock
- \((\theta,\omega,\alpha)\) sim-real 오차 RMS를 sliding window로 추적.
- RMS가 임계치 이하 상태를 hold 시간 이상 유지하면 `fit_done=True`.
- 이후 EKF update를 중단해 최종 파라미터를 고정(`fit:LOCK`).

#### D) 터미널/로그 상태 확장
- one-line 갱신(`\r + line clear`)은 유지.
- 상태 문자열에 `LS`, `fit:RUN/LOCK/OFF` 표시.
- CSV/meta에 `ls_cost`, `fit_done`, `fit_complete`, `fit_final_params` 기록.

### Calibration JSON의 CPR / radius 사용처

#### 1) CPR(counts per revolution)
- 용도: encoder count를 각도로 변환.
\[
\theta_{\text{enc}} = \theta_{\text{ref}} + \frac{2\pi}{CPR}(enc-enc_{\text{ref}})
\]
- `est/theta`가 없을 때 fallback 실측 각도 생성에 사용됨.
- 즉, CPR은 **센서 단위(count)→물리 단위(rad)** 스케일링 파라미터.

#### 2) radius (mean_radius_m)
- 용도: calibration 결과에서 얻은 실제 기구학 반경을 `cfg.radius_m`로 로딩.
- 최신 코드에서는 **시각 형상/IMU body 배치**는 `cfg.link_L`를 유지하고, **동역학 COM/관성 계산**은 `cfg.radius_m`(실측 반경)를 사용한다.
- 정리하면 radius는 **실험 계측 기하값**, CPR은 **encoder 각도 환산값**.

### COM(질량중심) 처리 로직 (구체화)

최신 `chrono_pendulum.py`는 link를 균일 막대로만 보는 대신, **link + IMU 복합체**로 COM/관성을 계산함. 이때 COM/관성 계산의 길이 스케일은 `radius_m`를 사용한다.

```text
motor pivot (body ref)
   o-----------------------------  (link axis, -y)
      [uniform link COM]
                     [IMU center]
=> composite COM = mass-weighted average
```

#### 계산식
- 질량중심:
\[
\mathbf{r}_{com}=\frac{m_l\mathbf{r}_l + m_i\mathbf{r}_i}{m_l+m_i}
\]
- z축 관성(평행축 정리):
\[
I_{zz,com}=I_{zz,l}^{center}+m_l d_l^2 + I_{zz,i}^{center}+m_i d_i^2
\]
- 이 값을 `SetFrameCOMToRef`와 `SetInertiaXX`에 반영.

> 참고: Chrono Python 바인딩에서 `SetFrameCOMToRef`가 없으면 경고 후 기본 COM으로 안전 fallback.

### 주요 출력
- `run_logs/chrono_run_N.csv`
- `run_logs/chrono_run_N.meta.json`

---

## 2) `calibration.py`

### 역할
- IMU orientation + encoder 기반 calibration 수행
- full rotation 감지 기반으로 회전/정지 제어
- 사용자 입력 최대 PWM 한계 하에서 1 PWM step 증가 적용
- 정지 후 overshoot 회전량을 반대 방향으로 보정
- mean CPR, IMU 기반 `r` 추정치 출력 및 JSON 저장

### 핵심 로직
- IMU 수신 확인 후 초기 orientation 기준점 설정
- 단일 방향으로 PWM을 1 step씩 증가
- full rotation 2회 감지 시 즉시 정지
- 정지 후 추가 회전량을 측정해 반대 방향으로 보정

### CLI 주요 파라미터
- `--max-pwm-hard-limit`
- `--loop-hz`
- `--imu-wait-sec`, `--stop-settle-sec`

---

## 3) `plot_pendulum.py`

### 역할
- 시뮬레이션/실측 비교 시각화
- 전기 모델(전압/전류/전력) 비교
- online calibration 비용/파라미터 추세 확인

### 특징
- pandas가 없어도 동작하도록 CSV fallback 로더 내장
- `sim_time` 없으면 `wall_time` 기반 시간축 구성
- 컬럼명이 조금 달라도(`current_A` vs `current_ma`) 유연 매핑

---

## 4) `RL_fitting.py`

### 역할
- 오프라인 파라미터 최적화 (PPO/SAC)
- 실측 로그를 재현하는 파라미터 집합 탐색
- 최적 파라미터/학습 산출물 저장

---

## 5) `imu_viewer.py`

### 역할
- IMU orientation/가속도 기반 3D/2D 시각화
- 링크 자세 및 tip trajectory 직관적 확인

---

## 6) `pendulum_stack.sh`

### 역할
- Host 통합 메뉴 엔트리 포인트
- IMU viewer / calibration / RL / chrono / plot 실행
- calibration 메뉴 실행 (max PWM은 calibration.py 내부에서 입력)
