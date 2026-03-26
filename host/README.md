# Host Layer 상세 문서

이 폴더는 **디지털 트윈 시뮬레이션, 캘리브레이션, 로그 분석, RL 파인튜닝**의 중심 계층입니다.

## 1) `chrono_pendulum.py`

### 역할
- PyChrono 기반 1-DOF 물리 모델 생성
- ROS2로 하드웨어 신호(`/cmd/u`, `/hw/*`, `/ina219/*`) 수신
- 시뮬레이터 상태(`/sim/*`) 퍼블리시
- 온라인 EKF-like 파라미터 업데이트
- 지연 보상(auto delay compensation)

### 동작 요약
1. 초기 파라미터(`J,b,tau_c,mgl,k_t,i0,R,k_e`) 로딩
2. 입력 PWM 수신 후 지연 보상 큐를 거쳐 모델에 적용
3. 시뮬레이션 상태(각도/각속도/각가속도) 계산
4. 실측 신호와의 오차를 기반으로 online fitting 수행
5. CSV + meta JSON 기록

### 주요 출력
- `run_logs/chrono_run_N.csv`
- `run_logs/chrono_run_N.meta.json`

---

## 2) `system_identification.py`

### 역할
- **Adaptive calibration** 수행
- IMU yaw + encoder를 이용해 회전량 추적
- 안전 제약(PWM 상한, 최대 회전수) 하에서 sweep
- 원점 복귀(return-to-origin)
- CPR 추정치 산출 및 calibration JSON 저장

### 핵심 로직
- settle 구간에서 기준 yaw/encoder 확보
- 단일 방향으로 PWM을 단계적으로 증가
- full rotation 2회 감지 시 sweep 정지
- 종료 후 yaw 기준 복귀 제어

### CLI 주요 파라미터
- `--max-calib-pwm`
- `--sweep-pwm-step`
- `--sweep-hold-sec`
- `--return-kp`, `--return-timeout-sec`, `--return-tol-rad`

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
- calibration role 및 safety 파라미터 입력 지원
