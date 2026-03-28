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
