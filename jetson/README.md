# Jetson Layer 상세 문서

이 폴더는 실제 하드웨어 측 ROS 노드 운영, 실시간 명령 발행, 라이브 모니터링을 담당합니다.

## 1) `pendulum_stack.sh`

### 역할
Jetson 운영 자동화 스크립트로 다음 기능을 제공합니다.
- ROS 환경 로딩 (`/opt/ros/humble`, workspace setup)
- Arduino/IMU 시리얼 포트 자동 탐색
- 브리지/IMU 노드 start/stop/status/logs
- live monitor 실행
- keyboard controller 실행

### 운영 포인트
- PID 파일 기반 프로세스 관리
- 로그를 `~/jetson_ros_logs` 아래 누적 저장
- 실패 시 tail 로그 출력으로 디버깅 지원

---

## 2) `pendulum_controller.py`

### 역할
- 키보드 입력을 ROS 토픽 `/cmd/u`로 발행하는 제어기
- 수동 입력 + 자동 파형 프리셋(sin/square/burst/prbs)

### 입력 키(예시)
- `W/S` 또는 방향키: PWM 증감
- `Space`, `X`: 0으로 정지
- `1~4`: 고정 PWM preset
- `5~8`: 자동 파형 모드
- `[ ]`: step 크기 조절
- `- =`: max PWM 제한 조절

### 출력
- `/cmd/u` (Float32)
- `/cmd/keyboard_state` (String debug)

---

## 3) `live_monitor.py`

### 역할
- `/cmd/u`, `/hw/pwm_applied`, `/hw/enc`, `/ina219/*`를 구독
- 터미널 한 줄 대시보드 형태로 실시간 표시

### 장점
- SSH 환경에서도 상태를 빠르게 확인 가능
- 실기 테스트 중 과입력/전류 급증 감지에 유용

---

## Host- Jetson 협업 구조

1. Jetson이 하드웨어 센서/액추에이터 토픽 제공
2. Host가 이를 받아 시뮬레이션/식별/분석 수행