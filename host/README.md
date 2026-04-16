# chrono_pendulum `host/` 실행/구조 안내

이 문서는 `pendulum_stack.sh` 메뉴 엔트리와 실제 실행 코드의 매핑을 정리합니다.

## 메뉴 엔트리 매핑 (`pendulum_stack.sh`)

- **1) IMU Viewer**
  - 함수: `run_imu_viewer`
  - 실행: `python3 imu_viewer.py`
  - 기능: IMU/엔코더 실시간 시각화

- **2) Model Calibration**
  - 함수: `run_system_identification`
  - 실행:
    - free-decay: `python3 calibration.py --mode free_decay ...`
    - manual CPR/r: `python3 calibration.py`
  - 기능: 캘리브레이션 데이터 수집/저장

- **3) Stage1 Identification (CMA-ES Headless)**
  - 함수: `run_stage1_pem_identification`
  - 실행: `python3 stage1_cmaes_chrono.py ...`
  - 기능:
    - 기본: free-decay 기반 `b_eq`, `tau_eq` 식별
    - 선택(y): motor input 포함 `K_i`, `b_eq`, `tau_eq` 동시 식별
  - 출력: Stage1 reports + `model_parameter.latest.json` 업데이트

- **4) Stage2 Identification (SINDy)**
  - 함수: `run_stage2_sindy_identification`
  - 실행: `python3 stage2_sindy_entry.py ...`
  - 기능:
    - residual-acceleration 기반 SINDy 식별
    - rollout-aware 안정성 진단
    - runtime용 residual torque 항으로 매핑 후 `model_parameter.latest.json` 업데이트

- **5) Stage3 Identification (PPO)**
  - 함수: `run_stage3_ppo_identification`
  - 실행: `python3 stage3_ppo_entry.py ...`
  - 기능: PPO 기반 후속 식별/탐색

- **6) Chrono Pendulum**
  - 함수: `run_chrono_pendulum`
  - 실행: `python3 chrono_pendulum.py ...`
  - 기능: Host/Jetson 모드 시뮬레이션 런타임

- **7) Plot Data**
  - 함수: `run_plot`
  - 실행: `python3 plot_pendulum.py --csv ...`
  - 기능: 선택 CSV의 시계열 플롯

- **8) Replay Runs**
  - 함수: `run_replay_validation`
  - 실행: `python3 replay_pendulum_cli.py --csv ...`
  - 기능: CSV 재생 + 3D/IMU 동작 검증

- **9) Offline Benchmark**
  - 함수: `run_offline_benchmark_pem_sindy_ppo`
  - 실행: `python3 offline_id_pem_sindy_ppo.py ...`
  - 기능: 오프라인 벤치마크 파이프라인

---

## 모델 파라미터 파일 정책

- `model_parameter.template.json`
  - **새 학습 시작용 빈 템플릿**
  - 실행 중 덮어쓰지 않고 기준 스켈레톤으로 유지

- `model_parameter.latest.json`
  - **실제 학습/실행에서 업데이트되는 최신 파일**
  - Stage1/Stage2 결과 반영 대상

---

## 이번 정리에서 반영한 obsolete/legacy 제거

- `chrono_core/model_parameter_io.py`에서 legacy schema(`model_init`, `best_params`, flat keys) 파싱 제거
- Stage2 SINDy의 deterministic STLSQ fallback 제거 (PySINDy 2.1.0 기준 필수 경로로 고정)

