# Stage 1 (CMA-ES) 설계 초안

이 문서는 아래 요구사항을 기준으로 Stage 1을 **순차 구현**하기 위한 작업 계획이다.

- A: Stage1에서는 `b_eq`, `tau_eq`만 식별
- B/F/H: 실제 Chrono(Headless) 롤아웃 기반 CMA-ES + 병렬 평가
- C/D: `model_parameter.json` 기반 토크식/파라미터 주입 및 stage별 업데이트
- E: SINDy에서 사용할 known dynamics(중력 토크 포함)의 JSON 명시
- G: ask/tell 루프 구조 준수

---

## 0) 최종 목표(요약)

1. **Headless Chrono rollout evaluator**를 만든다.  
2. `cmaes` 라이브러리의 ask/tell + `ProcessPoolExecutor`로 `b_eq`, `tau_eq`를 최적화한다.  
3. 결과를 `model_parameter.json`에 업데이트하고, replay 가능한 CSV를 저장한다.  
4. Stage2 SINDy가 동일한 JSON에서 known dynamics를 읽어 residual 식을 추가할 수 있게 스키마를 고정한다.

---

## 1) model_parameter.json 스키마 (초안)

```json
{
  "version": 1,
  "known": {
    "mass_total_kg": 0.22,
    "inertia_total_kgm2": 0.00137,
    "l_com_total_m": 0.1652,
    "gravity_mps2": 9.81,
    "r_imu_m": 0.249793
  },
  "torque_model": {
    "motor": {
      "enabled": true,
      "equation": "tau_motor = K_i * I_filtered_A",
      "params": {"K_i": 1.0e-5}
    },
    "resistance": {
      "enabled": true,
      "equation": "tau_res = b_eq*omega + tau_eq*tanh(omega/eps)",
      "params": {"b_eq": 0.02, "tau_eq": 0.01, "eps": 0.05}
    },
    "residual_terms": []
  },
  "stage_outputs": {
    "stage1": null,
    "stage2": null,
    "stage3": null
  }
}
```

### 설계 원칙

- Stage1은 `resistance.params.b_eq`, `resistance.params.tau_eq`만 갱신
- Stage2는 `residual_terms`에 식/계수 append
- Stage3는 필요 시 motor/resistance 파라미터 재보정

---

## 2) Stage1용 Headless Chrono evaluator 설계

### 입력

- free-decay train CSV (real trajectory)
- 초기조건: `theta0`, `omega0` (CSV 첫 샘플)
- candidate params: `[b_eq, tau_eq]`
- model_parameter.json (known 항 + 현재 토크모델)

### 동작

1. `PendulumModel`을 headless로 생성 (현재 `ChBodyEasyBox`, `ChLinkMotorRotationTorque`, `ChLinkLockLock` 유지)
2. 모터 입력은 free-decay 상황에 맞게 `I=0`(또는 CSV 기반 보정 전류 사용 옵션)
3. 각 시간 샘플 간격(`dt`)에 맞춰 Chrono를 적분
4. `theta_sim`, `omega_sim` 시계열 출력

### 출력

- scalar loss (예: `w_theta * MSE(theta) + w_omega * MSE(omega)`)
- rollout csv (real + sim 동시 저장)

---

## 3) CMA-ES loop (ask/tell)

```python
from cmaes import CMA

optimizer = CMA(mean=[b0, tau0], sigma=0.03, bounds=[[b_min, b_max], [tau_min, tau_max]])

for gen in range(max_generations):
    candidates = [optimizer.ask() for _ in range(optimizer.population_size)]
    # 병렬 평가
    # losses = pool.map(evaluate, candidates)
    optimizer.tell(list(zip(candidates, losses)))
```

### 병렬화

- `ProcessPoolExecutor(max_workers=N)` 기반
- worker는 독립 Chrono system 인스턴스를 생성
- 멀티 trajectory는 worker 내부에서 평균 loss 계산

---

## 4) 단계별 구현 순서

### Step 1 (환경/인터페이스)

- `model_parameter.json` 로더/검증기 작성
- `chrono_pendulum.py`에 JSON 주입 경로 정리 (토크식 파라미터 읽기)

### Step 2 (headless rollout core)

- `backend/stage1/cmaes_chrono.py`(신규)에서 CSV 1개 evaluator 구현
- 단일 candidate 평가 및 rollout CSV 생성

### Step 3 (CMA-ES)

- ask/tell + 병렬 map + best params 저장
- `stage1_result.json` + `model_parameter.json` 업데이트

### Step 4 (pendulum_stack 연동)

- 메뉴에 Stage1-CMAES 추가
- 완료 후 replay option 8에서 결과 CSV 확인 가능하도록 경로 출력

### Step 5 (Stage2 연동 준비)

- `model_parameter.json`에 `known` + `residual_terms` 반영 API 추가
- SINDy가 `known gravity torque = m*g*l_com*sin(theta)`를 읽을 수 있게 문서화

---

## 5) 리스크/주의사항

- 렌더링 on 상태는 평가 속도를 크게 떨어뜨리므로 Stage1은 반드시 headless
- wall-time 동기화 모드와 offline trajectory 적분 모드는 분리 필요
- Chrono 적분 step과 CSV dt 정합이 핵심 (substep 정책 고정 필요)

