---
title: SO-101 Push-T DP-CNN Sim-to-Real 중단 원인 및 개선 방안
date: 2026-08-27
tags:
  - SO-101
  - Push-T
  - DP-CNN
  - sim-to-real
  - physical-rollout
  - safety
status: evidence-bounded
---

# SO-101 Push-T DP-CNN Sim-to-Real 중단 원인 및 개선 방안

## 최종 결론

이번 작업에서는 200-episode 데이터로 학습한 DP-CNN checkpoint를 현재 실행 환경에서 복구하고, 실제 카메라 입력으로 authentic `DiffusionUnetHybridImagePolicy` 추론을 수행한 뒤 SO-101의 물리 단일 스텝 직전까지 sim-to-real 안전 파이프라인을 구축했다.

그러나 실제 모터 명령은 실행하지 않고 중단했다. 중단 이유는 checkpoint 손상이나 추론 실패가 아니라, DP-CNN이 출력한 시뮬레이터의 2차원 pusher 위치를 실제 5-DOF SO-101의 안전한 관절 경로로 변환하는 단계에서 다음 조건을 동시에 만족하지 못했기 때문이다.

1. 물리 workspace 내부
2. IK 수렴 및 허용 FK residual
3. 현재 자세와 동일한 IK branch 유지(최대 관절 변화 8도 이하)
4. 전체 이동 경로에서 최소 충돌 여유 1 mm 초과

후속 코드 검토 결과, 현재 collision checker가 실제 end-effector가 대신해야 할 가상 `pusher` sphere까지 장애물로 취급하는 모델링 모순도 확인했다. 또한 planner는 시작 관절과 목표 관절 사이를 직선으로만 보간하며 장애물을 우회하는 waypoint 경로를 생성하지 않는다. 따라서 현재 결과는 “DP-CNN을 실제 로봇에 사용할 수 없다”는 결론이 아니라, **현재 task-frame bridge와 contact/collision-aware motion planner가 물리 rollout에 충분하지 않다**는 한계다.

컴퓨터 포맷 일정과 남은 구현·물리 검증 비용을 고려하여 안전 기준을 임의로 낮추거나 급하게 모터를 움직이지 않고 이 지점에서 연구 범위를 종료했다. 종료 시 모든 SO-101 모터의 `Torque_Enable=0`을 확인했으며 policy가 생성한 실제 motor write는 수행하지 않았다.

---

## 1. 이번 단계에서 달성한 내용

### 1.1 네 모델 checkpoint 보존

다음 네 200-episode 모델의 원본 checkpoint를 변경하거나 덮어쓰지 않고 보존했다.

- DP-CNN
- DP-Transformer
- IBC
- LSTM-GMM

이번 물리 진단에서는 DP-CNN만 사용했다. 다른 세 모델은 물리 실행 fallback으로 사용하지 않았다.

### 1.2 DP-CNN current-runtime 복구

과거 runtime-lock 및 robomimic `CropRandomizer` 호환성 문제로 checkpoint를 현재 환경에서 직접 로드하지 못한 RED 상태를 분리했다. 이후 recovered artifact를 통해 authentic `DiffusionUnetHybridImagePolicy`를 로드하고 실제 frozen-policy inference가 실행되는 GREEN 경로를 확보했다. Fixture/random adapter가 아니라 recovered DP-CNN policy를 사용하도록 lineage와 runtime digest를 고정했다.

주요 lineage 증거:

```text
04_experiments/so101_pusht_benchmark/inference/sim_to_real_rollout/
  derived-evidence/v4-lineage-manifest.json
  derived-evidence/lineage-v4.json
  derived-evidence/compact-lineage-v4.json
```

### 1.3 실제 입력과 안전 경계 연결

다음 입력과 검사를 실제 장치 또는 고정 물리 모델에 연결했다.

- 실제 카메라 fresh capture
- 현재 SO-101 관절값 read
- 실제 관절과 MuJoCo joint의 등가성 receipt
- 실제 장면의 pusher/T pose
- task-frame 좌표 변환
- Cartesian workspace 검사
- 5-DOF IK와 FK residual 검사
- IK branch delta 검사
- table/object/self swept collision 검사
- owner-signed production policy
- 한 명령만 허용하는 execution budget
- deadman/HOLD/emergency STOP/torque-off 경계

### 1.4 물리 actuation 전 fail-closed 유지

Production shadow는 authentic policy output을 계산했지만 gate를 통과하지 못하면 즉시 HOLD로 종료했다.

```text
cycles_completed=0
motor_writes_performed=false
actuation_performed=false
terminal_state=HOLD
```

최종 shadow 증거:

```text
04_experiments/so101_pusht_benchmark/inference/sim_to_real_rollout/
  production-shadow-v4-ready-20260827/ledger.jsonl
  production-shadow-v4-ready-20260827/terminal_receipt.json
```

---

## 2. 목표했던 최종 상태와 실제 중단점

목표는 사용자가 visible xterm에서 다음 상태를 확인하고 Space를 누르고 있는 동안에만 DP-CNN의 단일 physical step을 실행하는 것이었다.

```text
ARMED
Space = deadman
Space release = HOLD
Esc = emergency STOP
Ctrl+C = abort
```

그러나 ARMED 상태는 성공한 production shadow와 유효 authorization에 바인딩되어야 한다. Shadow가 안전한 IK/collision proposal을 만들지 못했기 때문에 authorization과 ARMED terminal을 발급하지 않았다. UI만 억지로 띄우거나 safety gate를 우회하지 않았다.

```text
실제 카메라 + 현재 관절값
        ↓
Authentic DP-CNN 추론
        ↓
absolute mocap XY action
        ↓
simulator-to-physical task-frame 변환
        ↓
Cartesian workspace / contact Z 검사
        ↓
5-DOF SO-101 IK
        ↓
branch / FK / swept collision 검사  ← 실제 중단점
        ↓
authorization
        ↓
ARMED terminal
```

모델 로드, 이미지 입력, inference, 좌표 변환 자체가 최종 blocker는 아니었다.

---

## 3. 관찰된 실패 유형

### 3.1 Workspace 범위 초과

일부 DP-CNN action은 기존 물리 workspace 밖으로 변환됐다. Contact Z와 workspace를 owner-signed v4, v5, v6 policy로 단계적으로 재검토했다.

```text
authority/inputs/dp-cnn-v4-clearance-policy-20260827/
authority/inputs/dp-cnn-v5-clearance-policy-20260827/
authority/inputs/dp-cnn-v6-workspace-policy-20260827/
```

v6에서는 X/Y 허용범위를 넓혀 workspace 탈락을 줄였지만, 통과한 목표도 IK, branch 또는 collision gate에서 탈락했다. Workspace를 넓히는 것은 목표를 허용할 뿐 로봇의 기구학이나 충돌 형상을 바꾸지 않는다.

### 3.2 IK 불수렴

일부 목표는 현재 관절 seed에서 DLS IK가 허용 residual까지 수렴하지 못했다. 한 production shadow 사례에서는 약 `0.194740 m` residual이 남았다.

```text
R_CLIPPING_REQUIRED: residual 0.194740 m, singularity 0.049226
```

표면 terminal code는 상위 fail-closed 코드였지만 rejection detail이 가리키는 실제 원인은 IK residual이었다.

### 3.3 IK branch 불연속

IK 해가 있어도 현재 자세에서 다른 elbow/shoulder branch로 전환해야 하는 후보가 많았다. 200개 authentic closed-loop 후보 중 다수는 최대 관절 변화가 약 20~95도였고 signed policy의 단일-step 제한은 8도였다. 이를 임의로 늘리면 단일 스텝이 아니라 팔 전체의 큰 자세 전환이 될 수 있으므로 완화하지 않았다.

### 3.4 Swept collision clearance 0 mm

Focused fixed-point/midpoint 탐색으로 branch delta를 `3.26~5.26도`까지 낮춘 후보를 찾았다. 이는 8도 제한을 만족한다. 그러나 시작과 목표 사이의 직선 관절 보간 경로에서 최소 clearance가 `0 mm`로 계산되어 reject됐다. 현재 자세와 가까운 IK 해만 찾는 것으로는 해결되지 않았다.

---

## 4. 수행한 탐색과 그 의미

### 4.1 약 100,000개 IK 후보

원래 target과 안전한 초기 자세 조합을 찾기 위해 약 100,000개 deterministic IK 후보를 검사했다. workspace, IK, branch, collision 기준을 동시에 통과한 후보는 없었다.

### 4.2 Authentic DP-CNN closed-loop 후보 200개

서로 다른 collision-free joint seed 200개에 대해 다음을 반복했다.

1. 해당 seed의 FK로 `agent_pos` 구성
2. 동일 카메라 history로 authentic DP-CNN 재추론
3. 새 action을 physical XYZ로 변환
4. IK 수렴 확인
5. branch delta 확인
6. swept collision 확인

모든 기준을 통과한 후보는 없었다.

### 4.3 Focused fixed-point/midpoint 탐색

200개 중 branch 초과가 가장 작은 후보를 선택해 IK solution과 seed의 midpoint를 다시 closed-loop seed로 사용했다. Branch delta는 policy 한도 아래까지 감소했지만 직선 보간 경로의 clearance가 0 mm로 남았다.

따라서 다음처럼 해석한다.

> DP-CNN 출력 전체가 기구학적으로 무조건 불가능한 것은 아니지만, 현재 collision 의미와 직선 관절 보간 planner로는 안전한 실행 경로를 증명할 수 없다.

---

## 5. 근본 원인

### 5.1 학습 action과 실제 로봇 action의 차이

DP-CNN의 native action은 시뮬레이터 pusher의 `absolute mocap XY: float32[2]`다. 실제 SO-101은 5개 body joint를 움직여야 한다.

```text
DP-CNN absolute XY
  → physical Cartesian XYZ
  → 5-DOF IK
  → collision-free joint trajectory
```

DP-CNN은 학습 중 실제 SO-101의 관절 범위, elbow branch, 자기 충돌, 테이블 충돌, 현재 관절 seed를 직접 보지 않았다. 이 차이를 runtime bridge와 planner가 흡수해야 하지만 현재 구현은 그 역할을 완전히 수행하지 못한다.

### 5.2 가상 pusher와 실제 tool의 중복 표현

MuJoCo overlay에는 반지름 12 mm의 별도 가상 pusher sphere가 있다.

```xml
<body name="pusher" mocap="true" pos=".28 0 .045">
  <geom name="pusher" type="sphere" size=".012" .../>
</body>
```

동시에 collision checker는 다음을 모두 robot-object obstacle로 포함한다.

```python
objects = ["pusher", "push_t_bar", "push_t_stem"]
```

하지만 physical bridge는 DP-CNN의 pusher XY를 실제 gripper의 목표 XY로 사용한다. 결과적으로 다음 두 조건을 동시에 요구한다.

```text
gripper를 pusher 위치로 이동시킨다.
gripper와 pusher 사이에는 1 mm보다 큰 여유를 유지한다.
```

실제 tool이 시뮬레이터 pusher 역할을 대신한다면 이 sphere는 장애물이 아니라 task-state proxy다. 이를 장애물로 유지하면 목표 부근에서 0 mm clearance가 발생할 가능성이 매우 높다.

### 5.3 의도된 접촉과 위험한 충돌을 구분하지 않음

Push-T는 tool이 T 블록을 미는 접촉 작업이다. 현재 checker는 팔/손목이 T와 부딪히는 위험한 충돌과 지정된 tool-tip이 T를 미는 의도된 접촉을 구분하지 않는다. 모든 object contact를 동일하게 금지하면 실제 Push-T 동작 자체를 표현하기 어렵다.

### 5.4 장애물 회피 경로를 생성하지 않음

현재 `swept_collision_proof()`는 시작과 끝 관절값을 선형 보간해 검사한다.

```text
q(t) = q_start + t × (q_goal - q_start)
```

경로가 충돌하면 다른 경로를 찾지 않고 reject한다. 따라서 시작과 목표가 각각 안전해도 중간에 elbow나 wrist가 테이블 또는 물체에 접근하면 실패한다.

---

## 6. 왜 안전 기준을 낮추지 않았는가

빠른 우회는 collision clearance를 0 이하로 내리거나, branch delta 8도를 수십 도로 늘리거나, collision gate를 끄는 것이다. 그러나 이는 원인을 해결하지 않는다.

- `0 mm` clearance는 실제 접촉 또는 geometry 중첩 가능성을 뜻한다.
- 큰 branch jump는 단일 저속 step이 아닌 전체 팔 자세 전환이 될 수 있다.
- collision gate를 끄면 table/self/object 충돌을 구분할 수 없다.
- workspace 확장은 도달 가능성을 만들지 않는다.

따라서 숫자를 바꿔 GREEN처럼 보이게 하지 않고 실제 motor write 전에 fail-closed로 종료했다.

---

## 7. 재학습 없이 가능한 개선 방향

### 7.1 Collision semantics 수정

가상 pusher proxy와 실제 장애물을 분리한다.

| 충돌쌍 | 권장 처리 |
|---|---|
| 로봇 body ↔ table | 항상 금지 |
| 로봇 self collision | 항상 금지 |
| 팔/손목 ↔ T 블록 | 항상 금지 |
| 지정된 tool-tip ↔ T 블록 | 접촉 단계에서만 제한 허용 |
| 실제 tool ↔ 가상 pusher proxy | 장애물 검사에서 제외 |

이는 안전 기준 제거가 아니라 task-state proxy 때문에 생긴 중복 collision을 제거하고 위험한 접촉과 의도된 접촉을 구분하는 변경이다.

### 7.2 Waypoint motion planner

단일 직선 관절 보간 대신 다음 Cartesian waypoint를 사용한다.

```text
현재 자세
  → 안전 높이로 상승
  → 목표 XY 상공으로 이동
  → 접촉 높이까지 하강
  → 짧은 수평 push
```

각 구간에서 joint domain, IK residual, branch continuity, table/self collision, 접촉 단계의 allowed-contact 조건을 검증한다. DP-CNN checkpoint와 원래 action을 바꾸지 않고 중간 장애물을 우회할 수 있다.

### 7.3 Receding-horizon 미세 스텝

DP-CNN의 action chunk 전체를 한 번에 실행하지 않고 다음 loop를 사용한다.

1. 현재 이미지와 관절값 capture
2. DP-CNN 재추론
3. action 방향으로 허용된 작은 Cartesian step 제안
4. IK/collision 검사
5. Space를 누른 동안 한 step만 실행
6. HOLD 후 다시 capture

현재 bridge는 임의 clipping을 금지하므로 작은 step 변환은 숨겨진 clipping이 아니라 owner-signed execution policy와 receipt로 명시해야 한다.

### 7.4 보조안: Feasible-action projection

원래 action 주변에서 가장 가까운 reachable/collision-free 목표를 찾는 action shield를 둘 수 있다. 다만 원래 action을 변경하므로 허용 오차, 방향 보존, 최대 투영 거리와 실패 조건을 명확히 기록해야 한다. 앞의 방법으로 해결되지 않을 때만 권장한다.

### 7.5 비권장안

- collision threshold를 0 또는 음수로 설정
- branch limit를 근거 없이 수십 도로 확대
- scene object를 모두 collision model에서 제거
- fixture/random policy를 실제 DP-CNN처럼 사용
- 다른 model checkpoint로 알리지 않고 fallback

---

## 8. 개선 후 필요한 검증 순서

1. 가상 pusher 제외 전후 collision pair와 clearance 비교
2. tool-tip allowed-contact 단위 테스트
3. table/self/body-to-object collision 회귀 테스트
4. waypoint planner의 simulation QA
5. authentic DP-CNN 200-seed shadow 재실행
6. fresh camera + fresh joints 동일-process production shadow
7. `SHADOW_COMPLETE`, `cycles_completed=1`, `motor_writes_performed=false` 확인
8. shadow ledger에 바인딩된 single-step authorization 발급
9. torque-off 상태에서 ARMED terminal QA
10. 장면을 비운 상태에서 저속 Space-held 단일 step
11. T 블록을 배치하고 emergency STOP 접근 상태에서 접촉 QA

물리 QA 전까지 motor write는 0으로 유지해야 한다.

---

## 9. 종료 상태와 포맷 전 보존 자료

### 9.1 안전 종료 상태

```text
shoulder_pan: Torque_Enable = 0
shoulder_lift: Torque_Enable = 0
elbow_flex: Torque_Enable = 0
wrist_flex: Torque_Enable = 0
wrist_roll: Torque_Enable = 0
gripper: Torque_Enable = 0
```

### 9.2 보존 우선순위

1. 프로젝트 Git repository와 현재 변경사항
2. 네 200-episode model checkpoint
3. dataset manifest와 train/validation/test split digest
4. current-runtime DP-CNN recovered artifact
5. runtime lock과 dependency 정보
6. `sim_to_real_rollout`의 lineage, policy, receipt, ledger
7. 카메라 calibration과 joint-equivalence 자료
8. 본 문서

`04_experiments/`는 Git에서 제외된 runtime evidence와 대용량 artifact가 포함될 수 있으므로 Git repository만 백업해서는 충분하지 않다. Checkpoint와 sim-to-real evidence를 별도로 복사하고 hash를 확인해야 한다.

---

## 10. 연구 결과의 해석

### 확인된 것

- 200-episode DP-CNN checkpoint는 current runtime에서 authentic policy로 복구 가능하다.
- 실제 카메라 입력으로 DP-CNN inference가 가능하다.
- Simulation action을 physical Cartesian target으로 변환하는 receipted bridge가 동작한다.
- 실제 관절 seed에 대한 IK/branch/FK/collision fail-closed 경계가 동작한다.
- 위험한 후보에서는 authorization과 motor write가 발급되지 않는다.

### 확인하지 못한 것

- contact-aware planner를 통한 collision-free physical path
- `SHADOW_COMPLETE` 1-cycle production shadow
- 유효 authorization에 바인딩된 ARMED terminal
- 실제 DP-CNN motor single-step
- T 블록의 물리 이동 성공

### 보고서용 최종 한계 문장

> 본 연구에서는 200-episode DP-CNN의 current-runtime 복구와 실제 영상 기반 추론, receipted sim-to-real 변환 및 물리 actuation 직전의 안전 검증까지 완료했다. 그러나 시뮬레이터의 virtual pusher와 실제 SO-101 tool 사이의 collision 의미 중복, 의도된 tool contact와 위험한 object collision의 미분리, 그리고 직선 관절 보간만 지원하는 planner의 한계로 인해 정책 기준을 만족하는 physical single-step 경로를 증명하지 못했다. 이에 안전 기준을 임의로 완화하지 않고 motor write 없이 종료했으며, 후속 과제로 contact-aware allowed-collision matrix, waypoint IK planner, receding-horizon execution shield를 제안한다.

---

## 11. 최종 판단

```text
모델 출력이 존재한다
≠ IK 해가 존재한다
≠ 그 해까지의 경로가 안전하다
≠ 실제 로봇에 실행해도 된다
```

DP-CNN의 모델 lineage와 추론 자체는 보존됐고 실패 지점은 action bridge 이후의 physical motion-planning/contact-model 경계로 좁혀졌다. 컴퓨터 포맷 전에 새로운 planner를 급하게 구현하고 실제 로봇으로 검증하기보다, 현재 증거와 한계를 명확히 남긴 뒤 종료하는 것이 안전성과 연구 재현성 측면에서 적절하다.
