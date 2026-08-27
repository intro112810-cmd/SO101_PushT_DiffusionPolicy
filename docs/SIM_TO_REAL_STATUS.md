# Sim-to-Real 최종 상태 — 2026-08-27

## 판정

**Physical rollout 미완료 / 안전 종료**

| 항목 | 결과 |
|---|---:|
| Joint/FK pose | 15 / 15 |
| 최대 FK residual | 0.0 m |
| Camera fit / held-out RMSE | 0.328042 / 1.183719 px |
| Space deadman/HOLD, Esc STOP | 비작동 QA 확인 |
| Authentic DP-CNN | recovered v4 load/inference 확인 |
| Production shadow | HOLD, 0 cycle |
| Motor writes / actuation | false / false |
| Authorization / ARMED | 미발급 |
| Physical T-block movement | 미실행 |
| 종료 torque | 6개 motor 모두 disabled |

## 직접 중단 원인

Production shadow의 대표 rejection detail은 IK residual `0.194740 m`이다. 100,000개 IK 후보와 200개 authentic DP-CNN closed-loop seed를 추가 검사했지만 workspace, IK, branch 8도, swept clearance 1 mm를 동시에 통과한 후보가 없었다. Focused 후보는 branch delta를 3.26~5.26도로 줄였지만 swept clearance가 0 mm였다.

## 구조적 한계

1. DP-CNN은 simulation pusher의 absolute XY를 출력하지만 실제 SO-101은 5-DOF collision-free trajectory가 필요하다.
2. Virtual pusher sphere가 robot-object obstacle에 포함되어 실제 tool target과 collision 의미가 중복된다.
3. 의도된 tool-tip contact와 위험한 arm/object collision을 구분하지 않는다.
4. `swept_collision_proof()`는 직선 joint interpolation을 검사할 뿐 우회 waypoint를 생성하지 않는다.

## 재학습 없는 후속안

- Virtual pusher proxy를 obstacle에서 분리한다.
- Tool-tip ↔ T 접촉만 contact phase에서 제한적으로 허용한다.
- Lift → translate → descend → short push waypoint planner를 사용한다.
- 매 fresh observation마다 작은 receding-horizon step만 제안한다.
- Action projection은 owner-signed 최대 편차와 fail-closed 조건이 있을 때만 보조적으로 사용한다.

Collision threshold를 0으로 낮추거나 branch limit를 근거 없이 확대하는 방법은 사용하지 않는다.

## 증거

- [`../sim_to_real/evidence/shadow/terminal_receipt.json`](../sim_to_real/evidence/shadow/terminal_receipt.json)
- [`../sim_to_real/evidence/shadow/ledger.jsonl`](../sim_to_real/evidence/shadow/ledger.jsonl)
- [`../sim_to_real/evidence/lineage/compact-lineage-v4.json`](../sim_to_real/evidence/lineage/compact-lineage-v4.json)
- [`../sim_to_real/final_results/FINAL_STATUS.json`](../sim_to_real/final_results/FINAL_STATUS.json)
- [`SIM_TO_REAL_FINAL_2026-08-27.md`](SIM_TO_REAL_FINAL_2026-08-27.md)
