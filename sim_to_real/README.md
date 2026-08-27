# Sim-to-Real 최종 Handoff

## 실제 포함 구조

```text
sim_to_real/
├── evidence/
│   ├── lineage/
│   ├── recovered_dp_cnn/
│   ├── shadow/
│   └── runtime/
└── final_results/
    └── FINAL_STATUS.json
```

이 폴더는 성공한 physical rollout이 아니라 **실제 입력 기반 production shadow가 안전 gate에서 HOLD로 종료된 증거**를 보존한다.

## 최종 상태

- Authentic recovered DP-CNN v4 inference: 확인
- Fresh camera/joint/scene evidence: 확인
- Production shadow: `cycles_completed=0`, `terminal_state=HOLD`
- Policy motor writes: 0
- Physical actuation: 없음
- Authorization / ARMED terminal: 없음
- 종료 torque: 여섯 motor 모두 disabled

## 안전 경계

- 학습 checkpoint에는 hardware 권한이 없다.
- Gripper는 policy writer payload에 포함하지 않는다.
- Clipping, 자동 offset, 근거 없는 branch/range 확대를 허용하지 않는다.
- 다른 모델로 fallback하지 않는다.
- 이 package의 non-hardware 검증은 robot port를 열지 않는다.
- Private signing key는 포함하지 않는다.

## 읽기 순서

1. [`final_results/FINAL_STATUS.json`](final_results/FINAL_STATUS.json)
2. [`../docs/SIM_TO_REAL_STATUS.md`](../docs/SIM_TO_REAL_STATUS.md)
3. [`../docs/SIM_TO_REAL_FINAL_2026-08-27.md`](../docs/SIM_TO_REAL_FINAL_2026-08-27.md)
4. [`evidence/shadow/terminal_receipt.json`](evidence/shadow/terminal_receipt.json)
5. [`evidence/shadow/ledger.jsonl`](evidence/shadow/ledger.jsonl)
