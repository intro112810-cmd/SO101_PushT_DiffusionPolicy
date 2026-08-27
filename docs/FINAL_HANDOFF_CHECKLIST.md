# 2026-08-27 최종 Handoff Checklist

## 1. 먼저 실행

```bash
cd /home/intro/InternLab/InTro_Lab_Handoff/InTro_SO101_PushT_Reproducibility_Package
bash docs/verify_package.sh
bash docs/run_non_hardware_checks.sh
```

첫 명령은 package/NAS checksum과 final handoff manifest를 검사한다. 두 번째 명령은 robot에 연결하지 않고 CLI import/help만 확인한다.

## 2. 최종 판정

- Simulation benchmark: 완료
- 네 200ep checkpoint: NAS 보존
- Recovered DP-CNN v4: authentic load/inference 확인
- Physical production shadow: HOLD, 0 cycle
- Policy motor write: 0
- Torque: 종료 시 모두 disabled
- Physical Push-T: 미완료

## 3. 읽을 문서

1. [`../README.md`](../README.md)
2. [`../docs/EXPERIMENT_SUMMARY.md`](../docs/EXPERIMENT_SUMMARY.md)
3. [`../docs/SIM_TO_REAL_FINAL_2026-08-27.md`](../docs/SIM_TO_REAL_FINAL_2026-08-27.md)
4. [`../sim_to_real/final_results/FINAL_STATUS.json`](../sim_to_real/final_results/FINAL_STATUS.json)
5. [`ARTIFACT_STORAGE.md`](ARTIFACT_STORAGE.md)

## 4. 금지

- Safety threshold를 낮춰 shadow를 강제로 통과시키지 않는다.
- 다른 모델을 DP-CNN fallback으로 사용하지 않는다.
- `owner-signing-private-key.pem`을 package에 복사하지 않는다.
- 일반 재현 절차에서 robot motor command를 실행하지 않는다.

## 5. 재개 조건

재학습 없이 재개하려면 virtual pusher proxy collision 분리, tool-tip allowed contact, waypoint IK, receding-horizon shield를 먼저 simulation에서 검증한다. 그 후 fresh evidence로 production shadow를 처음부터 다시 발급한다.
