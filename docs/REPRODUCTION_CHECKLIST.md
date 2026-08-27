# 최종 재현·인수 확인 목록

## 패키지 무결성

- [ ] `bash docs/verify_package.sh`가 `PACKAGE_OK`와 `FINAL_HANDOFF_OK`를 출력한다.
- [ ] `integrity/FINAL_HANDOFF_MANIFEST.tsv`의 모든 in-package SHA-256이 일치한다.
- [ ] NAS 대형 artifact를 복원한 뒤 `integrity/SHA256SUMS_NAS`가 통과한다.
- [ ] private signing key가 package/NAS에 포함되지 않았음을 확인한다.

## 데이터·네 모델

- [ ] 200 episode / 43,314 frame / 180:20 split을 확인한다.
- [ ] DP-CNN, DP-Transformer, IBC, LSTM-GMM training receipt가 모두 completed다.
- [ ] 네 final checkpoint와 inference bundle의 NAS 경로·checksum을 확인한다.
- [ ] Simulation 비교 결과와 physical candidate identity를 혼동하지 않는다.

## Recovered DP-CNN

- [ ] `local-dp_cnn-recovered-v4-seed0` compact lineage가 valid다.
- [ ] Policy class가 authentic `DiffusionUnetHybridImagePolicy`다.
- [ ] Recovered artifact는 96x96 `cam_top` single-camera physical-shadow candidate임을 확인한다.
- [ ] 대형 `policy.safetensors`는 NAS/reference-only이며 package에는 metadata와 digest만 둔다.

## Sim-to-Real 최종 판정

- [x] Joint/FK evidence 확보
- [x] Camera registration evidence 확보
- [x] Keyboard interlock 비작동 QA 확보
- [x] Fresh synchronized sample과 authentic inference 확보
- [x] Production shadow가 fail-closed HOLD로 종료됨
- [x] `motor_writes_performed=false`, `actuation_performed=false` 확인
- [x] 종료 시 여섯 motor `Torque_Enable=0` 확인
- [ ] `SHADOW_COMPLETE` 1 cycle — **미달성**
- [ ] Single-step authorization — **미발급**
- [ ] ARMED terminal / physical step — **미실행**

## 포맷 전 백업

- [ ] 이 package 전체를 복사한다.
- [ ] sibling `NAS_Artifacts/InTro_SO101_PushT/` 전체를 별도 매체에 복사한다.
- [ ] 네 checkpoint, recovered DP-CNN tensor, dataset store를 hash 검증한다.
- [ ] `sim_to_real/evidence/`와 최종 보고서를 함께 보존한다.
- [ ] 원본 Obsidian report와 project Git repository도 별도 보존한다.

세부 경로는 [`ARTIFACT_STORAGE.md`](ARTIFACT_STORAGE.md), 중단 원인은 [`../docs/SIM_TO_REAL_FINAL_2026-08-27.md`](../docs/SIM_TO_REAL_FINAL_2026-08-27.md)를 참조한다.
