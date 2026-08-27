# 대용량 Artifact 저장·포맷 전 백업 정책

## 두 개의 필수 root

```text
Package:
/home/intro/InternLab/InTro_Lab_Handoff/InTro_SO101_PushT_Reproducibility_Package

NAS artifacts:
/home/intro/InternLab/InTro_Lab_Handoff/NAS_Artifacts/InTro_SO101_PushT
```

Package에는 코드, 문서, 작은 JSON/ledger/receipt와 checksum을 둔다. Dataset, 네 checkpoint, policy tensor와 대용량 runtime은 sibling NAS root에 둔다. **둘 중 하나만 복사하면 재현 패키지가 완전하지 않다.**

## 보존 대상

| NAS 경로 | 내용 |
|---|---|
| `datasets/native_store_200ep/` | 200ep canonical dataset |
| `models/*/training/final_checkpoint.ckpt` | 네 모델 최종 checkpoint |
| `models/*/bundle/policy.safetensors` | 네 inference tensor |
| `runtime/pushT-so100.git/` | pinned upstream identity |

DP-CNN source `latest.ckpt`는 NAS `models/dp_cnn/training/final_checkpoint.ckpt`와 동일 content digest로 복원한다. Recovered v4 metadata는 package `sim_to_real/evidence/recovered_dp_cnn/`에 있고 대형 tensor는 NAS `models/dp_cnn/bundle/policy.safetensors`에 있다.

## 포맷 전 절차

1. Package root와 NAS artifact root를 서로 다른 외장/NAS 위치에 각각 복사한다.
2. 복사본에서 `sha256sum --check integrity/SHA256SUMS_GIT`를 실행한다.
3. NAS 복사본 root에서 package의 `integrity/SHA256SUMS_NAS`를 검사한다.
4. 네 checkpoint, recovered DP-CNN tensor, dataset manifest가 존재하는지 확인한다.
5. `owner-signing-private-key.pem`은 handoff에 포함하지 않는다.
6. 원 프로젝트 Git repository와 Obsidian 최종 보고서도 별도로 보존한다.

## 복원

1. 두 root를 위 상대 구조로 복원한다.
2. `bash docs/verify_package.sh`를 실행한다.
3. Recovered loader가 source layout을 요구하면 DP-CNN final checkpoint를 `checkpoints/latest.ckpt`로 **복사**하고 digest `a7224ec4c8cd7172185f0160ede1944895eb9c5e6417cddc3d53ab40c3af84a8`을 확인한다.
4. 일반 재현 검증에서는 robot hardware command를 실행하지 않는다.

상세 mapping은 `integrity/NAS_ARTIFACT_MANIFEST.tsv`와 `sim_to_real/evidence/EVIDENCE_INDEX.json`에 있다.
