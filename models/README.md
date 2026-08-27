# 최종 모델과 대형 Artifact

네 모델의 training receipt, resolved config, bundle metadata와 evaluation metrics는 package에 포함한다. 대형 checkpoint와 policy tensor는 sibling NAS artifact root에 보존하고 SHA-256으로 연결한다.

| 모델 | 학습 | 최종 checkpoint |
|---|---:|---|
| DP-CNN | 400,000 updates | NAS 보존 |
| DP-Transformer | 400,000 updates | NAS 보존 |
| IBC | 100,000 updates | NAS 보존 |
| LSTM-GMM | 300,000 updates | NAS 보존 |

## Recovered DP-CNN

Physical-shadow candidate는 `local-dp_cnn-recovered-v4-seed0`이다. Authentic class는 `DiffusionUnetHybridImagePolicy`, observation은 96x96 `cam_top` single camera다. 이는 224x224 dual-camera four-model simulation comparison과 별도 실행 identity다.

Recovered tensor와 source `latest.ckpt`는 대형 artifact이므로 package에 중복 저장하지 않는다. 대신 `sim_to_real/evidence/recovered_dp_cnn/` metadata와 `integrity/NAS_ARTIFACT_MANIFEST.tsv`의 경로/checksum으로 보존한다. “latest”라는 이름을 이유로 source checkpoint를 삭제하면 lineage가 깨지므로 절대 삭제하지 않는다.
