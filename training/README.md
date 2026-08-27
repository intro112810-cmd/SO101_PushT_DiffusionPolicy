# 네 정책 학습

동일한 200 episode 데이터와 고정 split에서 다음 정책을 학습한다.

- DP-CNN
- DP-Transformer
- IBC
- LSTM-GMM

각 정책은 smoke test와 full training을 분리하고, dataset, split,
runtime, model profile과 checkpoint identity를 receipt에 기록한다.

```bash
./training/model_smoke.sh dp_cnn --fixture --output /path/to/smoke

./training/train_model.sh dp_cnn \
  --paper-view /path/to/frozen_dataset \
  --output /path/to/training_output \
  --artifact-id ARTIFACT_ID \
  --artifact-index configs/provenance/artifact_index.json \
  --full-production --max-updates 400000
```

