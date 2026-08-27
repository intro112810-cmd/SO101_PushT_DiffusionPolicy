# 데이터셋 구성

이 폴더에는 최종 200 episode 정본과 재현에 필요한 manifest를 둔다.

## 예정 구성

```text
data/datasets/
├── DATASET_CARD.md
├── native_store_200ep/
├── frozen_split_200ep/
├── episode_manifest.json
├── split_manifest.json
└── checksums.sha256
```

## 확인된 정본 정보

- episode: 200
- frame: 43,314
- camera: `cam_top`, `cam_side`
- state: `agent_pos[5]`
- action: `absolute_mocap_xy[2]`
- FPS: 10
- canonical digest:
  `2c8563c716699cfb4c3fd05741abf4fc0d1b7098e4dcf1b43ebad2f5351dada3`

실제 데이터 복사는 source dataset 감사, split, 용량과 checksum을 다시
확인한 뒤 수행한다.

