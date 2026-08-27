# SO-101 Push-T 데이터 수집 운영 안내

## 기록 항목

- 상단 카메라 `uint8[224,224,3]`
- 측면 카메라 `uint8[224,224,3]`
- 다섯 관절 상태 `float32[5]`
- absolute mocap XY 행동 `float32[2]`
- episode ID
- frame index
- 10 Hz timestamp

## 수집 전 확인

1. runtime 버전을 확인한다.
2. F710의 axis와 button 수를 확인한다.
3. 상단·측면 카메라가 모두 열리는지 확인한다.
4. 화면 출력 환경을 확인한다.
5. dataset 저장 위치와 남은 용량을 확인한다.
6. preflight가 통과하기 전에는 수집을 시작하지 않는다.

```bash
./data/collection/preflight.sh --dataset-root /path/to/dataset
```

## 조작 원칙

- 물체와 접촉하기 전 경로를 안정적으로 맞춘다.
- 지나치게 빠른 방향 전환을 피한다.
- 목표 위치에 도달한 뒤 최종 상태를 짧게 유지한다.
- 잘못된 조작은 승인 데이터에 포함하지 않는다.
- episode 사이에 물체와 로봇을 초기 상태로 복원한다.

## 수집 후 확인

- 양쪽 camera frame 수가 일치하는지 확인한다.
- state와 action row 수가 일치하는지 확인한다.
- timestamp가 증가하는지 확인한다.
- NaN 또는 무한대가 없는지 확인한다.
- episode 경계가 겹치지 않는지 확인한다.
- 승인된 episode만 canonical store로 import한다.

실제 수집은 preflight와 동일한 config 및 dataset root로 실행한다.

```bash
./data/collection/start_collection.sh --dataset-root /path/to/dataset
```

