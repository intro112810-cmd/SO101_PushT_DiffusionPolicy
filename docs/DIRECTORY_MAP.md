# 재현 패키지 폴더 구성 안내

## 기본 원칙

이 패키지는 기존 작업 공간의 전체 복사본이 아니다. 검증된 정본과
최종 재현 경로만 포함한다.

- 기존 파일은 이동하거나 수정하지 않는다.
- source, data, model, result를 분리한다.
- 중간 cache와 실패한 임시 실행은 포함하지 않는다.
- 데이터와 모델에는 checksum과 manifest를 함께 둔다.
- 실제 로봇 명령은 비작동 검증 경로와 분리한다.
- 모든 운영 문서는 한글로 작성한다.

## 폴더 구조

```text
InTro_SO101_PushT_Reproducibility_Package/
├── README.md
├── docs/
├── src/
├── data/collection/
├── data/datasets/
├── simulation/
├── 05_references/
├── training/
├── models/
├── evaluation/
├── sim_to_real/
├── results/
│   ├── figures/
│   └── videos/
├── docs/
├── third_party/
└── integrity/
```

## 폴더별 역할

### `src`

Python package, 실행 script, config, environment lock, type stub, test를
포함한다. 가상환경, cache, session 기록은 포함하지 않는다.

### `data/collection`

F710 입력, dual-camera 수집, episode 기록, collection dashboard,
episode 검증과 운영 절차를 포함한다.

### `data/datasets`

최종 200 episode native store, 고정 split, dataset card, episode manifest,
checksum을 포함한다.

### `simulation`

MuJoCo 환경 검증, 고정 seed rollout, 정책 재현, 영상 및 receipt 생성
절차를 포함한다.

### `training`

네 정책의 model profile, smoke test, full training, resume, runtime
검증 스크립트를 포함한다.

### `05_references`

Native Push-T collection과 evaluation에 필요한 고정 runtime source를
포함한다. 원 저작권과 라이선스는 `third_party`에서 관리한다.

### `models`

각 정책의 최종 checkpoint 하나, inference bundle, normalizer, resolved
config, training receipt를 포함한다.

### `evaluation`

100-seed 평가, 성공 판정, dxy/dyaw 계산, 네 모델 비교표 생성을
포함한다.

### `sim_to_real`

관절·카메라 calibration, read-only sample, shadow, authorization,
single-step, bounded rollout, ledger와 검증 결과를 포함한다.

### `results`

README에서 직접 보여줄 대표 figure와 영상을 포함한다. 원본 결과
전체가 아니라 연구 결론을 설명하는 검증된 대표 자산만 둔다.

### `docs`

연구 질문, 실험 설계, 데이터 수집 지침, 학습 지침, 평가 지침,
문제 해결 기록과 후속 과제를 포함한다.

### `third_party`

사용한 외부 구성요소의 버전, 고정 revision, 라이선스와 설치 방법을
포함한다.

### `integrity`

전체 파일 checksum, dataset lineage, model lineage, runtime identity와
최종 재현 receipt를 포함한다.

