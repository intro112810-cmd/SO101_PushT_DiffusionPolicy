# 소프트웨어 요구사항

## 운영 환경

- Linux x86-64
- NVIDIA GPU와 호환 CUDA runtime
- Python 3.10
- MuJoCo 3.3.7
- LeRobot 0.4.4
- Gymnasium 1.2.2
- Feetech Servo SDK 1.0.0

## 환경 정의

```text
src/package/environments/sim-runtime.lock
src/package/environments/paper-baselines.conda-lock.yml
src/package/pyproject.toml
src/package/uv.lock
```

수집·시뮬레이션 평가 환경과 정책 학습 환경은 분리한다. 설치 후에는
lock digest와 실제 package version이 일치하는지 확인한다.

## 기본 검증

```bash
cd src/package
PYTHONPATH=src python -m so101_pusht_benchmark.cli --help
```

정확한 runtime이 준비된 환경에서는 다음 검사를 수행한다.

```bash
PYTHONPATH=src python -m so101_pusht_benchmark.cli inspect-env \
  --native-pusht-so100
```

