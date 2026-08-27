# 시뮬레이션 재현

이 폴더에는 MuJoCo 환경 검증과 저장 정책 rollout 절차를 둔다.

## 재현 단계

1. runtime identity 확인
2. MuJoCo 환경 reset
3. 상단·측면 camera render
4. 관절 상태와 action schema 확인
5. 고정 seed에서 정책 rollout
6. trajectory, metric, video 저장
7. 기존 receipt와 결과 비교

실행 스크립트는 정본 소스코드를 복사한 뒤 추가한다.

```bash
./simulation/verify_environment.sh /path/to/evidence
./simulation/rollout_policy.sh dp_cnn /path/to/output \
  --artifact-id ARTIFACT_ID \
  --artifact-index configs/provenance/artifact_index.json
```

