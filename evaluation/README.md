# 평가 및 모델 비교

모든 정책은 동일한 환경 seed, 최대 step, 성공 판정과 metric으로
평가한다.

## 주요 지표

- 성공률
- terminal/minimum dxy
- terminal/minimum dyaw
- steps-to-success
- episode duration
- inference latency
- 성공·실패 seed 목록

결과는 CSV, JSON, Markdown과 대표 영상으로 함께 보존한다.

