# 외부 구성요소 고지

이 패키지는 아래 공개 구성요소를 사용한다. 각 구성요소의 원 저작권과
라이선스 조건은 `third_party/licenses/`에 보존한다.

| 구성요소 | 고정 버전 또는 revision | 사용 범위 | 라이선스 |
|---|---|---|---|
| Diffusion Policy | `5ba07ac6661db573af695b419a7947ecb704690f` | DP-CNN, DP-Transformer, IBC 정책 구현 | MIT |
| LeRobot | `0.4.4`, revision `e40b58a8dfa9e7b86918c374791599d070518d11` | 데이터 형식과 SO-101 연동 | Apache-2.0 |
| robomimic | `62ed2de905caeb9133136e4d14d810a8b6baa96c` | LSTM-GMM 정책 구현 | MIT |
| SO-101 reference assets | `fda892cba81032c46c40976a48c9ceadbf40a9ca` | 로봇 모델과 hardware reference | 원본 라이선스 참조 |
| MuJoCo | `3.3.7` | 물리 시뮬레이션 | Apache-2.0 |
| Gymnasium | `1.2.2` | 환경 인터페이스 | MIT |
| Feetech Servo SDK | `1.0.0` | MotorBus 통신 | 배포 package 라이선스 참조 |

본 연구에서 작성한 부분은 SO-101 Push-T 데이터 수집, canonical dataset,
공통 학습·평가 계약, artifact 검증, 시뮬레이션 재현, 실제 관절·카메라
정합과 안전 진단 제어 경로다.

