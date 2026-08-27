# 연구 수행 내용

## 1. SO-101 기반 환경 구축

SO-101 follower의 MotorBus, motor ID, baudrate, firmware와 calibration을
확인하고 안전 범위를 기록하였다. camera와 gamepad 입력을 연결해
시뮬레이션 조작과 데이터 수집 환경을 구성하였다.

## 2. 데이터 수집 체계

상단·측면 camera, 다섯 관절 상태와 평면 절대 목표 행동을 10 Hz로
기록하는 수집 경로를 구축하였다. episode 단위 승인, fault 기록,
atomic publish, dashboard와 정제 절차를 구성하였다.

## 3. Canonical dataset

서로 다른 수집 세션을 독립적으로 검사하고 200개 episode로 병합하였다.
43,314 frame의 native store와 불변 split을 생성하고 manifest 및
checksum으로 identity를 고정하였다.

## 4. 네 정책 비교

DP-CNN, DP-Transformer, IBC, LSTM-GMM을 동일 dataset과 evaluation
contract에 연결하였다. 학습, checkpoint, inference bundle, 100-seed
평가와 결과 시각화를 하나의 재현 경로로 구성하였다.

## 5. Sim-to-Real 최종 결과

실제 관절 상태와 시뮬레이션 관절 frame의 관계를 15개 pose로 검사하고 실제 camera와 table 좌표계를 held-out view로 검증했다. 200ep DP-CNN checkpoint를 current runtime의 authentic `DiffusionUnetHybridImagePolicy` v4 artifact로 복구해 fresh camera/joint/scene 입력으로 inference했다.

Production shadow는 IK residual, branch continuity, swept collision을 모두 만족하지 못해 fail-closed HOLD로 종료됐다. `cycles_completed=0`, `motor_writes_performed=false`, `actuation_performed=false`이며 authorization과 ARMED terminal은 발급하지 않았다. 종료 시 여섯 motor torque가 모두 0임을 확인했다.

중단 원인은 model load가 아니라 virtual pusher collision 의미 중복, 의도된 tool contact 미분리, 직선 joint interpolation만 지원하는 planner의 한계다. 재학습 없는 후속안은 allowed-contact matrix, waypoint IK, receding-horizon micro-step이다.

