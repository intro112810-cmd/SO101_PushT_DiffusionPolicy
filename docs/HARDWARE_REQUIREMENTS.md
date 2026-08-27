# 하드웨어 요구사항

## 데이터 수집

- Logitech F710 gamepad
- NVIDIA GPU가 장착된 Linux workstation
- 상단 및 측면 관측을 제공하는 두 camera
- Push-T MuJoCo runtime을 표시할 수 있는 graphical session

## 실제 로봇 진단

- SO-101 follower
- Feetech STS3215 motor 6개
- USB serial MotorBus
- 고정된 overhead camera
- 물리 Push-T 작업판
- Red T와 Green T
- camera registration용 ChArUco board
- Space와 Esc 입력이 가능한 keyboard

## 장치 확인 원칙

장치 node 번호만으로 camera나 MotorBus를 식별하지 않는다. USB identity,
calibration digest, resolution과 hardware profile을 함께 확인한다.
카메라 위치가 바뀌면 기존 camera registration은 유효하지 않다.

