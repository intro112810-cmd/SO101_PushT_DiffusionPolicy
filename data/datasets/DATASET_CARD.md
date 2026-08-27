# SO-101 Push-T 200 Episode Dataset

## 개요

MuJoCo 기반 SO-101 Push-T 과제에서 수집한 200개 human demonstration으로
구성한다.

| 항목 | 값 |
|---|---|
| Episode 수 | 200 |
| Frame 수 | 43,314 |
| FPS | 10 |
| Camera | `cam_top`, `cam_side` |
| Camera 형식 | `uint8[224,224,3]` |
| State | `float32[5]` |
| Action | `float32[2]` absolute mocap XY |
| Canonical digest | `2c8563c716699cfb4c3fd05741abf4fc0d1b7098e4dcf1b43ebad2f5351dada3` |

## 관절 순서

1. Rotation
2. Pitch
3. Elbow
4. Wrist Pitch
5. Wrist Roll

## 품질 검사

- 두 camera의 frame 길이 일치
- state와 action row 수 일치
- episode 및 frame identifier의 순서 확인
- timestamp 단조 증가
- 비정상 수치 검사
- episode 경계 검사
- source와 split digest 고정

## 사용 범위

정책 학습, validation, 고정 seed simulation 평가에 사용한다. 실제
로봇 제어 권한을 부여하는 데이터로 해석하지 않는다.

