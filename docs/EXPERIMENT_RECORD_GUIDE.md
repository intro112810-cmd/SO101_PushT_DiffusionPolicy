# 실험 기록 작성 기준

## 목적

실험 기록은 결과만 나열하지 않고 조건, 입력, 실행 방법, 관찰,
정량 결과, 한계와 재현 방법을 함께 남긴다.

## 실험별 필수 항목

1. 실험 목적
2. 수행 날짜
3. hardware와 runtime
4. dataset와 split identity
5. model과 checkpoint identity
6. 실행 명령
7. seed와 step 제한
8. 정량 결과
9. 대표 성공 사례
10. 대표 실패 사례
11. figure와 영상
12. 해석
13. 알려진 한계
14. 결과 파일 checksum

## Figure 작성 기준

- Figure 번호와 제목을 붙인다.
- 무엇을 비교하는지 caption에 적는다.
- 축, 단위, seed 수, episode 수를 명시한다.
- 성공 사례만 선택하지 않고 실패 사례도 함께 보여준다.
- 원본 수치 파일의 상대경로를 함께 적는다.
- 이미지에는 실험 해석에 필요한 정보만 표시한다.

예시:

```markdown
### Figure 3. 동일 초기 상태에서의 40개 확률적 행동 궤적

고정된 초기 observation에서 policy sampling seed만 변경하였다.
각 선은 첫 40 step의 absolute mocap XY 행동을 나타낸다.

![행동 궤적](../results/figures/DP_Transformer_multimodal_40rollouts.png)

원본 수치:
`../results/metrics/DP_Transformer_multimodal_40rollouts.json`
```

## 영상 작성 기준

- 파일명에 모델, 성공/실패, 평가 조건을 포함한다.
- 재생 속도를 원래 시간 대비 몇 배속인지 적는다.
- episode 전환과 최종 pose 확인 시간을 적는다.
- 영상에 사용한 rollout ID 또는 seed 목록을 별도 manifest로 둔다.
- README에는 HTML video와 일반 링크를 함께 제공한다.

## 결과 표현 기준

확인되지 않은 성능을 완료된 결과처럼 표현하지 않는다.

다음 세 상태를 구분한다.

- `확인`: 정본 receipt와 원본 수치가 존재함
- `중간 결과`: 특정 prototype 또는 제한된 조건의 결과
- `예정`: 최종 검증 후 추가할 내용


