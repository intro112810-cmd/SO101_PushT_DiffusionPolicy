# 실험 결과 자산 정리 목록

## 명명 규칙

```text
모델_평가조건_내용_추가정보.확장자
```

예:

```text
DP_Transformer_100seed_success_4x.mp4
DP_Transformer_100seed_failure_4x.mp4
DP_Transformer_multimodal_40rollouts.png
four_model_100seed_comparison.png
sim_to_real_camera_registration_heldout.png
sim_to_real_single_step_final.mp4
```

## 우선 배치할 Figure

| 대상 파일 | README 용도 | 상태 |
|---|---|---|
| `so101_pusht_3d_simulation_environment.png` | 직접 구축한 3D SO-101 MuJoCo 환경 | 포함 |
| `DP_Transformer_multimodal_40rollouts.png` | 확률적 policy의 행동 분포 | 포함 |
| `DP_Transformer_100seed_benchmark.png` | 100-seed 정량 결과 | 포함 |
| `dataset_representative_trajectories.png` | 수집 데이터 분포 | 포함 |
| `dataset_final_state_review.png` | 수집 데이터 최종 상태 | 포함 |
| `camera_registration_fit.png` | 실제 카메라 fit 결과 | 포함 |
| `camera_registration_heldout.png` | 실제 카메라 held-out 결과 | 포함 |
| `joint_fk_15poses.png` | 관절·FK 검증 | 포함 |

## 우선 배치할 영상

| 대상 파일 | README 용도 | 상태 |
|---|---|---|
| `DP_Transformer_success_reel.mp4` | 대표 성공 rollout | 포함 |
| `DP_Transformer_failure_reel.mp4` | 대표 실패 rollout | 포함 |
| `DP_Transformer_representative_rollout.mp4` | DP-Transformer 동작 특성 | 포함 |
| `IBC_representative_rollout.mp4` | IBC 동작 특성 | 포함 |
| `LSTM_GMM_representative_rollout.mp4` | LSTM-GMM 동작 특성 | 포함 |
| `DP_CNN_representative_rollout.mp4` | DP-CNN 동작 특성 | 포함 |
| `sim_to_real_shadow.mp4` | motor write 없는 실제 입력 검증 | 최종 검증 후 추가 |
| `sim_to_real_single_step.mp4` | 승인된 단일 명령 결과 | 실제 실행 후 추가 |

## 배치 원칙

- 외부 project의 예시 이미지와 영상은 복사하지 않는다.
- 직접 생성한 simulation, dataset, calibration, rollout 자산만 사용한다.
- 원본은 기존 작업 공간에 유지한다.
- 인계 패키지에는 검증된 복사본만 둔다.
- 복사 후 SHA-256을 생성해 원본과 동일한지 확인한다.
- README에 연결된 자산은 이름을 변경한 후 링크를 함께 갱신한다.


## Sim-to-Real 최종 상태

`sim_to_real_shadow.mp4`와 `sim_to_real_single_step.mp4`는 생성되지 않았다. Production shadow는 0 cycle HOLD로 종료됐고 physical step이 없었으므로, 빈 placeholder 영상으로 대체하지 않는다. 증거는 `sim_to_real/evidence/shadow/`의 ledger와 terminal receipt다.

## Attributed Paper Reference

| Asset | Source | Purpose |
|---|---|---|
| `paper_reference/diffusion_policy_figure3_multimodal_behavior.svg` | Diffusion Policy arXiv v5, Figure 3 | Compare original planar Push-T with this project’s 3D SO-101 adaptation |

Source and attribution metadata: [`paper_reference/README.md`](paper_reference/README.md).

## README Embedded Media

| Asset | Purpose |
|---|---|
| `figures/four_model_simulation_comparison.png` | Shared quantitative comparison across all four models |
| `videos/previews/DP_CNN_representative_preview.gif` | DP-CNN clean synchronized three-view success preview |
| `videos/previews/DP_Transformer_representative_preview.gif` | DP-Transformer clean synchronized three-view success preview |
| `videos/previews/IBC_representative_preview.gif` | IBC clean synchronized three-view failure preview |
| `videos/previews/LSTM_GMM_representative_preview.gif` | LSTM-GMM clean synchronized three-view failure preview |
| `videos/previews/three_view_preview_receipt.json` | Source hashes, dark-title removal threshold and output GIF validation |
