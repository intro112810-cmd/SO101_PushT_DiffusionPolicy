from __future__ import annotations

import pytest

from so101_pusht_benchmark.evaluation.professor_artifacts import (
    MODEL_ORDER,
    figure_filename,
    get_model_spec,
    reel_filename,
    validate_metrics_receipt,
)


def test_four_approved_model_specs_have_distinct_artifacts_and_labels() -> None:
    assert MODEL_ORDER == ("dp_cnn", "dp_transformer", "ibc", "lstm_gmm")
    specs = [get_model_spec(model) for model in MODEL_ORDER]

    assert [spec.label for spec in specs] == [
        "DP-CNN",
        "DP-Transformer",
        "IBC",
        "LSTM-GMM",
    ]
    assert len({spec.artifact_id for spec in specs}) == 4


def test_final_asset_names_are_model_specific() -> None:
    assert (
        reel_filename("ibc", success=True)
        == "2026-08-20_ibc_success_three_view_120s_4x_hold2s.mp4"
    )
    assert (
        reel_filename("lstm_gmm", success=False)
        == "2026-08-20_lstm_gmm_failure_three_view_120s_4x_hold2s.mp4"
    )
    assert (
        figure_filename("dp_cnn")
        == "2026-08-20_dp_cnn_paper_fixed_state_40steps.png"
    )


def test_metrics_validator_requires_exact_fixed_seed_contract() -> None:
    valid = {
        "model": "ibc",
        "evaluation_seeds": list(range(100000, 100100)),
        "eval/success_rate": 0.06,
        "eval/mean_dxy": 0.04,
        "eval/mean_dyaw": 20.0,
        "optimizer_updates": 100000,
        "rollouts": [
            {
                "seed": seed,
                "success": seed < 100006,
                "steps": 300,
                "dxy": 0.04,
                "dyaw": 20.0,
            }
            for seed in range(100000, 100100)
        ],
    }

    summary = validate_metrics_receipt(valid, expected_model="ibc")

    assert summary.success_count == 6
    assert summary.success_rate == 0.06

    invalid = {**valid, "evaluation_seeds": list(range(100000, 100099))}
    with pytest.raises(ValueError, match="evaluation seeds"):
        validate_metrics_receipt(invalid, expected_model="ibc")
