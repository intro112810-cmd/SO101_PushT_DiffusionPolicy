from __future__ import annotations

from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.contracts import (
    ContractError,
    build_dry_run_contract,
)


def follower_receipt() -> dict[str, object]:
    return {
        "schema": 1,
        "mode": "read_only_follower_state",
        "positions_degrees": {
            "shoulder_pan": -1.0,
            "shoulder_lift": -90.0,
            "elbow_flex": 106.0,
            "wrist_flex": 32.0,
            "wrist_roll": -10.0,
            "gripper": 1.0,
        },
        "raw_encoder": {
            "shoulder_pan": 2035,
            "shoulder_lift": 1001,
            "elbow_flex": 3108,
            "wrist_flex": 2491,
            "wrist_roll": 2087,
            "gripper": 1476,
        },
        "motor_writes_performed": False,
        "actuation_performed": False,
    }


def shadow_receipt() -> dict[str, object]:
    return {
        "schema": 1,
        "mode": "physical_frame_shadow_only",
        "model": "dp_cnn",
        "artifact_id": "local-dp_cnn-recovered-v3-seed0",
        "frame_sha256": "a" * 64,
        "checkpoint_image_contract": "CCW90 RGB uint8[96,96,3]",
        "agent_pos": [0.0, -1.5, 1.6, 0.5, -0.1],
        "agent_pos_source": "read_only_follower_degrees_direct_radians_provisional",
        "predicted_actions": [[0.25 + index * 0.001, 0.01] for index in range(8)],
        "deployment_valid": False,
        "actuation_performed": False,
        "follower_motor_writes_performed": False,
        "follower_actuation_performed": False,
    }


def test_valid_receipts_build_non_actuating_dry_run_contract() -> None:
    receipt = build_dry_run_contract(follower_receipt(), shadow_receipt())

    assert receipt["mode"] == "sim_to_real_dry_run"
    assert receipt["checkpoint_image_contract"] == "CCW90 RGB uint8[96,96,3]"
    assert len(cast("list[object]", receipt["predicted_actions"])) == 8
    assert receipt["deployment_valid"] is False
    assert receipt["motor_writes_performed"] is False
    assert receipt["actuation_performed"] is False


def test_tampered_follower_receipt_is_rejected_before_inference() -> None:
    follower = follower_receipt()
    follower["motor_writes_performed"] = True

    with pytest.raises(ContractError, match="read-only"):
        build_dry_run_contract(follower, shadow_receipt())
