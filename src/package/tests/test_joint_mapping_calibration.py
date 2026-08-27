from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image

from so101_pusht_benchmark.sim_to_real.joint_mapping import (
    affine_map_without_clipping,
    build_joint_mapping_receipt,
    calibrated_degree_range,
)


BENCHMARK = Path(__file__).resolve().parents[1]
MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
CALIBRATION = {
    "shoulder_pan": {"range_min": 1022, "range_max": 3091},
    "shoulder_lift": {"range_min": 849, "range_max": 3194},
    "elbow_flex": {"range_min": 700, "range_max": 3095},
    "wrist_flex": {"range_min": 999, "range_max": 3264},
    "wrist_roll": {"range_min": 300, "range_max": 4095},
    "gripper": {"range_min": 1458, "range_max": 2963},
}
FOLLOWER_RECEIPT = {
    "schema": 1,
    "mode": "read_only_follower_state",
    "positions_degrees": {
        "shoulder_pan": -1.89010989010989,
        "shoulder_lift": -89.71428571428571,
        "elbow_flex": 106.41758241758242,
        "wrist_flex": 31.604395604395606,
        "wrist_roll": -9.714285714285714,
        "gripper": 1.196013289036545,
    },
    "motor_writes_performed": False,
    "actuation_performed": False,
}
MUJOCO_RANGES = {
    "shoulder_pan": (-1.9198621771937616, 1.9198621771937634),
    "shoulder_lift": (-1.7453292519943224, 1.7453292519943366),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.6580628494556928, 1.6580627293335335),
    "wrist_roll": (-2.7438472969992493, 2.841206309382605),
}


def test_affine_mapping_preserves_out_of_range_value_without_clipping() -> None:
    degree_min, degree_max = calibrated_degree_range(700, 3095)

    mapped = affine_map_without_clipping(
        106.41758241758242,
        degree_min=degree_min,
        degree_max=degree_max,
        q_min=-1.69,
        q_max=1.69,
    )

    assert degree_min == pytest.approx(-105.27472527472527)
    assert degree_max == pytest.approx(105.27472527472527)
    assert mapped == pytest.approx(1.7083465553235908)
    assert mapped > 1.69


def test_current_elbow_marks_both_physical_and_mujoco_blockers() -> None:
    receipt = build_joint_mapping_receipt(
        calibration=CALIBRATION,
        follower_receipt=FOLLOWER_RECEIPT,
        mujoco_ranges=MUJOCO_RANGES,
    )

    elbow = receipt["joints"]["elbow_flex"]

    assert elbow["physical_degree"] == pytest.approx(106.41758241758242)
    assert elbow["physical_degree_range"][1] == pytest.approx(105.27472527472527)
    assert elbow["mapped_q_radians"] == pytest.approx(1.7083465553235908)
    assert elbow["mujoco_range_radians"][1] == pytest.approx(1.69)
    assert elbow["valid"] is False
    assert elbow["blockers"] == [
        "physical_calibration_range_exceeded",
        "mujoco_joint_range_exceeded",
    ]


def test_invalid_mapping_receipt_fails_closed_without_auditing_equivalence() -> None:
    receipt = build_joint_mapping_receipt(
        calibration=CALIBRATION,
        follower_receipt=FOLLOWER_RECEIPT,
        mujoco_ranges=MUJOCO_RANGES,
    )

    assert receipt["joint_order"] == list(MOTOR_ORDER)
    assert receipt["mapping_status"] == "invalid_blocked"
    assert receipt["joint_frame_equivalence"] == "not_audited_single_static_pose"
    assert receipt["deployment_valid"] is False
    assert receipt["clipping_performed"] is False
    assert receipt["motor_writes_performed"] is False
    assert receipt["actuation_performed"] is False
    assert receipt["blockers"] == [
        "elbow_flex:physical_calibration_range_exceeded",
        "elbow_flex:mujoco_joint_range_exceeded",
    ]


def test_mapping_requires_exact_five_mujoco_joint_ranges() -> None:
    incomplete_ranges = dict(MUJOCO_RANGES)
    incomplete_ranges.pop("wrist_roll")

    with pytest.raises(ValueError, match="exactly five"):
        build_joint_mapping_receipt(
            calibration=CALIBRATION,
            follower_receipt=FOLLOWER_RECEIPT,
            mujoco_ranges=incomplete_ranges,
        )


def test_joint_mapping_surface_contains_no_motor_write_path() -> None:
    surface = (
        BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/joint_mapping.py",
        BENCHMARK / "scripts/calibrate_joint_mapping_read_only.py",
    )
    forbidden = (
        "send_action",
        "sync_write",
        "Goal_Position",
        "enable_torque",
        "Torque_Enable",
        "configure_motors",
    )

    source_parts = [path.read_text(encoding="utf-8") for path in surface]
    source = "\n".join(source_parts)

    assert all(symbol not in source for symbol in forbidden)
    assert "DIAGNOSTIC INVALID - NO COMMAND" in source_parts[1]


def test_read_only_cli_generates_invalid_receipt_and_diagnostic_png(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    follower_path = tmp_path / "follower.json"
    output_dir = tmp_path / "output"
    calibration_path.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    follower_path.write_text(json.dumps(FOLLOWER_RECEIPT), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "scripts/calibrate_joint_mapping_read_only.py"),
            "--calibration-file",
            str(calibration_path),
            "--follower-state",
            str(follower_path),
            "--output-dir",
            str(output_dir),
            "--fixture-only",
        ],
        check=True,
        env={
            **os.environ,
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": str(BENCHMARK / "src"),
        },
        timeout=30,
    )

    receipt = json.loads((output_dir / "joint_mapping_receipt.json").read_text(encoding="utf-8"))
    with Image.open(output_dir / "physical_front_joint_mapping_diagnostic.png") as diagnostic:
        assert diagnostic.size == (480, 480)
    assert receipt["mapping_status"] == "invalid_blocked"
    assert receipt["evidence_scope"] == "test_fixture_only"
    assert receipt["deployment_valid"] is False
    assert receipt["clipping_performed"] is False
