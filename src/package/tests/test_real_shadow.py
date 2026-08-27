from __future__ import annotations

import numpy as np
import pytest

from so101_pusht_benchmark.real_shadow import (
    physical_degrees_to_shadow_agent_pos,
    physical_crop_to_checkpoint_image,
    validate_shadow_agent_pos,
)


def test_physical_crop_adapter_outputs_checkpoint_rgb_contract() -> None:
    bgr = np.zeros((400, 400, 3), dtype=np.uint8)
    bgr[:, :, 0] = 10
    bgr[:, :, 1] = 20
    bgr[:, :, 2] = 30

    image = physical_crop_to_checkpoint_image(bgr)

    assert image.shape == (96, 96, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert tuple(image[0, 0]) == (30, 20, 10)


def test_physical_crop_adapter_rotates_counterclockwise_to_training_view() -> None:
    bgr = np.zeros((400, 400, 3), dtype=np.uint8)
    bgr[:200, :200] = (10, 20, 240)

    image = physical_crop_to_checkpoint_image(bgr)

    assert tuple(image[80, 16]) == (240, 20, 10)
    assert tuple(image[16, 16]) == (0, 0, 0)


def test_shadow_agent_position_requires_exact_five_float32_values() -> None:
    position = np.zeros(5, dtype=np.float32)

    assert validate_shadow_agent_pos(position) is position

    with pytest.raises(ValueError, match="float32\\[5\\]"):
        validate_shadow_agent_pos(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="float32\\[5\\]"):
        validate_shadow_agent_pos(np.zeros(5, dtype=np.float64))


def test_physical_degrees_map_to_simulator_joint_order() -> None:
    physical = {
        "shoulder_pan": -1.0,
        "shoulder_lift": -90.0,
        "elbow_flex": 106.0,
        "wrist_flex": 32.0,
        "wrist_roll": -10.0,
        "gripper": 4.0,
    }

    mapped = physical_degrees_to_shadow_agent_pos(physical)

    assert mapped.dtype == np.float32
    assert mapped.shape == (5,)
    np.testing.assert_allclose(
        mapped,
        np.deg2rad([-1.0, -90.0, 106.0, 32.0, -10.0]),
        rtol=1e-6,
    )


def test_physical_joint_mapping_rejects_incomplete_receipt() -> None:
    with pytest.raises(ValueError, match="exactly"):
        physical_degrees_to_shadow_agent_pos(
            {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
            }
        )
