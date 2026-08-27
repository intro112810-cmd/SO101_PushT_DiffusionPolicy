from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from so101_pusht_benchmark.core.contract import ContractError, Observation, PolicyInput


def good_image() -> NDArray[np.uint8]:
    return np.zeros((96, 96, 3), dtype=np.uint8)


def test_observation_contract_freezes_image_state_and_joint_order() -> None:
    observation = Observation.parse(
        {
            "observation.images.front": good_image(),
            "observation.state": np.zeros(15, dtype=np.float32),
        }
    )
    assert observation.front is not None
    assert observation.front.shape == (96, 96, 3)
    assert observation.state == (0.0,) * 15
    assert observation.joint_names == Observation.JOINT_NAMES


def test_schema3_observation_contract_topdown() -> None:
    observation = Observation.parse(
        {
            "observation.images.topdown": good_image(),
            "observation.state": np.zeros(15, dtype=np.float32),
        }
    )
    assert observation.topdown is not None
    assert observation.topdown.shape == (96, 96, 3)
    assert observation.front is None
    assert observation.state == (0.0,) * 15


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(14, dtype=np.float32),
        np.array([0.0] * 14 + [math.nan], dtype=np.float32),
        np.zeros(16, dtype=np.float32),
    ],
)
def test_bad_state_is_rejected(state: NDArray[np.float32]) -> None:
    with pytest.raises(ContractError):
        Observation.parse({"observation.images.front": good_image(), "observation.state": state})


def test_policy_allowlist_rejects_telemetry_and_unknown_keys() -> None:
    with pytest.raises(ContractError):
        PolicyInput.parse(
            {"action": np.array([0.2, 0.1, 0.05], dtype=np.float32), "telemetry.coverage": 1.0}
        )
    with pytest.raises(ContractError):
        PolicyInput.parse({"action": np.array([0.2, 0.1, 0.05], dtype=np.float32), "extra": 1})


def test_action_is_float32_three_meter_coordinates_and_bounded() -> None:
    action = PolicyInput.parse({"action": np.array([0.2, -0.1, 0.05], dtype=np.float32)})
    assert action.action == pytest.approx((0.2, -0.1, 0.05))
    assert action.bounds == ((0.18, 0.38), (-0.16, 0.16), (0.030, 0.100))
    for invalid in (
        np.array([float("inf"), 0.0, 0.05], dtype=np.float32),
        np.array([2.0, 0.0, 0.05], dtype=np.float32),
        np.array([0.2, 0.0, 0.029], dtype=np.float32),
        np.array([0.2, 0.0, 0.101], dtype=np.float32),
        np.array([0.2, 0.0], dtype=np.float32),
    ):
        with pytest.raises(ContractError):
            PolicyInput.parse({"action": invalid})
