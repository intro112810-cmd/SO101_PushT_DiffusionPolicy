from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from so101_pusht_benchmark.core.contract import (
    ContractError,
    Observation,
    PolicyInput,
    TimingContract,
)


def image() -> NDArray[np.uint8]:
    return np.zeros((96, 96, 3), dtype=np.uint8)


def test_observation_and_action_require_exact_numpy_float32_arrays() -> None:
    with pytest.raises(ContractError, match=r"numpy\.ndarray"):
        Observation.parse({"observation.images.front": image(), "observation.state": [0.0] * 15})
    with pytest.raises(ContractError, match="float32"):
        Observation.parse(
            {
                "observation.images.front": image(),
                "observation.state": np.zeros(15, dtype=np.float64),
            }
        )
    with pytest.raises(ContractError, match=r"numpy\.ndarray"):
        PolicyInput.parse({"action": [0.2, 0.0, 0.05]})
    with pytest.raises(ContractError, match="float32"):
        PolicyInput.parse({"action": np.zeros(3, dtype=np.bool_)})


def test_frame_index_rejects_float_and_bool() -> None:
    with pytest.raises(ContractError):
        TimingContract.create(cast("int", 1.0), 0.1)
    bad_bool = True
    with pytest.raises(ContractError):
        TimingContract.create(cast("int", bad_bool), 0.1)
