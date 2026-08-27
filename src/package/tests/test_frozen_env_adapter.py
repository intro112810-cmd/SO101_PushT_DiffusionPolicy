from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from so101_pusht_benchmark.evaluation.frozen_env import (
    ActionContractError,
    FrozenPushTAdapter,
    JOINT_ORDER,
)


def _raw_observation() -> dict[str, object]:
    result: dict[str, object] = {
        "cam_top": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_side": np.ones((224, 224, 3), dtype=np.uint8),
    }
    result.update({name: float(index) for index, name in enumerate(JOINT_ORDER)})
    return result


@dataclass
class _RawEnvironment:
    steps: int = 0
    closes: int = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
        return _raw_observation(), {"seed": seed}

    def step(
        self, action: object
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        del action
        self.steps += 1
        return _raw_observation(), -0.01, True, True, {"dxy": 0.01, "dyaw": 2.0}

    def close(self) -> None:
        self.closes += 1


def test_adapter_orders_native_observation_and_termination_metrics() -> None:
    raw = _RawEnvironment()
    adapter = FrozenPushTAdapter(cast("object", raw))
    observation, info = adapter.reset(seed=100000)
    assert tuple(observation) == ("cam_top", "cam_side", "agent_pos")
    assert observation["cam_top"].dtype == np.uint8
    assert observation["cam_side"].shape == (224, 224, 3)
    assert observation["agent_pos"].dtype == np.float32
    assert observation["agent_pos"].tolist() == [0, 1, 2, 3, 4]
    assert info == {"seed": 100000}

    result = adapter.step(np.array([0.25, -0.25], dtype=np.float32))
    assert result.terminated is True
    assert result.truncated is True
    assert result.info == {"dxy": 0.01, "dyaw": 2.0}
    adapter.close()
    assert raw.steps == 1
    assert raw.closes == 1


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(3, dtype=np.float32),
        np.array([np.nan, 0], dtype=np.float32),
        np.array([1.01, 0], dtype=np.float32),
        np.zeros(2, dtype=np.float64),
        [0.0, 0.0],
    ],
)
def test_invalid_action_is_typed_and_never_steps(action: object) -> None:
    raw = _RawEnvironment()
    adapter = FrozenPushTAdapter(cast("object", raw))
    with pytest.raises(ActionContractError):
        adapter.step(action)
    assert raw.steps == 0
