from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from so101_pusht_benchmark.integrations.paper_baselines.configs import (
    PROFILES,
    PolicyNamespaceError,
    workspace_config,
)
from so101_pusht_benchmark.integrations.paper_baselines.runner import PaperBaselineRunner
from so101_pusht_benchmark.sim.env import PushTEnv


@dataclass(frozen=True, slots=True)
class _Result:
    observation: dict[str, NDArray[np.generic]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


def _native_observation() -> dict[str, NDArray[np.generic]]:
    return {
        "cam_top": np.full((224, 224, 3), 255, dtype=np.uint8),
        "cam_side": np.zeros((224, 224, 3), dtype=np.uint8),
        "agent_pos": np.arange(5, dtype=np.float32),
    }


class _Environment:
    def __init__(self, observation: dict[str, NDArray[np.generic]]) -> None:
        self.observation = observation
        self.steps = 0
        self.closed = False
        self.reset_calls = 0

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
        assert seed == 100000
        self.reset_calls += 1
        return self.observation, {"seed": seed}

    def step(self, action: object) -> _Result:
        value = cast("NDArray[np.float32]", action)
        assert value.shape == (2,)
        assert value.dtype == np.dtype(np.float32)
        self.steps += 1
        return _Result(
            self.observation,
            0.0,
            True,
            False,
            {"dxy": 0.0, "dyaw": 0.0},
        )

    def close(self) -> None:
        self.closed = True


class _Policy(BaseImagePolicy):
    def __init__(self, action: torch.Tensor | None = None) -> None:
        module_init = cast("Callable[[torch.nn.Module], None]", torch.nn.Module.__init__)
        module_init(self)
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.action = torch.zeros((1, 1, 2), dtype=torch.float32) if action is None else action
        self.calls = 0

    def reset(self) -> None:
        return None

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls += 1
        assert tuple(obs_dict) == ("cam_top", "cam_side", "agent_pos")
        assert obs_dict["cam_top"].shape == (1, 2, 3, 224, 224)
        assert obs_dict["cam_side"].shape == (1, 2, 3, 224, 224)
        assert obs_dict["agent_pos"].shape == (1, 2, 5)
        assert all(value.dtype is torch.float32 for value in obs_dict.values())
        assert torch.equal(obs_dict["cam_top"], torch.ones_like(obs_dict["cam_top"]))
        assert torch.equal(obs_dict["cam_side"], torch.zeros_like(obs_dict["cam_side"]))
        return {"action": self.action}


def test_native_runner_requires_explicit_factory_and_rejects_legacy_default(tmp_path: Path) -> None:
    with pytest.raises(PolicyNamespaceError, match="native environment factory is required"):
        PaperBaselineRunner(tmp_path)
    with pytest.raises(PolicyNamespaceError, match="legacy custom simulator"):
        PaperBaselineRunner(tmp_path, options={"native_env_factory": PushTEnv})
    with pytest.raises(PolicyNamespaceError, match="legacy env_factory option"):
        PaperBaselineRunner(
            tmp_path,
            options={"env_factory": lambda: _Environment(_native_observation())},
        )


def test_all_workspace_profiles_target_native_runner_without_legacy_fallback() -> None:
    for name in PROFILES:
        config = workspace_config(name, "/artifact/native-view", 0)
        task = cast("dict[str, object]", config["task"])
        runner = cast("dict[str, object]", task["env_runner"])
        assert runner["_target_"] == (
            "so101_pusht_benchmark.integrations.paper_baselines.runner.PaperBaselineRunner"
        )
        assert runner["options"] == {"native_env_factory": None}
        assert "PushTEnv" not in repr(runner)
        assert "sim.env" not in repr(runner)


def test_native_runner_converts_hwc_uint8_to_ordered_chw_float32(tmp_path: Path) -> None:
    environment = _Environment(_native_observation())
    policy = _Policy()
    runner = PaperBaselineRunner(
        tmp_path,
        evaluation_seeds=(100000,),
        n_obs_steps=2,
        n_action_steps=1,
        options={"max_steps": 1, "native_env_factory": lambda: environment},
    )
    result = runner.run(policy)
    assert policy.calls == 1
    assert environment.steps == 1
    assert environment.closed
    assert result["eval/success_rate"] == 1.0


def _missing_side() -> dict[str, NDArray[np.generic]]:
    value = _native_observation()
    value.pop("cam_side")
    return value


def _wrong_order() -> dict[str, NDArray[np.generic]]:
    value = _native_observation()
    return {
        "cam_side": value["cam_side"],
        "cam_top": value["cam_top"],
        "agent_pos": value["agent_pos"],
    }


def _extra_key() -> dict[str, NDArray[np.generic]]:
    value = _native_observation()
    value["unknown"] = np.zeros(1, dtype=np.float32)
    return value


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_missing_side, "keys/order"),
        (_wrong_order, "keys/order"),
        (_extra_key, "keys/order"),
        (
            lambda: {**_native_observation(), "cam_top": np.zeros((3, 224, 224), dtype=np.uint8)},
            "cam_top.*HWC uint8",
        ),
        (
            lambda: {
                **_native_observation(),
                "cam_side": np.zeros((224, 224, 3), dtype=np.float32),
            },
            "cam_side.*HWC uint8",
        ),
        (
            lambda: {**_native_observation(), "agent_pos": np.zeros(15, dtype=np.float32)},
            "agent_pos.*float32.*5",
        ),
        (
            lambda: {**_native_observation(), "agent_pos": np.zeros(5, dtype=np.float64)},
            "agent_pos.*float32.*5",
        ),
    ],
)
def test_native_runner_rejects_malformed_observation_before_policy_inference(
    tmp_path: Path,
    mutate: Callable[[], dict[str, NDArray[np.generic]]],
    match: str,
) -> None:
    environment = _Environment(mutate())
    policy = _Policy()
    runner = PaperBaselineRunner(
        tmp_path,
        evaluation_seeds=(100000,),
        options={"max_steps": 1, "native_env_factory": lambda: environment},
    )
    with pytest.raises(PolicyNamespaceError, match=match):
        runner.run(policy)
    assert policy.calls == 0
    assert environment.steps == 0
    assert environment.closed
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("action", "error", "match"),
    [
        (torch.zeros((1, 1, 2), dtype=torch.float64), ValueError, "exact float32"),
        (torch.tensor([[[np.nan, 0.0]]], dtype=torch.float32), ValueError, "finite"),
        (torch.tensor([[[np.inf, 0.0]]], dtype=torch.float32), ValueError, "finite"),
        (torch.tensor([[[1.01, 0.0]]], dtype=torch.float32), ValueError, "bounds"),
        (torch.zeros((1, 2), dtype=torch.float32), RuntimeError, "shape"),
        (torch.zeros((1, 1, 3), dtype=torch.float32), RuntimeError, "shape"),
    ],
    ids=("float64", "nan", "inf", "out-of-range", "wrong-rank", "wrong-width"),
)
def test_native_runner_rejects_invalid_policy_action_before_environment_step(
    tmp_path: Path,
    action: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    environment = _Environment(_native_observation())
    runner = PaperBaselineRunner(
        tmp_path,
        evaluation_seeds=(100000,),
        options={"max_steps": 1, "native_env_factory": lambda: environment},
    )

    with pytest.raises(error, match=match):
        runner.run(_Policy(action))

    assert environment.steps == 0
    assert environment.closed
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_full_production_config_injects_frozen_factory_marker() -> None:
    from so101_pusht_benchmark.integrations.paper_baselines.configs import workspace_config
    from so101_pusht_benchmark.training.launcher import full_production_config

    production = full_production_config(workspace_config("dp_cnn", "/artifact/native-view", 0))
    runner = cast("dict[str, object]", production["task"]["env_runner"])
    assert runner["options"]["native_env_factory"] == "frozen"
    assert runner["evaluation_seeds"] == []


def test_runner_accepts_frozen_factory_marker(tmp_path: Path) -> None:
    runner = PaperBaselineRunner(
        tmp_path,
        evaluation_seeds=(100000,),
        n_obs_steps=2,
        n_action_steps=1,
        options={"max_steps": 1, "native_env_factory": "frozen"},
    )
    environment = runner.env_factory()
    assert type(environment).__name__ == "FrozenPushTAdapter"


def test_runner_defers_rollout_with_empty_seeds(tmp_path: Path) -> None:
    environment = _Environment(_native_observation())
    runner = PaperBaselineRunner(
        tmp_path,
        evaluation_seeds=(),
        n_obs_steps=2,
        n_action_steps=1,
        options={"max_steps": 1, "native_env_factory": lambda: environment},
    )
    result = runner.run(_Policy())
    assert result == {"rollouts": [], "deferred": True}
    assert environment.reset_calls == 0
