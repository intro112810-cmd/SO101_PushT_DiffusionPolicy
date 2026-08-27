from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import subprocess
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from so101_pusht_benchmark.evaluation.frozen_env import FrozenStep
from so101_pusht_benchmark.integrations.paper_baselines.configs import PROFILES
from so101_pusht_benchmark.integrations.paper_baselines.runner import (
    PaperBaselineRunner,
    policy_seed,
)
from so101_pusht_benchmark.native_runtime import NativeRuntimeError, NativeRuntimeReport
from so101_pusht_benchmark.training.artifacts import ArtifactIndex
from so101_pusht_benchmark.training.evaluator import (
    EvaluationDependencies,
    EvaluationRequest,
    evaluate_bundle,
)
from so101_pusht_benchmark.training import evaluator as evaluator_module

_NumpyState = tuple[str, NDArray[np.uint32], int, int, float]
_numpy_get_state = cast("Callable[[], _NumpyState]", vars(np.random)["get_state"])
_numpy_random = cast("Callable[[], float]", vars(np.random)["random"])
_python_random = cast("Callable[[], float]", vars(random)["random"])


def test_evaluation_timing_wraps_synchronized_runner_result() -> None:
    times = iter((10.0, 12.5))
    synchronization: list[str] = []

    result = evaluator_module.timed_runner_result(
        lambda: {"eval/success_rate": 0.5},
        lambda: next(times),
        lambda: synchronization.append("sync"),
    )

    assert result == {"eval/success_rate": 0.5, "wall_time_s": 2.5}
    assert synchronization == ["sync", "sync"]


def _observation() -> dict[str, NDArray[np.generic]]:
    return {
        "cam_top": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_side": np.zeros((224, 224, 3), dtype=np.uint8),
        "agent_pos": np.zeros(5, dtype=np.float32),
    }


@dataclass
class _Environment:
    terminate_at: int = 2

    def __post_init__(self) -> None:
        self.seeds: list[int] = []
        self.actions: list[list[float]] = []
        self.steps = 0
        self.closed = 0

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
        assert seed is not None
        self.seeds.append(seed)
        self.steps = 0
        return _observation(), {}

    def step(self, action: object) -> FrozenStep:
        value = cast("NDArray[np.float32]", action)
        self.actions.append(value.tolist())
        self.steps += 1
        return FrozenStep(
            _observation(),
            -0.01,
            self.steps >= self.terminate_at,
            self.steps >= 300,
            {"dxy": 0.02, "dyaw": 3.0},
        )

    def close(self) -> None:
        self.closed += 1


class _Policy(BaseImagePolicy):
    def __init__(self, horizon: int) -> None:
        torch.nn.Module.__init__(self)
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.horizon = horizon

    def reset(self) -> None:
        return None

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del obs_dict
        action = torch.zeros((1, self.horizon, 2), dtype=torch.float32)
        action[0, :, 0] = torch.arange(self.horizon, dtype=torch.float32) / 100
        return {"action": action.to(self.anchor.device)}


class _ObservationTraceEnvironment(_Environment):
    steps: int

    def _observation_at_step(self) -> dict[str, NDArray[np.generic]]:
        value = int(self.steps)
        return {
            "cam_top": np.full((224, 224, 3), value, dtype=np.uint8),
            "cam_side": np.full((224, 224, 3), value, dtype=np.uint8),
            "agent_pos": np.full(5, value, dtype=np.float32),
        }

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
        super().reset(seed)
        return self._observation_at_step(), {}

    def step(self, action: object) -> FrozenStep:
        value = cast("NDArray[np.float32]", action)
        self.actions.append(value.tolist())
        self.steps += 1
        return FrozenStep(
            self._observation_at_step(),
            -0.01,
            self.steps >= self.terminate_at,
            False,
            {"dxy": 0.02, "dyaw": 3.0},
        )


class _RecurrentObservationTracePolicy(RobomimicImagePolicy):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.observed_steps: list[int] = []

    def reset(self) -> None:
        return None

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = obs_dict["agent_pos"][:, 0, ...]
        assert state.shape == (1, 5)
        self.observed_steps.append(int(state[0, 0].item()))
        return {"action": torch.zeros((1, 1, 2), dtype=torch.float32)}


def test_lstm_runner_forwards_current_observation_not_oldest_history(tmp_path: Path) -> None:
    environment = _ObservationTraceEnvironment(terminate_at=3)
    policy = _RecurrentObservationTracePolicy()
    runner = PaperBaselineRunner(
        tmp_path / "lstm-current-observation",
        evaluation_seeds=(100000,),
        n_obs_steps=10,
        n_action_steps=1,
        options={
            "native_env_factory": cast("Callable[[], object]", lambda: environment),
            "max_steps": 300,
        },
    )

    runner.run(policy)

    assert policy.observed_steps == [0, 1, 2]


@pytest.mark.parametrize("model", tuple(PROFILES))
def test_runner_uses_profile_prefix_and_native_metrics(tmp_path: Path, model: str) -> None:
    profile = PROFILES[model]
    environment = _Environment(terminate_at=profile.executed_actions + 1)
    runner = PaperBaselineRunner(
        tmp_path / model,
        evaluation_seeds=(100000,),
        n_obs_steps=profile.observation_steps,
        n_action_steps=profile.executed_actions,
        options={
            "native_env_factory": cast("Callable[[], object]", lambda: environment),
            "max_steps": 300,
        },
    )
    result = runner.run(_Policy(profile.horizon))
    rollout = cast("list[dict[str, object]]", result["rollouts"])[0]
    assert environment.seeds == [100000]
    assert len(environment.actions) == profile.executed_actions + 1
    assert rollout == {
        "seed": 100000,
        "policy_seed": policy_seed(100000),
        "success": True,
        "dxy": 0.02,
        "dyaw": 3.0,
        "duration_s": (profile.executed_actions + 1) / 10,
        "steps": profile.executed_actions + 1,
        "terminated": True,
        "truncated": False,
    }
    assert environment.closed == 1


class _StochasticPolicy(_Policy):
    def __init__(self) -> None:
        super().__init__(1)
        self.generator = torch.Generator()

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del obs_dict
        value = (
            _python_random()
            + _numpy_random()
            + float(torch.rand(()).item())
            + float(torch.rand((), generator=self.generator).item())
        ) / 4
        return {"action": torch.tensor([[[value, 0.0]]], dtype=torch.float32)}


class _ActionMetricEnvironment(_Environment):
    def step(self, action: object) -> FrozenStep:
        value = cast("NDArray[np.float32]", action)
        self.actions.append(value.tolist())
        self.steps += 1
        return FrozenStep(
            _observation(),
            -0.01,
            True,
            False,
            {"dxy": float(value[0]), "dyaw": float(value[1])},
        )


def test_policy_rng_is_seed_derived_repeatable_and_globally_isolated(tmp_path: Path) -> None:
    python_before = random.getstate()
    numpy_before = _numpy_get_state()
    torch_before = torch.get_rng_state().clone()

    results: list[bytes] = []
    result_objects: list[dict[str, object]] = []
    for name in ("first", "second"):
        runner = PaperBaselineRunner(
            tmp_path / name,
            evaluation_seeds=(100000,),
            n_obs_steps=2,
            n_action_steps=1,
            options={"native_env_factory": _ActionMetricEnvironment, "max_steps": 1},
        )
        result = runner.run(_StochasticPolicy())
        result_objects.append(result)
        results.append(json.dumps(result, sort_keys=True, separators=(",", ":")).encode())
    assert results[0] == results[1]
    assert (tmp_path / "first/failure_traces.json").read_bytes() == (
        tmp_path / "second/failure_traces.json"
    ).read_bytes()
    assert random.getstate() == python_before
    numpy_after = _numpy_get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)

    different = PaperBaselineRunner(
        tmp_path / "different",
        evaluation_seeds=(100001,),
        n_obs_steps=2,
        n_action_steps=1,
        options={"native_env_factory": _ActionMetricEnvironment, "max_steps": 1},
    ).run(_StochasticPolicy())
    first_rollout = cast("list[dict[str, object]]", result_objects[0]["rollouts"])[0]
    different_rollout = cast("list[dict[str, object]]", different["rollouts"])[0]
    assert different_rollout["policy_seed"] != first_rollout["policy_seed"]
    assert different_rollout["dxy"] != first_rollout["dxy"]


def test_evaluator_native_runtime_failure_precedes_artifact_or_environment(
    tmp_path: Path,
) -> None:
    class _UntouchedIndex:
        touched = False

        def verify(self, artifact_id: str, label: str) -> Path:
            del artifact_id, label
            self.touched = True
            raise AssertionError("artifact access must not occur")

    index = _UntouchedIndex()
    environments = 0

    def wrong_runtime() -> NativeRuntimeReport:
        raise NativeRuntimeError(
            "native pushT-so100 runtime mismatch: MuJoCo 3.8.1; Gymnasium 1.3.0"
        )

    def environment_factory() -> object:
        nonlocal environments
        environments += 1
        return _Environment()

    with pytest.raises(NativeRuntimeError, match=r"MuJoCo 3\.8\.1"):
        evaluate_bundle(
            tmp_path / "bundle",
            tmp_path / "output",
            cast("ArtifactIndex", index),
            EvaluationRequest("artifact"),
            EvaluationDependencies(environment_factory, wrong_runtime),
        )
    assert index.touched is False
    assert environments == 0
    assert not (tmp_path / "output").exists()


def test_fixture_evaluation_rejects_before_bundle_environment_or_output(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    index_path = artifact_root / "index.json"
    fixture_record = {
        "deployment_scope": "simulation_only",
        "training_eligible": False,
        "comparison_eligible": False,
        "result_status": "ineligible_fixture",
    }
    index_path.write_text(
        json.dumps({"schema": 1, "artifacts": {"fixture": fixture_record}}),
        encoding="utf-8",
    )
    index = ArtifactIndex(index_path, artifact_root)
    environments = 0

    def trusted_runtime() -> NativeRuntimeReport:
        return cast("NativeRuntimeReport", {"status": "compatible"})

    def environment_factory() -> object:
        nonlocal environments
        environments += 1
        return _Environment()

    before = index_path.read_bytes()
    output = artifact_root / "evaluation"
    with pytest.raises(RuntimeError, match="production"):
        evaluate_bundle(
            artifact_root / "missing.safetensors",
            output,
            index,
            EvaluationRequest("fixture"),
            EvaluationDependencies(environment_factory, trusted_runtime),
        )
    assert environments == 0
    assert index_path.read_bytes() == before
    assert not output.exists()


def test_cli_wrong_runtime_subprocess_has_zero_artifact_or_environment_side_effects(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[3]
    paper_python = (
        project / "04_experiments/so101_pusht_benchmark/cache/envs/paper-baselines/bin/python"
    )
    index = tmp_path / "must-not-be-read.json"
    index.write_text("runtime preflight must precede artifact parsing\n", encoding="utf-8")
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            str(paper_python),
            "-m",
            "so101_pusht_benchmark.cli",
            "evaluate-model",
            "--model",
            "ibc",
            "--bundle",
            str(tmp_path / "bundle.safetensors"),
            "--output",
            str(output),
            "--artifact-id",
            "never-read",
            "--artifact-index",
            str(index),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "native pushT-so100 runtime mismatch" in completed.stdout
    assert "Gymnasium: expected 1.2.2, found 1.3.0" in completed.stdout
    assert "MuJoCo: expected 3.3.7, found 3.8.1" in completed.stdout
    assert index.read_text(encoding="utf-8") == "runtime preflight must precede artifact parsing\n"
    assert not output.exists()
    assert not any(tmp_path.glob(".output.tmp-*"))


def test_default_seed_order_cap_and_cleanup_on_failure(tmp_path: Path) -> None:
    environment = _Environment(terminate_at=301)
    runner = PaperBaselineRunner(
        tmp_path / "cap",
        evaluation_seeds=(100000,),
        options={
            "native_env_factory": cast("Callable[[], object]", lambda: environment),
            "max_steps": 3,
        },
    )
    assert PaperBaselineRunner(
        tmp_path / "defaults",
        options={"native_env_factory": cast("Callable[[], object]", lambda: environment)},
    ).evaluation_seeds == tuple(range(100000, 100100))
    result = runner.run(_Policy(16))
    assert cast("list[dict[str, object]]", result["rollouts"])[0]["steps"] == 3
    assert environment.closed == 1

    class _BadPolicy(_Policy):
        def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            del obs_dict
            return {"action": torch.tensor([[[2.0, 0.0]]], dtype=torch.float32)}

    bad = _Environment()
    bad_runner = PaperBaselineRunner(
        tmp_path / "bad",
        evaluation_seeds=(100000,),
        options={"native_env_factory": cast("Callable[[], object]", lambda: bad)},
    )
    with pytest.raises(ValueError, match="bounds"):
        bad_runner.run(_BadPolicy(1))
    assert bad.steps == 0
    assert bad.closed == 1
    assert not (tmp_path / "bad").exists()
