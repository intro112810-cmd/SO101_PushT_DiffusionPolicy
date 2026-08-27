from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import yaml

from so101_pusht_benchmark.task.spec import TaskSpec, TaskSpecError

CONFIG = Path(__file__).parents[1] / "configs/benchmark/pusht_v1.yaml"


def config() -> dict[str, object]:
    parsed = yaml.safe_load(CONFIG.read_text())
    assert isinstance(parsed, dict)
    return cast("dict[str, object]", parsed)


def test_config_rejects_unknown_and_missing_keys() -> None:
    unknown = config()
    unknown["unexpected"] = 1
    with pytest.raises(TaskSpecError, match="unknown"):
        TaskSpec.parse(unknown)
    missing = config()
    del missing["reset"]
    with pytest.raises(TaskSpecError, match="missing"):
        TaskSpec.parse(missing)


def test_config_rejects_coercions_and_wrong_reset_schema() -> None:
    raw = config()
    raw["horizon"] = "300"
    with pytest.raises(TaskSpecError):
        TaskSpec.parse(raw)
    reset = deepcopy(config())
    assert isinstance(reset["reset"], dict)
    reset["reset"]["development_yaw"] = [0.0]
    with pytest.raises(TaskSpecError):
        TaskSpec.parse(reset)


def test_exact_model_budget_and_evaluation_seeds_are_frozen() -> None:
    spec = TaskSpec.from_yaml(CONFIG)
    assert spec.models["DP-CNN"] == (2, 16, 8)
    assert spec.models["DP-Transformer"] == (2, 16, 8)
    assert spec.models["IBC"] == (2, 2, 1)
    assert spec.models["LSTM-GMM"] == (10, 10, 1)
    assert spec.training_updates == 100_000
    assert spec.evaluation_seeds == tuple(range(100000, 100100))


def test_seed_overlap_and_strata_semantics_fail_closed() -> None:
    raw = config()
    assert isinstance(raw["data"], dict)
    raw["data"]["training_seeds"] = [100000, 1, 2]
    with pytest.raises(TaskSpecError, match="seed"):
        TaskSpec.parse(raw)
    raw = config()
    assert isinstance(raw["quotas"], dict)
    raw["quotas"]["strata"] = ["wrong"] * 4
    with pytest.raises(TaskSpecError, match="strata"):
        TaskSpec.parse(raw)


def test_safety_readiness_accepts_locked_numeric_envelope() -> None:
    spec = TaskSpec.from_yaml(CONFIG)
    spec.require_safety_ready()
