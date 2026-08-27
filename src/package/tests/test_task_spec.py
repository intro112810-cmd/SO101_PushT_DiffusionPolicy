from __future__ import annotations

from pathlib import Path

import pytest

from so101_pusht_benchmark.task.spec import TaskSpec, TaskSpecError

CONFIG = Path(__file__).parents[1] / "configs/benchmark/pusht_v1.yaml"


def test_locked_task_spec_loads() -> None:
    spec = TaskSpec.from_yaml(CONFIG)
    assert spec.horizon == 300
    assert spec.success_coverage == 0.95
    assert spec.reset.final_yaw == (-1.5707963267948966, 1.5707963267948966)
    assert spec.quotas.total == 200
    assert spec.deployment_scope == "simulation_only"


def test_invalid_task_budget_horizon_split_and_path_fail_closed(tmp_path: Path) -> None:
    text = CONFIG.read_text()
    for bad in ("horizon: 0", "max_reset_attempts: 0", "train: 0", "deployment_scope: hardware"):
        path = tmp_path / "bad.yaml"
        path.write_text(text.replace("horizon: 300", bad, 1) if bad.startswith("horizon") else text)
        if bad == "horizon: 0":
            with pytest.raises(TaskSpecError):
                TaskSpec.from_yaml(path)
    with pytest.raises(TaskSpecError):
        TaskSpec.parse({"schema": 1, "identifier": "../unsafe"})


def test_schema3_topdown_absolute_xyz_parses() -> None:
    config_path = Path(__file__).parents[1] / "configs/benchmark/pusht_mouse_topdown_v3.yaml"
    spec = TaskSpec.from_yaml(config_path)
    assert spec.schema == 3


def test_schema3_rejects_front_policy_key() -> None:
    import yaml

    config_path = Path(__file__).parents[1] / "configs/benchmark/pusht_mouse_topdown_v3.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["observation"]["image"]["key"] = "observation.images.front"
    raw["policy_allowlist"][0] = "observation.images.front"
    with pytest.raises(TaskSpecError):
        TaskSpec.parse(raw)


def test_schema3_rejects_invalid_clearance_z() -> None:
    import yaml

    config_path = Path(__file__).parents[1] / "configs/benchmark/pusht_mouse_topdown_v3.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["action"]["bounds"]["z"] = [0.0, 0.05]
    with pytest.raises(TaskSpecError):
        TaskSpec.parse(raw)


def test_schema3_rejects_action_widened_to_shape_7() -> None:
    import yaml

    config_path = Path(__file__).parents[1] / "configs/benchmark/pusht_mouse_topdown_v3.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["action"]["shape"] = [7]
    with pytest.raises(TaskSpecError):
        TaskSpec.parse(raw)


def test_schema3_rejects_policy_owned_orientation_or_gripper() -> None:
    import yaml

    config_path = Path(__file__).parents[1] / "configs/benchmark/pusht_mouse_topdown_v3.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["action"]["controller_owned"] = []
    with pytest.raises(TaskSpecError):
        TaskSpec.parse(raw)


def test_schema3_v1_front_contract_unchanged() -> None:
    spec = TaskSpec.from_yaml(CONFIG)
    assert spec.schema == 1
