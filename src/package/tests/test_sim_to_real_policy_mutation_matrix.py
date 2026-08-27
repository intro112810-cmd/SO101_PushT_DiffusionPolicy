from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeGuard
from typing_extensions import assert_never

import pytest
import yaml

from sim_to_real_policy_helpers import YamlMapping, YamlValue, mapping, raw_policy
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
PathPart = str | int
ValuePath = tuple[PathPart, ...]


def _mapping_paths(value: YamlValue, path: ValuePath = ()) -> list[ValuePath]:
    if not isinstance(value, dict | list):
        return []
    match value:
        case dict():
            result: list[ValuePath] = [path]
            for key, item in value.items():
                result.extend(_mapping_paths(item, (*path, key)))
            return result
        case list():
            result = []
            for index, item in enumerate(value):
                result.extend(_mapping_paths(item, (*path, index)))
            return result
        case _:
            assert_never(value)


def _numeric_paths(value: YamlValue, path: ValuePath = ()) -> list[ValuePath]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [path]
    if not isinstance(value, dict | list):
        return []
    match value:
        case dict():
            result: list[ValuePath] = []
            for key, item in value.items():
                result.extend(_numeric_paths(item, (*path, key)))
            return result
        case list():
            result = []
            for index, item in enumerate(value):
                result.extend(_numeric_paths(item, (*path, index)))
            return result
        case _:
            assert_never(value)


def _is_list(value: YamlValue) -> TypeGuard[list[YamlValue]]:
    return isinstance(value, list)


def _at(raw: YamlMapping, path: ValuePath) -> YamlValue:
    value: YamlValue = raw
    for part in path:
        if _is_list(value):
            assert isinstance(part, int)
            value = value[part]
        else:
            assert isinstance(part, str)
            value = mapping(value)[part]
    return value


def _set(raw: YamlMapping, path: ValuePath, replacement: YamlValue) -> None:
    parent = _at(raw, path[:-1])
    leaf = path[-1]
    if isinstance(parent, list):
        assert isinstance(leaf, int)
        parent[leaf] = replacement
    else:
        mapping(parent)[str(leaf)] = replacement


def _delete(raw: YamlMapping, path: ValuePath) -> None:
    parent = _at(raw, path[:-1])
    assert isinstance(parent, dict)
    del parent[str(path[-1])]


def _reject(raw: YamlMapping, destination: Path) -> None:
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(RolloutViolation) as caught:
        load_fixture_safety_policy(destination, now=NOW)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


def test_independent_complete_mutation_matrix(tmp_path: Path) -> None:
    original = raw_policy()
    destination = tmp_path / "mutated.yaml"
    count = 0

    for parent_path in _mapping_paths(original):
        parent = _at(original, parent_path)
        assert isinstance(parent, dict)
        for key in tuple(parent):
            mutated = deepcopy(original)
            _delete(mutated, (*parent_path, key))
            _reject(mutated, destination)
            count += 1
        mutated = deepcopy(original)
        target = _at(mutated, parent_path)
        assert isinstance(target, dict)
        target["unknown_mutation_field"] = True
        _reject(mutated, destination)
        count += 1

    threshold_paths = [path for path in _numeric_paths(mapping(original["thresholds"])) if path]
    for path in threshold_paths:
        full_path = ("thresholds", *path)
        original_value = _at(original, full_path)
        for replacement in (0, -1, float("nan"), float("inf"), True):
            if (
                not isinstance(replacement, bool)
                and isinstance(original_value, int | float)
                and float(replacement) == float(original_value)
            ):
                continue
            mutated = deepcopy(original)
            _set(mutated, full_path, replacement)
            _reject(mutated, destination)
            count += 1

    assert count >= 237
    print(f"independent_policy_mutations={count}")
