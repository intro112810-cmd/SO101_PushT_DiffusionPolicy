from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeGuard

import pytest
import yaml

from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.policy_schema import (
    YamlValue,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

ROOT = Path(__file__).parents[1]
APPROVED = ROOT / "tests/fixtures/sim_to_real/approved_policy.yaml"
YamlMapping = dict[str, YamlValue]


def _is_object_sequence(value: YamlValue) -> TypeGuard[Sequence[YamlValue]]:
    return isinstance(value, list)


def _is_object_mapping(value: YamlValue) -> TypeGuard[Mapping[YamlValue, YamlValue]]:
    return isinstance(value, dict)


def _valid_sequence(values: Sequence[YamlValue]) -> bool:
    return all(is_value(item) for item in values)


def _valid_mapping(values: Mapping[YamlValue, YamlValue]) -> bool:
    return all(isinstance(key, str) and is_value(item) for key, item in values.items())


def is_value(value: YamlValue) -> TypeGuard[YamlValue]:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if _is_object_sequence(value):
        return _valid_sequence(value)
    if _is_object_mapping(value):
        return _valid_mapping(value)
    return False


def is_mapping(value: YamlValue) -> TypeGuard[YamlMapping]:
    return is_value(value) and isinstance(value, dict)


def raw_policy() -> YamlMapping:
    value: YamlValue = yaml.safe_load(APPROVED.read_text(encoding="utf-8"))
    assert is_mapping(value)
    return value


def mapping(value: YamlValue) -> YamlMapping:
    assert isinstance(value, dict)
    return value


def set_path(raw: YamlMapping, path: tuple[str, ...], value: YamlValue) -> None:
    target = raw
    for key in path[:-1]:
        target = mapping(target[key])
    target[path[-1]] = value


def reject_policy(raw: YamlMapping, path: Path, now: datetime) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(RolloutViolation) as caught:
        load_fixture_safety_policy(path, now=now)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
