"""Duplicate-safe YAML loading with exact recursive value types."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml

from .policy_schema import (
    TOP_FIELDS,
    YamlMapping,
    YamlValue,
    mapping_value,
    policy_unauthorized,
)
from .rollout_codes import RolloutViolation

__all__: tuple[str, ...] = ()


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects mapping-key replacement."""


class _LoaderApi(Protocol):
    def construct_object(self, node: yaml.Node, deep: bool = False) -> YamlValue: ...
    def get_single_data(self) -> YamlValue: ...
    def dispose(self) -> None: ...


def _loader_api(value: _LoaderApi) -> _LoaderApi:
    return value


def _unique_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode
) -> dict[YamlValue, YamlValue]:
    api = _loader_api(loader)
    result: dict[YamlValue, YamlValue] = {}
    for key_node, value_node in node.value:
        key = api.construct_object(key_node, deep=False)
        if key in result:
            raise policy_unauthorized(f"duplicate YAML key: {key}")
        result[key] = api.construct_object(value_node, deep=True)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def load_yaml_document(path: Path) -> YamlMapping:
    try:
        content = path.read_text(encoding="utf-8")
        loader = _UniqueSafeLoader(content)
        api = _loader_api(loader)
        try:
            loaded = api.get_single_data()
        finally:
            api.dispose()
        return mapping_value(loaded, "policy", TOP_FIELDS)
    except RolloutViolation:
        raise
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise policy_unauthorized("policy is malformed or cannot be read") from exc
