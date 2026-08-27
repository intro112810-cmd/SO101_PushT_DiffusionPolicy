"""Canonical typed content identity for rollout boundary records."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import TypeAlias

BoundaryValue: TypeAlias = str | float | int | list[float] | list[str] | None
__all__ = ("BoundaryValue", "digest_content")
_FLOAT_FIELDS = frozenset({"created_at", "camera_timestamp", "joint_timestamp", "valid_until"})
_FLOAT_VECTOR_FIELDS = frozenset({"body_degrees", "target_xy", "accepted_body_degrees"})


def _normalize_content(
    content: Mapping[str, BoundaryValue],
) -> dict[str, BoundaryValue]:
    normalized: dict[str, BoundaryValue] = {}
    for key, value in content.items():
        if key in _FLOAT_FIELDS and isinstance(value, (float, int)) and not isinstance(value, bool):
            normalized[key] = float(value)
        elif key in _FLOAT_VECTOR_FIELDS and isinstance(value, list):
            numbers = [float(item) for item in value if isinstance(item, (float, int))]
            normalized[key] = numbers if len(numbers) == len(value) else value
        else:
            normalized[key] = value
    return normalized


def digest_content(content: Mapping[str, BoundaryValue]) -> str:
    """Hash canonical typed values rather than their input spelling."""
    encoded = json.dumps(
        _normalize_content(content),
        allow_nan=True,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
