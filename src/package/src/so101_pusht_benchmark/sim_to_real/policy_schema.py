"""Private exhaustive schema primitives for the policy trust boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import math
from typing import TypeGuard

from .rollout_codes import RolloutCode, RolloutViolation

__all__: tuple[str, ...] = ()
YamlScalar = None | bool | int | float | str
YamlValue = YamlScalar | list["YamlValue"] | dict[str, "YamlValue"]
YamlMapping = dict[str, YamlValue]
JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TOP_FIELDS = frozenset(
    {
        "schema",
        "policy_version",
        "policy_id",
        "artifact_scope",
        "approval_status",
        "approved_by",
        "approved_at",
        "valid_from",
        "expires_at",
        "canonical_digest",
        "thresholds",
        "owner_approval",
    }
)
SECTION_FIELDS = {
    "thresholds": frozenset(
        {
            "workspace",
            "joint_domains",
            "timing",
            "camera",
            "kinematics",
            "slew",
            "provider",
            "watchdog",
            "acknowledgement",
            "post_state",
            "shadow",
            "single_step",
            "bounded_rollout",
            "operator",
        }
    ),
    "workspace": frozenset({"polygon_xy_m", "contact_z_m", "tool_orientation_rpy_rad"}),
    "joint_domains": frozenset({"joint_order", "physical_degrees", "mapped_radians"}),
    "timing": frozenset(
        {
            "sample_max_age_seconds",
            "sample_max_skew_seconds",
            "max_policy_age_seconds",
            "authorization_max_age_seconds",
            "authorization_ttl_seconds",
        }
    ),
    "camera": frozenset(
        {"max_reprojection_error_px", "min_correspondences", "max_correspondence_error_px"}
    ),
    "kinematics": frozenset(
        {
            "max_fk_residual_m",
            "max_ik_residual_m",
            "min_singularity_metric",
            "max_branch_delta_degrees",
        }
    ),
    "collision": frozenset({"minimum_clearance_m", "max_joint_step_radians", "max_path_samples"}),
    "slew": frozenset({"max_cartesian_delta_m", "max_joint_delta_degrees"}),
    "provider": frozenset({"exact_goal_required", "max_abs_error_degrees"}),
    "watchdog": frozenset({"timeout_seconds"}),
    "acknowledgement": frozenset({"required", "timeout_seconds", "max_position_error_degrees"}),
    "post_state": frozenset({"max_age_seconds", "max_tracking_error_degrees"}),
    "shadow": frozenset({"min_cycles", "max_cycle_latency_seconds", "max_error_count"}),
    "single_step": frozenset({"max_commands"}),
    "bounded_rollout": frozenset(
        {"max_commands", "max_duration_seconds", "max_path_length_m", "max_error_count"}
    ),
    "operator": frozenset(
        {"deadman_required", "stop_required", "stop_behavior", "acknowledgement_required"}
    ),
    "owner_approval": frozenset(
        {"scheme", "approval_id", "signer_id", "policy_digest", "binding_signature"}
    ),
}


def policy_unauthorized(detail: str) -> RolloutViolation:
    return RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, detail)


def _is_object_sequence(value: YamlValue) -> TypeGuard[Sequence[YamlValue]]:
    return isinstance(value, list)


def _is_object_mapping(value: YamlValue) -> TypeGuard[Mapping[YamlValue, YamlValue]]:
    return isinstance(value, dict)


def _valid_sequence(values: Sequence[YamlValue]) -> bool:
    return all(is_yaml_value_internal(item) for item in values)


def _valid_mapping(values: Mapping[YamlValue, YamlValue]) -> bool:
    return all(
        isinstance(key, str) and is_yaml_value_internal(item) for key, item in values.items()
    )


def is_yaml_value_internal(value: YamlValue) -> TypeGuard[YamlValue]:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if _is_object_sequence(value):
        return _valid_sequence(value)
    if _is_object_mapping(value):
        return _valid_mapping(value)
    return False


def is_mapping_internal(value: YamlValue) -> TypeGuard[YamlMapping]:
    return is_yaml_value_internal(value) and isinstance(value, dict)


def mapping_value(value: YamlValue, label: str, fields: frozenset[str]) -> YamlMapping:
    if not is_mapping_internal(value) or frozenset(value) != fields:
        raise policy_unauthorized(f"{label} fields are incomplete or unknown")
    return value


def text_value(value: YamlValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise policy_unauthorized(f"{label} must be a nonempty string")
    return value


def boolean_value(value: YamlValue, label: str) -> bool:
    if not isinstance(value, bool):
        raise policy_unauthorized(f"{label} must be boolean")
    return value


def positive_number(value: YamlValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise policy_unauthorized(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise policy_unauthorized(f"{label} must be a positive finite number")
    return result


def positive_integer(value: YamlValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise policy_unauthorized(f"{label} must be a positive integer")
    return value


def finite_number(value: YamlValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise policy_unauthorized(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise policy_unauthorized(f"{label} must be finite")
    return result


def timestamp_value(value: YamlValue, label: str) -> datetime:
    raw = text_value(value, label)
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise policy_unauthorized(f"{label} must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise policy_unauthorized(f"{label} must include a timezone")
    return result


def sequence_value(value: YamlValue, label: str) -> list[YamlValue]:
    if not is_yaml_value_internal(value) or not isinstance(value, list):
        raise policy_unauthorized(f"{label} must be a sequence")
    return value


def vector_value(value: YamlValue, label: str, length: int) -> tuple[float, ...]:
    items = sequence_value(value, label)
    if len(items) != length:
        raise policy_unauthorized(f"{label} has invalid length")
    return tuple(finite_number(item, label) for item in items)


def numeric_range_value(value: YamlValue, label: str) -> tuple[float, float]:
    values = vector_value(value, label, 2)
    minimum, maximum = values[0], values[1]
    if minimum >= maximum:
        raise policy_unauthorized(f"{label} minimum must be below maximum")
    return minimum, maximum


def orientation_value(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> int:
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (cross > 0) - (cross < 0)


def segments_intersect(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
    fourth: tuple[float, float],
) -> bool:
    return orientation_value(first, second, third) != orientation_value(
        first, second, fourth
    ) and orientation_value(third, fourth, first) != orientation_value(third, fourth, second)


def polygon_value(value: YamlValue) -> tuple[tuple[float, float], ...]:
    points = tuple(
        (point[0], point[1])
        for point in (
            vector_value(item, "polygon point", 2) for item in sequence_value(value, "polygon_xy_m")
        )
    )
    if len(points) < 3 or len(set(points)) != len(points):
        raise policy_unauthorized("workspace polygon is incomplete or has duplicate vertices")
    area = sum(
        point[0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * point[1]
        for index, point in enumerate(points)
    )
    if area == 0:
        raise policy_unauthorized("workspace polygon has zero area")
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if second in {first, first + 1} or (first == 0 and second == len(points) - 1):
                continue
            if segments_intersect(
                points[first],
                points[(first + 1) % len(points)],
                points[second],
                points[(second + 1) % len(points)],
            ):
                raise policy_unauthorized("workspace polygon self-intersects")
    return points
