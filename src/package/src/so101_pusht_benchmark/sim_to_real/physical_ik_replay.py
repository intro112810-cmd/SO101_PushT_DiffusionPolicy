"""Deserialize collision-bound physical-IK evidence for semantic replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from .physical_ik_collision import Clearance, CollisionSample, ObstacleTransform
from .physical_ik_fk import BodyDegrees, BodyRadians
from .physical_ik_proposal import PhysicalIKProposal
from .rollout_codes import RolloutCode, RolloutViolation


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return cast("Mapping[str, object]", value)


def _items(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return cast("list[object]", value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return float(value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return value


def _floats(value: object, count: int, label: str) -> tuple[float, ...]:
    items = _items(value, label)
    if len(items) != count:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return tuple(_number(item, label) for item in items)


def _clearance(value: object) -> Clearance:
    items = _items(value, "IK clearance")
    if len(items) != 4:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "IK clearance")
    return (
        _text(items[0], "IK clearance category"),
        _number(items[1], "IK clearance distance"),
        _integer(items[2], "IK clearance robot geom"),
        _integer(items[3], "IK clearance other geom"),
    )


def _obstacle_transform(value: object) -> ObstacleTransform:
    items = _items(value, "IK obstacle transform")
    if len(items) != 2:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "IK obstacle transform")
    return (
        _text(items[0], "IK obstacle name"),
        cast(
            "tuple[float, float, float, float, float, float, float]",
            _floats(items[1], 7, "IK obstacle pose"),
        ),
    )


def _collision_sample(value: object) -> CollisionSample:
    document = _mapping(value, "IK collision sample")
    clearances = _items(document.get("clearances"), "IK clearances")
    if len(clearances) != 3:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "IK clearances")
    return CollisionSample(
        _integer(document.get("index"), "IK collision index"),
        _number(document.get("fraction"), "IK collision fraction"),
        cast("BodyRadians", _floats(document.get("body_radians"), 5, "IK collision body")),
        cast("tuple[float, float, float]", _floats(document.get("site_xyz"), 3, "IK site")),
        cast(
            "tuple[Clearance, Clearance, Clearance]", tuple(_clearance(item) for item in clearances)
        ),
        _number(document.get("minimum_clearance_m"), "IK clearance minimum"),
        _text(document.get("pose_digest"), "IK pose digest"),
        cast(
            "tuple[ObstacleTransform, ObstacleTransform]",
            tuple(
                _obstacle_transform(item)
                for item in _items(document.get("obstacle_transforms"), "IK obstacle transforms")
            ),
        ),
        _text(document.get("digest"), "IK collision digest"),
    )


def parse_physical_ik_proposal(
    document: Mapping[str, object], *, declared_hash: str
) -> PhysicalIKProposal:
    """Reconstruct the corrected typed proposal, including every collision sample."""
    body = cast("BodyDegrees", _floats(document.get("body_degrees"), 5, "IK body"))
    raw_path = _items(document.get("swept_path"), "IK swept path")
    path = tuple(
        cast("tuple[float, float, float]", _floats(point, 3, "IK swept point"))
        for point in raw_path
    )
    collisions = tuple(
        _collision_sample(item)
        for item in _items(document.get("collision_samples"), "IK collision samples")
    )
    obstacle_transforms = tuple(
        _obstacle_transform(item)
        for item in _items(document.get("obstacle_transforms"), "IK obstacle transforms")
    )
    return PhysicalIKProposal(
        body,
        _number(document.get("fk_residual_m"), "IK residual"),
        _number(document.get("singularity_metric"), "IK singularity"),
        _number(document.get("branch_delta_degrees"), "IK branch delta"),
        path,
        document.get("clipping_performed") is True,
        document.get("gripper_present") is True,
        _text(document.get("joint_equivalence_digest"), "IK joint digest"),
        declared_hash,
        collisions,
        _text(document.get("model_digest"), "IK model digest"),
        _text(document.get("policy_digest"), "IK policy digest"),
        _text(document.get("scene_pose_digest"), "IK scene pose digest"),
        obstacle_transforms,
    )
