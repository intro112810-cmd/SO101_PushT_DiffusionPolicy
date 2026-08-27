"""Registered task-frame SE(2) parsing, inversion, hashing, and geometry.

Camera evidence declares a transform from physical-table coordinates to the
simulator task frame. Checkpoint actions are simulator-frame values, so the
physical target is obtained with the rigid inverse. Evidence and transform
hashes remain content-addressed. No IK, writer, clipping, or silent repair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final, cast

from so101_pusht_benchmark.sim_to_real.replay_types import CAMERA_REGISTRATION_DIGEST
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

CAMERA_DIGEST: Final = CAMERA_REGISTRATION_DIGEST
CANONICAL_SE2: Final = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_ORTHONORMAL_TOLERANCE: Final = 1e-6

Se2 = tuple[float, float, float, float, float, float]
Point2 = tuple[float, float]
Polygon = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TransformMaterial:
    """Content-addressed physical-to-simulator SE(2) plus its camera digest."""

    physical_to_sim_se2: Se2
    camera_digest: str


@dataclass(frozen=True, slots=True)
class SimulatorXY:
    """A point expressed in the simulator action frame."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PhysicalTableXY:
    """A point expressed in the physical table frame, in metres."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CartesianPoint3:
    """A finite physical Cartesian target in metres."""

    x: float
    y: float
    z: float


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def se2_hash(material: TransformMaterial) -> str:
    """Return the transform hash bound to direction, values, and evidence."""
    payload = {
        "physical_to_sim_se2": list(material.physical_to_sim_se2),
        "camera_digest": material.camera_digest,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def registration_evidence_digest(raw: Mapping[str, object]) -> str:
    """Hash registration evidence without its detached digest declaration."""
    payload = {key: value for key, value in raw.items() if key != "camera_digest"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


BOUND_TRANSFORM_HASH: Final = se2_hash(TransformMaterial(CANONICAL_SE2, CAMERA_DIGEST))


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, f"{label} must be finite")
    return number


def parse_rigid_se2(raw: object) -> Se2:
    if not isinstance(raw, list):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "SE(2) must be a 2x3 matrix")
    typed = cast("list[object]", raw)
    if len(typed) != 6:
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "SE(2) must have six values")
    v0 = _finite_number(typed[0], "SE(2) entry")
    v1 = _finite_number(typed[1], "SE(2) entry")
    v2 = _finite_number(typed[2], "SE(2) entry")
    v3 = _finite_number(typed[3], "SE(2) entry")
    v4 = _finite_number(typed[4], "SE(2) entry")
    v5 = _finite_number(typed[5], "SE(2) entry")
    r00, r01, _, r10, r11, _ = (v0, v1, v2, v3, v4, v5)
    column0 = r00 * r00 + r10 * r10
    column1 = r01 * r01 + r11 * r11
    cross = r00 * r01 + r10 * r11
    determinant = r00 * r11 - r01 * r10
    orthonormal = (
        abs(column0 - 1.0) <= _ORTHONORMAL_TOLERANCE
        and abs(column1 - 1.0) <= _ORTHONORMAL_TOLERANCE
        and abs(cross) <= _ORTHONORMAL_TOLERANCE
        and abs(determinant - 1.0) <= _ORTHONORMAL_TOLERANCE
    )
    if not orthonormal:
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "SE(2) rotation block is invalid")
    return (v0, v1, v2, v3, v4, v5)


def invert_se2(transform: Se2) -> Se2:
    """Return the exact rigid inverse of a physical-to-simulator transform."""
    r00, r01, tx, r10, r11, ty = transform
    inverse_tx = -(r00 * tx + r10 * ty)
    inverse_ty = -(r01 * tx + r11 * ty)
    return (r00, r10, inverse_tx, r01, r11, inverse_ty)


def _se2_close(left: Se2, right: Se2) -> bool:
    return all(
        abs(left_value - right_value) <= _ORTHONORMAL_TOLERANCE
        for left_value, right_value in zip(left, right, strict=True)
    )


def parse_se2_material(raw: Mapping[str, object]) -> TransformMaterial:
    """Parse rigid, direction-explicit, content-addressed registration evidence."""
    values = parse_rigid_se2(raw.get("physical_to_sim_se2"))
    declared_camera_digest = raw.get("camera_digest", CAMERA_DIGEST)
    if not isinstance(declared_camera_digest, str):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "camera digest must be a string")
    normalized_digest = declared_camera_digest.lower()
    if len(normalized_digest) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_digest
    ):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "camera digest must be SHA-256")
    if registration_evidence_digest(raw) != normalized_digest:
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "camera registration hash drift")
    declared_inverse = raw.get("sim_to_physical_se2")
    if declared_inverse is not None:
        inverse_values = parse_rigid_se2(declared_inverse)
        if not _se2_close(inverse_values, invert_se2(values)):
            raise RolloutViolation(
                RolloutCode.R_TRANSFORM_INVALID, "declared SE(2) inverse mismatch"
            )
    return TransformMaterial(values, normalized_digest)


def physical_to_simulator(
    material: TransformMaterial,
    point: PhysicalTableXY,
) -> SimulatorXY:
    """Map a physical-table point through the direction declared by evidence."""
    r00, r01, tx, r10, r11, ty = material.physical_to_sim_se2
    return SimulatorXY(
        r00 * point.x + r01 * point.y + tx,
        r10 * point.x + r11 * point.y + ty,
    )


def simulator_to_physical(
    material: TransformMaterial,
    point: SimulatorXY,
) -> PhysicalTableXY:
    """Map a simulator action to its physical target using the rigid inverse."""
    r00, r01, tx, r10, r11, ty = material.physical_to_sim_se2
    shifted_x = point.x - tx
    shifted_y = point.y - ty
    return PhysicalTableXY(
        r00 * shifted_x + r10 * shifted_y,
        r01 * shifted_x + r11 * shifted_y,
    )


def apply_se2(material: TransformMaterial, xy: Sequence[float]) -> tuple[float, float]:
    """Map simulator XY to physical-table XY; retained as a tuple adapter."""
    point = simulator_to_physical(material, SimulatorXY(float(xy[0]), float(xy[1])))
    return point.x, point.y


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    tolerance: float = 1e-12,
) -> bool:
    cross = (end[1] - start[1]) * (point[0] - start[0]) - (end[0] - start[0]) * (
        point[1] - start[1]
    )
    if abs(cross) > tolerance:
        return False
    return (
        min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
    )


def point_in_polygon(point: Point2, polygon: Polygon) -> bool:
    """Return whether ``point`` is inside or on the owner-policy polygon."""
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
    x, y = point
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        current_x, current_y = polygon[current]
        previous_x, previous_y = polygon[previous]
        if (current_y > y) != (previous_y > y):
            crossing = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def check_workspace_violation(
    polygon: Polygon,
    target: CartesianPoint3,
) -> None:
    """Reject any Cartesian target outside the owner-policy workspace polygon."""
    if not point_in_polygon((target.x, target.y), polygon):
        raise RolloutViolation(
            RolloutCode.R_WORKSPACE_VIOLATION,
            "physical Cartesian target is outside the owner-policy workspace",
        )
