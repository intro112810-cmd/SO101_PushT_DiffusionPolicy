"""Receipted physical Cartesian proposal bridge without IK or clipping.

Pipelines checkpoint ``absolute_mocap_xy:float32[2]`` into a frozen
``CartesianProposalReceipt``: parse physical-to-simulator SE(2), invert it for XY,
attach owner-policy contact Z and tool orientation, reject workspace
violations, then apply the owner-policy slew limit as an explicit visible
transformation. Any input that would require clipping rejects instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.sim_to_real.policy_types import (
    FixtureApprovedSafetyPolicy,
    ProductionApprovedSafetyPolicy,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame import (
    CartesianPoint3,
    Polygon,
    SimulatorXY,
    TransformMaterial,
    check_workspace_violation,
    parse_se2_material,
    se2_hash,
    simulator_to_physical,
)

Float32Vector = NDArray[np.float32]


class IKPlanner(Protocol):
    """The physical IK boundary this bridge must never invoke."""

    def solve(self, target: tuple[float, float, float]) -> object: ...


@dataclass(frozen=True, slots=True)
class MocapXY:
    """Exact checkpoint action contract: finite float32[2] in [-1,1]^2."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PreviousAppliedPose:
    """The last applied physical pose used for the slew decision."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class CartesianProposalReceipt:
    """Frozen, receipted Cartesian proposal with raw and applied values."""

    raw_xy: tuple[float, float]
    raw_xyz: tuple[float, float, float]
    applied_xyz: tuple[float, float, float]
    tool_rpy: tuple[float, float, float]
    transform_hash: str
    camera_digest: str
    policy_digest: str
    clipping_performed: bool
    ik_called: bool


@dataclass(frozen=True, slots=True)
class BridgeInput:
    """Boundary input assembled by the caller before the bridge runs.

    ``ik`` exists only as a capability sentinel: the bridge never stores it,
    invokes it, or promotes it. Passing any planner is structurally ignored;
    the typed field simply makes that boundary explicit for callers.
    """

    camera_corpus: Mapping[str, object]
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy
    raw_xy: MocapXY
    previous_applied: PreviousAppliedPose | None
    ik: IKPlanner | None


def _finite_xyz(values: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, f"{label} must have three values")
    x, y, z = float(values[0]), float(values[1]), float(values[2])
    if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, f"{label} must be finite")
    return x, y, z


def _finite_xy(x: float, y: float) -> None:
    if not math.isfinite(x) or not math.isfinite(y):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "mocap XY must be finite")


def _bounds(x: float, y: float) -> None:
    if x < -1.0 or x > 1.0 or y < -1.0 or y > 1.0:
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "mocap XY is outside [-1,1]^2")


def _slew_delta_meters(
    current: CartesianPoint3,
    previous: PreviousAppliedPose | None,
) -> float:
    if previous is None:
        return 0.0
    return math.hypot(
        current.x - previous.x,
        current.y - previous.y,
        current.z - previous.z,
    )


def _pending_jump(
    raw_pose: CartesianPoint3,
    previous: PreviousAppliedPose | None,
) -> float:
    """Return the requested Cartesian jump without rate-limiting it."""
    return _slew_delta_meters(raw_pose, previous)


def parse_mocap_xy(raw: object) -> MocapXY:
    """Parse the checkpoint action into a finite, in-domain mocap value."""
    if not isinstance(raw, np.ndarray):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "mocap action must be float32[2]")
    array = cast("NDArray[np.generic]", cast("object", raw))
    if array.shape != (2,) or array.dtype != np.dtype(np.float32):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "mocap action must be float32[2]")
    x, y = float(array[0]), float(array[1])
    _finite_xy(x, y)
    _bounds(x, y)
    return MocapXY(x, y)


def build_task_frame_bridge(
    source: BridgeInput,
) -> CartesianProposalReceipt:
    """Transform mocap XY into a receipted physical Cartesian proposal."""
    material: TransformMaterial = parse_se2_material(source.camera_corpus)
    workspace = source.policy.workspace
    polygon: Polygon = workspace.polygon_xy_m
    contact_z: float = workspace.contact_z_m
    tool_rpy: tuple[float, float, float] = workspace.tool_orientation_rpy_rad
    if len(tool_rpy) != 3 or not all(math.isfinite(value) for value in tool_rpy):
        raise RolloutViolation(
            RolloutCode.R_TRANSFORM_INVALID, "policy tool orientation is undefined"
        )
    max_cartesian_delta_m = source.policy.slew.max_cartesian_delta_m
    if not math.isfinite(max_cartesian_delta_m) or max_cartesian_delta_m <= 0.0:
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "policy slew limit is undefined")
    physical_xy = simulator_to_physical(
        material,
        SimulatorXY(source.raw_xy.x, source.raw_xy.y),
    )
    raw_pose = CartesianPoint3(physical_xy.x, physical_xy.y, contact_z)
    if not math.isfinite(raw_pose.x) or not math.isfinite(raw_pose.y):
        raise RolloutViolation(RolloutCode.R_TRANSFORM_INVALID, "mapped target is non-finite")
    check_workspace_violation(polygon, raw_pose)
    jump = _pending_jump(raw_pose, source.previous_applied)
    if jump > max_cartesian_delta_m:
        raise RolloutViolation(
            RolloutCode.R_CLIPPING_REQUIRED,
            "requested Cartesian jump exceeds the policy slew limit",
        )
    applied_pose = raw_pose
    raw_xyz: tuple[float, float, float] = _finite_xyz(
        (raw_pose.x, raw_pose.y, raw_pose.z), "raw XYZ"
    )
    applied_xyz: tuple[float, float, float] = _finite_xyz(
        (applied_pose.x, applied_pose.y, applied_pose.z), "applied XYZ"
    )
    if applied_xyz != raw_xyz:
        raise RolloutViolation(
            RolloutCode.R_CLIPPING_REQUIRED, "slew must not change the requested pose"
        )
    return CartesianProposalReceipt(
        raw_xy=(source.raw_xy.x, source.raw_xy.y),
        raw_xyz=raw_xyz,
        applied_xyz=applied_xyz,
        tool_rpy=tool_rpy,
        transform_hash=se2_hash(material),
        camera_digest=material.camera_digest,
        policy_digest=source.policy.canonical_digest,
        clipping_performed=False,
        ik_called=False,
    )
