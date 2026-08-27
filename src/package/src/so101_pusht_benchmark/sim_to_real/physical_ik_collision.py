"""Deterministic collision evidence for the pinned physical IK model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.sim.scene import OVERLAY, UPSTREAM
from so101_pusht_benchmark.sim_to_real.physical_ik_fk import (
    BodyRadians,
    MuJoCoWorkspace,
    forward_site,
)
from so101_pusht_benchmark.sim_to_real.policy_types import CollisionPolicy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

if TYPE_CHECKING:
    from .physical_ik_scene_pose import SceneObjectPoseReceipt

_FLOAT_WIDTH = 17
_CATEGORY_ORDER = ("table", "object", "self")


class _CollisionModel(Protocol):
    ngeom: int
    geom_group: NDArray[np.int32]
    geom_bodyid: NDArray[np.int32]
    geom_contype: NDArray[np.int32]
    geom_conaffinity: NDArray[np.int32]


_GeomDistance = Callable[[object, object, int, int, float, None], float]
_NameToId = Callable[[object, object, str], int]


class _ObjectTypes(Protocol):
    mjOBJ_GEOM: object


class _CollisionMuJoCo(Protocol):
    mjtObj: _ObjectTypes
    mj_name2id: _NameToId
    mj_geomDistance: _GeomDistance


Clearance = tuple[str, float, int, int]
ObstacleTransform = tuple[str, tuple[float, float, float, float, float, float, float]]


@dataclass(frozen=True, slots=True)
class _Interpolation:
    start: BodyRadians
    end: BodyRadians
    segments: int
    pose_digest: str
    obstacle_transforms: tuple[ObstacleTransform, ObstacleTransform]


def _float(value: float) -> float:
    return float(f"{value:.{_FLOAT_WIDTH}g}")


def _digest(document: dict[str, object]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CollisionSample:
    """One content-addressed joint/FK/clearance sample along the swept path."""

    index: int
    fraction: float
    body_radians: BodyRadians
    site_xyz: tuple[float, float, float]
    clearances: tuple[Clearance, Clearance, Clearance]
    minimum_clearance_m: float
    pose_digest: str
    obstacle_transforms: tuple[ObstacleTransform, ObstacleTransform]
    digest: str

    def content_document(self) -> dict[str, object]:
        return {
            "index": self.index,
            "fraction": _float(self.fraction),
            "body_radians": [_float(value) for value in self.body_radians],
            "site_xyz": [_float(value) for value in self.site_xyz],
            "clearances": [
                [category, _float(distance), robot, other]
                for category, distance, robot, other in self.clearances
            ],
            "minimum_clearance_m": _float(self.minimum_clearance_m),
            "pose_digest": self.pose_digest,
            "obstacle_transforms": [
                [name, [_float(value) for value in transform]]
                for name, transform in self.obstacle_transforms
            ],
        }

    def to_document(self) -> dict[str, object]:
        document = self.content_document()
        document["digest"] = self.digest
        return document

    def valid_digest(self) -> bool:
        return self.digest == _digest(self.content_document())


def pinned_model_digest() -> str:
    """Hash every owned/upstream byte consumed to construct the physical model."""
    members = [OVERLAY, UPSTREAM, *(UPSTREAM.parent / "assets").glob("*.stl")]
    digest = hashlib.sha256()
    for path in sorted(members, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _geom_id(workspace: MuJoCoWorkspace, name: str) -> int:
    mujoco = cast("_CollisionMuJoCo", workspace.scene.mujoco)
    geom_type = mujoco.mjtObj.mjOBJ_GEOM
    return int(mujoco.mj_name2id(workspace.scene.model, geom_type, name))


def _distance(workspace: MuJoCoWorkspace, first: int, second: int) -> float:
    mujoco = cast("_CollisionMuJoCo", workspace.scene.mujoco)
    value = float(
        mujoco.mj_geomDistance(
            workspace.scene.model, workspace.scene.data, first, second, math.inf, None
        )
    )
    if not math.isfinite(value):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "collision distance must be finite")
    return value


def _closest_pairs(workspace: MuJoCoWorkspace) -> tuple[Clearance, Clearance, Clearance]:
    model = cast("_CollisionModel", workspace.scene.model)
    robot = [
        index
        for index in range(model.ngeom)
        if int(model.geom_group[index]) == 3
        and int(model.geom_contype[index]) != 0
        and int(model.geom_conaffinity[index]) != 0
        and 2 <= int(model.geom_bodyid[index]) <= 7
    ]
    table = _geom_id(workspace, "table")
    objects = [_geom_id(workspace, name) for name in ("pusher", "push_t_bar", "push_t_stem")]
    objects = [
        geom
        for geom in objects
        if int(model.geom_contype[geom]) != 0 and int(model.geom_conaffinity[geom]) != 0
    ]
    table_pairs = [
        (_distance(workspace, geom, table), geom, table)
        for geom in robot
        if int(model.geom_bodyid[geom]) > 2
    ]
    object_pairs = [
        (_distance(workspace, geom, obstacle), geom, obstacle)
        for geom in robot
        for obstacle in objects
    ]
    self_pairs = [
        (_distance(workspace, first, second), first, second)
        for offset, first in enumerate(robot)
        for second in robot[offset + 1 :]
        if abs(int(model.geom_bodyid[first]) - int(model.geom_bodyid[second])) > 1
    ]
    groups = (table_pairs, object_pairs, self_pairs)
    if any(not group for group in groups):
        raise RolloutViolation(RolloutCode.R_COLLISION, "pinned collision geometry is incomplete")
    return cast(
        "tuple[Clearance, Clearance, Clearance]",
        tuple(
            (category, *min(group)) for category, group in zip(_CATEGORY_ORDER, groups, strict=True)
        ),
    )


def _sample(
    workspace: MuJoCoWorkspace, interpolation: _Interpolation, index: int
) -> CollisionSample:
    fraction = index / interpolation.segments
    radians = cast(
        "BodyRadians",
        tuple(
            left + (right - left) * fraction
            for left, right in zip(interpolation.start, interpolation.end, strict=True)
        ),
    )
    position = forward_site(workspace, radians)
    site = (float(position[0]), float(position[1]), float(position[2]))
    clearances = _closest_pairs(workspace)
    minimum = min(clearance[1] for clearance in clearances)
    pending = CollisionSample(
        index,
        fraction,
        radians,
        site,
        clearances,
        minimum,
        interpolation.pose_digest,
        interpolation.obstacle_transforms,
        "",
    )
    return CollisionSample(
        index,
        fraction,
        radians,
        site,
        clearances,
        minimum,
        interpolation.pose_digest,
        interpolation.obstacle_transforms,
        _digest(pending.content_document()),
    )


def swept_collision_proof(
    workspace: MuJoCoWorkspace,
    start: BodyRadians,
    end: BodyRadians,
    policy: CollisionPolicy | None,
    scene_pose: SceneObjectPoseReceipt | None,
) -> tuple[CollisionSample, ...]:
    """Interpolate a bounded joint path and reject any insufficient clearance."""
    from .physical_ik_scene_pose import SceneObjectPoseReceipt, apply_scene_object_pose

    if policy is None:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "collision policy is missing")
    pose = scene_pose if isinstance(scene_pose, SceneObjectPoseReceipt) else None
    apply_scene_object_pose(workspace, pose)
    maximum_delta = max(abs(right - left) for left, right in zip(start, end, strict=True))
    segments = max(1, math.ceil(maximum_delta / policy.max_joint_step_radians))
    if segments + 1 > policy.max_path_samples:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "collision interpolation budget is insufficient"
        )
    assert pose is not None
    interpolation = _Interpolation(start, end, segments, pose.digest, pose.obstacle_transforms)
    samples = tuple(_sample(workspace, interpolation, index) for index in range(segments + 1))
    for sample in samples:
        if sample.minimum_clearance_m <= policy.minimum_clearance_m:
            closest = min(sample.clearances, key=lambda clearance: clearance[1])
            raise RolloutViolation(
                RolloutCode.R_COLLISION,
                (
                    f"sample {sample.index} {closest[0]} clearance "
                    f"{sample.minimum_clearance_m:.9f} m is insufficient; "
                    f"body_radians={sample.body_radians}; pair={closest[2:]}; "
                    f"fraction={sample.fraction}; start={start}; end={end}"
                ),
            )
    return samples
