"""Authenticated, content-addressed object poses for physical-IK collision checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, InitVar
import hashlib
import json
import math
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .physical_ik_collision import pinned_model_digest
from .physical_ik_fk import MuJoCoWorkspace
from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .rollout_codes import RolloutCode, RolloutViolation
from .task_frame import point_in_polygon

Transform = tuple[float, float, float, float, float, float, float]
_FIELDS = frozenset(
    {
        "schema",
        "sample_id",
        "sample_timestamp",
        "sample_digest",
        "device_digest",
        "camera_registration_digest",
        "policy_digest",
        "model_digest",
        "pusher_transform",
        "push_t_transform",
        "digest",
    }
)
_SHA = frozenset("0123456789abcdef")
_SEAL = object()


class _PoseModel(Protocol):
    jnt_qposadr: NDArray[np.int32]
    jnt_range: NDArray[np.float64]
    body_pos: NDArray[np.float64]


class _PoseData(Protocol):
    qpos: NDArray[np.float64]
    mocap_pos: NDArray[np.float64]
    mocap_quat: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ScenePoseExpectations:
    """Externally authenticated identities and verification time for one pose."""

    pose_digest: str
    sample_id: str
    sample_timestamp: float
    sample_digest: str
    device_digest: str
    camera_registration_digest: str
    model_digest: str
    planning_timestamp: float


def _sha(value: str) -> bool:
    return len(value) == 64 and all(character in _SHA for character in value)


def scene_pose_content_digest(document: Mapping[str, object]) -> str:
    """Hash a scene-pose document excluding its declared digest."""
    content = {key: value for key, value in document.items() if key != "digest"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SceneObjectPoseReceipt:
    """Parser-sealed scene pose bound to one synchronized physical sample."""

    _construction_seal: InitVar[object]
    sample_id: str
    sample_timestamp: float
    sample_digest: str
    device_digest: str
    camera_registration_digest: str
    policy_digest: str
    model_digest: str
    pusher_transform: Transform
    push_t_transform: Transform
    digest: str
    _authority: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _construction_seal: object) -> None:
        """Seal construction to the authenticated parser."""
        if _construction_seal is not _SEAL:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "scene pose is not authenticated")
        object.__setattr__(self, "_authority", _SEAL)

    def authenticated(self) -> bool:
        return self._authority is _SEAL

    @property
    def obstacle_transforms(self) -> tuple[tuple[str, Transform], tuple[str, Transform]]:
        return (("pusher", self.pusher_transform), ("push_t", self.push_t_transform))


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"scene pose {label}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"scene pose {label}")
    result = float(value)
    if not math.isfinite(result):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"scene pose {label}")
    return result


def _transform(value: object, label: str) -> Transform:
    if not isinstance(value, (list, tuple)):
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, f"scene pose {label}")
    items = cast("Sequence[object]", value)
    if len(items) != 7:
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, f"scene pose {label}")
    values = tuple(_number(item, label) for item in items)
    return cast("Transform", values)


def _joint(workspace: MuJoCoWorkspace, name: str) -> tuple[int, tuple[float, float]]:
    scene = workspace.scene
    joint_id = scene.mujoco.mj_name2id(scene.model, scene.mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"scene pose joint {name}")
    model = cast("_PoseModel", scene.model)
    return int(model.jnt_qposadr[joint_id]), (
        float(model.jnt_range[joint_id][0]),
        float(model.jnt_range[joint_id][1]),
    )


def _yaw(transform: Transform) -> float:
    _, _, _, qw, qx, qy, qz = transform
    if qx != 0.0 or qy != 0.0 or not math.isclose(qw * qw + qz * qz, 1.0, abs_tol=1e-12):
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "scene pose quaternion")
    return 2.0 * math.atan2(qz, qw)


def _validate_domain(
    workspace: MuJoCoWorkspace,
    pusher: Transform,
    push_t: Transform,
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
) -> None:
    if not point_in_polygon((pusher[0], pusher[1]), policy.workspace.polygon_xy_m):
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "pusher pose outside workspace")
    if pusher[2] < policy.workspace.contact_z_m or _yaw(pusher) != 0.0:
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "pusher transform is out of domain")
    model = cast("_PoseModel", workspace.scene.model)
    body_id = workspace.scene.mujoco.mj_name2id(
        workspace.scene.model, workspace.scene.mujoco.mjtObj.mjOBJ_BODY, "push_t"
    )
    if body_id < 0 or push_t[2] != float(model.body_pos[body_id][2]):
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "push_t height is out of domain")
    values = (push_t[0], push_t[1], _yaw(push_t))
    for value, name in zip(values, ("push_t_x", "push_t_y", "push_t_yaw"), strict=True):
        _, domain = _joint(workspace, name)
        if not domain[0] <= value <= domain[1]:
            raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, f"{name} pose is out of domain")


def parse_scene_object_pose_receipt(
    document: Mapping[str, object],
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
    workspace: MuJoCoWorkspace,
    expectations: ScenePoseExpectations,
) -> SceneObjectPoseReceipt:
    """Authenticate all bindings and domains before returning pose authority."""
    if frozenset(document) != _FIELDS or document.get("schema") != "so101-scene-object-pose-v1":
        raise RolloutViolation(RolloutCode.R_MISSING, "scene pose fields are incomplete")
    sample_id = _text(document.get("sample_id"), "sample_id")
    timestamp = _number(document.get("sample_timestamp"), "sample_timestamp")
    pusher = _transform(document.get("pusher_transform"), "pusher_transform")
    push_t = _transform(document.get("push_t_transform"), "push_t_transform")
    _validate_domain(workspace, pusher, push_t, policy)
    declared = _text(document.get("digest"), "digest")
    if sample_id != expectations.sample_id or timestamp != expectations.sample_timestamp:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "scene pose sample identity drift")
    bindings = (
        (_text(document.get("sample_digest"), "sample_digest"), expectations.sample_digest),
        (_text(document.get("device_digest"), "device_digest"), expectations.device_digest),
        (
            _text(document.get("camera_registration_digest"), "camera_registration_digest"),
            expectations.camera_registration_digest,
        ),
        (_text(document.get("policy_digest"), "policy_digest"), policy.canonical_digest),
        (_text(document.get("model_digest"), "model_digest"), expectations.model_digest),
    )
    if any(not _sha(actual) or actual != expected for actual, expected in bindings):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "scene pose evidence binding drift")
    if expectations.model_digest != pinned_model_digest() or not _sha(expectations.pose_digest):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "scene pose model identity drift")
    if declared != expectations.pose_digest or declared != scene_pose_content_digest(document):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "scene pose content digest drift")
    age = _number(expectations.planning_timestamp, "planning_timestamp") - timestamp
    if age < 0.0 or age > policy.timing.sample_max_age_seconds:
        raise RolloutViolation(RolloutCode.R_STALE, "scene pose is stale or future")
    return SceneObjectPoseReceipt(
        _SEAL,
        sample_id,
        timestamp,
        bindings[0][0],
        bindings[1][0],
        bindings[2][0],
        bindings[3][0],
        bindings[4][0],
        pusher,
        push_t,
        declared,
    )


def apply_scene_object_pose(
    workspace: MuJoCoWorkspace, receipt: SceneObjectPoseReceipt | None
) -> None:
    """Apply only parser-authenticated physical obstacle transforms."""
    if receipt is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "scene object pose receipt is required")
    if type(receipt) is not SceneObjectPoseReceipt or not receipt.authenticated():
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "scene pose is not authenticated")
    data = cast("_PoseData", workspace.scene.data)
    data.mocap_pos[0] = receipt.pusher_transform[:3]
    data.mocap_quat[0] = receipt.pusher_transform[3:]
    yaw = _yaw(receipt.push_t_transform)
    for value, name in zip(
        (*receipt.push_t_transform[:2], yaw),
        ("push_t_x", "push_t_y", "push_t_yaw"),
        strict=True,
    ):
        qpos, _ = _joint(workspace, name)
        data.qpos[qpos] = value
