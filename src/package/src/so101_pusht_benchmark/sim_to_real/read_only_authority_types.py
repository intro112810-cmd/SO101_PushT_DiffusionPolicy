"""Sealed typed contract for signed read-only acquisition authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, InitVar
from datetime import datetime
import json
from pathlib import Path
from typing import NewType, Protocol

from .rollout_codes import RolloutCode, RolloutViolation

AUTHORITY_SCHEMA = "so101-read-only-evidence-acquisition-authority-v1"
AUTHORITY_SCHEME = "rsa-pkcs1v15-sha256-v1"
AUTHORITY_SCOPE = "read_only_evidence_acquisition"
AUTHORITY_FIELDS = frozenset(
    {
        "schema",
        "authority_version",
        "authority_id",
        "artifact_scope",
        "approved_by",
        "approved_at",
        "valid_from",
        "expires_at",
        "source_lineage_authority_digest",
        "provider_digest",
        "runtime",
        "profile",
        "follower",
        "camera",
        "thresholds",
        "permissions",
        "scheme",
        "trust_anchor_sha256",
        "authority_digest",
    }
)
CAMERA_PERMISSIONS = (
    "open_existing_capture",
    "observe_properties",
    "read_frames",
    "release_capture",
)
FOLLOWER_PERMISSIONS = (
    "direct_bus_connect",
    "sync_read:Present_Position",
    "disconnect:disable_torque=false",
)
MANUAL_POSITIONING_FOLLOWER_PERMISSIONS = (
    "direct_bus_connect",
    "sync_read:Torque_Enable",
    "sync_read:Present_Position",
    "disconnect:disable_torque=false",
)
FORBIDDEN_CAPABILITIES = (
    "Goal_Position",
    "sync_write",
    "torque_write",
    "configuration_write",
    "calibration_write",
    "SOFollower.connect",
    "SOFollower.configure",
    "SOFollower.calibrate",
    "SOFollower.send_action",
    "single_step",
    "bounded_rollout",
    "arming",
)


class _ConstructionSeal(Protocol):
    """Opaque verified-parser capability."""


READ_ONLY_CONSTRUCTION_SEAL: _ConstructionSeal = object()
CameraReadinessTimeoutSeconds = NewType("CameraReadinessTimeoutSeconds", float)
JointConnectTimeoutSeconds = NewType("JointConnectTimeoutSeconds", float)
PairCompletionTimeoutSeconds = NewType("PairCompletionTimeoutSeconds", float)
ShutdownGraceSeconds = NewType("ShutdownGraceSeconds", float)


@dataclass(frozen=True, slots=True)
class ReadOnlyRuntimePolicy:
    """Signed distribution, module ownership, origin, and content identity."""

    feetech_servo_sdk_distribution: str
    feetech_servo_sdk_version: str
    pyserial_distribution: str
    pyserial_version: str
    scservo_sdk_distribution: str
    scservo_sdk_module: str
    scservo_sdk_origin: Path
    scservo_sdk_origin_sha256: str


@dataclass(frozen=True, slots=True)
class ReadOnlyTimingPolicy:
    """Independent liveness budgets and post-read acceptance thresholds."""

    camera_readiness_timeout_seconds: CameraReadinessTimeoutSeconds
    joint_connect_timeout_seconds: JointConnectTimeoutSeconds
    sample_pair_completion_timeout_seconds: PairCompletionTimeoutSeconds
    shutdown_grace_seconds: ShutdownGraceSeconds
    sample_max_age_seconds: float
    sample_max_skew_seconds: float


@dataclass(frozen=True, slots=True)
class ReadOnlyCapturePolicy:
    camera_priming_frame_count: int
    accepted_sample_pair_count: int


@dataclass(frozen=True, slots=True)
class ReadOnlyCameraPolicy:
    max_reprojection_error_px: float
    min_correspondences: int
    max_correspondence_error_px: float


@dataclass(frozen=True, slots=True)
class ReadOnlyKinematicsPolicy:
    max_fk_residual_m: float


@dataclass(frozen=True, slots=True)
class ProductionReadOnlyAcquisitionAuthority:
    """Parser-sealed authority with no actuation fields or command budgets."""

    _construction_seal: InitVar[_ConstructionSeal]
    schema: str
    authority_id: str
    artifact_scope: str
    approved_by: str
    approved_at: datetime
    valid_from: datetime
    expires_at: datetime
    source_lineage_authority_digest: str
    provider_digest: str
    runtime: ReadOnlyRuntimePolicy
    profile_path: Path
    profile_digest: str
    follower_device_path: Path
    follower_device_digest: str
    calibration_id: str
    calibration_path: Path
    calibration_digest: str
    camera_device_path: Path
    camera_device_digest: str
    camera_width: int
    camera_height: int
    camera_fps: float
    timing: ReadOnlyTimingPolicy
    capture: ReadOnlyCapturePolicy
    camera: ReadOnlyCameraPolicy
    kinematics: ReadOnlyKinematicsPolicy
    camera_permissions: tuple[str, ...]
    follower_permissions: tuple[str, ...]
    forbidden_capabilities: tuple[str, ...]
    canonical_digest: str
    trust_anchor_sha256: str
    _authority_marker: _ConstructionSeal = field(init=False, repr=False, compare=False)

    def __post_init__(self, _construction_seal: _ConstructionSeal) -> None:
        """Reject construction outside the detached-signature parser."""
        if _construction_seal is not READ_ONLY_CONSTRUCTION_SEAL:
            raise RolloutViolation(
                RolloutCode.R_POLICY_UNAUTHORIZED,
                "read-only acquisition authority is parser-private",
            )
        object.__setattr__(self, "_authority_marker", READ_ONLY_CONSTRUCTION_SEAL)

    @property
    def identity_digest(self) -> str:
        """Expose the signed authority identity to acquisition receipt builders."""
        return self.canonical_digest

    def has_read_only_authority_marker(self) -> bool:
        return self._authority_marker is READ_ONLY_CONSTRUCTION_SEAL


def canonical_authority_bytes(value: Mapping[str, object]) -> bytes:
    """Return the exact detached-signature representation."""
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def require_read_only_acquisition_authority(
    value: object,
) -> ProductionReadOnlyAcquisitionAuthority:
    """Reject raw, fixture, actuation, and caller-constructed values."""
    if type(value) is not ProductionReadOnlyAcquisitionAuthority:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "read-only authority required")
    authority = value
    if (
        not authority.has_read_only_authority_marker()
        or authority.artifact_scope != AUTHORITY_SCOPE
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "read-only authority required")
    return authority
