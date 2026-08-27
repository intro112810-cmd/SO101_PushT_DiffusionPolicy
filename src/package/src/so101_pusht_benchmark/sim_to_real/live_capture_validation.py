"""Fail-closed authority, identity, profile, and timing checks for live reads."""

from __future__ import annotations

import math
from typing import Protocol

from .live_capture_identity import ApprovedLiveIdentity
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority
from .live_capture_types import (
    AdapterIdentity,
    CameraObservation,
    DeviceIdentityProbe,
    DigestFile,
    LiveCaptureConfiguration,
    SampleCaptureWindow,
    TimedCameraRead,
    TimedJointRead,
)
from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "finite",
    "require_adapter_identity",
    "require_camera_profile",
    "require_live_identity",
    "require_preflight",
    "require_window",
)


def finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RolloutViolation(RolloutCode.R_NONFINITE, label)
    return result


def _digest(value: str | None, label: str) -> str:
    if value is None or len(value) != 64:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label) from exc
    return value.lower()


LiveIdentity = ApprovedLiveIdentity | ProductionReadOnlyAcquisitionAuthority


class CameraProfileAuthority(Protocol):
    @property
    def camera_width(self) -> int: ...

    @property
    def camera_height(self) -> int: ...

    @property
    def camera_fps(self) -> float: ...


def require_live_identity(value: object) -> LiveIdentity:
    if type(value) is ApprovedLiveIdentity:
        identity = value
        approved = identity.has_production_authority_marker()
        valid_scope = identity.artifact_scope == "production"
    elif type(value) is ProductionReadOnlyAcquisitionAuthority:
        identity = value
        approved = identity.has_read_only_authority_marker()
        valid_scope = identity.artifact_scope == "read_only_evidence_acquisition"
    else:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "approved live identity required")
    if not valid_scope or not approved:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "approved live identity required")
    for label, digest in (
        ("provider identity", identity.provider_digest),
        ("profile identity", identity.profile_digest),
        ("camera identity", identity.camera_device_digest),
        ("follower identity", identity.follower_device_digest),
        ("calibration identity", identity.calibration_digest),
        ("identity evidence", identity.identity_digest),
    ):
        _digest(digest, label)
    return identity


def require_preflight(
    identity: LiveIdentity,
    configuration: LiveCaptureConfiguration,
    device_probe: DeviceIdentityProbe,
    profile_digest: DigestFile,
    calibration_digest: DigestFile,
) -> None:
    if type(identity) is ProductionReadOnlyAcquisitionAuthority and (
        configuration.profile_path.resolve(strict=False) != identity.profile_path
        or configuration.calibration_file.absolute() != identity.calibration_path
        or configuration.camera_device.absolute() != identity.camera_device_path
        or configuration.follower_device.absolute() != identity.follower_device_path
        or configuration.follower_calibration_id != identity.calibration_id
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "approved canonical path mismatch")
    checks = (
        (
            _digest(profile_digest(configuration.profile_path), "profile digest"),
            identity.profile_digest,
        ),
        (
            _digest(calibration_digest(configuration.calibration_file), "calibration digest"),
            identity.calibration_digest,
        ),
    )
    if any(observed != expected for observed, expected in checks):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "approved file identity mismatch")
    for path, expected in (
        (configuration.camera_device, identity.camera_device_digest),
        (configuration.follower_device, identity.follower_device_digest),
    ):
        observed = device_probe(path)
        if observed is None:
            raise RolloutViolation(RolloutCode.R_MISSING, f"approved device is missing: {path}")
        if _digest(observed, "device identity") != expected:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "approved device identity mismatch")
    configured = (
        configuration.camera_width,
        configuration.camera_height,
        configuration.camera_fps,
    )
    approved = (identity.camera_width, identity.camera_height, identity.camera_fps)
    if configured != approved:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "approved camera profile mismatch")


def require_adapter_identity(
    observed: AdapterIdentity,
    identity: LiveIdentity | AdapterIdentity,
    *,
    camera: bool,
) -> None:
    if isinstance(identity, AdapterIdentity):
        expected_device = identity.device_digest
        expected_provider = identity.provider_digest
        expected_calibration = identity.calibration_digest
    else:
        expected_device = (
            identity.camera_device_digest if camera else identity.follower_device_digest
        )
        expected_provider = identity.provider_digest
        expected_calibration = None if camera else identity.calibration_digest
    if (
        _digest(observed.provider_digest, "adapter provider identity") != expected_provider
        or _digest(observed.device_digest, "adapter device identity") != expected_device
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "adapter identity mismatch")
    if camera:
        if observed.calibration_digest is not None:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "camera adapter identity mismatch")
    elif _digest(observed.calibration_digest, "adapter calibration identity") != (
        expected_calibration
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "adapter identity mismatch")


def require_camera_profile(
    observed: CameraObservation,
    identity: CameraProfileAuthority,
) -> None:
    if (
        observed.width != identity.camera_width
        or observed.height != identity.camera_height
        or not math.isclose(observed.fps, identity.camera_fps, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise RolloutViolation(RolloutCode.CAMERA_UNREGISTERED, "observed camera profile drift")


def require_window(
    camera: TimedCameraRead,
    joint: TimedJointRead,
    *,
    sample_id: str,
    now: float,
    previous: SampleCaptureWindow | None,
) -> SampleCaptureWindow:
    camera_start = finite(camera.started_at, "camera read start")
    camera_end = finite(camera.completed_at, "camera read completion")
    joint_start = finite(joint.started_at, "joint read start")
    joint_end = finite(joint.completed_at, "joint read completion")
    if (
        not camera.read_id
        or not joint.read_id
        or not (camera_start <= camera_end <= now)
        or not (joint_start <= joint_end <= now)
    ):
        raise RolloutViolation(RolloutCode.R_STALE, "invalid live capture window")
    if previous is not None and (
        camera.read_id == previous.camera_read_id
        or joint.read_id == previous.joint_read_id
        or min(camera_start, joint_start)
        <= max(previous.camera_completed_at, previous.joint_completed_at)
    ):
        raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "non-monotonic provider read")
    camera_midpoint = (camera_start + camera_end) / 2.0
    joint_midpoint = (joint_start + joint_end) / 2.0
    return SampleCaptureWindow(
        sample_id,
        camera.read_id,
        camera_start,
        camera_end,
        camera_end - camera_start,
        joint.read_id,
        joint_start,
        joint_end,
        joint_end - joint_start,
        abs(camera_midpoint - joint_midpoint),
    )
