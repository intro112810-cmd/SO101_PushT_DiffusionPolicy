"""Post-read freshness, skew, distinctness, and sample construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .live_capture_types import (
    SampleCaptureWindow,
    TimedCameraRead,
    TimedJointRead,
)
from .live_capture_validation import require_window
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority
from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_record_types import PhysicalSample
from .sample_capture import CameraFrame, JointState, build_sample, load_policy_limits

__all__ = ("PairAcceptanceRequest", "accept_pair")


@dataclass(frozen=True, slots=True)
class PairAcceptanceRequest:
    authority: ProductionReadOnlyAcquisitionAuthority
    camera_read: TimedCameraRead
    joint_read: TimedJointRead
    now: float
    index: int
    previous: SampleCaptureWindow | None
    frame_digests: set[str]


def accept_pair(request: PairAcceptanceRequest) -> tuple[PhysicalSample, SampleCaptureWindow]:
    """Build one sample only after both complete provider reads pass policy."""
    sample_id = f"sample-{request.index:03d}"
    window = require_window(
        request.camera_read,
        request.joint_read,
        sample_id=sample_id,
        now=request.now,
        previous=request.previous,
    )
    if not request.camera_read.frame_bytes:
        raise RolloutViolation(RolloutCode.R_MISSING, "empty camera frame")
    frame_digest = hashlib.sha256(request.camera_read.frame_bytes).hexdigest()
    if frame_digest in request.frame_digests:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "duplicate camera bytes")
    request.frame_digests.add(frame_digest)
    authority = request.authority
    limits = load_policy_limits(
        authority.timing.sample_max_age_seconds,
        authority.timing.sample_max_skew_seconds,
    )
    sample = build_sample(
        CameraFrame(
            (window.camera_started_at + window.camera_completed_at) / 2.0,
            frame_digest,
        ),
        JointState(
            (window.joint_started_at + window.joint_completed_at) / 2.0,
            request.joint_read.body_degrees,
            authority.follower_device_digest,
            authority.calibration_digest,
        ),
        now=request.now,
        sample_id=sample_id,
        policy=limits,
    )
    return sample, window
