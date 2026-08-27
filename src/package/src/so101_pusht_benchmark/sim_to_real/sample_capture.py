"""Read-only synchronized physical sample acquisition.

The camera and joint providers are injected behind narrow protocols so the
acquisition path is testable without hardware, without sleeps, and without
importing any motor-write or robot-lifecycle symbol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import time
from typing import Protocol

from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_identity import digest_content
from .rollout_record_types import BodyDegrees, PhysicalSample

__all__ = (
    "BodyDegrees",
    "CameraFrame",
    "CameraSource",
    "Clock",
    "DuplicateFrameSource",
    "JointSource",
    "JointState",
    "SampleLimits",
    "StaticCameraSource",
    "StaticJointSource",
    "acquire_samples",
    "build_sample",
    "load_policy_limits",
    "sample_as_record",
)


class CameraSource(Protocol):
    """A provider of camera frames bound to a monotonic wall-clock reading."""

    def next_frame(self) -> CameraFrame: ...


class JointSource(Protocol):
    """A provider of five-motor body-joint states bound to a wall-clock reading."""

    def next_state(self) -> JointState: ...


class Clock(Protocol):
    """A monotonic time source; production uses ``time.monotonic``."""

    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class CameraFrame:
    timestamp: float
    digest: str


@dataclass(frozen=True, slots=True)
class JointState:
    timestamp: float
    body_degrees: BodyDegrees
    device_digest: str
    calibration_digest: str


@dataclass(frozen=True, slots=True)
class SampleLimits:
    max_age: float
    max_skew: float


@dataclass(slots=True)
class StaticCameraSource:
    frames: tuple[bytes, ...]
    digests: tuple[str, ...]
    timestamps: tuple[float, ...]
    _index: int = field(init=False, default=0, repr=False)

    def __init__(self, frames: Sequence[bytes], timestamps: Sequence[float]) -> None:
        self.frames = tuple(frames)
        self.digests = tuple(digest_content({"frame": frame.decode("latin-1")}) for frame in frames)
        self.timestamps = tuple(timestamps)
        if len(self.timestamps) != len(self.frames):
            raise RolloutViolation(RolloutCode.R_MISSING, "camera timestamp count mismatch")
        self._index = 0

    def next_frame(self) -> CameraFrame:
        index = self._index
        if index >= len(self.frames):
            raise RolloutViolation(RolloutCode.R_MISSING, "camera stream exhausted")
        self._index = index + 1
        return CameraFrame(self.timestamps[index], self.digests[index])


@dataclass(slots=True)
class DuplicateFrameSource:
    frames: tuple[bytes, ...]
    base: float
    _index: int = field(init=False, default=0, repr=False)

    def __init__(self, frames: Sequence[bytes], base: float) -> None:
        self.frames = tuple(frames)
        self.base = base
        self._index = 0

    def next_frame(self) -> CameraFrame:
        index = self._index
        if index >= len(self.frames):
            raise RolloutViolation(RolloutCode.R_MISSING, "camera stream exhausted")
        self._index = index + 1
        digest = digest_content({"frame": self.frames[index].decode("latin-1")})
        return CameraFrame(self.base + index * 0.01, digest)


@dataclass(slots=True)
class StaticJointSource:
    states: tuple[JointState, ...]
    _index: int = field(init=False, default=0, repr=False)

    def __init__(self, states: Sequence[JointState], count: int) -> None:
        if len(states) != count:
            raise RolloutViolation(RolloutCode.R_MISSING, "joint stream length")
        self.states = tuple(states)
        self._index = 0

    def next_state(self) -> JointState:
        index = self._index
        if index >= len(self.states):
            raise RolloutViolation(RolloutCode.R_MISSING, "joint stream exhausted")
        self._index = index + 1
        return self.states[index]


def load_policy_limits(max_age: float, max_skew: float) -> SampleLimits:
    if not math.isfinite(max_age) or max_age <= 0:
        raise RolloutViolation(RolloutCode.R_NONFINITE, "max_age")
    if not math.isfinite(max_skew) or max_skew < 0:
        raise RolloutViolation(RolloutCode.R_NONFINITE, "max_skew")
    return SampleLimits(max_age=max_age, max_skew=max_skew)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutViolation(RolloutCode.R_NONFINITE, label)
    result = float(value)
    if not math.isfinite(result):
        raise RolloutViolation(RolloutCode.R_NONFINITE, label)
    return result


def build_sample(
    camera: CameraFrame,
    joint: JointState,
    *,
    now: float,
    sample_id: str,
    policy: SampleLimits,
) -> PhysicalSample:
    """Validate timing/skew and mint one content-addressed physical sample."""
    camera_timestamp = _finite(camera.timestamp, "camera_timestamp")
    joint_timestamp = _finite(joint.timestamp, "joint_timestamp")
    now = _finite(now, "now")
    if not sample_id:
        raise RolloutViolation(RolloutCode.R_MISSING, "sample_id")
    if abs(camera_timestamp - joint_timestamp) > policy.max_skew:
        raise RolloutViolation(RolloutCode.R_STALE, "camera/joint skew")
    freshness = min(camera_timestamp, joint_timestamp)
    if now - freshness > policy.max_age:
        raise RolloutViolation(RolloutCode.R_STALE, "sample too old")
    body = tuple(_finite(value, "body_degrees") for value in joint.body_degrees)
    if len(body) != 5:
        raise RolloutViolation(RolloutCode.R_MISSING, "body_degrees length")
    payload = {
        "kind": "physical_sample",
        "record_id": sample_id,
        "created_at": now,
        "camera_timestamp": camera_timestamp,
        "joint_timestamp": joint_timestamp,
        "frame_digest": camera.digest.lower(),
        "body_degrees": list(body),
        "device_digest": joint.device_digest.lower(),
        "calibration_digest": joint.calibration_digest.lower(),
    }
    return PhysicalSample(
        sample_id,
        now,
        digest_content(payload),
        camera_timestamp,
        joint_timestamp,
        camera.digest.lower(),
        body,
        joint.device_digest.lower(),
        joint.calibration_digest.lower(),
    )


def sample_as_record(sample: PhysicalSample) -> dict[str, object]:
    """Serialize the canonical record consumed by :mod:`rollout_records`."""
    return {
        "kind": "physical_sample",
        "record_id": sample.record_id,
        "created_at": sample.created_at,
        "digest": sample.digest,
        "camera_timestamp": sample.camera_timestamp,
        "joint_timestamp": sample.joint_timestamp,
        "frame_digest": sample.frame_digest,
        "body_degrees": list(sample.body_degrees),
        "device_digest": sample.device_digest,
        "calibration_digest": sample.calibration_digest,
    }


def acquire_samples(
    camera: CameraSource,
    joint: JointSource,
    *,
    count: int,
    policy: SampleLimits,
    clock: Clock = time.monotonic,
) -> list[PhysicalSample]:
    """Acquire ``count`` genuine samples and reject duplicate/stale/skewed data."""
    if count < 1:
        raise RolloutViolation(RolloutCode.R_MISSING, "count")
    samples: list[PhysicalSample] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    previous_now: float | None = None
    for index in range(count):
        frame = camera.next_frame()
        state = joint.next_state()
        now = _finite(clock(), "now")
        if previous_now is not None and now <= previous_now:
            raise RolloutViolation(RolloutCode.R_STALE, "non-monotonic clock")
        previous_now = now
        sample = build_sample(
            frame,
            state,
            now=now,
            sample_id=f"sample-{index:03d}",
            policy=policy,
        )
        identity = (sample.frame_digest, sample.body_degrees)
        if identity in seen:
            raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "duplicate snapshot")
        seen.add(identity)
        samples.append(sample)
    return samples
