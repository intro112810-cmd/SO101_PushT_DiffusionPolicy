"""Narrow types for governed read-only physical sample providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority
from .live_capture_runtime import RuntimeDependencyReceipt, RuntimePreflight
from .rollout_record_types import BodyDegrees, PhysicalSample
from .sample_capture import Clock

__all__ = (
    "AdapterIdentity",
    "CameraObservation",
    "DeviceIdentityProbe",
    "DigestFile",
    "LiveCameraFactory",
    "LiveCameraReader",
    "LiveCaptureConfiguration",
    "LiveCaptureProviders",
    "LiveCaptureRequest",
    "LiveCaptureResult",
    "LiveJointFactory",
    "LiveJointReader",
    "PhaseEvidence",
    "PrimingEvidence",
    "ProcessCleanupEvidence",
    "SampleCaptureWindow",
    "TimedCameraRead",
    "TimedJointRead",
)


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    """Identity exposed before an adapter receives permission to open a device."""

    provider_digest: str
    device_digest: str
    calibration_digest: str | None


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """Read-only camera properties observed after opening an existing capture."""

    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class TimedCameraRead:
    """One byte-exact frame with timestamps bracketing the provider read."""

    read_id: str
    frame_bytes: bytes
    started_at: float
    completed_at: float


@dataclass(frozen=True, slots=True)
class TimedJointRead:
    """One direct-bus joint result with timestamps bracketing the sync read."""

    read_id: str
    body_degrees: BodyDegrees
    started_at: float
    completed_at: float


@dataclass(frozen=True, slots=True)
class LiveCaptureConfiguration:
    """Approved paths and observed camera profile used by live adapters."""

    profile_path: Path
    camera_device: Path
    follower_device: Path
    calibration_file: Path
    camera_width: int
    camera_height: int
    camera_fps: float
    follower_calibration_id: str = ""


@dataclass(frozen=True, slots=True)
class SampleCaptureWindow:
    """The camera and joint read intervals used to build one sample."""

    sample_id: str
    camera_read_id: str
    camera_started_at: float
    camera_completed_at: float
    camera_duration_seconds: float
    joint_read_id: str
    joint_started_at: float
    joint_completed_at: float
    joint_duration_seconds: float
    midpoint_skew_seconds: float


@dataclass(frozen=True, slots=True)
class PrimingEvidence:
    """Exactly one discarded camera read used only for readiness."""

    read_id: str
    phase_started_at: float
    deadline: float
    worker_started_at: float
    read_started_at: float
    read_completed_at: float
    readiness_completed_at: float


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    phase: str
    sample_index: int | None
    phase_started_at: float
    deadline: float
    camera_worker_started_at: float | None
    joint_worker_started_at: float | None
    camera_completed_at: float | None
    joint_completed_at: float | None
    timeout_observed_at: float | None


@dataclass(frozen=True, slots=True)
class ProcessCleanupEvidence:
    role: str
    cooperative_stop_requested: bool
    adapter_closed: bool
    terminate_sent: bool
    kill_sent: bool
    process_reaped: bool
    exit_code: int | None
    cleanup_errors: tuple[str, ...]
    child_failure_phase: str | None = None
    child_failure_observed_at: float | None = None
    child_error_type: str | None = None
    child_error_message: str | None = None
    child_traceback: str | None = None


@dataclass(frozen=True, slots=True)
class LiveCaptureResult:
    """Two validated pairs plus readiness, phase, and reaped-cleanup evidence."""

    samples: tuple[PhysicalSample, PhysicalSample]
    windows: tuple[SampleCaptureWindow, SampleCaptureWindow]
    priming: PrimingEvidence | None = None
    phases: tuple[PhaseEvidence, ...] = ()
    cleanup: tuple[ProcessCleanupEvidence, ...] = ()
    runtime_dependency: RuntimeDependencyReceipt | None = None


class LiveCameraReader(Protocol):
    """Existing-capture read capability with no property mutation API."""

    identity: AdapterIdentity

    def open(self) -> CameraObservation: ...

    def next_frame(self) -> TimedCameraRead: ...

    def close(self) -> None: ...


class LiveJointReader(Protocol):
    """Persistent direct-bus Present_Position read capability."""

    identity: AdapterIdentity

    def open(self) -> None: ...

    def next_state(self) -> TimedJointRead: ...

    def close(self) -> None: ...


class LiveCameraFactory(Protocol):
    def __call__(self, configuration: LiveCaptureConfiguration, /) -> LiveCameraReader: ...


class LiveJointFactory(Protocol):
    def __call__(self, configuration: LiveCaptureConfiguration, /) -> LiveJointReader: ...


class DeviceIdentityProbe(Protocol):
    def __call__(self, path: Path, /) -> str | None: ...


class DigestFile(Protocol):
    def __call__(self, path: Path, /) -> str: ...


@dataclass(frozen=True, slots=True)
class LiveCaptureRequest:
    """Production authorities and approved hardware configuration for one capture."""

    policy: (
        ProductionApprovedSafetyPolicy
        | ProductionReadOnlyAcquisitionAuthority
        | FixtureApprovedSafetyPolicy
        | None
    )
    identity: object
    configuration: LiveCaptureConfiguration


@dataclass(frozen=True, slots=True)
class LiveCaptureProviders:
    """Injected read-only capabilities used only after authority preflight."""

    camera_factory: LiveCameraFactory
    joint_factory: LiveJointFactory
    device_probe: DeviceIdentityProbe
    profile_digest: DigestFile
    calibration_digest: DigestFile
    clock: Clock
    runtime_preflight: RuntimePreflight
