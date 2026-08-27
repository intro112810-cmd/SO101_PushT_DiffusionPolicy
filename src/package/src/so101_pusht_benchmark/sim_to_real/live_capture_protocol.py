"""Typed command, event, and process seams for supervised live acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Protocol

from .live_capture_types import (
    AdapterIdentity,
    CameraObservation,
    LiveCameraFactory,
    LiveCaptureConfiguration,
    LiveJointFactory,
    TimedCameraRead,
    TimedJointRead,
)
from .live_capture_runtime import RuntimeDependencyReceipt, RuntimePreflight
from .sample_capture import Clock

__all__ = (
    "ArmSample",
    "CameraReady",
    "CameraWorkerSpec",
    "ExpectedCameraProfile",
    "JointReady",
    "JointWorkerSpec",
    "ProviderArmed",
    "ProviderCallStarted",
    "ProviderClosed",
    "ProviderCommand",
    "ProviderEvent",
    "ProviderFailed",
    "ProviderProcess",
    "ProviderProcessRuntime",
    "ProviderProtocolError",
    "ProviderRole",
    "ProviderRuntimeReady",
    "ReleaseSample",
    "StartProvider",
    "StopProvider",
    "WorkerCompleted",
    "WorkerSpec",
)


class ProviderProtocolError(RuntimeError):
    """An internal worker command or event violated the typed protocol."""


@unique
class ProviderRole(str, Enum):
    CAMERA = "camera"
    JOINT = "joint"


@dataclass(frozen=True, slots=True)
class ArmSample:
    sample_index: int


@dataclass(frozen=True, slots=True)
class ReleaseSample:
    sample_index: int


@dataclass(frozen=True, slots=True)
class StartProvider:
    requested_at: float


@dataclass(frozen=True, slots=True)
class StopProvider:
    requested_at: float


ProviderCommand = ArmSample | ReleaseSample | StartProvider | StopProvider


@dataclass(frozen=True, slots=True)
class ProviderRuntimeReady:
    role: ProviderRole
    worker_started_at: float
    completed_at: float
    dependency: RuntimeDependencyReceipt


@dataclass(frozen=True, slots=True)
class CameraReady:
    role: ProviderRole
    phase_started_at: float
    worker_started_at: float
    completed_at: float
    observation: CameraObservation
    priming_read_id: str
    priming_started_at: float
    priming_completed_at: float


@dataclass(frozen=True, slots=True)
class JointReady:
    role: ProviderRole
    phase_started_at: float
    worker_started_at: float
    completed_at: float


@dataclass(frozen=True, slots=True)
class ProviderArmed:
    role: ProviderRole
    sample_index: int
    observed_at: float


@dataclass(frozen=True, slots=True)
class ProviderCallStarted:
    role: ProviderRole
    sample_index: int | None
    observed_at: float


@dataclass(frozen=True, slots=True)
class WorkerCompleted:
    role: ProviderRole
    sample_index: int
    observed_at: float
    camera_read: TimedCameraRead | None
    joint_read: TimedJointRead | None


@dataclass(frozen=True, slots=True)
class ProviderFailed:
    role: ProviderRole
    phase: str
    sample_index: int | None
    observed_at: float
    error_type: str
    error_message: str
    traceback_text: str


@dataclass(frozen=True, slots=True)
class ProviderClosed:
    role: ProviderRole
    observed_at: float
    adapter_closed: bool
    cleanup_error: str | None


ProviderEvent = (
    ProviderRuntimeReady
    | CameraReady
    | JointReady
    | ProviderArmed
    | ProviderCallStarted
    | WorkerCompleted
    | ProviderFailed
    | ProviderClosed
)


@dataclass(frozen=True, slots=True)
class ExpectedCameraProfile:
    camera_width: int
    camera_height: int
    camera_fps: float


@dataclass(frozen=True, slots=True)
class CameraWorkerSpec:
    configuration: LiveCaptureConfiguration
    factory: LiveCameraFactory
    expected_identity: AdapterIdentity
    expected_profile: ExpectedCameraProfile
    clock: Clock
    runtime_preflight: RuntimePreflight


@dataclass(frozen=True, slots=True)
class JointWorkerSpec:
    configuration: LiveCaptureConfiguration
    factory: LiveJointFactory
    expected_identity: AdapterIdentity
    clock: Clock
    runtime_preflight: RuntimePreflight


WorkerSpec = CameraWorkerSpec | JointWorkerSpec


class ProviderProcess(Protocol):
    role: ProviderRole

    def start(self) -> None: ...

    def send(self, command: ProviderCommand) -> None: ...

    def receive(self) -> ProviderEvent: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float) -> bool: ...

    def exit_code(self) -> int | None: ...

    def child_failure(self) -> ProviderFailed | None: ...

    def close(self) -> None: ...


class ProviderProcessRuntime(Protocol):
    def spawn(self, spec: WorkerSpec) -> ProviderProcess: ...

    def wait(
        self,
        processes: tuple[ProviderProcess, ...],
        timeout: float,
    ) -> tuple[ProviderProcess, ...]: ...
