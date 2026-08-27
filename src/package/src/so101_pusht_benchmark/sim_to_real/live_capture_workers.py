"""Provider-process entry points that exclusively own one read-only resource."""

from __future__ import annotations

from collections.abc import Callable
import traceback
from typing import Protocol

from .live_capture_protocol import (
    ArmSample,
    CameraReady,
    CameraWorkerSpec,
    JointReady,
    JointWorkerSpec,
    ProviderArmed,
    ProviderCallStarted,
    ProviderClosed,
    ProviderCommand,
    ProviderEvent,
    ProviderFailed,
    ProviderProtocolError,
    ProviderRole,
    ProviderRuntimeReady,
    ReleaseSample,
    StartProvider,
    StopProvider,
    WorkerCompleted,
)
from .live_capture_validation import require_adapter_identity, require_camera_profile

__all__ = ("run_camera_worker", "run_joint_worker")


class CommandReceiver(Protocol):
    def __call__(self) -> ProviderCommand: ...


class EventSender(Protocol):
    def __call__(self, event: ProviderEvent) -> None: ...


def _await_release(
    role: ProviderRole,
    index: int,
    clock: Callable[[], float],
    receive: CommandReceiver,
    send: EventSender,
) -> None:
    command = receive()
    if not isinstance(command, ArmSample) or command.sample_index != index:
        raise ProviderProtocolError("provider received an invalid arm command")
    send(ProviderArmed(role, index, clock()))
    command = receive()
    if not isinstance(command, ReleaseSample) or command.sample_index != index:
        raise ProviderProtocolError("provider received an invalid release command")


def _await_stop(receive: CommandReceiver) -> None:
    command = receive()
    if not isinstance(command, StopProvider):
        raise ProviderProtocolError("provider received work after its two-read budget")


def run_camera_worker(
    spec: CameraWorkerSpec,
    receive: CommandReceiver,
    send: EventSender,
) -> None:
    """Preflight, create, prime, read twice, and close one child-owned camera."""
    camera = None
    phase = "runtime_preflight"
    sample_index: int | None = None
    worker_started_at = spec.clock()
    try:
        dependency = spec.runtime_preflight()
        send(
            ProviderRuntimeReady(
                ProviderRole.CAMERA,
                worker_started_at,
                spec.clock(),
                dependency,
            )
        )
        if not isinstance(receive(), StartProvider):
            raise ProviderProtocolError("camera start was not authorized")
        phase = "camera_readiness"
        phase_started_at = spec.clock()
        send(ProviderCallStarted(ProviderRole.CAMERA, None, phase_started_at))
        camera = spec.factory(spec.configuration)
        require_adapter_identity(camera.identity, spec.expected_identity, camera=True)
        observation = camera.open()
        require_camera_profile(observation, spec.expected_profile)
        priming = camera.next_frame()
        send(
            CameraReady(
                ProviderRole.CAMERA,
                phase_started_at,
                worker_started_at,
                spec.clock(),
                observation,
                priming.read_id,
                priming.started_at,
                priming.completed_at,
            )
        )
        for index in range(2):
            phase = "sample_pair"
            sample_index = index
            _await_release(ProviderRole.CAMERA, index, spec.clock, receive, send)
            send(ProviderCallStarted(ProviderRole.CAMERA, index, spec.clock()))
            frame = camera.next_frame()
            send(WorkerCompleted(ProviderRole.CAMERA, index, spec.clock(), frame, None))
        phase = "shutdown"
        sample_index = None
        _await_stop(receive)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        send(
            ProviderFailed(
                ProviderRole.CAMERA,
                phase,
                sample_index,
                spec.clock(),
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )
    finally:
        closed = False
        cleanup_error: str | None = None
        if camera is not None:
            try:
                camera.close()
                closed = True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        send(ProviderClosed(ProviderRole.CAMERA, spec.clock(), closed, cleanup_error))


def run_joint_worker(
    spec: JointWorkerSpec,
    receive: CommandReceiver,
    send: EventSender,
) -> None:
    """Preflight, create, connect once, read twice, and close one child-owned bus."""
    joint = None
    phase = "runtime_preflight"
    sample_index: int | None = None
    worker_started_at = spec.clock()
    try:
        dependency = spec.runtime_preflight()
        send(
            ProviderRuntimeReady(
                ProviderRole.JOINT,
                worker_started_at,
                spec.clock(),
                dependency,
            )
        )
        if not isinstance(receive(), StartProvider):
            raise ProviderProtocolError("joint start was not authorized")
        phase = "joint_connect"
        phase_started_at = spec.clock()
        send(ProviderCallStarted(ProviderRole.JOINT, None, phase_started_at))
        joint = spec.factory(spec.configuration)
        require_adapter_identity(joint.identity, spec.expected_identity, camera=False)
        joint.open()
        send(JointReady(ProviderRole.JOINT, phase_started_at, worker_started_at, spec.clock()))
        for index in range(2):
            phase = "sample_pair"
            sample_index = index
            _await_release(ProviderRole.JOINT, index, spec.clock, receive, send)
            send(ProviderCallStarted(ProviderRole.JOINT, index, spec.clock()))
            state = joint.next_state()
            send(WorkerCompleted(ProviderRole.JOINT, index, spec.clock(), None, state))
        phase = "shutdown"
        sample_index = None
        _await_stop(receive)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        send(
            ProviderFailed(
                ProviderRole.JOINT,
                phase,
                sample_index,
                spec.clock(),
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )
    finally:
        closed = False
        cleanup_error: str | None = None
        if joint is not None:
            try:
                joint.close()
                closed = True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        send(ProviderClosed(ProviderRole.JOINT, spec.clock(), closed, cleanup_error))
