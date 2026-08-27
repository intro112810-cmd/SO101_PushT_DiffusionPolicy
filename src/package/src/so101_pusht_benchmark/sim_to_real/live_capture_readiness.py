"""Event-driven readiness joins for camera priming and bus connection."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .live_capture_phase import PhaseContext, PhaseFaultError, next_event, provider_fault
from .live_capture_protocol import (
    CameraReady,
    JointReady,
    ProviderCallStarted,
    ProviderEvent,
    ProviderFailed,
    ProviderProcess,
    ProviderProcessRuntime,
    ProviderProtocolError,
    ProviderRole,
)
from .sample_capture import Clock

__all__ = ("ReadinessRequest", "await_camera_ready", "await_joint_ready")


@dataclass(frozen=True, slots=True)
class ReadinessRequest:
    runtime: ProviderProcessRuntime
    process: ProviderProcess
    context: PhaseContext
    clock: Clock
    events: list[ProviderEvent]


def await_camera_ready(request: ReadinessRequest) -> CameraReady:
    """Await one worker start followed by one discarded priming completion."""
    worker_started: float | None = None
    while True:
        try:
            event = next_event(
                request.runtime,
                (request.process,),
                request.context,
                frozenset({ProviderRole.CAMERA}),
                request.clock,
            )
        except PhaseFaultError as exc:
            raise PhaseFaultError(
                exc.primary,
                replace(exc.phase_evidence, camera_worker_started_at=worker_started),
                exc.active_roles,
            ) from exc
        request.events.append(event)
        if isinstance(event, ProviderCallStarted) and event.sample_index is None:
            worker_started = event.observed_at
            continue
        if isinstance(event, CameraReady):
            return event
        if isinstance(event, ProviderFailed):
            raise provider_fault(event, request.context, frozenset())
        raise ProviderProtocolError("unexpected camera readiness event")


def await_joint_ready(request: ReadinessRequest) -> JointReady:
    """Await one worker start followed by one direct-bus connection completion."""
    worker_started: float | None = None
    while True:
        try:
            event = next_event(
                request.runtime,
                (request.process,),
                request.context,
                frozenset({ProviderRole.JOINT}),
                request.clock,
            )
        except PhaseFaultError as exc:
            raise PhaseFaultError(
                exc.primary,
                replace(exc.phase_evidence, joint_worker_started_at=worker_started),
                exc.active_roles,
            ) from exc
        request.events.append(event)
        if isinstance(event, ProviderCallStarted) and event.sample_index is None:
            worker_started = event.observed_at
            continue
        if isinstance(event, JointReady):
            return event
        if isinstance(event, ProviderFailed):
            raise provider_fault(event, request.context, frozenset())
        raise ProviderProtocolError("unexpected joint readiness event")
