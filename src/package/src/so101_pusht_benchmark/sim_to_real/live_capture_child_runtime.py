"""Join both child dependency preflights before authorizing device creation."""

from __future__ import annotations

from dataclasses import dataclass

from .live_capture_phase import PhaseContext, PhaseFaultError, next_event, provider_fault
from .live_capture_protocol import (
    ProviderEvent,
    ProviderFailed,
    ProviderProcess,
    ProviderProcessRuntime,
    ProviderProtocolError,
    ProviderRole,
    ProviderRuntimeReady,
)
from .live_capture_types import PhaseEvidence
from .sample_capture import Clock

__all__ = ("ChildRuntimeRequest", "await_child_runtime_preflights")


@dataclass(frozen=True, slots=True)
class ChildRuntimeRequest:
    runtime: ProviderProcessRuntime
    processes: tuple[ProviderProcess, ProviderProcess]
    context: PhaseContext
    clock: Clock
    events: list[ProviderEvent]


def await_child_runtime_preflights(request: ChildRuntimeRequest) -> PhaseEvidence:
    """Accept only matching camera and joint child runtime identities."""
    ready: dict[ProviderRole, ProviderRuntimeReady] = {}
    active = {ProviderRole.CAMERA, ProviderRole.JOINT}
    while len(ready) < 2:
        event = next_event(
            request.runtime,
            request.processes,
            request.context,
            frozenset(active),
            request.clock,
        )
        request.events.append(event)
        if isinstance(event, ProviderRuntimeReady):
            ready[event.role] = event
            active.discard(event.role)
            continue
        if isinstance(event, ProviderFailed):
            raise provider_fault(event, request.context, frozenset(active))
        raise ProviderProtocolError("unexpected child runtime preflight event")
    camera = ready[ProviderRole.CAMERA]
    joint = ready[ProviderRole.JOINT]
    if camera.dependency != joint.dependency:
        raise PhaseFaultError(
            provider_fault(
                ProviderFailed(
                    ProviderRole.JOINT,
                    "runtime_preflight",
                    None,
                    request.clock(),
                    "RuntimeDependencyMismatch",
                    "camera and joint child runtime identities disagree",
                    "",
                ),
                request.context,
                frozenset(),
            ).primary,
            PhaseEvidence(
                "runtime_preflight",
                None,
                request.context.started,
                request.context.deadline,
                camera.worker_started_at,
                joint.worker_started_at,
                camera.completed_at,
                joint.completed_at,
                None,
            ),
            frozenset(),
        )
    return PhaseEvidence(
        "runtime_preflight",
        None,
        request.context.started,
        request.context.deadline,
        camera.worker_started_at,
        joint.worker_started_at,
        camera.completed_at,
        joint.completed_at,
        None,
    )
