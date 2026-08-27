"""Arm, release, and join one complete camera/joint sample pair."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .live_capture_failure import CapturePrimaryError
from .live_capture_phase import PhaseContext, PhaseFaultError, next_event, provider_fault
from .live_capture_protocol import (
    ArmSample,
    ProviderArmed,
    ProviderCallStarted,
    ProviderEvent,
    ProviderFailed,
    ProviderProcess,
    ProviderProcessRuntime,
    ProviderProtocolError,
    ProviderRole,
    ReleaseSample,
    WorkerCompleted,
)
from .live_capture_types import PhaseEvidence, TimedCameraRead, TimedJointRead
from .sample_capture import Clock

__all__ = ("ReadPairRequest", "read_pair")


@dataclass(frozen=True, slots=True)
class ReadPairRequest:
    runtime: ProviderProcessRuntime
    processes: tuple[ProviderProcess, ProviderProcess]
    index: int
    budget: float
    clock: Clock
    events: list[ProviderEvent]


def _send_command(
    process: ProviderProcess,
    command: ArmSample | ReleaseSample,
    context: PhaseContext,
    active: frozenset[ProviderRole],
    clock: Clock,
) -> None:
    try:
        process.send(command)
    except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
        observed = clock()
        raise PhaseFaultError(
            CapturePrimaryError(
                "ChildProcessError",
                f"provider IPC send failed ({type(exc).__name__})",
                context.phase,
                context.sample_index,
                process.role.value,
                observed,
            ),
            PhaseEvidence(
                context.phase,
                context.sample_index,
                context.started,
                context.deadline,
                None,
                None,
                None,
                None,
                observed,
            ),
            active,
        ) from exc


def _enrich_fault(
    fault: PhaseFaultError,
    worker_starts: dict[ProviderRole, float],
    completions: dict[ProviderRole, WorkerCompleted],
) -> PhaseFaultError:
    return PhaseFaultError(
        fault.primary,
        replace(
            fault.phase_evidence,
            camera_worker_started_at=worker_starts.get(ProviderRole.CAMERA),
            joint_worker_started_at=worker_starts.get(ProviderRole.JOINT),
            camera_completed_at=(
                completions[ProviderRole.CAMERA].observed_at
                if ProviderRole.CAMERA in completions
                else None
            ),
            joint_completed_at=(
                completions[ProviderRole.JOINT].observed_at
                if ProviderRole.JOINT in completions
                else None
            ),
        ),
        fault.active_roles,
    )


def read_pair(request: ReadPairRequest) -> tuple[TimedCameraRead, TimedJointRead, PhaseEvidence]:
    """Arm both workers, release both, and accept only two on-time completions."""
    started = request.clock()
    context = PhaseContext("sample_pair", request.index, started, started + request.budget)
    for process in request.processes:
        _send_command(process, ArmSample(request.index), context, frozenset(), request.clock)
    armed: set[ProviderRole] = set()
    while len(armed) < 2:
        event = next_event(
            request.runtime,
            request.processes,
            context,
            frozenset(),
            request.clock,
        )
        request.events.append(event)
        if isinstance(event, ProviderArmed) and event.sample_index == request.index:
            armed.add(event.role)
            continue
        if isinstance(event, ProviderFailed):
            raise provider_fault(event, context, frozenset())
        raise ProviderProtocolError("unexpected provider arm event")
    active = {ProviderRole.CAMERA, ProviderRole.JOINT}
    for process in request.processes:
        _send_command(
            process,
            ReleaseSample(request.index),
            context,
            frozenset(active),
            request.clock,
        )
    worker_starts: dict[ProviderRole, float] = {}
    completions: dict[ProviderRole, WorkerCompleted] = {}
    while len(completions) < 2:
        try:
            event = next_event(
                request.runtime,
                request.processes,
                context,
                frozenset(active),
                request.clock,
            )
        except PhaseFaultError as exc:
            raise _enrich_fault(exc, worker_starts, completions) from exc
        request.events.append(event)
        if isinstance(event, ProviderCallStarted) and event.sample_index == request.index:
            worker_starts[event.role] = event.observed_at
            continue
        if isinstance(event, WorkerCompleted) and event.sample_index == request.index:
            completions[event.role] = event
            active.discard(event.role)
            continue
        if isinstance(event, ProviderFailed):
            fault = provider_fault(event, context, frozenset(active))
            raise _enrich_fault(fault, worker_starts, completions) from fault
        raise ProviderProtocolError("unexpected provider sample event")
    camera_event = completions[ProviderRole.CAMERA]
    joint_event = completions[ProviderRole.JOINT]
    if camera_event.camera_read is None or joint_event.joint_read is None:
        raise ProviderProtocolError("provider returned the wrong result variant")
    return (
        camera_event.camera_read,
        joint_event.joint_read,
        PhaseEvidence(
            context.phase,
            request.index,
            context.started,
            context.deadline,
            worker_starts.get(ProviderRole.CAMERA),
            worker_starts.get(ProviderRole.JOINT),
            camera_event.observed_at,
            joint_event.observed_at,
            None,
        ),
    )
