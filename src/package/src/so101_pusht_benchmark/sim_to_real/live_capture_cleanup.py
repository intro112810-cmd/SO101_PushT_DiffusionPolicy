"""IPC-independent stop, escalation, child failure, and reap evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .live_capture_protocol import (
    ProviderClosed,
    ProviderEvent,
    ProviderProcess,
    ProviderProcessRuntime,
    ProviderRole,
    StopProvider,
)
from .live_capture_types import ProcessCleanupEvidence
from .sample_capture import Clock

__all__ = ("CleanupRequest", "cleanup_processes")


@dataclass(frozen=True, slots=True)
class CleanupRequest:
    runtime: ProviderProcessRuntime
    processes: tuple[ProviderProcess, ...]
    active_roles: frozenset[ProviderRole]
    shutdown_grace: float
    clock: Clock
    events: list[ProviderEvent]


def _collect_close_events(
    request: CleanupRequest,
    deadline: float,
    errors: dict[ProviderRole, list[str]],
) -> dict[ProviderRole, ProviderClosed]:
    closed: dict[ProviderRole, ProviderClosed] = {}
    while len(closed) < len(request.processes):
        now = request.clock()
        if now > deadline:
            break
        pending = tuple(process for process in request.processes if process.role not in closed)
        try:
            ready = request.runtime.wait(pending, max(0.0, deadline - now))
        except (OSError, RuntimeError) as exc:
            for process in pending:
                errors[process.role].append(f"cleanup wait: {type(exc).__name__}: {exc}")
            break
        if not ready:
            break
        for process in ready:
            try:
                event = process.receive()
            except (EOFError, OSError, RuntimeError) as exc:
                errors[process.role].append(f"cleanup receive: {type(exc).__name__}: {exc}")
                closed[process.role] = ProviderClosed(
                    process.role,
                    request.clock(),
                    False,
                    None,
                )
                continue
            request.events.append(event)
            if isinstance(event, ProviderClosed):
                closed[event.role] = event
    return closed


def cleanup_processes(request: CleanupRequest) -> tuple[ProcessCleanupEvidence, ...]:
    """Never let broken live IPC prevent terminate, kill, reap, or evidence."""
    errors: dict[ProviderRole, list[str]] = {process.role: [] for process in request.processes}
    cooperative: set[ProviderRole] = set()
    for process in request.processes:
        if process.role not in request.active_roles and process.is_alive():
            try:
                process.send(StopProvider(request.clock()))
                cooperative.add(process.role)
            except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
                errors[process.role].append(f"cooperative stop: {type(exc).__name__}: {exc}")
    close_deadline = request.clock() + request.shutdown_grace
    closed = _collect_close_events(request, close_deadline, errors)
    terminated: set[ProviderRole] = set()
    killed: set[ProviderRole] = set()
    reaped: dict[ProviderRole, bool] = {}
    for process in request.processes:
        if process.is_alive():
            try:
                process.terminate()
                terminated.add(process.role)
            except (OSError, RuntimeError) as exc:
                errors[process.role].append(f"terminate: {type(exc).__name__}: {exc}")
        reaped[process.role] = process.join(request.shutdown_grace)
    for process in request.processes:
        if not reaped[process.role] and process.is_alive():
            try:
                process.kill()
                killed.add(process.role)
            except (OSError, RuntimeError) as exc:
                errors[process.role].append(f"kill: {type(exc).__name__}: {exc}")
            reaped[process.role] = process.join(request.shutdown_grace)
    evidence: list[ProcessCleanupEvidence] = []
    for process in request.processes:
        close_event = closed.get(process.role)
        if close_event is not None and close_event.cleanup_error is not None:
            errors[process.role].append(close_event.cleanup_error)
        if not reaped[process.role]:
            errors[process.role].append("provider process was not reaped within shutdown policy")
        child_failure = None
        if reaped[process.role]:
            try:
                child_failure = process.child_failure()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                errors[process.role].append(f"child failure evidence: {type(exc).__name__}: {exc}")
        exit_code = process.exit_code() if reaped[process.role] else None
        if reaped[process.role]:
            try:
                process.close()
            except (OSError, RuntimeError) as exc:
                errors[process.role].append(f"process close: {type(exc).__name__}: {exc}")
        evidence.append(
            ProcessCleanupEvidence(
                process.role.value,
                process.role in cooperative,
                close_event.adapter_closed if close_event is not None else False,
                process.role in terminated,
                process.role in killed,
                reaped[process.role],
                exit_code,
                tuple(errors[process.role]),
                None if child_failure is None else child_failure.phase,
                None if child_failure is None else child_failure.observed_at,
                None if child_failure is None else child_failure.error_type,
                None if child_failure is None else child_failure.error_message,
                None if child_failure is None else child_failure.traceback_text,
            )
        )
    return tuple(evidence)
