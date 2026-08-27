"""Deadline and child-exit classification shared by live provider phases."""

from __future__ import annotations

from dataclasses import dataclass

from .live_capture_failure import CapturePrimaryError
from .live_capture_protocol import (
    CameraReady,
    JointReady,
    ProviderEvent,
    ProviderFailed,
    ProviderProcess,
    ProviderProcessRuntime,
    ProviderRole,
    ProviderRuntimeReady,
)
from .live_capture_types import PhaseEvidence
from .sample_capture import Clock

__all__ = (
    "PhaseContext",
    "PhaseFaultError",
    "next_event",
    "provider_fault",
)


@dataclass(frozen=True, slots=True)
class PhaseContext:
    phase: str
    sample_index: int | None
    started: float
    deadline: float


@dataclass(frozen=True, slots=True)
class _FaultReport:
    detail: str
    error_type: str
    observed: float
    role: ProviderRole | None


@dataclass(frozen=True, slots=True)
class PhaseFaultError(RuntimeError):
    primary: CapturePrimaryError
    phase_evidence: PhaseEvidence
    active_roles: frozenset[ProviderRole]

    def __str__(self) -> str:
        """Retain the primary provider or timeout detail."""
        return self.primary.detail


def _phase_fault(
    report: _FaultReport,
    context: PhaseContext,
    active: frozenset[ProviderRole],
) -> PhaseFaultError:
    return PhaseFaultError(
        CapturePrimaryError(
            report.error_type,
            report.detail,
            context.phase,
            context.sample_index,
            None if report.role is None else report.role.value,
            report.observed,
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
            report.observed if report.error_type == "TimeoutError" else None,
        ),
        active,
    )


def _event_time(event: ProviderEvent) -> float:
    if isinstance(event, ProviderRuntimeReady | CameraReady | JointReady):
        return event.completed_at
    return event.observed_at


def next_event(
    runtime: ProviderProcessRuntime,
    processes: tuple[ProviderProcess, ...],
    context: PhaseContext,
    active: frozenset[ProviderRole],
    clock: Clock,
) -> ProviderEvent:
    """Await one exact process event once; classify EOF without retry or polling."""
    now = clock()
    ready = runtime.wait(processes, max(0.0, context.deadline - now))
    if not ready:
        observed = clock()
        raise _phase_fault(
            _FaultReport(f"{context.phase} timed out", "TimeoutError", observed, None),
            context,
            active,
        )
    process = ready[0]
    try:
        event = process.receive()
    except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
        observed = clock()
        raise _phase_fault(
            _FaultReport(
                f"provider IPC ended ({type(exc).__name__}); exit_code={process.exit_code()}",
                "ChildProcessError",
                observed,
                process.role,
            ),
            context,
            active,
        ) from exc
    observed = _event_time(event)
    if observed > context.deadline:
        raise _phase_fault(
            _FaultReport(
                "provider result completed after its deadline",
                "TimeoutError",
                observed,
                event.role,
            ),
            context,
            active,
        )
    return event


def provider_fault(
    event: ProviderFailed,
    context: PhaseContext,
    active: frozenset[ProviderRole],
) -> PhaseFaultError:
    """Preserve the provider error while marking its call inactive."""
    return _phase_fault(
        _FaultReport(
            event.error_message,
            event.error_type,
            event.observed_at,
            event.role,
        ),
        context,
        active - {event.role},
    )
