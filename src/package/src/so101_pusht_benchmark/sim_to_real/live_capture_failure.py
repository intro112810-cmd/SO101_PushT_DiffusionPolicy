"""Canonical non-consumable evidence for terminal live-capture failures."""

from __future__ import annotations

from dataclasses import dataclass

from .live_capture_protocol import ProviderEvent
from .live_capture_types import PhaseEvidence, ProcessCleanupEvidence

__all__ = (
    "CapturePrimaryError",
    "LiveCaptureAttemptError",
    "LiveCaptureFailure",
    "terminal_failure_receipt",
)


@dataclass(frozen=True, slots=True)
class CapturePrimaryError:
    error_type: str
    detail: str
    phase: str
    sample_index: int | None
    role: str | None
    observed_at: float


@dataclass(frozen=True, slots=True)
class LiveCaptureFailure:
    attempt_id: str
    primary_error: CapturePrimaryError
    phases: tuple[PhaseEvidence, ...]
    events: tuple[ProviderEvent, ...]
    cleanup: tuple[ProcessCleanupEvidence, ...]
    completed_pair_count: int


@dataclass(frozen=True, slots=True)
class LiveCaptureAttemptError(RuntimeError):
    failure: LiveCaptureFailure

    def __str__(self) -> str:
        """Return the preserved primary acquisition error."""
        return self.failure.primary_error.detail


def _event_record(event: ProviderEvent) -> dict[str, str | int | float | bool | None]:
    record: dict[str, str | int | float | bool | None] = {
        "event": type(event).__name__,
        "role": event.role.value,
    }
    for name in (
        "sample_index",
        "observed_at",
        "phase_started_at",
        "worker_started_at",
        "completed_at",
        "priming_read_id",
        "priming_started_at",
        "priming_completed_at",
        "phase",
        "error_type",
        "error_message",
        "traceback_text",
        "adapter_closed",
        "cleanup_error",
    ):
        value = getattr(event, name, None)
        if isinstance(value, str | int | float | bool) or value is None:
            record[name] = value
    camera_read = getattr(event, "camera_read", None)
    joint_read = getattr(event, "joint_read", None)
    if camera_read is not None:
        record["read_id"] = camera_read.read_id
        record["read_started_at"] = camera_read.started_at
        record["read_completed_at"] = camera_read.completed_at
    if joint_read is not None:
        record["read_id"] = joint_read.read_id
        record["read_started_at"] = joint_read.started_at
        record["read_completed_at"] = joint_read.completed_at
    return record


def terminal_failure_receipt(
    failure: LiveCaptureFailure,
    *,
    policy_digest: str,
    identity_digest: str,
) -> dict[str, object]:  # noqa: OBJECT_OK
    """Serialize failure timing and cleanup without creating sample authority."""
    primary = failure.primary_error
    return {
        "schema": 1,
        "mode": "sim_to_real_physical_sample_capture_failure",
        "evidence_scope": "authorized_physical_diagnostic_failure",
        "terminal": True,
        "genuine_physical_samples": False,
        "consumable_sample_receipt": False,
        "count": 0,
        "completed_pair_count": failure.completed_pair_count,
        "attempt_id": failure.attempt_id,
        "policy_digest": policy_digest,
        "identity_evidence_digest": identity_digest,
        "primary_error": {
            "error_type": primary.error_type,
            "detail": primary.detail,
            "phase": primary.phase,
            "sample_index": primary.sample_index,
            "role": primary.role,
            "observed_at": primary.observed_at,
        },
        "cleanup_errors": [
            error for cleanup in failure.cleanup for error in cleanup.cleanup_errors
        ],
        "phases": [
            {
                "phase": phase.phase,
                "sample_index": phase.sample_index,
                "phase_started_at": phase.phase_started_at,
                "deadline": phase.deadline,
                "camera_worker_started_at": phase.camera_worker_started_at,
                "joint_worker_started_at": phase.joint_worker_started_at,
                "camera_completed_at": phase.camera_completed_at,
                "joint_completed_at": phase.joint_completed_at,
                "timeout_observed_at": phase.timeout_observed_at,
            }
            for phase in failure.phases
        ],
        "partial_events_quarantined": [_event_record(event) for event in failure.events],
        "cleanup": [
            {
                "role": cleanup.role,
                "cooperative_stop_requested": cleanup.cooperative_stop_requested,
                "adapter_closed": cleanup.adapter_closed,
                "terminate_sent": cleanup.terminate_sent,
                "kill_sent": cleanup.kill_sent,
                "process_reaped": cleanup.process_reaped,
                "exit_code": cleanup.exit_code,
                "cleanup_errors": list(cleanup.cleanup_errors),
                "child_failure_phase": cleanup.child_failure_phase,
                "child_failure_observed_at": cleanup.child_failure_observed_at,
                "child_error_type": cleanup.child_error_type,
                "child_error_message": cleanup.child_error_message,
                "child_traceback": cleanup.child_traceback,
            }
            for cleanup in failure.cleanup
        ],
        "motor_writes_performed": False,
        "actuation_performed": False,
    }
