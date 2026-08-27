"""Process-supervised orchestration of exactly two accepted read-only pairs."""

from __future__ import annotations

import hashlib

from .live_capture_acceptance import PairAcceptanceRequest, accept_pair
from .live_capture_cleanup import CleanupRequest, cleanup_processes
from .live_capture_failure import (
    CapturePrimaryError,
    LiveCaptureAttemptError,
    LiveCaptureFailure,
)
from .live_capture_pair import ReadPairRequest, read_pair
from .live_capture_phase import PhaseFaultError
from .live_capture_startup import StartupRequest, start_provider_processes
from .live_capture_protocol import (
    ProviderEvent,
    ProviderProcess,
    ProviderProcessRuntime,
    ProviderProtocolError,
    ProviderRole,
)
from .live_capture_types import (
    LiveCaptureConfiguration,
    LiveCaptureProviders,
    LiveCaptureResult,
    PhaseEvidence,
    PrimingEvidence,
    SampleCaptureWindow,
)
from .live_capture_validation import finite
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority
from .rollout_codes import RolloutViolation
from .rollout_record_types import PhysicalSample

__all__ = ("capture_with_process_supervision",)


def capture_with_process_supervision(
    authority: ProductionReadOnlyAcquisitionAuthority,
    configuration: LiveCaptureConfiguration,
    providers: LiveCaptureProviders,
    runtime: ProviderProcessRuntime,
) -> LiveCaptureResult:
    """Prime once, accept exactly two complete pairs, and reap every child."""
    clock = providers.clock
    attempt_started = finite(clock(), "capture attempt start")
    attempt_id = hashlib.sha256(
        f"{authority.canonical_digest}:{attempt_started:.9f}".encode()
    ).hexdigest()
    events: list[ProviderEvent] = []
    phases: list[PhaseEvidence] = []
    samples: list[PhysicalSample] = []
    windows: list[SampleCaptureWindow] = []
    frame_digests: set[str] = set()
    processes: list[ProviderProcess] = []
    active_roles: frozenset[ProviderRole] = frozenset()
    priming: PrimingEvidence | None = None
    primary: CapturePrimaryError | None = None
    timing = authority.timing
    try:
        camera_process, joint_process, priming = start_provider_processes(
            StartupRequest(
                authority,
                configuration,
                providers,
                runtime,
                events,
                phases,
                processes,
            )
        )
        pair_processes = (camera_process, joint_process)
        for index in range(2):
            camera_read, joint_read, phase = read_pair(
                ReadPairRequest(
                    runtime,
                    pair_processes,
                    index,
                    timing.sample_pair_completion_timeout_seconds,
                    clock,
                    events,
                )
            )
            phases.append(phase)
            sample, window = accept_pair(
                PairAcceptanceRequest(
                    authority,
                    camera_read,
                    joint_read,
                    finite(clock(), "capture completion"),
                    index,
                    windows[-1] if windows else None,
                    frame_digests,
                )
            )
            samples.append(sample)
            windows.append(window)
    except PhaseFaultError as exc:
        primary = exc.primary
        phases.append(exc.phase_evidence)
        active_roles = exc.active_roles
    except (EOFError, OSError, RuntimeError, RolloutViolation) as exc:
        primary = CapturePrimaryError(
            type(exc).__name__,
            str(exc),
            "capture",
            len(samples) if len(samples) < 2 else None,
            None,
            clock(),
        )
    cleanup = cleanup_processes(
        CleanupRequest(
            runtime,
            tuple(processes),
            active_roles,
            timing.shutdown_grace_seconds,
            clock,
            events,
        )
    )
    child_failure = next(
        (item for item in cleanup if item.child_error_type is not None),
        None,
    )
    if child_failure is not None:
        primary = CapturePrimaryError(
            child_failure.child_error_type or "ChildProcessError",
            child_failure.child_error_message or "provider child exited without an event",
            child_failure.child_failure_phase or "child_process",
            len(samples) if len(samples) < 2 else None,
            child_failure.role,
            child_failure.child_failure_observed_at or clock(),
        )
    cleanup_errors = tuple(error for item in cleanup for error in item.cleanup_errors)
    if primary is None and cleanup_errors:
        primary = CapturePrimaryError(
            "CleanupError",
            "provider cleanup did not complete cleanly",
            "shutdown",
            None,
            None,
            clock(),
        )
    if primary is not None:
        raise LiveCaptureAttemptError(
            LiveCaptureFailure(
                attempt_id,
                primary,
                tuple(phases),
                tuple(events),
                cleanup,
                len(samples),
            )
        )
    if len(samples) != 2 or len(windows) != 2 or priming is None:
        raise ProviderProtocolError("supervised capture completed without exactly two pairs")
    return LiveCaptureResult(
        (samples[0], samples[1]),
        (windows[0], windows[1]),
        priming,
        tuple(phases),
        cleanup,
    )
