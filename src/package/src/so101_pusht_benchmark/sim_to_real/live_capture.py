"""Governed process-supervised acquisition of two live read-only samples."""

from __future__ import annotations

from dataclasses import replace
import hashlib

from .live_capture_failure import (
    CapturePrimaryError,
    LiveCaptureAttemptError,
    LiveCaptureFailure,
)
from .live_capture_process import MultiprocessingProviderRuntime
from .live_capture_protocol import ProviderProcessRuntime
from .live_capture_supervisor import capture_with_process_supervision
from .live_capture_types import (
    LiveCaptureProviders,
    LiveCaptureRequest,
    PhaseEvidence,
    LiveCaptureResult,
)
from .live_capture_validation import finite, require_live_identity, require_preflight
from .read_only_authority import (
    ProductionReadOnlyAcquisitionAuthority,
    require_read_only_acquisition_authority,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .sample_capture import sample_as_record

__all__ = ("capture_live_samples", "live_receipt")


def _acquisition_authority(value: object) -> ProductionReadOnlyAcquisitionAuthority:
    return require_read_only_acquisition_authority(value)


def capture_live_samples(
    request: LiveCaptureRequest,
    providers: LiveCaptureProviders,
    *,
    process_runtime: ProviderProcessRuntime | None = None,
) -> LiveCaptureResult:
    """Preflight all authority and path identities before spawning providers."""
    authority = _acquisition_authority(request.policy)
    identity = require_live_identity(request.identity)
    if type(identity) is not ProductionReadOnlyAcquisitionAuthority:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "signed read-only acquisition authority required",
        )
    if identity.canonical_digest != authority.canonical_digest:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "live authorities disagree")
    configuration = request.configuration
    require_preflight(
        identity,
        configuration,
        providers.device_probe,
        providers.profile_digest,
        providers.calibration_digest,
    )
    runtime_started = finite(providers.clock(), "runtime preflight start")
    runtime_deadline = runtime_started + authority.timing.joint_connect_timeout_seconds
    try:
        runtime_dependency = providers.runtime_preflight()
    except RolloutViolation as exc:
        observed = finite(providers.clock(), "runtime preflight failure")
        attempt_id = hashlib.sha256(
            f"{authority.canonical_digest}:runtime:{runtime_started:.9f}".encode()
        ).hexdigest()
        raise LiveCaptureAttemptError(
            LiveCaptureFailure(
                attempt_id,
                CapturePrimaryError(
                    type(exc).__name__,
                    str(exc),
                    "runtime_preflight",
                    None,
                    None,
                    observed,
                ),
                (
                    PhaseEvidence(
                        "runtime_preflight",
                        None,
                        runtime_started,
                        runtime_deadline,
                        None,
                        None,
                        None,
                        None,
                        observed,
                    ),
                ),
                (),
                (),
                0,
            )
        ) from exc
    runtime = MultiprocessingProviderRuntime() if process_runtime is None else process_runtime
    result = capture_with_process_supervision(authority, configuration, providers, runtime)
    return replace(result, runtime_dependency=runtime_dependency)


def live_receipt(
    result: LiveCaptureResult,
    policy: ProductionReadOnlyAcquisitionAuthority | None,
    identity_value: object,
) -> dict[str, object]:
    """Publish consumable evidence only for a complete, reaped two-pair attempt."""
    authority = _acquisition_authority(policy)
    identity = require_live_identity(identity_value)
    if (
        type(identity) is not ProductionReadOnlyAcquisitionAuthority
        or identity.canonical_digest != authority.canonical_digest
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "live authorities disagree")
    if (
        len(result.samples) != 2
        or len(result.windows) != 2
        or result.priming is None
        or result.runtime_dependency is None
    ):
        raise RolloutViolation(RolloutCode.R_MISSING, "two complete live sample pairs required")
    if any(not cleanup.process_reaped for cleanup in result.cleanup):
        raise RolloutViolation(RolloutCode.R_MISSING, "provider cleanup is not reaped")
    windows = [
        {
            "sample_id": window.sample_id,
            "camera_read_id": window.camera_read_id,
            "camera_started_at": window.camera_started_at,
            "camera_completed_at": window.camera_completed_at,
            "camera_duration_seconds": window.camera_duration_seconds,
            "joint_read_id": window.joint_read_id,
            "joint_started_at": window.joint_started_at,
            "joint_completed_at": window.joint_completed_at,
            "joint_duration_seconds": window.joint_duration_seconds,
            "midpoint_skew_seconds": window.midpoint_skew_seconds,
        }
        for window in result.windows
    ]
    priming = result.priming
    return {
        "schema": 3,
        "mode": "sim_to_real_physical_sample_capture",
        "evidence_scope": "authorized_physical_diagnostic",
        "genuine_physical_samples": True,
        "count": 2,
        "policy_digest": authority.canonical_digest,
        "identity_evidence_digest": identity.identity_digest,
        "provider_digest": identity.provider_digest,
        "camera_device_digest": identity.camera_device_digest,
        "follower_device_digest": identity.follower_device_digest,
        "calibration_digest": identity.calibration_digest,
        "runtime_dependency": {
            "distribution": result.runtime_dependency.distribution,
            "version": result.runtime_dependency.version,
            "module": result.runtime_dependency.module,
            "module_origin": str(result.runtime_dependency.module_origin),
        },
        "samples": [sample_as_record(sample) for sample in result.samples],
        "capture_windows": windows,
        "camera_readiness": {
            "discarded_priming_frame_count": 1,
            "priming_read_id": priming.read_id,
            "phase_started_at": priming.phase_started_at,
            "deadline": priming.deadline,
            "worker_started_at": priming.worker_started_at,
            "read_started_at": priming.read_started_at,
            "read_completed_at": priming.read_completed_at,
            "readiness_completed_at": priming.readiness_completed_at,
        },
        "phase_timing": [
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
            for phase in result.phases
        ],
        "cleanup": [
            {
                "role": cleanup.role,
                "adapter_closed": cleanup.adapter_closed,
                "process_reaped": cleanup.process_reaped,
                "terminate_sent": cleanup.terminate_sent,
                "kill_sent": cleanup.kill_sent,
                "cleanup_errors": list(cleanup.cleanup_errors),
            }
            for cleanup in result.cleanup
        ],
        "authority_scope": authority.artifact_scope,
        "source_lineage_authority_digest": authority.source_lineage_authority_digest,
        "camera_permissions": list(authority.camera_permissions),
        "follower_permissions": list(authority.follower_permissions),
        "forbidden_capabilities": list(authority.forbidden_capabilities),
        "motor_writes_performed": False,
        "actuation_performed": False,
    }
