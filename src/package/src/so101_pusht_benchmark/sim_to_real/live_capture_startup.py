"""Child runtime handshake and provider readiness before sample arming."""

from __future__ import annotations

from dataclasses import dataclass

from .live_capture_child_runtime import ChildRuntimeRequest, await_child_runtime_preflights
from .live_capture_phase import PhaseContext
from .live_capture_protocol import (
    CameraWorkerSpec,
    ExpectedCameraProfile,
    JointWorkerSpec,
    ProviderEvent,
    ProviderProcess,
    ProviderProcessRuntime,
    StartProvider,
)
from .live_capture_readiness import ReadinessRequest, await_camera_ready, await_joint_ready
from .live_capture_types import (
    AdapterIdentity,
    LiveCaptureConfiguration,
    LiveCaptureProviders,
    PhaseEvidence,
    PrimingEvidence,
)
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority

__all__ = ("StartupRequest", "start_provider_processes")


@dataclass(frozen=True, slots=True)
class StartupRequest:
    authority: ProductionReadOnlyAcquisitionAuthority
    configuration: LiveCaptureConfiguration
    providers: LiveCaptureProviders
    runtime: ProviderProcessRuntime
    events: list[ProviderEvent]
    phases: list[PhaseEvidence]
    processes: list[ProviderProcess]


def _camera_spec(request: StartupRequest) -> CameraWorkerSpec:
    authority = request.authority
    return CameraWorkerSpec(
        request.configuration,
        request.providers.camera_factory,
        AdapterIdentity(authority.provider_digest, authority.camera_device_digest, None),
        ExpectedCameraProfile(
            authority.camera_width,
            authority.camera_height,
            authority.camera_fps,
        ),
        request.providers.clock,
        request.providers.runtime_preflight,
    )


def _joint_spec(request: StartupRequest) -> JointWorkerSpec:
    authority = request.authority
    return JointWorkerSpec(
        request.configuration,
        request.providers.joint_factory,
        AdapterIdentity(
            authority.provider_digest,
            authority.follower_device_digest,
            authority.calibration_digest,
        ),
        request.providers.clock,
        request.providers.runtime_preflight,
    )


def start_provider_processes(
    request: StartupRequest,
) -> tuple[ProviderProcess, ProviderProcess, PrimingEvidence]:
    """Authorize device creation only after both child runtimes agree."""
    clock = request.providers.clock
    timing = request.authority.timing
    camera = request.runtime.spawn(_camera_spec(request))
    joint = request.runtime.spawn(_joint_spec(request))
    request.processes.extend((camera, joint))
    runtime_started = clock()
    runtime_deadline = runtime_started + max(
        timing.camera_readiness_timeout_seconds,
        timing.joint_connect_timeout_seconds,
    )
    camera.start()
    joint.start()
    request.phases.append(
        await_child_runtime_preflights(
            ChildRuntimeRequest(
                request.runtime,
                (camera, joint),
                PhaseContext("runtime_preflight", None, runtime_started, runtime_deadline),
                clock,
                request.events,
            )
        )
    )
    start_requested = clock()
    camera.send(StartProvider(start_requested))
    joint.send(StartProvider(start_requested))
    camera_started = clock()
    joint_started = camera_started
    camera_deadline = camera_started + timing.camera_readiness_timeout_seconds
    joint_deadline = joint_started + timing.joint_connect_timeout_seconds
    joint_ready = await_joint_ready(
        ReadinessRequest(
            request.runtime,
            joint,
            PhaseContext("joint_connect", None, joint_started, joint_deadline),
            clock,
            request.events,
        )
    )
    request.phases.append(
        PhaseEvidence(
            "joint_connect",
            None,
            joint_started,
            joint_deadline,
            None,
            joint_ready.worker_started_at,
            None,
            joint_ready.completed_at,
            None,
        )
    )
    camera_ready = await_camera_ready(
        ReadinessRequest(
            request.runtime,
            camera,
            PhaseContext("camera_readiness", None, camera_started, camera_deadline),
            clock,
            request.events,
        )
    )
    priming = PrimingEvidence(
        camera_ready.priming_read_id,
        camera_started,
        camera_deadline,
        camera_ready.worker_started_at,
        camera_ready.priming_started_at,
        camera_ready.priming_completed_at,
        camera_ready.completed_at,
    )
    request.phases.append(
        PhaseEvidence(
            "camera_readiness",
            None,
            camera_started,
            camera_deadline,
            camera_ready.worker_started_at,
            None,
            camera_ready.completed_at,
            None,
            None,
        )
    )
    return camera, joint, priming
