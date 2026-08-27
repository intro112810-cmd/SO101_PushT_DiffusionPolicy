"""Process-supervised read-only acquisition timing and lifecycle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import active_children, get_context
from pathlib import Path
import time
from typing import Protocol

from read_only_authority_fakes import fixture_runtime_preflight, signed_test_authority
from so101_pusht_benchmark.sim_to_real.live_capture import capture_live_samples
from so101_pusht_benchmark.sim_to_real.live_capture_process import (
    MultiprocessingProviderRuntime,
)
from so101_pusht_benchmark.sim_to_real.live_capture_types import (
    AdapterIdentity,
    CameraObservation,
    LiveCaptureConfiguration,
    LiveCaptureProviders,
    LiveCaptureRequest,
    TimedCameraRead,
    TimedJointRead,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority import (
    CameraReadinessTimeoutSeconds,
    JointConnectTimeoutSeconds,
    PairCompletionTimeoutSeconds,
    ReadOnlyTimingPolicy,
    ShutdownGraceSeconds,
)


def test_acquisition_timing_policy_keeps_liveness_and_acceptance_distinct() -> None:
    timing = ReadOnlyTimingPolicy(
        camera_readiness_timeout_seconds=CameraReadinessTimeoutSeconds(5.0),
        joint_connect_timeout_seconds=JointConnectTimeoutSeconds(5.0),
        sample_pair_completion_timeout_seconds=PairCompletionTimeoutSeconds(0.2),
        shutdown_grace_seconds=ShutdownGraceSeconds(1.0),
        sample_max_age_seconds=0.2,
        sample_max_skew_seconds=0.04,
    )

    assert timing.camera_readiness_timeout_seconds == 5.0
    assert timing.joint_connect_timeout_seconds == 5.0
    assert timing.sample_pair_completion_timeout_seconds == 0.2
    assert timing.shutdown_grace_seconds == 1.0
    assert timing.sample_max_age_seconds == 0.2
    assert timing.sample_max_skew_seconds == 0.04


class _Signal(Protocol):
    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class _CameraFactory:
    identity: AdapterIdentity
    log_path: Path
    camera_started: tuple[_Signal, _Signal]
    joint_started: tuple[_Signal, _Signal]

    def __call__(self, configuration: LiveCaptureConfiguration) -> _ProcessCamera:
        del configuration
        return _ProcessCamera(
            self.identity,
            self.log_path,
            self.camera_started,
            self.joint_started,
        )


class _ProcessCamera:
    """Process-owned fake camera with event-gated accepted reads."""

    def __init__(
        self,
        identity: AdapterIdentity,
        log_path: Path,
        camera_started: tuple[_Signal, _Signal],
        joint_started: tuple[_Signal, _Signal],
    ) -> None:
        self.identity = identity
        self._log_path = log_path
        self._camera_started = camera_started
        self._joint_started = joint_started
        self._index = -1

    def _record(self, value: str) -> None:
        with self._log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{value}\n")

    def open(self) -> CameraObservation:
        self._record("open")
        return CameraObservation(640, 480, 30.0)

    def next_frame(self) -> TimedCameraRead:
        index = self._index
        self._index += 1
        now = time.monotonic()
        if index < 0:
            self._record("prime")
            return TimedCameraRead("camera-prime", b"discarded", now - 0.3, now)
        self._record(f"read:{index}")
        self._camera_started[index].set()
        if not self._joint_started[index].wait(1.0):
            raise RuntimeError("joint peer did not start")
        completed = time.monotonic()
        return TimedCameraRead(
            f"camera-{index:03d}",
            f"frame-{index}".encode(),
            completed,
            completed,
        )

    def close(self) -> None:
        self._record("close")


@dataclass(frozen=True, slots=True)
class _JointFactory:
    identity: AdapterIdentity
    log_path: Path
    camera_started: tuple[_Signal, _Signal]
    joint_started: tuple[_Signal, _Signal]

    def __call__(self, configuration: LiveCaptureConfiguration) -> _ProcessJoint:
        del configuration
        return _ProcessJoint(
            self.identity,
            self.log_path,
            self.camera_started,
            self.joint_started,
        )


class _ProcessJoint:
    """Process-owned fake direct bus with event-gated Present_Position reads."""

    def __init__(
        self,
        identity: AdapterIdentity,
        log_path: Path,
        camera_started: tuple[_Signal, _Signal],
        joint_started: tuple[_Signal, _Signal],
    ) -> None:
        self.identity = identity
        self._log_path = log_path
        self._camera_started = camera_started
        self._joint_started = joint_started
        self._index = 0

    def _record(self, value: str) -> None:
        with self._log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{value}\n")

    def open(self) -> None:
        self._record("connect")

    def next_state(self) -> TimedJointRead:
        index = self._index
        self._index += 1
        self._record(f"Present_Position:{index}")
        self._joint_started[index].set()
        if not self._camera_started[index].wait(1.0):
            raise RuntimeError("camera peer did not start")
        completed = time.monotonic()
        return TimedJointRead(
            f"joint-{index:03d}",
            (float(index), 1.0, 2.0, 3.0, 4.0),
            completed,
            completed,
        )

    def close(self) -> None:
        self._record("close")


def test_real_provider_processes_prime_then_read_two_concurrent_pairs_and_reap(
    tmp_path: Path,
) -> None:
    authority = signed_test_authority(tmp_path, provider_digest="1" * 64)
    configuration = LiveCaptureConfiguration(
        authority.profile_path,
        authority.camera_device_path,
        authority.follower_device_path,
        authority.calibration_path,
        640,
        480,
        30.0,
        authority.calibration_id,
    )
    context = get_context("fork")
    camera_started = (context.Event(), context.Event())
    joint_started = (context.Event(), context.Event())
    camera_log = tmp_path / "camera-calls.log"
    joint_log = tmp_path / "joint-calls.log"
    before = {process.pid for process in active_children()}
    result = capture_live_samples(
        LiveCaptureRequest(authority, authority, configuration),
        LiveCaptureProviders(
            _CameraFactory(
                AdapterIdentity(
                    authority.provider_digest,
                    authority.camera_device_digest,
                    None,
                ),
                camera_log,
                camera_started,
                joint_started,
            ),
            _JointFactory(
                AdapterIdentity(
                    authority.provider_digest,
                    authority.follower_device_digest,
                    authority.calibration_digest,
                ),
                joint_log,
                camera_started,
                joint_started,
            ),
            lambda path: (
                authority.camera_device_digest
                if path == authority.camera_device_path
                else authority.follower_device_digest
            ),
            lambda _path: authority.profile_digest,
            lambda _path: authority.calibration_digest,
            time.monotonic,
            fixture_runtime_preflight,
        ),
        process_runtime=MultiprocessingProviderRuntime(),
    )
    after = {process.pid for process in active_children()}

    assert result.priming is not None
    assert result.priming.read_completed_at - result.priming.read_started_at > 0.2
    assert camera_log.read_text(encoding="utf-8").splitlines() == [
        "open",
        "prime",
        "read:0",
        "read:1",
        "close",
    ]
    assert joint_log.read_text(encoding="utf-8").splitlines() == [
        "connect",
        "Present_Position:0",
        "Present_Position:1",
        "close",
    ]
    assert all(item.process_reaped and item.adapter_closed for item in result.cleanup)
    assert after == before
