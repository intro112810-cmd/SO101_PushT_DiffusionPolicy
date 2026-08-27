"""Governed synchronized live capture through injected read-only adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from live_capture_process_fakes import FakeProviderRuntime
from read_only_authority_fakes import fixture_runtime_preflight, signed_test_authority
from sim_to_real_policy_helpers import APPROVED
from so101_pusht_benchmark.sim_to_real.live_capture import (
    capture_live_samples,
    live_receipt,
)
from so101_pusht_benchmark.sim_to_real.live_capture_failure import LiveCaptureAttemptError
from so101_pusht_benchmark.sim_to_real.live_capture_protocol import (
    ProviderProcess,
    ProviderProcessRuntime,
    ProviderRole,
)
from so101_pusht_benchmark.sim_to_real.live_capture_runtime import RuntimePreflight
from so101_pusht_benchmark.sim_to_real.live_capture_identity import (
    PRODUCTION_LIVE_IDENTITY_SEAL,
    ApprovedLiveIdentity,
)
from so101_pusht_benchmark.sim_to_real.live_capture_types import (
    AdapterIdentity,
    CameraObservation,
    LiveCaptureConfiguration,
    LiveCaptureProviders,
    LiveCaptureRequest,
    LiveCaptureResult,
    TimedCameraRead,
    TimedJointRead,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.policy_types import (
    PRODUCTION_CONSTRUCTION_SEAL,
    ProductionApprovedSafetyPolicy,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority import (
    ProductionReadOnlyAcquisitionAuthority,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
SHA_PROVIDER = "1" * 64
SHA_CAMERA = "2" * 64
SHA_FOLLOWER = "3" * 64
SHA_CALIBRATION = "4" * 64
SHA_PROFILE = "5" * 64


def production_policy(
    *,
    max_age: float = 0.2,
    max_skew: float = 0.03,
) -> ProductionApprovedSafetyPolicy:
    fixture = load_fixture_safety_policy(APPROVED, now=NOW)
    return ProductionApprovedSafetyPolicy(
        PRODUCTION_CONSTRUCTION_SEAL,
        fixture.schema,
        fixture.policy_version,
        fixture.policy_id,
        "production",
        fixture.approved_by,
        fixture.approved_at,
        fixture.valid_from,
        fixture.expires_at,
        fixture.canonical_content,
        fixture.canonical_digest,
        fixture.workspace,
        fixture.joint_domains,
        replace(
            fixture.timing,
            sample_max_age_seconds=max_age,
            sample_max_skew_seconds=max_skew,
        ),
        fixture.camera,
        fixture.kinematics,
        fixture.collision,
        fixture.slew,
        fixture.provider,
        fixture.watchdog,
        fixture.acknowledgement,
        fixture.post_state,
        fixture.shadow,
        fixture.single_step,
        fixture.bounded_rollout,
        fixture.operator,
        fixture.owner_approval,
    )


def approved_identity() -> ApprovedLiveIdentity:
    return ApprovedLiveIdentity(
        PRODUCTION_LIVE_IDENTITY_SEAL,
        "live-read-identity-v1",
        "production",
        SHA_PROVIDER,
        SHA_PROFILE,
        SHA_CAMERA,
        SHA_FOLLOWER,
        SHA_CALIBRATION,
        640,
        480,
        30.0,
        "fixture-owner@example.invalid",
        "approval-001",
        "6" * 64,
    )


def _configuration(tmp_path: Path) -> LiveCaptureConfiguration:
    return LiveCaptureConfiguration(
        profile_path=tmp_path / "profile.yaml",
        camera_device=tmp_path / "camera-device",
        follower_device=tmp_path / "follower-device",
        calibration_file=tmp_path / "calibration.json",
        camera_width=640,
        camera_height=480,
        camera_fps=30.0,
        follower_calibration_id="follower-01",
    )


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeCamera:
    def __init__(
        self,
        log: list[tuple[str, object]],
        reads: list[TimedCameraRead],
        *,
        observation: CameraObservation | None = None,
        identity: AdapterIdentity | None = None,
        fail_at: int | None = None,
    ) -> None:
        self.log = log
        self.reads = reads
        self.observation = observation or CameraObservation(640, 480, 30.0)
        self.identity = identity or AdapterIdentity(SHA_PROVIDER, SHA_CAMERA, None)
        self.fail_at = fail_at
        self.index = -1

    def open(self) -> CameraObservation:
        self.log.append(("camera.open", None))
        return self.observation

    def next_frame(self) -> TimedCameraRead:
        self.log.append(("camera.read", self.index))
        if self.index < 0:
            self.index = 0
            return TimedCameraRead("camera-prime", b"discarded-prime", 999.9, 999.95)
        if self.fail_at == self.index:
            raise RuntimeError("camera read failed")
        value = self.reads[self.index]
        self.index += 1
        return value

    def close(self) -> None:
        self.log.append(("camera.close", None))


class FakeJoint:
    def __init__(
        self,
        log: list[tuple[str, object]],
        reads: list[TimedJointRead],
        *,
        identity: AdapterIdentity | None = None,
        fail_at: int | None = None,
    ) -> None:
        self.log = log
        self.reads = reads
        self.identity = identity or AdapterIdentity(SHA_PROVIDER, SHA_FOLLOWER, SHA_CALIBRATION)
        self.fail_at = fail_at
        self.index = 0

    def open(self) -> None:
        self.log.append(("joint.open", None))

    def next_state(self) -> TimedJointRead:
        self.log.append(("joint.read", self.index))
        if self.fail_at == self.index:
            raise RuntimeError("joint read failed")
        value = self.reads[self.index]
        self.index += 1
        return value

    def close(self) -> None:
        self.log.append(("joint.close", None))


def camera_reads() -> list[TimedCameraRead]:
    return [
        TimedCameraRead("camera-000", b"frame-a", 1000.000, 1000.002),
        TimedCameraRead("camera-001", b"frame-b", 1000.020, 1000.022),
    ]


def joint_reads() -> list[TimedJointRead]:
    return [
        TimedJointRead("joint-000", (0.0, 1.0, 2.0, 3.0, 4.0), 1000.003, 1000.005),
        TimedJointRead("joint-001", (0.1, 1.1, 2.1, 3.1, 4.1), 1000.023, 1000.025),
    ]


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    policy: ProductionApprovedSafetyPolicy | None = None
    completion_times: tuple[float, ...] = ()
    authority: ProductionReadOnlyAcquisitionAuthority | None = None
    bind_adapter_identities: bool = True
    process_runtime: ProviderProcessRuntime | None = None
    runtime_preflight: RuntimePreflight | None = None


def capture_fake(
    tmp_path: Path,
    camera: FakeCamera,
    joint: FakeJoint,
    *,
    settings: CaptureSettings | None = None,
) -> LiveCaptureResult:
    settings = CaptureSettings() if settings is None else settings
    policy = settings.policy
    authority = settings.authority
    selected = (
        signed_test_authority(
            tmp_path,
            provider_digest=SHA_PROVIDER,
            sample_max_age_seconds=(
                0.2 if policy is None else policy.timing.sample_max_age_seconds
            ),
            sample_max_skew_seconds=(
                0.03 if policy is None else policy.timing.sample_max_skew_seconds
            ),
        )
        if authority is None
        else authority
    )
    if settings.bind_adapter_identities:
        camera.identity = AdapterIdentity(
            selected.provider_digest,
            selected.camera_device_digest,
            None,
        )
        joint.identity = AdapterIdentity(
            selected.provider_digest,
            selected.follower_device_digest,
            selected.calibration_digest,
        )
    return capture_live_samples(
        LiveCaptureRequest(
            selected,
            selected,
            _configuration(tmp_path),
        ),
        LiveCaptureProviders(
            lambda _configuration: camera,
            lambda _configuration: joint,
            lambda path: (
                selected.camera_device_digest
                if path == selected.camera_device_path
                else selected.follower_device_digest
            ),
            lambda _path: selected.profile_digest,
            lambda _path: selected.calibration_digest,
            lambda: 1000.15,
            (
                fixture_runtime_preflight
                if settings.runtime_preflight is None
                else settings.runtime_preflight
            ),
        ),
        process_runtime=(
            FakeProviderRuntime() if settings.process_runtime is None else settings.process_runtime
        ),
    )


_capture = capture_fake


def test_valid_fake_live_capture_is_genuine_and_ordered(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    authority = signed_test_authority(tmp_path, provider_digest=SHA_PROVIDER)
    result = _capture(
        tmp_path,
        FakeCamera(log, camera_reads()),
        FakeJoint(log, joint_reads()),
        settings=CaptureSettings(authority=authority),
    )

    receipt = live_receipt(result, authority, authority)
    assert receipt["evidence_scope"] == "authorized_physical_diagnostic"
    assert receipt["genuine_physical_samples"] is True
    assert receipt["count"] == 2
    samples = cast("list[dict[str, object]]", receipt["samples"])
    assert [sample["record_id"] for sample in samples] == ["sample-000", "sample-001"]
    assert samples[0]["frame_digest"] != samples[1]["frame_digest"]
    assert receipt["capture_windows"] == [
        {
            "sample_id": "sample-000",
            "camera_read_id": "camera-000",
            "camera_started_at": 1000.0,
            "camera_completed_at": 1000.002,
            "camera_duration_seconds": 1000.002 - 1000.0,
            "joint_read_id": "joint-000",
            "joint_started_at": 1000.003,
            "joint_completed_at": 1000.005,
            "joint_duration_seconds": 1000.005 - 1000.003,
            "midpoint_skew_seconds": abs((1000.0 + 1000.002) / 2.0 - (1000.003 + 1000.005) / 2.0),
        },
        {
            "sample_id": "sample-001",
            "camera_read_id": "camera-001",
            "camera_started_at": 1000.02,
            "camera_completed_at": 1000.022,
            "camera_duration_seconds": 1000.022 - 1000.02,
            "joint_read_id": "joint-001",
            "joint_started_at": 1000.023,
            "joint_completed_at": 1000.025,
            "joint_duration_seconds": 1000.025 - 1000.023,
            "midpoint_skew_seconds": abs((1000.02 + 1000.022) / 2.0 - (1000.023 + 1000.025) / 2.0),
        },
    ]
    assert log == [
        ("camera.open", None),
        ("camera.read", -1),
        ("joint.open", None),
        ("camera.read", 0),
        ("joint.read", 0),
        ("camera.read", 1),
        ("joint.read", 1),
        ("camera.close", None),
        ("joint.close", None),
    ]


def test_fixture_policy_and_missing_identity_fail_before_factory(tmp_path: Path) -> None:
    fixture = load_fixture_safety_policy(APPROVED, now=NOW)
    factory_calls = 0

    def factory(_configuration: LiveCaptureConfiguration) -> FakeCamera:
        nonlocal factory_calls
        factory_calls += 1
        return FakeCamera([], camera_reads())

    providers = LiveCaptureProviders(
        factory,
        lambda _configuration: FakeJoint([], joint_reads()),
        lambda _path: None,
        lambda _path: SHA_PROFILE,
        lambda _path: SHA_CALIBRATION,
        SequenceClock([]),
        fixture_runtime_preflight,
    )
    requests = (
        LiveCaptureRequest(None, approved_identity(), _configuration(tmp_path)),
        LiveCaptureRequest(fixture, approved_identity(), _configuration(tmp_path)),
        LiveCaptureRequest(production_policy(), None, _configuration(tmp_path)),
    )
    for request in requests:
        with pytest.raises(RolloutViolation) as caught:
            capture_live_samples(request, providers)
        assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
    assert factory_calls == 0


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing-camera", RolloutCode.R_MISSING),
        ("camera-device-mismatch", RolloutCode.R_HASH_MISMATCH),
        ("profile-mismatch", RolloutCode.R_HASH_MISMATCH),
        ("calibration-mismatch", RolloutCode.R_HASH_MISMATCH),
    ],
)
def test_preflight_failure_occurs_before_adapter_construction(
    tmp_path: Path, mutation: str, code: RolloutCode
) -> None:
    factory_calls = 0
    authority = signed_test_authority(tmp_path, provider_digest=SHA_PROVIDER)
    camera_identity = {
        "missing-camera": None,
        "camera-device-mismatch": "9" * 64,
    }.get(mutation, authority.camera_device_digest)
    probes = {
        authority.camera_device_path: camera_identity,
        authority.follower_device_path: authority.follower_device_digest,
    }

    def factory(_configuration: LiveCaptureConfiguration) -> FakeCamera:
        nonlocal factory_calls
        factory_calls += 1
        return FakeCamera([], camera_reads())

    def probe(path: Path) -> str | None:
        return probes.get(path)

    with pytest.raises(RolloutViolation) as caught:
        capture_live_samples(
            LiveCaptureRequest(authority, authority, _configuration(tmp_path)),
            LiveCaptureProviders(
                factory,
                lambda _configuration: FakeJoint([], joint_reads()),
                probe,
                lambda _path: (
                    "9" * 64 if mutation == "profile-mismatch" else authority.profile_digest
                ),
                lambda _path: (
                    "9" * 64 if mutation == "calibration-mismatch" else authority.calibration_digest
                ),
                lambda: 1000.1,
                fixture_runtime_preflight,
            ),
            process_runtime=FakeProviderRuntime(),
        )
    assert caught.value.code is code
    assert factory_calls == 0


@pytest.mark.parametrize(
    "failure", ["camera-profile", "camera-duplicate", "joint-duplicate", "skew", "stale"]
)
def test_bad_live_sample_closes_camera_and_returns_no_result(tmp_path: Path, failure: str) -> None:
    log: list[tuple[str, object]] = []
    camera_values = camera_reads()
    joint_values = joint_reads()
    observation = CameraObservation(641, 480, 30.0) if failure == "camera-profile" else None
    if failure == "camera-duplicate":
        camera_values[1] = replace(camera_values[1], frame_bytes=b"frame-a")
    elif failure == "joint-duplicate":
        joint_values[1] = replace(joint_values[1], read_id="joint-000")
    elif failure == "skew":
        joint_values[0] = replace(joint_values[0], started_at=1000.100, completed_at=1000.102)
    elif failure == "stale":
        camera_values[0] = replace(camera_values[0], started_at=999.0, completed_at=999.002)
        joint_values[0] = replace(joint_values[0], started_at=999.003, completed_at=999.005)

    with pytest.raises(LiveCaptureAttemptError):
        _capture(
            tmp_path,
            FakeCamera(log, camera_values, observation=observation),
            FakeJoint(log, joint_values),
        )
    expected_cleanup = [("camera.close", None), ("joint.close", None)]
    assert log[-len(expected_cleanup) :] == expected_cleanup


def test_adapter_identity_mismatch_fails_before_device_open(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    camera = FakeCamera(
        log,
        camera_reads(),
        identity=AdapterIdentity("9" * 64, SHA_CAMERA, None),
    )

    with pytest.raises(LiveCaptureAttemptError) as caught:
        _capture(
            tmp_path,
            camera,
            FakeJoint(log, joint_reads()),
            settings=CaptureSettings(bind_adapter_identities=False),
        )
    assert "R_HASH_MISMATCH" in caught.value.failure.primary_error.detail
    assert ("camera.open", None) not in log
    assert not any(entry[0] == "camera.read" for entry in log)


@pytest.mark.parametrize("source", ["camera", "joint"])
def test_partial_failure_closes_resources_and_publishes_nothing(
    tmp_path: Path, source: str
) -> None:
    log: list[tuple[str, object]] = []
    camera = FakeCamera(log, camera_reads(), fail_at=1 if source == "camera" else None)
    joint = FakeJoint(log, joint_reads(), fail_at=1 if source == "joint" else None)

    with pytest.raises(LiveCaptureAttemptError):
        _capture(tmp_path, camera, joint)
    assert camera.index == (1 if source == "camera" else 2)
    assert joint.index == (2 if source == "camera" else 1)
    assert sum(entry[0] == "camera.read" for entry in log) == 3
    assert sum(entry[0] == "joint.read" for entry in log) == 2
    expected_cleanup = (
        [("camera.close", None)]
        if source == "camera"
        else [("camera.close", None), ("joint.close", None)]
    )
    assert log[-len(expected_cleanup) :] == expected_cleanup


def test_each_sample_records_both_worker_starts_after_explicit_arming(
    tmp_path: Path,
) -> None:
    result = _capture(
        tmp_path,
        FakeCamera([], camera_reads()),
        FakeJoint([], joint_reads()),
    )

    pair_phases = [phase for phase in result.phases if phase.phase == "sample_pair"]
    assert len(pair_phases) == 2
    assert all(phase.camera_worker_started_at is not None for phase in pair_phases)
    assert all(phase.joint_worker_started_at is not None for phase in pair_phases)


@pytest.mark.parametrize("slow_side", ["camera", "joint"])
def test_overlapping_slow_read_retains_exact_durations_and_midpoint_skew(
    tmp_path: Path,
    slow_side: str,
) -> None:
    camera_values = camera_reads()
    joint_values = joint_reads()
    camera_values[1] = replace(camera_values[1], started_at=1000.1, completed_at=1000.102)
    joint_values[1] = replace(joint_values[1], started_at=1000.103, completed_at=1000.105)
    if slow_side == "camera":
        camera_values[0] = replace(camera_values[0], started_at=1000.0, completed_at=1000.08)
        joint_values[0] = replace(joint_values[0], started_at=1000.04, completed_at=1000.04)
    else:
        camera_values[0] = replace(camera_values[0], started_at=1000.04, completed_at=1000.04)
        joint_values[0] = replace(joint_values[0], started_at=1000.0, completed_at=1000.08)

    result = _capture(
        tmp_path,
        FakeCamera([], camera_values),
        FakeJoint([], joint_values),
        settings=CaptureSettings(policy=production_policy(max_skew=0.04)),
    )

    window = result.windows[0]
    assert max(window.camera_duration_seconds, window.joint_duration_seconds) == pytest.approx(0.08)
    assert window.midpoint_skew_seconds == pytest.approx(0.0)


def test_midpoint_skew_accepts_exact_004_boundary_and_rejects_above_it(
    tmp_path: Path,
) -> None:
    camera_values = [
        TimedCameraRead("camera-000", b"frame-a", 1000.0, 1000.0),
        TimedCameraRead("camera-001", b"frame-b", 1000.1, 1000.1),
    ]
    boundary_joint = [
        TimedJointRead("joint-000", (0.0, 1.0, 2.0, 3.0, 4.0), 1000.04, 1000.04),
        TimedJointRead("joint-001", (0.1, 1.1, 2.1, 3.1, 4.1), 1000.14, 1000.14),
    ]
    result = _capture(
        tmp_path,
        FakeCamera([], camera_values),
        FakeJoint([], boundary_joint),
        settings=CaptureSettings(
            policy=production_policy(max_skew=0.04),
            completion_times=(1000.041, 1000.141),
        ),
    )
    assert result.windows[0].midpoint_skew_seconds == pytest.approx(0.04)

    above = [
        replace(
            value,
            started_at=value.started_at + 0.000001,
            completed_at=value.completed_at + 0.000001,
        )
        for value in boundary_joint
    ]
    with pytest.raises(LiveCaptureAttemptError, match="camera/joint skew"):
        _capture(
            tmp_path,
            FakeCamera([], camera_values),
            FakeJoint([], above),
            settings=CaptureSettings(
                policy=production_policy(max_skew=0.04),
                completion_times=(1000.042, 1000.142),
            ),
        )


def test_coordinator_timeout_terminates_and_reaps_without_thread_ownership(
    tmp_path: Path,
) -> None:
    class TimeoutRuntime(FakeProviderRuntime):
        def wait(
            self,
            processes: tuple[ProviderProcess, ...],
            timeout: float,
        ) -> tuple[ProviderProcess, ...]:
            ready = super().wait(processes, timeout)
            if any(process.provider_started for process in self.processes):
                return tuple(
                    process for process in ready if process.role is not ProviderRole.CAMERA
                )
            return ready

    runtime = TimeoutRuntime()
    with pytest.raises(LiveCaptureAttemptError, match="camera_readiness timed out"):
        _capture(
            tmp_path,
            FakeCamera([], camera_reads()),
            FakeJoint([], joint_reads()),
            settings=CaptureSettings(process_runtime=runtime),
        )

    assert len(runtime.processes) == 2
    assert runtime.processes[0].terminate_calls == 1
    assert all(process.join_calls == 1 for process in runtime.processes)
    assert not any(process.is_alive() for process in runtime.processes)


def test_one_side_exception_preserves_primary_detail_without_retry(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    camera = FakeCamera(log, camera_reads(), fail_at=0)
    joint = FakeJoint(log, joint_reads())

    with pytest.raises(LiveCaptureAttemptError) as caught:
        _capture(tmp_path, camera, joint)

    assert caught.value.failure.primary_error.error_type == "RuntimeError"
    assert caught.value.failure.primary_error.detail == "camera read failed"
    assert camera.index == 0
    assert joint.index == 1
    assert sum(entry[0] == "camera.read" for entry in log) == 2
