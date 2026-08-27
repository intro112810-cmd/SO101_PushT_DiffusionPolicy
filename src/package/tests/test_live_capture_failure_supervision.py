"""Timeout, quarantine, escalation, and terminal failure evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from live_capture_process_fakes import (
    FakeProviderProcess,
    FakeProviderRuntime,
)
from so101_pusht_benchmark.sim_to_real.live_capture_failure import (
    LiveCaptureAttemptError,
    terminal_failure_receipt,
)
from so101_pusht_benchmark.sim_to_real.live_capture_protocol import (
    ProviderProcess,
    ProviderRole,
    WorkerSpec,
)
from test_live_sample_capture import (
    CaptureSettings,
    FakeCamera,
    FakeJoint,
    camera_reads,
    capture_fake as _capture,
    joint_reads,
)


class _SelectiveTimeoutRuntime(FakeProviderRuntime):
    """Hide one exact process event stream to force a bounded phase timeout."""

    def __init__(self, phase: str, role: ProviderRole) -> None:
        super().__init__()
        self._phase = phase
        self._role = role

    def wait(
        self,
        processes: tuple[ProviderProcess, ...],
        timeout: float,
    ) -> tuple[ProviderProcess, ...]:
        ready = super().wait(processes, timeout)
        sample_started = any(process.release_count > 0 for process in self.processes)
        provider_started = any(process.provider_started for process in self.processes)
        if self._phase == "joint_connect" and provider_started and not sample_started:
            return tuple(process for process in ready if process.role is not ProviderRole.JOINT)
        if self._phase == "sample_pair" and sample_started:
            return tuple(process for process in ready if process.role is not self._role)
        return ready


@pytest.mark.parametrize("role", [ProviderRole.CAMERA, ProviderRole.JOINT])
def test_blocked_sample_side_times_out_without_second_pair_and_reaps_peer(
    tmp_path: Path,
    role: ProviderRole,
) -> None:
    runtime = _SelectiveTimeoutRuntime("sample_pair", role)
    camera = FakeCamera([], camera_reads())
    joint = FakeJoint([], joint_reads())

    with pytest.raises(LiveCaptureAttemptError) as caught:
        _capture(
            tmp_path,
            camera,
            joint,
            settings=CaptureSettings(process_runtime=runtime),
        )

    failure = caught.value.failure
    assert failure.primary_error.phase == "sample_pair"
    assert failure.completed_pair_count == 0
    assert camera.index == 1
    assert joint.index == 1
    assert all(cleanup.process_reaped for cleanup in failure.cleanup)
    assert not any(process.is_alive() for process in runtime.processes)


def test_joint_connect_timeout_never_arms_sample_and_reaps_both(tmp_path: Path) -> None:
    runtime = _SelectiveTimeoutRuntime("joint_connect", ProviderRole.JOINT)
    camera = FakeCamera([], camera_reads())
    joint = FakeJoint([], joint_reads())

    with pytest.raises(LiveCaptureAttemptError) as caught:
        _capture(
            tmp_path,
            camera,
            joint,
            settings=CaptureSettings(process_runtime=runtime),
        )

    assert caught.value.failure.primary_error.phase == "joint_connect"
    assert camera.index == 0
    assert joint.index == 0
    assert all(process.release_count == 0 for process in runtime.processes)
    assert all(cleanup.process_reaped for cleanup in caught.value.failure.cleanup)


def test_second_sample_failure_quarantines_first_pair_and_is_never_consumable(
    tmp_path: Path,
) -> None:
    camera = FakeCamera([], camera_reads(), fail_at=1)

    with pytest.raises(LiveCaptureAttemptError) as caught:
        _capture(tmp_path, camera, FakeJoint([], joint_reads()))

    failure = caught.value.failure
    receipt = terminal_failure_receipt(
        failure,
        policy_digest="a" * 64,
        identity_digest="b" * 64,
    )
    assert failure.completed_pair_count == 1
    assert camera.index == 1
    assert receipt["count"] == 0
    assert receipt["genuine_physical_samples"] is False
    assert receipt["consumable_sample_receipt"] is False
    assert receipt["completed_pair_count"] == 1


class _CloseFailCamera(FakeCamera):
    def close(self) -> None:
        super().close()
        raise RuntimeError("camera cleanup failed")


def test_primary_provider_error_is_separate_from_cleanup_error(tmp_path: Path) -> None:
    camera = _CloseFailCamera([], camera_reads(), fail_at=0)

    with pytest.raises(LiveCaptureAttemptError) as caught:
        _capture(tmp_path, camera, FakeJoint([], joint_reads()))

    failure = caught.value.failure
    assert failure.primary_error.detail == "camera read failed"
    cleanup_errors = [error for item in failure.cleanup for error in item.cleanup_errors]
    assert cleanup_errors == ["RuntimeError: camera cleanup failed"]


class _StubbornProcess(FakeProviderProcess):
    """Ignore terminate so cleanup must kill before the second reap."""

    def terminate(self) -> None:
        self.terminate_calls += 1


class _KillEscalationRuntime(FakeProviderRuntime):
    def spawn(self, spec: WorkerSpec) -> ProviderProcess:
        process = _StubbornProcess(spec)
        self.processes.append(process)
        return process

    def wait(
        self,
        processes: tuple[ProviderProcess, ...],
        timeout: float,
    ) -> tuple[ProviderProcess, ...]:
        del processes, timeout
        return ()


def test_timeout_escalates_terminate_kill_and_reap_truthfully(tmp_path: Path) -> None:
    runtime = _KillEscalationRuntime()

    with pytest.raises(LiveCaptureAttemptError) as caught:
        _capture(
            tmp_path,
            FakeCamera([], camera_reads()),
            FakeJoint([], joint_reads()),
            settings=CaptureSettings(process_runtime=runtime),
        )

    cleanup = caught.value.failure.cleanup[0]
    assert cleanup.terminate_sent is True
    assert cleanup.kill_sent is True
    assert cleanup.process_reaped is True
    assert cleanup.adapter_closed is False
    assert runtime.processes[0].join_calls == 2
    assert not runtime.processes[0].is_alive()
