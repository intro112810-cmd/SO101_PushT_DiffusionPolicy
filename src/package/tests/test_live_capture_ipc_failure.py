"""Child EOF and BrokenPipe fallback to IPC-independent crash journals."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from live_capture_process_fakes import FakeProviderProcess, FakeProviderRuntime
from so101_pusht_benchmark.sim_to_real.live_capture_failure import (
    LiveCaptureAttemptError,
    terminal_failure_receipt,
)
from so101_pusht_benchmark.sim_to_real.live_capture_protocol import (
    ProviderCommand,
    ProviderEvent,
    ProviderFailed,
    ProviderProcess,
    StartProvider,
    WorkerSpec,
)
from test_live_sample_capture import (
    CaptureSettings,
    FakeCamera,
    FakeJoint,
    camera_reads,
    capture_fake,
    joint_reads,
)


class _EofStartupProcess(FakeProviderProcess):
    """Exit before a startup event while retaining a child crash journal."""

    @property
    def has_event(self) -> bool:
        return True

    def start(self) -> None:
        self.terminate()

    def receive(self) -> ProviderEvent:
        raise EOFError

    def child_failure(self) -> ProviderFailed | None:
        return ProviderFailed(
            self.role,
            "runtime_preflight",
            None,
            1000.15,
            "ModuleNotFoundError",
            "No module named 'scservo_sdk'",
            "Traceback: ModuleNotFoundError: scservo_sdk",
        )


class _EofStartupRuntime(FakeProviderRuntime):
    def spawn(self, spec: WorkerSpec) -> ProviderProcess:
        process = _EofStartupProcess(spec) if not self.processes else FakeProviderProcess(spec)
        self.processes.append(process)
        return process


def test_child_startup_eof_uses_journal_phase_exit_and_reap(tmp_path: Path) -> None:
    runtime = _EofStartupRuntime()

    with pytest.raises(LiveCaptureAttemptError) as caught:
        capture_fake(
            tmp_path,
            FakeCamera([], camera_reads()),
            FakeJoint([], joint_reads()),
            settings=CaptureSettings(process_runtime=runtime),
        )

    failure = caught.value.failure
    camera_cleanup = next(item for item in failure.cleanup if item.role == "camera")
    assert failure.primary_error.error_type == "ModuleNotFoundError"
    assert failure.primary_error.phase == "runtime_preflight"
    assert camera_cleanup.exit_code == 0
    assert camera_cleanup.process_reaped is True
    assert "scservo_sdk" in (camera_cleanup.child_traceback or "")


class _BrokenStartProcess(FakeProviderProcess):
    """Lose command IPC while retaining an independent child failure journal."""

    def send(self, command: ProviderCommand) -> None:
        if isinstance(command, StartProvider):
            raise BrokenPipeError("child command socket closed")
        super().send(command)

    def child_failure(self) -> ProviderFailed | None:
        return ProviderFailed(
            self.role,
            "provider_startup",
            None,
            1000.15,
            "ModuleNotFoundError",
            "No module named 'scservo_sdk'",
            "Traceback: ModuleNotFoundError: scservo_sdk",
        )


class _BrokenStartRuntime(FakeProviderRuntime):
    def spawn(self, spec: WorkerSpec) -> ProviderProcess:
        process = _BrokenStartProcess(spec) if not self.processes else FakeProviderProcess(spec)
        self.processes.append(process)
        return process


def test_broken_pipe_uses_child_journal_and_terminal_evidence(tmp_path: Path) -> None:
    runtime = _BrokenStartRuntime()
    log: list[tuple[str, object]] = []

    with pytest.raises(LiveCaptureAttemptError) as caught:
        capture_fake(
            tmp_path,
            FakeCamera(log, camera_reads()),
            FakeJoint(log, joint_reads()),
            settings=CaptureSettings(process_runtime=runtime),
        )

    failure = caught.value.failure
    receipt = terminal_failure_receipt(
        failure,
        policy_digest="a" * 64,
        identity_digest="b" * 64,
    )
    assert log == []
    assert failure.primary_error.error_type == "ModuleNotFoundError"
    assert failure.primary_error.phase == "provider_startup"
    assert all(cleanup.process_reaped for cleanup in failure.cleanup)
    cleanup_records = receipt["cleanup"]
    assert isinstance(cleanup_records, list)
    records = cast("list[dict[str, object]]", cleanup_records)
    child_traceback = records[0]["child_traceback"]
    assert isinstance(child_traceback, str)
    assert "scservo_sdk" in child_traceback
    assert receipt["genuine_physical_samples"] is False
