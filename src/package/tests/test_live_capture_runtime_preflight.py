"""Feetech dependency and child-startup containment before device creation."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import ModuleType

import pytest

from read_only_authority_fakes import fixture_runtime_preflight
from so101_pusht_benchmark.sim_to_real.live_capture_child_failure import read_child_failure
from so101_pusht_benchmark.sim_to_real.live_capture_failure import LiveCaptureAttemptError
from so101_pusht_benchmark.sim_to_real.live_capture_process import (
    MultiprocessingProviderRuntime,
)
from so101_pusht_benchmark.sim_to_real.live_capture_protocol import ProviderRole
from so101_pusht_benchmark.sim_to_real.live_capture_types import CameraObservation
from so101_pusht_benchmark.sim_to_real.live_capture_runtime import (
    RuntimeDependencyReceipt,
    RuntimeInspector,
    verify_feetech_runtime,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from test_live_sample_capture import (
    CaptureSettings,
    FakeCamera,
    FakeJoint,
    camera_reads,
    capture_fake,
    joint_reads,
)


def _module(path: Path) -> ModuleType:
    module = ModuleType("scservo_sdk")
    module.__file__ = str(path)
    return module


@pytest.mark.parametrize("observed", [None, "1.0.1", "0.9.9"])
def test_missing_or_drifted_distribution_rejects(observed: str | None, tmp_path: Path) -> None:
    module_path = tmp_path / "scservo_sdk.py"
    module_path.write_text("# fixture\n", encoding="utf-8")
    inspector = RuntimeInspector(
        lambda _name: observed,
        lambda _name: ("feetech-servo-sdk",),
        lambda _name: _module(module_path),
    )

    with pytest.raises(RolloutViolation) as caught:
        verify_feetech_runtime(inspector)

    assert caught.value.code is RolloutCode.R_PROVIDER_MISMATCH


@pytest.mark.parametrize("failure", ["owner", "module"])
def test_module_owner_or_import_failure_rejects(failure: str, tmp_path: Path) -> None:
    module_path = tmp_path / "scservo_sdk.py"
    module_path.write_text("# fixture\n", encoding="utf-8")
    inspector = RuntimeInspector(
        lambda _name: "1.0.0",
        lambda _name: ("other-distribution",) if failure == "owner" else ("feetech-servo-sdk",),
        lambda _name: None if failure == "module" else _module(module_path),
    )

    with pytest.raises(RolloutViolation) as caught:
        verify_feetech_runtime(inspector)

    assert caught.value.code is RolloutCode.R_PROVIDER_MISMATCH


def test_exact_distribution_module_and_origin_pass(tmp_path: Path) -> None:
    module_path = tmp_path / "scservo_sdk.py"
    module_path.write_text("# fixture\n", encoding="utf-8")

    receipt = verify_feetech_runtime(
        RuntimeInspector(
            lambda _name: "1.0.0",
            lambda _name: ("feetech-servo-sdk",),
            lambda _name: _module(module_path),
        )
    )

    assert receipt == RuntimeDependencyReceipt(
        "feetech-servo-sdk",
        "1.0.0",
        "scservo_sdk",
        module_path,
    )


class _MarkerCamera(FakeCamera):
    def __init__(self, marker: Path) -> None:
        super().__init__([], camera_reads())
        self._marker = marker

    def open(self) -> CameraObservation:
        self._marker.write_text("camera opened\n", encoding="utf-8")
        return super().open()


class _MarkerJoint(FakeJoint):
    def __init__(self, marker: Path) -> None:
        super().__init__([], joint_reads())
        self._marker = marker

    def open(self) -> None:
        self._marker.write_text("joint opened\n", encoding="utf-8")
        super().open()


class _ParentPassChildCrash:
    """Pass in the coordinator, then fail in each forked child copy."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> RuntimeDependencyReceipt:
        if self.calls == 0:
            self.calls += 1
            return fixture_runtime_preflight()
        raise ModuleNotFoundError("No module named 'scservo_sdk'")


def test_tampered_child_failure_journal_becomes_typed_evidence(tmp_path: Path) -> None:
    journal = tmp_path / "failure.json"
    journal.write_text("{not-json", encoding="utf-8")

    failure = read_child_failure(journal, ProviderRole.JOINT)

    assert failure is not None
    assert failure.error_type == "ChildFailureJournalError"
    assert failure.role is ProviderRole.JOINT


def test_child_startup_crash_preserves_traceback_exit_reap_and_opens_no_device(
    tmp_path: Path,
) -> None:
    camera_marker = tmp_path / "camera-opened"
    joint_marker = tmp_path / "joint-opened"
    temporary_root = Path(tempfile.gettempdir())
    before = set(temporary_root.glob("so101-live-child-*"))

    with pytest.raises(LiveCaptureAttemptError) as caught:
        capture_fake(
            tmp_path,
            _MarkerCamera(camera_marker),
            _MarkerJoint(joint_marker),
            settings=CaptureSettings(
                process_runtime=MultiprocessingProviderRuntime(),
                runtime_preflight=_ParentPassChildCrash(),
            ),
        )

    failure = caught.value.failure
    after = set(temporary_root.glob("so101-live-child-*"))
    assert not camera_marker.exists()
    assert not joint_marker.exists()
    assert failure.primary_error.error_type == "ModuleNotFoundError"
    assert failure.primary_error.phase == "runtime_preflight"
    assert all(cleanup.process_reaped for cleanup in failure.cleanup)
    assert all(cleanup.exit_code is not None for cleanup in failure.cleanup)
    assert any("scservo_sdk" in (cleanup.child_traceback or "") for cleanup in failure.cleanup)
    assert after == before
