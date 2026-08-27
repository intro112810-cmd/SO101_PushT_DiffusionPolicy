"""Explicit live-mode authority and publication gates for sample capture."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from live_capture_process_fakes import FakeProviderRuntime
from read_only_authority_fakes import fixture_runtime_preflight, signed_test_authority
from so101_pusht_benchmark.sim_to_real.live_capture_cli import (
    LiveCaptureDependencies,
    publish_capture_receipt,
    run_capture_cli,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority import (
    ProductionReadOnlyAcquisitionAuthority,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import CANONICAL_ROLLOUT_ROOT
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from test_live_sample_capture import (
    FakeCamera,
    FakeJoint,
    approved_identity,
    camera_reads,
    joint_reads,
)


def _profile(path: Path, calibration: Path) -> None:
    path.write_text(
        f"""schema: 1
deployment_scope: real_hardware_preflight
follower:
  role: follower
  port: {path.parent / "follower-device"}
  calibration_id: follower-01
  calibration_file: {calibration}
leader:
  role: leader
  port: {path.parent / "leader-device"}
  calibration_id: leader-01
  calibration_file: {path.parent / "leader-calibration.json"}
camera:
  role: front
  device: {path.parent / "camera-device"}
  width: 640
  height: 480
  fps: 30
  crop_x: 0
  crop_y: 0
  crop_size: 400
  saved_width: 400
  saved_height: 400
  latest_frame: {path.parent / "latest.jpg"}
safety:
  max_relative_target_degrees: 5.0
  require_workspace_confirmation: true
sim_to_real:
  physical_camera_registration_calibrated: false
  action_bridge: absent
  diagnostic_governance: real_diagnostic_rollout
  control_plane: separate_control_plane
  training_identity_authority: forbidden
""",
        encoding="utf-8",
    )


def _dependencies(
    tmp_path: Path,
    log: list[tuple[str, object]],
    *,
    fail_camera: bool = False,
) -> LiveCaptureDependencies:
    profile = tmp_path / "profile.yaml"
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}\n", encoding="utf-8")
    _profile(profile, calibration)
    authority = signed_test_authority(tmp_path, provider_digest="1" * 64)
    camera = FakeCamera(log, camera_reads(), fail_at=1 if fail_camera else None)
    joint = FakeJoint(
        log,
        joint_reads(),
        identity=replace(
            FakeJoint([], joint_reads()).identity,
            device_digest=authority.follower_device_digest,
            calibration_digest=authority.calibration_digest,
        ),
    )
    camera.identity = replace(
        camera.identity,
        device_digest=authority.camera_device_digest,
    )
    identities = {
        tmp_path / "camera-device": authority.camera_device_digest,
        tmp_path / "follower-device": authority.follower_device_digest,
    }

    def probe(path: Path) -> str | None:
        return identities.get(path)

    return LiveCaptureDependencies(
        policy_loader=lambda _path: authority,
        identity_loader=lambda _path: approved_identity(),
        camera_factory=lambda _configuration: camera,
        joint_factory=lambda _configuration: joint,
        device_probe=probe,
        profile_digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        calibration_digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        clock=lambda: 1000.15,
        process_runtime=FakeProviderRuntime(),
        runtime_preflight=fixture_runtime_preflight,
        acquisition_authority_loader=lambda _path, _signature: authority,
    )


def _argv(tmp_path: Path, output: Path) -> list[str]:
    return [
        "--live",
        "--profile",
        str(tmp_path / "profile.yaml"),
        "--acquisition-authority",
        str(tmp_path / "authority.json"),
        "--authority-signature",
        str(tmp_path / "authority.sig"),
        "--count",
        "2",
        "--output",
        str(output),
    ]


def test_live_mode_requires_provider_before_device_or_output_open(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = CANONICAL_ROLLOUT_ROOT / "samples/missing-provider.json"
    exit_code = run_capture_cli(_argv(tmp_path, output), live_dependencies=None)

    assert exit_code == 2
    assert "R_POLICY_UNAUTHORIZED" in capsys.readouterr().err


def test_live_mode_rejects_noncanonical_output_before_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loader_calls = 0
    dependencies = _dependencies(tmp_path, [])

    original_loader = dependencies.acquisition_authority_loader
    assert original_loader is not None

    def load_authority(
        path: Path,
        signature: Path,
    ) -> ProductionReadOnlyAcquisitionAuthority:
        nonlocal loader_calls
        loader_calls += 1
        return original_loader(path, signature)

    dependencies = replace(dependencies, acquisition_authority_loader=load_authority)
    exit_code = run_capture_cli(
        _argv(tmp_path, tmp_path / "not-canonical.json"),
        live_dependencies=dependencies,
    )

    assert exit_code == 2
    assert "canonical rollout root" in capsys.readouterr().err
    assert loader_calls == 0


def test_missing_feetech_runtime_publishes_terminal_receipt_before_camera_factory(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    published: list[tuple[Path, dict[str, object], bool]] = []
    output = CANONICAL_ROLLOUT_ROOT / "samples/missing-feetech.json"
    dependencies = replace(
        _dependencies(tmp_path, log),
        runtime_preflight=lambda: (_ for _ in ()).throw(
            RolloutViolation(
                RolloutCode.R_PROVIDER_MISMATCH,
                "feetech-servo-sdk==1.0.0 / scservo_sdk is unavailable",
            )
        ),
    )

    exit_code = run_capture_cli(
        _argv(tmp_path, output),
        live_dependencies=dependencies,
        publisher=lambda path, receipt, production: published.append((path, receipt, production)),
    )

    assert exit_code == 2
    assert log == []
    assert len(published) == 1
    assert "runtime_preflight" in repr(published[0][1]["primary_error"])
    assert published[0][1]["genuine_physical_samples"] is False


def test_fake_live_happy_publishes_genuine_only_after_capture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log: list[tuple[str, object]] = []
    published: list[tuple[Path, dict[str, object], bool]] = []
    output = CANONICAL_ROLLOUT_ROOT / "samples/fake-live-happy.json"

    exit_code = run_capture_cli(
        _argv(tmp_path, output),
        live_dependencies=_dependencies(tmp_path, log),
        publisher=lambda path, receipt, production: published.append((path, receipt, production)),
    )

    assert exit_code == 0
    assert published[0][0] == output
    assert published[0][1]["evidence_scope"] == "authorized_physical_diagnostic"
    assert published[0][1]["genuine_physical_samples"] is True
    assert published[0][2] is True
    assert log[-2:] == [("camera.close", None), ("joint.close", None)]
    assert "authorized_physical_diagnostic" in capsys.readouterr().out


def test_terminal_failure_receipt_publication_is_atomic_and_rejects_leaf_tamper(
    tmp_path: Path,
) -> None:
    receipt: dict[str, object] = {
        "terminal": True,
        "genuine_physical_samples": False,
        "primary_error": {"phase": "runtime_preflight"},
    }
    output = tmp_path / "attempt.terminal-failure.json"

    publish_capture_receipt(output, receipt, False)
    original = output.read_bytes()
    with pytest.raises(RolloutViolation):
        publish_capture_receipt(output, {"tampered": True}, False)
    assert output.read_bytes() == original

    symlink = tmp_path / "symlink.terminal-failure.json"
    symlink.symlink_to(output)
    with pytest.raises(ValueError, match="symlink"):
        publish_capture_receipt(symlink, receipt, False)


def test_fake_live_partial_failure_publishes_terminal_nonconsumable_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    published: list[tuple[Path, dict[str, object], bool]] = []
    output = CANONICAL_ROLLOUT_ROOT / "samples/fake-live-failure.json"

    def publish(path: Path, receipt: dict[str, object], production: bool) -> None:
        published.append((path, receipt, production))

    exit_code = run_capture_cli(
        _argv(tmp_path, output),
        live_dependencies=_dependencies(tmp_path, [], fail_camera=True),
        publisher=publish,
    )

    assert exit_code == 2
    assert len(published) == 1
    assert published[0][0].name == "fake-live-failure.terminal-failure.json"
    assert published[0][1]["genuine_physical_samples"] is False
    assert published[0][1]["completed_pair_count"] == 1
    assert published[0][2] is True
    assert "camera read failed" in capsys.readouterr().err
