from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
from typing import cast, Protocol

import numpy as np
from numpy.typing import NDArray
import pytest

from so101_pusht_benchmark.sim_to_real.camera_registration import audit_camera_registration
from so101_pusht_benchmark.sim_to_real import camera_registration_capture as capture_module
from so101_pusht_benchmark.sim_to_real.receipt_routing import ReceiptPathIdentity
from so101_pusht_benchmark.sim_to_real.camera_registration_capture import (
    CameraObservation,
    CaptureDependencies,
    RegistrationAuthority,
    RegistrationCaptureRequest,
    run_guided_capture,
)
from so101_pusht_benchmark.sim_to_real.camera_registration_target import (
    PLACEMENTS,
    SQUARE_SIZE_MM,
    TARGET_ASSET,
    table_corners,
)
from so101_pusht_benchmark.sim_to_real.camera_registration_vision import (
    DetectedView,
    FittedGeometry,
    detect_checkerboard,
    encode_png,
    fit_geometry,
    reject_repeated_pose,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority_types import ReadOnlyCameraPolicy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame import registration_evidence_digest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture_camera_registration.py"
GENERATOR = ROOT / "scripts/generate_camera_registration_target.py"
CAMERA_SOURCE = ROOT / "src/so101_pusht_benchmark/sim_to_real/camera_registration_capture_cli.py"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class _Board(Protocol):
    def generateImage(
        self, size: tuple[int, int], margin_size: int = 0, border_bits: int = 1
    ) -> NDArray[np.uint8]: ...


class _Aruco(Protocol):
    DICT_5X5_100: int

    def getPredefinedDictionary(self, identifier: int) -> object: ...

    def CharucoBoard(
        self, size: tuple[int, int], square_length: float, marker_length: float, dictionary: object
    ) -> _Board: ...


class _Cv2(Protocol):
    aruco: _Aruco
    INTER_AREA: int

    def resize(
        self, source: NDArray[np.uint8], size: tuple[int, int], interpolation: int
    ) -> NDArray[np.uint8]: ...


def _authority() -> RegistrationAuthority:
    return RegistrationAuthority(
        SHA_A,
        SHA_B,
        SHA_C,
        Path("/profile.yaml"),
        SHA_D,
        Path("/camera"),
        SHA_A,
        SHA_B,
        640,
        480,
        30.0,
        (100, 0, 400, 400),
        ReadOnlyCameraPolicy(1.5, 12, 2.0),
    )


class FakeCamera:
    def __init__(
        self,
        *,
        start_capture: int = 0,
        observation: CameraObservation | None = None,
    ) -> None:
        self.start_capture = start_capture
        self.observation = observation or CameraObservation(640, 480, 30.0)
        self.open_count = 0
        self.read_count = 0
        self.close_count = 0

    def open(self) -> CameraObservation:
        self.open_count += 1
        return self.observation

    def read(self) -> NDArray[np.uint8]:
        marker = 255 if self.read_count == 0 else self.start_capture + self.read_count - 1
        self.read_count += 1
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        frame[0, 0, 0] = marker
        return frame

    def close(self) -> None:
        self.close_count += 1


class FakeVision:
    def __init__(self, *, fail_at: int | None = None, failure: str = "detection") -> None:
        self.fail_at = fail_at
        self.failure = failure
        self.detect_count = 0

    @staticmethod
    def _project(points: list[tuple[float, float, float]]) -> NDArray[np.float64]:
        return np.asarray([[500.0 * x + 320.0, -500.0 * y + 240.0] for x, y, _z in points])

    def detect(self, frame: NDArray[np.uint8], /) -> tuple[NDArray[np.float64], float]:
        marker = int(frame[0, 0, 0])
        if self.fail_at is not None and self.detect_count == self.fail_at:
            self.detect_count += 1
            raise ValueError(
                "frame is too blurred for registration"
                if self.failure == "blur"
                else "checkerboard corner detection failed"
            )
        self.detect_count += 1
        placement = PLACEMENTS[marker]
        if placement.role is not None:
            return self._project(table_corners(placement)), 500.0
        base = np.asarray(
            [[180.0 + column * 35.0, 120.0 + row * 35.0] for row in range(5) for column in range(7)]
        )
        return base + np.asarray([marker * 10.0, marker * 6.0]), 500.0

    def encode(self, frame: NDArray[np.uint8], /) -> bytes:
        return encode_png(frame)

    def reject_repeat(
        self,
        _candidate: NDArray[np.float64],
        _accepted: list[DetectedView],
        _resolution: tuple[int, int],
        /,
    ) -> None:
        return

    def fit(self, views: list[DetectedView], _resolution: tuple[int, int], /) -> FittedGeometry:
        if len(views) != len(PLACEMENTS):
            raise ValueError("insufficient views")
        return FittedGeometry(
            (500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        result = self.value
        self.value += timedelta(milliseconds=1)
        return result


def _default_probe(_path: Path) -> str:
    return SHA_A


def _dependencies(
    vision: FakeVision,
    *,
    probe: Callable[[Path], str | None] | None = None,
) -> CaptureDependencies:
    return CaptureDependencies(
        probe or _default_probe,
        lambda _placement: None,
        vision,
        Clock(),
    )


@dataclass(frozen=True, slots=True)
class RunOptions:
    resume: bool = False
    measured: float = 25.0
    probe: Callable[[Path], str | None] | None = None


def _run(
    root: Path,
    camera: FakeCamera,
    vision: FakeVision,
    options: RunOptions | None = None,
) -> dict[str, object]:
    selected = options or RunOptions()
    return run_guided_capture(
        RegistrationCaptureRequest(root, selected.measured, selected.resume, production=False),
        _authority(),
        camera,
        _dependencies(vision, probe=selected.probe),
    )


def test_printable_target_is_the_board_only_pdf_contract() -> None:
    asset = ROOT / TARGET_ASSET
    assert TARGET_ASSET.as_posix() == "docs/assets/camera_registration_charuco_board_only_a4.pdf"
    assert asset.is_file()
    assert asset.stat().st_size > 0


def test_printable_target_declares_exact_physical_scale() -> None:
    content = (ROOT / "docs/assets/camera_registration_charuco_a4.svg").read_text(encoding="utf-8")
    assert 'width="210mm" height="297mm"' in content
    assert f"square = exactly {SQUARE_SIZE_MM:.1f} mm" in content
    assert "DICT_5X5_100" in content
    assert 'id="marker-23-' in content


def test_target_generator_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "target.svg"
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output)],
        cwd=ROOT,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        output.read_bytes()
        == (ROOT / "docs/assets/camera_registration_charuco_a4.svg").read_bytes()
    )


def test_generated_charuco_is_detected_by_installed_opencv() -> None:
    cv2 = cast("_Cv2", import_module("cv2"))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((8, 6), 0.025, 0.018, dictionary).generateImage((400, 300), 0, 1)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[90:390, 120:520] = np.repeat(board[:, :, None], 3, axis=2)
    corners, sharpness = detect_checkerboard(frame)
    assert corners.shape == (35, 2)
    assert sharpness >= 80.0


def test_small_charuco_returns_all_corners_in_original_coordinates() -> None:
    cv2 = cast("_Cv2", import_module("cv2"))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.resize(
        cv2.aruco.CharucoBoard((8, 6), 0.025, 0.018, dictionary).generateImage((400, 300), 0, 1),
        (140, 105),
        interpolation=cv2.INTER_AREA,
    )
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[187:292, 250:390] = np.repeat(board[:, :, None], 3, axis=2)

    corners, _sharpness = detect_checkerboard(frame)

    expected = np.asarray(
        [
            [250.0 + column * 140.0 / 8.0, 187.0 + row * 105.0 / 6.0]
            for row in range(1, 6)
            for column in range(1, 8)
        ]
    )
    assert corners.shape == (35, 2)
    assert np.allclose(corners, expected, atol=2.0)
    assert float(corners.max()) < 400.0


@pytest.mark.parametrize("measured", [float("nan"), float("inf"), 24.874, 25.126])
def test_invalid_square_measurement_fails_before_camera_open(
    tmp_path: Path, measured: float
) -> None:
    camera = FakeCamera()
    with pytest.raises(ValueError, match="print scale invalid"):
        _run(tmp_path / "session", camera, FakeVision(), RunOptions(measured=measured))
    assert camera.open_count == 0


def test_publish_uses_verified_resolved_parent_for_lexical_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lexical_parent = tmp_path / "lexical"
    resolved_parent = tmp_path / "resolved"
    resolved_parent.mkdir()
    lexical = lexical_parent / "receipt.json"
    resolved = resolved_parent / "receipt.json"
    identity = ReceiptPathIdentity(lexical, resolved, True)

    def locate(_path: Path) -> ReceiptPathIdentity:
        return identity

    monkeypatch.setattr(capture_module, "locate_receipt_path", locate)

    def validate(
        current: ReceiptPathIdentity, *, _production: bool = False, **_kwargs: bool
    ) -> ReceiptPathIdentity:
        return current

    monkeypatch.setattr(capture_module, "validate_receipt_identity", validate)

    capture_module.publish(lexical, b"receipt")

    assert resolved.read_bytes() == b"receipt"
    assert not lexical.exists()


def test_publish_removes_output_when_lexical_identity_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lexical = tmp_path / "lexical/receipt.json"
    resolved = tmp_path / "resolved/receipt.json"
    resolved.parent.mkdir()
    identity = ReceiptPathIdentity(lexical, resolved, True)
    calls = 0

    def validate(
        _identity: ReceiptPathIdentity, *, _production: bool = False, **_kwargs: bool
    ) -> ReceiptPathIdentity:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise capture_module.ReceiptRoutingError("identity drift")
        return identity

    def locate(_path: Path) -> ReceiptPathIdentity:
        return identity

    monkeypatch.setattr(capture_module, "locate_receipt_path", locate)
    monkeypatch.setattr(capture_module, "validate_receipt_identity", validate)

    with pytest.raises(capture_module.ReceiptRoutingError, match="identity drift"):
        capture_module.publish(lexical, b"receipt")

    assert not resolved.exists()


def test_publish_removes_output_when_revalidation_rollout_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lexical = tmp_path / "lexical/receipt.json"
    resolved = tmp_path / "resolved/receipt.json"
    resolved.parent.mkdir()
    identity = ReceiptPathIdentity(lexical, resolved, True)
    calls = 0

    def validate(
        _identity: ReceiptPathIdentity, *, _production: bool = False, **_kwargs: bool
    ) -> ReceiptPathIdentity:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "identity drift")
        return identity

    def locate(_path: Path) -> ReceiptPathIdentity:
        return identity

    monkeypatch.setattr(capture_module, "locate_receipt_path", locate)
    monkeypatch.setattr(capture_module, "validate_receipt_identity", validate)

    with pytest.raises(RolloutViolation, match="identity drift"):
        capture_module.publish(lexical, b"receipt")

    assert not resolved.exists()


def test_square_measurement_tolerance_accepts_25_mm(tmp_path: Path) -> None:
    summary = _run(tmp_path / "session", FakeCamera(), FakeVision(), RunOptions(measured=25.0))
    assert summary["audited"] is True


def test_injected_fake_happy_captures_raw_png_and_unsigned_corpus(tmp_path: Path) -> None:
    camera = FakeCamera()
    summary = _run(tmp_path / "session", camera, FakeVision())
    assert summary["audited"] is True
    assert summary["authoritative"] is False
    assert summary["publication_status"] == "capture_complete_owner_signature_required"
    assert summary["raw_capture_count"] == 11
    assert camera.read_count == 12  # exactly one prime plus eleven accepted events
    assert camera.close_count == 1
    corpus = json.loads((tmp_path / "session/corpus.json").read_text(encoding="utf-8"))
    assert len(corpus["members"]) == 5
    assert len(corpus["fit_correspondences"]) == 105
    assert len(corpus["held_out_correspondences"]) == 70
    assert corpus["physical_to_sim"]["direction"] == "physical_table_to_simulation_table"
    assert corpus["camera_to_table"]["translation_units"] == "meters"
    for member in (tmp_path / "session/members").glob("*.png"):
        content = member.read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("failure", ["blur", "detection"])
def test_injected_bad_frame_is_partial_resumable_and_unsigned(tmp_path: Path, failure: str) -> None:
    camera = FakeCamera()
    with pytest.raises(ValueError, match=r"blurred|detection failed"):
        _run(tmp_path / "session", camera, FakeVision(fail_at=0, failure=failure))
    assert camera.close_count == 1
    assert (tmp_path / "session/records/000-session-header.json").is_file()
    assert not (tmp_path / "session/corpus.json").exists()
    assert not (tmp_path / "session/capture-summary.json").exists()


def test_insufficient_intrinsic_views_fail_geometry_fit() -> None:
    corners = np.zeros((35, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="six nondegenerate intrinsic views"):
        fit_geometry([DetectedView(PLACEMENTS[0], corners)], (640, 480))


def test_repeated_pose_is_rejected() -> None:
    corners = np.arange(70, dtype=np.float64).reshape(35, 2)
    accepted = [DetectedView(PLACEMENTS[0], corners)]
    with pytest.raises(ValueError, match="repeated target pose"):
        reject_repeated_pose(corners.copy(), accepted, (640, 480))


def test_identity_drift_fails_before_capture_event(tmp_path: Path) -> None:
    camera = FakeCamera()
    with pytest.raises(ValueError, match="device identity drift"):
        _run(
            tmp_path / "session",
            camera,
            FakeVision(),
            RunOptions(probe=lambda _path: SHA_B),
        )
    assert camera.read_count == 1  # the one authorized prime only
    assert list((tmp_path / "session/members").iterdir()) == []


def test_raw_member_tamper_blocks_partial_resume_before_camera_open(tmp_path: Path) -> None:
    root = tmp_path / "session"
    with pytest.raises(ValueError, match="detection failed"):
        _run(root, FakeCamera(), FakeVision(fail_at=1))
    member = root / "members/intrinsic-01.png"
    member.write_bytes(member.read_bytes() + b"tamper")
    resumed = FakeCamera(start_capture=1)
    with pytest.raises(ValueError, match="raw member tamper"):
        _run(root, resumed, FakeVision(), RunOptions(resume=True))
    assert resumed.open_count == 0


def test_partial_resume_rehashes_prior_members_and_completes(tmp_path: Path) -> None:
    root = tmp_path / "session"
    with pytest.raises(ValueError, match="detection failed"):
        _run(root, FakeCamera(), FakeVision(fail_at=3))
    resumed = FakeCamera(start_capture=3)
    summary = _run(root, resumed, FakeVision(), RunOptions(resume=True))
    assert summary["raw_capture_count"] == 11
    assert resumed.read_count == 9  # one new prime plus eight remaining views
    assert len(list((root / "members").glob("*.png"))) == 11


def test_frame_unit_and_direction_tamper_rejects(tmp_path: Path) -> None:
    root = tmp_path / "session"
    _run(root, FakeCamera(), FakeVision())
    corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    corpus["camera_to_table"]["direction"] = "table_to_camera"
    corpus["camera_digest"] = registration_evidence_digest(corpus)
    with pytest.raises(RolloutViolation, match="frame direction"):
        audit_camera_registration(
            corpus,
            corpus_root=root,
            source_scope="production",
            thresholds=_authority().camera_policy,
        )
    corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    corpus["physical_to_sim"]["physical_units"] = "millimeters"
    corpus["camera_digest"] = registration_evidence_digest(corpus)
    with pytest.raises(RolloutViolation, match="units or frame direction"):
        audit_camera_registration(
            corpus,
            corpus_root=root,
            source_scope="production",
            thresholds=_authority().camera_policy,
        )


def test_symlink_session_root_is_rejected_before_camera_open(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    camera = FakeCamera()
    with pytest.raises(ValueError, match="symlink"):
        _run(alias, camera, FakeVision())
    assert camera.open_count == 0


def test_camera_surface_has_no_setters_or_configuration_writes() -> None:
    tree = ast.parse(CAMERA_SOURCE.read_text(encoding="utf-8"))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint({"set", "configure", "calibrate", "sync_write", "write"})


def test_cli_requires_square_measurement_and_rejects_invalid_before_camera_open() -> None:
    camera = FakeCamera()
    from so101_pusht_benchmark.sim_to_real.camera_registration_capture_cli import run

    result = run(
        [
            "--profile",
            "/missing",
            "--acquisition-authority",
            "/missing",
            "--authority-signature",
            "/missing",
            "--trust-anchor",
            "/missing",
            "--output-dir",
            "/missing",
            "--measured-square-mm",
            "nan",
        ],
        camera=camera,
    )
    assert result == 2
    assert camera.open_count == 0


def test_cli_help_is_available_without_hardware() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--preflight-only" in result.stdout
    assert "--resume" in result.stdout
    assert "--measured-square-mm" in result.stdout
    assert "--measured-ruler-mm" not in result.stdout
