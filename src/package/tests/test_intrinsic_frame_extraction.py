from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from so101_pusht_benchmark.sim_to_real.camera_registration_target import (
    Placement,
    local_inner_corners,
)
from so101_pusht_benchmark.sim_to_real.camera_registration_vision import (
    DetectedView,
    fit_intrinsic_views,
)
from so101_pusht_benchmark.sim_to_real.intrinsic_extraction import (
    ExtractionError,
    build_candidate_pool,
    scan_frames,
)
from so101_pusht_benchmark.sim_to_real.intrinsic_extraction_pipeline import run_extraction
from so101_pusht_benchmark.sim_to_real.intrinsic_extraction_types import (
    DecodedFrame,
    ExtractionDependencies,
    ExtractionRequest,
    FitEvaluation,
    FitQuality,
    HeldoutMetrics,
)


def _corners(
    center_x: float,
    center_y: float,
    scale: float,
    projective_x: float = 0.0,
    projective_y: float = 0.0,
) -> NDArray[np.float64]:
    return np.asarray(
        [
            (
                center_x + scale * (column - 3.0) * (1.0 + projective_x * (row - 2.0)),
                center_y + scale * (row - 2.0) * (1.0 + projective_y * (column - 3.0)),
            )
            for row in range(5)
            for column in range(7)
        ],
        dtype=np.float64,
    )


def _pose(index: int) -> NDArray[np.float64]:
    projective = (-0.12, -0.06, 0.0, 0.06, 0.12)
    return _corners(
        100.0 + 88.0 * (index % 6),
        90.0 + 72.0 * (index // 6),
        10.0 + 2.5 * (index % 5),
        projective[index % 5],
        projective[(index * 2) % 5],
    )


def _frame(index: int, *, width: int = 640, height: int = 480) -> DecodedFrame:
    return DecodedFrame(
        index,
        index,
        1,
        30,
        index / 30.0,
        np.full((height, width, 3), index, dtype=np.uint8),
    )


def test_fit_intrinsic_views_passes_contiguous_float32_points_to_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = np.asarray(local_inner_corners(), dtype=np.float64)[:, :2]
    views = [
        DetectedView(
            Placement(f"synthetic-{index}", "intrinsics", None, "test"),
            np.column_stack(
                (
                    320.0
                    + (index - 2.5) * 8.0
                    + base[:, 0] * (600.0 + index * 12.0)
                    + base[:, 1] * index,
                    240.0
                    + (index - 2.5) * 6.0
                    + base[:, 1] * (580.0 - index * 9.0)
                    + base[:, 0] * (index - 2.5),
                )
            ),
        )
        for index in range(6)
    ]

    class FakeCv2:
        def calibrateCamera(
            self,
            object_points: list[NDArray[np.float32]],
            image_points: list[NDArray[np.float32]],
            _size: tuple[int, int],
            _matrix: None,
            _distortion: None,
        ) -> tuple[
            float,
            NDArray[np.float64],
            NDArray[np.float64],
            list[NDArray[np.float64]],
            list[NDArray[np.float64]],
        ]:
            assert all(points.dtype == np.float32 for points in object_points)
            assert all(points.dtype == np.float32 for points in image_points)
            assert all(points.flags.c_contiguous for points in object_points)
            assert all(points.flags.c_contiguous for points in image_points)
            return 0.5, np.eye(3), np.zeros((1, 5)), [], []

    def fake_cv2() -> FakeCv2:
        return FakeCv2()

    monkeypatch.setattr(
        "so101_pusht_benchmark.sim_to_real.camera_registration_vision._cv2", fake_cv2
    )
    matrix, distortion = fit_intrinsic_views(views, (640, 480))

    assert matrix.dtype == np.float64
    assert distortion.dtype == np.float64


def test_scan_records_every_decoded_frame_and_each_rejection_reason() -> None:
    frames = [_frame(index) for index in range(6)]

    def detect(image: NDArray[np.uint8]) -> tuple[NDArray[np.float64], float]:
        marker = int(image[0, 0, 0])
        if marker == 1:
            raise ValueError("frame is too blurred for registration")
        if marker == 2:
            return _corners(320.0, 240.0, 20.0)[:34], 200.0
        if marker == 3:
            corners = _corners(320.0, 240.0, 20.0)
            corners[0, 0] = np.nan
            return corners, 200.0
        if marker == 4:
            return _corners(-20.0, 240.0, 20.0), 200.0
        return _corners(320.0 + marker, 240.0, 20.0), 200.0

    result = scan_frames(frames, detect)

    assert result.summary.total_decoded == 6
    assert result.summary.complete_35_corner == 4
    assert result.summary.eligible == 2
    assert result.summary.blur_rejection == 1
    assert result.summary.incomplete_detection == 1
    assert result.summary.nonfinite == 1
    assert result.summary.out_of_bounds == 1
    assert [record.frame_index for record in result.records] == list(range(6))


def test_pool_is_stable_removes_duplicates_and_reserves_heldout_views() -> None:
    frames = [_frame(index) for index in range(31)]

    def detect(image: NDArray[np.uint8]) -> tuple[NDArray[np.float64], float]:
        marker = int(image[0, 0, 0])
        return (_pose(0) + 0.2 if marker == 1 else _pose(max(0, marker - 1))), 1000.0 - marker

    scanned = scan_frames(frames, detect)
    first = build_candidate_pool(scanned.candidates, (640, 480))
    second = build_candidate_pool(scanned.candidates, (640, 480))

    assert first.duplicate_count == 1
    assert len(first.heldout) == 6
    assert first.fit_sizes == (6, 12, 18, 24)
    assert [item.frame_index for item in first.fit_order] == [
        item.frame_index for item in second.fit_order
    ]
    assert first.minimum_pool_distance >= 0.035
    assert first.coverage.centroid_x_span >= 0.2
    assert first.coverage.centroid_y_span >= 0.15
    assert first.coverage.scale_ratio >= 1.35
    assert first.coverage.projective_span >= 0.08


def test_pool_adds_terminal_fit_size_without_exceeding_cap() -> None:
    frames = [_frame(index) for index in range(32)]

    def detect(image: NDArray[np.uint8]) -> tuple[NDArray[np.float64], float]:
        marker = int(image[0, 0, 0])
        return (_pose(0) + 0.2 if marker == 1 else _pose(max(0, marker - 1))), 1000.0 - marker

    scanned = scan_frames(frames, detect)
    assert build_candidate_pool(scanned.candidates[:13], (640, 480)).fit_sizes == (6,)
    assert build_candidate_pool(scanned.candidates[:14], (640, 480)).fit_sizes == (6, 7)
    assert build_candidate_pool(scanned.candidates, (640, 480)).fit_sizes == (6, 12, 18, 24, 25)


def test_insufficient_diversity_fails() -> None:
    frames = [_frame(index) for index in range(20)]

    def detect(image: NDArray[np.uint8]) -> tuple[NDArray[np.float64], float]:
        marker = int(image[0, 0, 0])
        return _corners(320.0 + marker, 240.0, 20.0), 200.0

    with pytest.raises(ExtractionError, match="fit and held-out"):
        build_candidate_pool(scan_frames(frames, detect).candidates, (640, 480))


def _pipeline_dependencies() -> ExtractionDependencies:
    def decode(_path: Path) -> tuple[DecodedFrame, ...]:
        return tuple(_frame(index) for index in range(36))

    def detect(image: NDArray[np.uint8]) -> tuple[NDArray[np.float64], float]:
        marker = int(image[0, 0, 0])
        return _pose(marker), 500.0 + marker

    def calibrate(
        fit_corners: tuple[NDArray[np.float64], ...],
        heldout_corners: tuple[NDArray[np.float64], ...],
        _size: tuple[int, int],
    ) -> FitEvaluation:
        count = len(fit_corners)
        assert len(heldout_corners) == 6
        heldout_rms = {6: 0.40, 12: 0.20, 18: 0.24, 24: 0.25, 27: 0.05, 30: 0.10}[count]
        return FitEvaluation(
            FitQuality(
                0.25,
                (500.0, 0.0, 320.0, 0.0, 505.0, 240.0, 0.0, 0.0, 1.0),
                (0.01, -0.02, 0.0, 0.0, 0.001),
            ),
            HeldoutMetrics(heldout_rms, 0.15, 0.14, 0.30, 0.40, 210),
        )

    return ExtractionDependencies(decode, detect, calibrate)


def test_pipeline_selects_best_heldout_count_and_is_byte_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"synthetic-video")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    left = tmp_path / "left"
    right = tmp_path / "right"

    for output in (left, right):
        receipt = run_extraction(
            ExtractionRequest(source, digest, output), _pipeline_dependencies()
        )
        assert receipt.fit_frame_count == 27
        assert receipt.heldout_frame_count == 6

    left_files = {
        path.relative_to(left): path.read_bytes() for path in left.rglob("*") if path.is_file()
    }
    right_files = {
        path.relative_to(right): path.read_bytes() for path in right.rglob("*") if path.is_file()
    }
    assert left_files == right_files
    receipt = json.loads((left / "extraction-receipt.json").read_text(encoding="utf-8"))
    assert receipt["fit_frame_count"] == 27
    assert receipt["heldout_frame_count"] == 6
    assert [metric["fit_frame_count"] for metric in receipt["fit_count_comparison"]] == [
        6,
        12,
        18,
        24,
        27,
    ]
    assert len(list(left.glob("fit-[0-9][0-9].png"))) == 27
    assert len(list(left.glob("heldout-[0-9][0-9].png"))) == 6
    assert len(list(left.glob("fit-contact-sheet-*.png"))) == 5
    assert len(list(left.glob("heldout-contact-sheet-*.png"))) == 1


def test_pipeline_failure_writes_no_success_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"synthetic-video")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "failed"
    dependencies = _pipeline_dependencies()

    def repeated(_image: NDArray[np.uint8]) -> tuple[NDArray[np.float64], float]:
        return _corners(320.0, 240.0, 20.0), 500.0

    failing = ExtractionDependencies(dependencies.decode, repeated, dependencies.calibrate)
    with pytest.raises(ExtractionError, match="fit and held-out"):
        run_extraction(ExtractionRequest(source, digest, output), failing)

    assert not (output / "extraction-receipt.json").exists()
