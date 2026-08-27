"""Typed values for deterministic offline intrinsic-frame extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Corners = NDArray[np.float64]
Decoder = Callable[[Path], Iterable["DecodedFrame"]]
Detector = Callable[[Image], tuple[Corners, float]]
Calibrator = Callable[[tuple[Corners, ...], tuple[Corners, ...], tuple[int, int]], "FitEvaluation"]


class FrameStatus(str, Enum):
    """Exhaustive per-frame scan outcomes."""

    ELIGIBLE = "eligible"
    BLUR = "blur_rejection"
    INCOMPLETE = "incomplete_detection"
    NONFINITE = "nonfinite"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One sequentially decoded source frame with native timing."""

    frame_index: int
    pts: int
    time_base_numerator: int
    time_base_denominator: int
    timestamp_seconds: float
    image: Image


@dataclass(frozen=True, slots=True)
class SelectionFeatures:
    """Coverage features derived from all 35 normalized corners."""

    centroid_x: float
    centroid_y: float
    scale: float
    projective_x: float
    projective_y: float
    normalized_geometry: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CandidateFrame:
    """A complete, finite, sharp, in-bounds intrinsic candidate."""

    frame_index: int
    pts: int
    time_base_numerator: int
    time_base_denominator: int
    timestamp_seconds: float
    sharpness: float
    corners: Corners
    features: SelectionFeatures


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Machine-readable result for every decoded frame."""

    frame_index: int
    pts: int
    time_base_numerator: int
    time_base_denominator: int
    timestamp_seconds: float
    status: FrameStatus
    sharpness: float | None


@dataclass(frozen=True, slots=True)
class ScanSummary:
    """Complete mutually exclusive scan reason counts."""

    total_decoded: int
    complete_35_corner: int
    eligible: int
    blur_rejection: int
    incomplete_detection: int
    nonfinite: int
    out_of_bounds: int


@dataclass(frozen=True, slots=True)
class ScanResult:
    records: tuple[FrameRecord, ...]
    candidates: tuple[CandidateFrame, ...]
    summary: ScanSummary
    resolution: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Coverage:
    centroid_x_span: float
    centroid_y_span: float
    scale_ratio: float
    projective_span: float


@dataclass(frozen=True, slots=True)
class CandidatePool:
    heldout: tuple[CandidateFrame, ...]
    fit_order: tuple[CandidateFrame, ...]
    fit_sizes: tuple[int, ...]
    duplicate_count: int
    eligible_unique_count: int
    pairwise_matrix: tuple[tuple[float, ...], ...]
    minimum_pool_distance: float
    coverage: Coverage


@dataclass(frozen=True, slots=True)
class FitQuality:
    rms_reprojection_error_px: float
    intrinsics: tuple[float, ...]
    distortion: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HeldoutMetrics:
    rms_error_px: float
    mean_error_px: float
    median_error_px: float
    p95_error_px: float
    max_error_px: float
    corner_count: int


@dataclass(frozen=True, slots=True)
class FitEvaluation:
    fit: FitQuality
    heldout: HeldoutMetrics


@dataclass(frozen=True, slots=True)
class CountEvaluation:
    fit_frame_count: int
    evaluation: FitEvaluation


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    source_video: Path
    expected_sha256: str
    output_directory: Path


@dataclass(frozen=True, slots=True)
class ExtractionDependencies:
    decode: Decoder
    detect: Detector
    calibrate: Calibrator


@dataclass(frozen=True, slots=True)
class ExtractionReceipt:
    source_sha256: str
    fit_frame_count: int
    heldout_frame_count: int
    total_decoded: int
    minimum_pool_distance: float
    evaluation: FitEvaluation
