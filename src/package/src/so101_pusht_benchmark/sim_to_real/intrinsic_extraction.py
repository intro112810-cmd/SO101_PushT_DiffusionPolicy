"""All-frame classification and deterministic geometry-based view selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from .camera_registration_vision import MIN_BLUR_VARIANCE, MIN_POSE_DISTANCE, pose_distance
from .intrinsic_extraction_types import (
    CandidateFrame,
    CandidatePool,
    Corners,
    Coverage,
    DecodedFrame,
    Detector,
    FrameRecord,
    FrameStatus,
    ScanResult,
    ScanSummary,
    SelectionFeatures,
)

HELDOUT_VIEW_COUNT: Final = 6
FIT_SET_SIZES: Final = (6, 12, 18, 24, 30)
MIN_CENTROID_X_SPAN: Final = 0.20
MIN_CENTROID_Y_SPAN: Final = 0.15
MIN_SCALE_RATIO: Final = 1.35
MIN_PROJECTIVE_SPAN: Final = 0.08


@dataclass(frozen=True, slots=True)
class ExtractionError(Exception):
    reason: str

    def __str__(self) -> str:
        """Return the boundary-safe failure reason."""
        return self.reason


def _features(corners: Corners, resolution: tuple[int, int]) -> SelectionFeatures:
    width, height = resolution
    normalized = corners / np.asarray([width, height], dtype=np.float64)
    centroid = normalized.mean(axis=0)
    quad = normalized[[0, 6, 34, 28]]
    area = 0.5 * abs(
        float(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)))
        - float(np.dot(quad[:, 1], np.roll(quad[:, 0], -1)))
    )
    top = float(np.linalg.norm(normalized[6] - normalized[0]))
    bottom = float(np.linalg.norm(normalized[34] - normalized[28]))
    left = float(np.linalg.norm(normalized[28] - normalized[0]))
    right = float(np.linalg.norm(normalized[34] - normalized[6]))
    epsilon = np.finfo(np.float64).tiny
    return SelectionFeatures(
        float(centroid[0]),
        float(centroid[1]),
        math.sqrt(area),
        math.log(max(top, epsilon) / max(bottom, epsilon)),
        math.log(max(left, epsilon) / max(right, epsilon)),
        tuple(float(value) for value in normalized.reshape(-1)),
    )


def _record(frame: DecodedFrame, status: FrameStatus, sharpness: float | None) -> FrameRecord:
    return FrameRecord(
        frame.frame_index,
        frame.pts,
        frame.time_base_numerator,
        frame.time_base_denominator,
        frame.timestamp_seconds,
        status,
        sharpness,
    )


def scan_frames(frames: Iterable[DecodedFrame], detect: Detector) -> ScanResult:
    """Evaluate every decoded frame exactly once with the current detector contract."""
    records: list[FrameRecord] = []
    candidates: list[CandidateFrame] = []
    resolution: tuple[int, int] | None = None
    complete = blur = incomplete = nonfinite = out_of_bounds = 0
    for frame in frames:
        height, width = frame.image.shape[:2]
        observed = (width, height)
        if resolution is None:
            resolution = observed
        if observed != resolution:
            raise ExtractionError("decoded video resolution changed during the complete scan")
        try:
            corners, sharpness = detect(frame.image)
        except ValueError as error:
            if "too blurred" in str(error):
                blur += 1
                records.append(_record(frame, FrameStatus.BLUR, None))
            else:
                incomplete += 1
                records.append(_record(frame, FrameStatus.INCOMPLETE, None))
            continue
        corners = np.asarray(corners, dtype=np.float64)
        sharpness = float(sharpness)
        if corners.shape != (35, 2):
            incomplete += 1
            records.append(_record(frame, FrameStatus.INCOMPLETE, sharpness))
            continue
        complete += 1
        if not math.isfinite(sharpness) or not np.isfinite(corners).all():
            nonfinite += 1
            records.append(_record(frame, FrameStatus.NONFINITE, sharpness))
            continue
        if sharpness < MIN_BLUR_VARIANCE:
            blur += 1
            records.append(_record(frame, FrameStatus.BLUR, sharpness))
            continue
        if (
            float(corners[:, 0].min()) < 0.0
            or float(corners[:, 1].min()) < 0.0
            or float(corners[:, 0].max()) >= width
            or float(corners[:, 1].max()) >= height
        ):
            out_of_bounds += 1
            records.append(_record(frame, FrameStatus.OUT_OF_BOUNDS, sharpness))
            continue
        records.append(_record(frame, FrameStatus.ELIGIBLE, sharpness))
        candidates.append(
            CandidateFrame(
                frame.frame_index,
                frame.pts,
                frame.time_base_numerator,
                frame.time_base_denominator,
                frame.timestamp_seconds,
                sharpness,
                corners,
                _features(corners, observed),
            )
        )
    if resolution is None:
        raise ExtractionError("source video decoded zero frames")
    return ScanResult(
        tuple(records),
        tuple(candidates),
        ScanSummary(
            len(records),
            complete,
            len(candidates),
            blur,
            incomplete,
            nonfinite,
            out_of_bounds,
        ),
        resolution,
    )


def _geometry_distance(left: CandidateFrame, right: CandidateFrame) -> float:
    delta = np.asarray(left.features.normalized_geometry) - np.asarray(
        right.features.normalized_geometry
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def _unique_candidates(
    candidates: tuple[CandidateFrame, ...], resolution: tuple[int, int]
) -> tuple[CandidateFrame, ...]:
    unique: list[CandidateFrame] = []
    for candidate in candidates:
        if all(
            pose_distance(candidate.corners, accepted.corners, resolution) >= MIN_POSE_DISTANCE
            for accepted in unique
        ):
            unique.append(candidate)
    return tuple(unique)


def _farthest_order(unique: tuple[CandidateFrame, ...]) -> tuple[CandidateFrame, ...]:
    geometry = np.asarray([item.features.normalized_geometry for item in unique])
    center = geometry.mean(axis=0)
    seed_scores = np.sqrt(np.mean(np.square(geometry - center), axis=1))
    seed = max(
        range(len(unique)),
        key=lambda index: (seed_scores[index], unique[index].sharpness, -unique[index].frame_index),
    )
    ordered_indices = [seed]
    while len(ordered_indices) < len(unique):
        remaining = [index for index in range(len(unique)) if index not in ordered_indices]
        ordered_indices.append(
            max(
                remaining,
                key=lambda index: (
                    min(
                        _geometry_distance(unique[index], unique[accepted])
                        for accepted in ordered_indices
                    ),
                    unique[index].sharpness,
                    -unique[index].frame_index,
                ),
            )
        )
    return tuple(unique[index] for index in ordered_indices)


def _coverage(candidates: tuple[CandidateFrame, ...]) -> Coverage:
    x = [candidate.features.centroid_x for candidate in candidates]
    y = [candidate.features.centroid_y for candidate in candidates]
    scales = [candidate.features.scale for candidate in candidates]
    projective_x = [candidate.features.projective_x for candidate in candidates]
    projective_y = [candidate.features.projective_y for candidate in candidates]
    return Coverage(
        max(x) - min(x),
        max(y) - min(y),
        max(scales) / min(scales),
        max(max(projective_x) - min(projective_x), max(projective_y) - min(projective_y)),
    )


def _pairwise(
    candidates: tuple[CandidateFrame, ...], resolution: tuple[int, int]
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(pose_distance(left.corners, right.corners, resolution) for right in candidates)
        for left in candidates
    )


def build_candidate_pool(
    candidates: tuple[CandidateFrame, ...], resolution: tuple[int, int]
) -> CandidatePool:
    """Deduplicate, farthest-order, and reserve six geometry-diverse held-out views."""
    unique = _unique_candidates(candidates, resolution)
    minimum_required = HELDOUT_VIEW_COUNT + FIT_SET_SIZES[0]
    if len(unique) < minimum_required:
        raise ExtractionError("recording lacks diverse fit and held-out intrinsic poses")
    ordered = _farthest_order(unique)
    heldout = ordered[:HELDOUT_VIEW_COUNT]
    fit_order = ordered[HELDOUT_VIEW_COUNT:]
    capped_size = min(len(fit_order), FIT_SET_SIZES[-1])
    fit_sizes = tuple(
        sorted({size for size in FIT_SET_SIZES if size <= capped_size} | {capped_size})
    )
    matrix = _pairwise(ordered, resolution)
    minimum = min(matrix[row][column] for row in range(len(ordered)) for column in range(row))
    coverage = _coverage(ordered)
    if minimum < MIN_POSE_DISTANCE or (
        coverage.centroid_x_span < MIN_CENTROID_X_SPAN
        or coverage.centroid_y_span < MIN_CENTROID_Y_SPAN
        or coverage.scale_ratio < MIN_SCALE_RATIO
        or coverage.projective_span < MIN_PROJECTIVE_SPAN
    ):
        raise ExtractionError(
            "recording lacks meaningful centroid, scale, and projective diversity"
        )
    return CandidatePool(
        heldout,
        fit_order,
        fit_sizes,
        len(candidates) - len(unique),
        len(unique),
        matrix,
        minimum,
        coverage,
    )
