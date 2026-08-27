"""Deterministic JSON documents for offline intrinsic extraction evidence."""

from __future__ import annotations

import json
from pathlib import Path

from .intrinsic_extraction_types import (
    CandidateFrame,
    CandidatePool,
    CountEvaluation,
    FrameRecord,
    ScanResult,
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def record_document(record: FrameRecord) -> dict[str, int | float | str | None]:
    return {
        "frame_index": record.frame_index,
        "pts": record.pts,
        "time_base_numerator": record.time_base_numerator,
        "time_base_denominator": record.time_base_denominator,
        "timestamp_seconds": record.timestamp_seconds,
        "status": record.status.value,
        "sharpness": record.sharpness,
    }


def summary_document(scan: ScanResult, pool: CandidatePool) -> dict[str, int]:
    summary = scan.summary
    return {
        "total_decoded": summary.total_decoded,
        "complete_35_corner": summary.complete_35_corner,
        "eligible": summary.eligible,
        "blur_rejection": summary.blur_rejection,
        "incomplete_detection": summary.incomplete_detection,
        "nonfinite": summary.nonfinite,
        "out_of_bounds": summary.out_of_bounds,
        "near_duplicate_pose": pool.duplicate_count,
        "eligible_unique": pool.eligible_unique_count,
    }


def coverage_document(pool: CandidatePool) -> dict[str, float]:
    coverage = pool.coverage
    return {
        "centroid_x_span": coverage.centroid_x_span,
        "centroid_y_span": coverage.centroid_y_span,
        "scale_ratio": coverage.scale_ratio,
        "projective_span": coverage.projective_span,
        "minimum_pool_pose_distance": pool.minimum_pool_distance,
    }


def evaluation_document(result: CountEvaluation) -> dict[str, object]:
    fit = result.evaluation.fit
    heldout = result.evaluation.heldout
    return {
        "fit_frame_count": result.fit_frame_count,
        "calibration_rms_px": fit.rms_reprojection_error_px,
        "intrinsics": list(fit.intrinsics),
        "distortion": list(fit.distortion),
        "finite_positive_focal_lengths": True,
        "heldout": {
            "rms_error_px": heldout.rms_error_px,
            "mean_error_px": heldout.mean_error_px,
            "median_error_px": heldout.median_error_px,
            "p95_error_px": heldout.p95_error_px,
            "max_error_px": heldout.max_error_px,
            "corner_count": heldout.corner_count,
        },
    }


def member_document(
    candidate: CandidateFrame,
    metadata: tuple[str, int, str, str],
    source: tuple[Path, str],
) -> dict[str, object]:
    role, rank, png_path, png_sha256 = metadata
    source_path, source_sha256 = source
    features = candidate.features
    return {
        "role": role,
        "selection_rank": rank,
        "source_video": str(source_path),
        "source_sha256": source_sha256,
        "frame_index": candidate.frame_index,
        "pts": candidate.pts,
        "time_base_numerator": candidate.time_base_numerator,
        "time_base_denominator": candidate.time_base_denominator,
        "timestamp_seconds": candidate.timestamp_seconds,
        "corners_px": candidate.corners.tolist(),
        "sharpness": candidate.sharpness,
        "png_path": png_path,
        "png_sha256": png_sha256,
        "selection_features": {
            "centroid_x": features.centroid_x,
            "centroid_y": features.centroid_y,
            "scale": features.scale,
            "projective_x": features.projective_x,
            "projective_y": features.projective_y,
            "normalized_full_35_corner_geometry": list(features.normalized_geometry),
        },
    }
