"""End-to-end deterministic publication of non-authoritative extraction evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Final

from .intrinsic_extraction import ExtractionError, build_candidate_pool, scan_frames
from .intrinsic_extraction_evidence import (
    coverage_document,
    evaluation_document,
    json_bytes,
    member_document,
    record_document,
    summary_document,
)
from .intrinsic_extraction_io import contact_sheets, png_bytes, selected_images
from .intrinsic_extraction_types import (
    CountEvaluation,
    ExtractionDependencies,
    ExtractionReceipt,
    ExtractionRequest,
    FitEvaluation,
)

SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_evaluation(evaluation: FitEvaluation) -> None:
    fit = evaluation.fit
    heldout = evaluation.heldout
    values = (
        fit.rms_reprojection_error_px,
        *fit.intrinsics,
        *fit.distortion,
        heldout.rms_error_px,
        heldout.mean_error_px,
        heldout.median_error_px,
        heldout.p95_error_px,
        heldout.max_error_px,
    )
    if (
        len(fit.intrinsics) != 9
        or len(fit.distortion) < 5
        or heldout.corner_count <= 0
        or not all(math.isfinite(value) for value in values)
        or fit.intrinsics[0] <= 0.0
        or fit.intrinsics[4] <= 0.0
    ):
        raise ExtractionError("intrinsic fit did not produce finite positive calibration geometry")


def run_extraction(
    request: ExtractionRequest,
    dependencies: ExtractionDependencies,
) -> ExtractionReceipt:
    """Verify, scan, compare fit sizes, then publish a fresh evidence directory."""
    source = request.source_video.resolve()
    if not source.is_file() or source.is_symlink():
        raise ExtractionError("source video must be an existing non-symlink regular file")
    expected = request.expected_sha256.lower()
    if SHA256_PATTERN.fullmatch(expected) is None:
        raise ExtractionError("expected SHA256 must be exactly 64 lowercase hexadecimal characters")
    source_sha256 = _sha256_file(source)
    if source_sha256 != expected:
        raise ExtractionError(
            f"source SHA256 mismatch: expected {expected}, observed {source_sha256}"
        )
    output = request.output_directory.absolute()
    if output.exists() or output.is_symlink():
        raise ExtractionError("output directory must be a fresh non-existing child path")
    scan = scan_frames(dependencies.decode(source), dependencies.detect)
    pool = build_candidate_pool(scan.candidates, scan.resolution)
    heldout_corners = tuple(item.corners for item in pool.heldout)
    comparisons = tuple(
        CountEvaluation(
            size,
            dependencies.calibrate(
                tuple(item.corners for item in pool.fit_order[:size]),
                heldout_corners,
                scan.resolution,
            ),
        )
        for size in pool.fit_sizes
    )
    for comparison in comparisons:
        _validate_evaluation(comparison.evaluation)
    winner = min(
        comparisons,
        key=lambda item: (
            item.evaluation.heldout.rms_error_px,
            item.evaluation.heldout.p95_error_px,
            item.fit_frame_count,
        ),
    )
    fit_members = pool.fit_order[: winner.fit_frame_count]
    evidence_members = fit_members + pool.heldout
    images = selected_images(
        source,
        tuple(item.frame_index for item in evidence_members),
        dependencies.decode,
    )
    fit_images = images[: len(fit_members)]
    heldout_images = images[len(fit_members) :]
    output.mkdir(parents=False)
    artifacts: dict[str, bytes] = {}
    member_records: list[dict[str, object]] = []
    for role, members, member_images in (
        ("fit", fit_members, fit_images),
        ("heldout", pool.heldout, heldout_images),
    ):
        for rank, (candidate, image) in enumerate(
            zip(members, member_images, strict=True), start=1
        ):
            name = f"{role}-{rank:02d}.png"
            png = png_bytes(image)
            artifacts[name] = png
            member_records.append(
                member_document(
                    candidate,
                    (role, rank, name, hashlib.sha256(png).hexdigest()),
                    (source, source_sha256),
                )
            )
        for sheet_index, sheet in enumerate(contact_sheets(member_images), start=1):
            artifacts[f"{role}-contact-sheet-{sheet_index:02d}.png"] = sheet
    artifacts["selected-frames.json"] = json_bytes(member_records)
    artifacts["scan-records.jsonl"] = b"".join(
        json.dumps(record_document(record), allow_nan=False, sort_keys=True).encode() + b"\n"
        for record in scan.records
    )
    artifacts["rejection-summary.json"] = json_bytes(summary_document(scan, pool))
    artifacts["pairwise-pose-distance.json"] = json_bytes(
        {
            "frame_indices": [item.frame_index for item in pool.heldout + pool.fit_order],
            "matrix": [list(row) for row in pool.pairwise_matrix],
            "coverage": coverage_document(pool),
        }
    )
    comparison_documents = [evaluation_document(item) for item in comparisons]
    artifacts["fit-count-comparison.json"] = json_bytes(comparison_documents)
    for name, content in artifacts.items():
        (output / name).write_bytes(content)
    receipt_document = {
        "schema": "so101-offline-intrinsic-extraction-v2",
        "authoritative": False,
        "source_video": str(source),
        "source_sha256": source_sha256,
        "resolution": list(scan.resolution),
        "total_decoded": scan.summary.total_decoded,
        "fit_frame_count": winner.fit_frame_count,
        "heldout_frame_count": len(pool.heldout),
        "fit_count_comparison": comparison_documents,
        "decision": {
            "primary_metric": "minimum heldout rms_error_px",
            "stable_tie_break": ["minimum heldout p95_error_px", "smaller fit frame count"],
            "winning_fit_frame_count": winner.fit_frame_count,
        },
        "rejections": summary_document(scan, pool),
        "coverage": coverage_document(pool),
        "winning_evaluation": evaluation_document(winner),
        "artifacts": {
            name: hashlib.sha256(content).hexdigest() for name, content in sorted(artifacts.items())
        },
    }
    (output / "extraction-receipt.json").write_bytes(json_bytes(receipt_document))
    return ExtractionReceipt(
        source_sha256,
        winner.fit_frame_count,
        len(pool.heldout),
        scan.summary.total_decoded,
        pool.minimum_pool_distance,
        winner.evaluation,
    )
